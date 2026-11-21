"""The contract between System 2 and System 1: a goal is a set of measurable conditions on the
world plus constraints on how to get there. There are no skills or task names: "pick up the
cube" is whatever sequence of goals System 2 chooses, e.g. jaws open and 6 cm above it, then at
it, then holding it.

Condition types (all must hold, on consecutive fresh observations):
  reach        {"a": "gripper", "b": <entity> | "xyz": [x, y, z], "offset": [dx, dy, dz], "tol": m, "tol_z": m}
  distance     {"a": <entity>, "b": <entity>, "axes": "xy" | "xyz", "min": m, "max": m}
  gripper      {"state": "open" | "closed" | "holding", "object": <entity, for holding>}
  within       {"a": <entity>, "b": <container entity>, "margin": m}
  height       {"a": <entity>, "min": m, "max": m}
  yaw_aligned  {"a": "gripper", "b": <entity>, "tol_deg": deg}
Constraint types:
  min_height   {"z": m}                  the TCP stays at or above z
  region       {"x": [lo, hi], "y": [lo, hi], "z": [lo, hi]}   the TCP stays inside
  avoid        {"entity": <name>, "clearance": m}  stay clear of it unless above it by clearance
  keep_holding {"object": <entity>}      fail at once if the grip is lost
  top_down     {"value": true | false}   true (default): jaws as vertical as reachable (the arm leans
                                         only as much as it must); false: orientation free (rarely useful)
  max_step     {"m": m}                  largest single TCP motion (default 0.04)
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .world import World, jaw_yaw_error

CONDITIONS = {'reach', 'distance', 'gripper', 'within', 'height', 'yaw_aligned'}
CONSTRAINTS = {'min_height', 'region', 'avoid', 'keep_holding', 'top_down', 'max_step'}
JAW_M = 0.03       # metres-equivalent cost of a fully wrong jaw opening
DEG_M = 0.0004     # metres-equivalent cost per degree of jaw misalignment
_ids = itertools.count(1)


class GoalError(ValueError):
    pass


@dataclass
class State:
    """What conditions are evaluated on; candidates predict a modified copy."""
    tcp: np.ndarray
    jaw: float
    jaw_cmd: float | None
    jaw_yaw: float
    holding: frozenset
    pos: dict          # entity -> xyz
    known: dict        # entity -> bool (visible, held, or a recent belief)
    entities: dict     # entity -> Entity (static info: kind, size, height, yaw)

    @classmethod
    def from_world(cls, world: World, belief_ttl: float) -> 'State':
        now = world.t
        pos = {n: e.xyz.copy() for n, e in world.entities.items()}
        # Points and containers are static: the arm hiding them does not make them move.
        known = {n: e.kind in ('point', 'container') or e.visible or e.attached or e.age(now) <= belief_ttl
                 for n, e in world.entities.items()}
        g = world.gripper
        pos['gripper'], known['gripper'] = g.tcp.copy(), True
        held = frozenset(n for n, e in world.entities.items() if e.attached)
        return cls(g.tcp.copy(), g.jaw, g.jaw_cmd, g.jaw_yaw_deg, held, pos, known, dict(world.entities))

    def moved(self, delta=None, jaw_cmd=None, yaw=None) -> 'State':
        s = State(self.tcp.copy(), self.jaw, self.jaw_cmd, self.jaw_yaw, self.holding, dict(self.pos),
                  self.known, self.entities)
        if delta is not None:
            s.tcp = s.tcp + delta
            s.pos['gripper'] = s.tcp
            for n in s.holding:
                s.pos[n] = s.pos[n] + delta
        if jaw_cmd is not None:
            s.jaw_cmd = jaw_cmd
            s.jaw = jaw_cmd
            if jaw_cmd > self.jaw + 0.05:
                s.holding = frozenset()  # opening releases
        if yaw is not None:
            s.jaw_yaw = yaw
        return s


@dataclass
class Eval:
    met: bool | None   # None: not observable right now
    cost: float        # metres-equivalent distance from being met (0 when met)
    text: str


def _vec(v, n, name):
    try:
        a = np.asarray(v, float).reshape(n)
    except (TypeError, ValueError):
        raise GoalError(f'{name} must be a list of {n} numbers')
    if not np.isfinite(a).all():
        raise GoalError(f'{name} must be finite')
    return a


def _num(c, key, default=None, lo=None, hi=None):
    v = c.get(key, default)
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise GoalError(f'{key} must be a number')
    if not math.isfinite(v) or (lo is not None and v < lo) or (hi is not None and v > hi):
        raise GoalError(f'{key}={v} outside [{lo}, {hi}]')
    return v


def holds_object(s: State, name: str) -> bool:
    return name in s.holding


APPROACH_Z = 0.05   # come down onto objects (and into containers) from this far above
GRASP_MIN_Z = 0.018


def between_jaws(s: State, name: str) -> bool:
    p, e = s.pos[name], s.entities[name]
    return np.hypot(*(p[:2] - s.tcp[:2])) < 0.012 and abs(p[2] - s.tcp[2]) < max(0.015, e.height / 2 + 0.005)


def approach(s: State, p: np.ndarray, stop_z: float, hover: float = APPROACH_Z) -> tuple[np.ndarray, float]:
    """Next TCP waypoint and remaining path length for reaching p from above: line up over it at
    hover height first, then go straight down to stop_z."""
    xy = float(np.hypot(*(p[:2] - s.tcp[:2])))
    high = max(stop_z + hover, min(s.tcp[2], stop_z + 0.12))
    if xy > 0.008:
        wp = np.array([p[0], p[1], high])
        return wp, xy + abs(s.tcp[2] - high) + (high - stop_z)
    return np.array([p[0], p[1], stop_z]), xy + abs(s.tcp[2] - stop_z)


class Condition:
    def __init__(self, spec: dict, names: set[str]):
        self.spec = dict(spec)
        self.type = spec.get('type')
        if self.type not in CONDITIONS:
            raise GoalError(f'unknown condition type {self.type!r}; use one of {sorted(CONDITIONS)}')
        c = spec

        def ent(key, default=None, allow_gripper=True):
            v = c.get(key, default)
            if v is None:
                raise GoalError(f'{self.type}: "{key}" is required')
            if v not in names and not (allow_gripper and v == 'gripper'):
                raise GoalError(f'{self.type}: unknown entity {v!r}; tracked: {sorted(names)}')
            return v

        if self.type == 'reach':
            self.a = ent('a', 'gripper')
            if 'xyz' in c:
                self.b, self.xyz = None, _vec(c['xyz'], 3, 'xyz')
            else:
                self.b, self.xyz = ent('b'), None
            self.offset = _vec(c.get('offset', [0, 0, 0]), 3, 'offset')
            if np.abs(self.offset).max() > 0.3:
                raise GoalError('offset components must be within 0.3 m')
            self.tol = _num(c, 'tol', 0.01, 0.003, 0.1)
            self.tol_z = _num(c, 'tol_z', self.tol, 0.003, 0.1)
        elif self.type == 'distance':
            self.a, self.b = ent('a'), ent('b')
            self.axes = c.get('axes', 'xy')
            if self.axes not in ('xy', 'xyz'):
                raise GoalError('distance: axes must be "xy" or "xyz"')
            self.min, self.max = _num(c, 'min', 0.0, 0, 1), _num(c, 'max', 10.0, 0, 10)
            if self.min >= self.max:
                raise GoalError('distance: min must be below max')
        elif self.type == 'gripper':
            self.state = c.get('state')
            if self.state not in ('open', 'closed', 'holding'):
                raise GoalError('gripper: state must be open, closed or holding')
            self.object = ent('object', allow_gripper=False) if self.state == 'holding' else None
        elif self.type == 'within':
            self.a, self.b = ent('a', allow_gripper=False), ent('b', allow_gripper=False)
            self.margin = _num(c, 'margin', 0.01, 0, 0.1)
        elif self.type == 'height':
            self.a = ent('a')
            self.min, self.max = _num(c, 'min', -1.0, -1, 1), _num(c, 'max', 1.0, -1, 1)
        elif self.type == 'yaw_aligned':
            self.a, self.b = ent('a', 'gripper'), ent('b', allow_gripper=False)
            self.tol_deg = _num(c, 'tol_deg', 8, 2, 45)

    def target(self, s: State) -> np.ndarray | None:
        """Where the gripper should go, for conditions that pin one (used to generate candidates)."""
        if self.type == 'reach':
            base = self.xyz if self.b is None else s.pos[self.b]
            goal = base + self.offset
            if self.a == 'gripper':
                return goal
            if self.a in s.holding:  # move the held object there: the TCP goes by the same displacement
                return s.tcp + (goal - s.pos[self.a])
        if self.type == 'gripper' and self.state == 'holding' and self.object not in s.holding and s.known[self.object]:
            return approach(s, s.pos[self.object], self._grasp_z(s))[0]
        if self.type == 'within' and self.a in s.holding and s.known[self.b]:
            b, e = s.pos[self.b], s.entities[self.b]
            return np.array([b[0], b[1], e.height + APPROACH_Z])
        if self.type == 'height' and self.a in ('gripper', *s.holding):
            z = s.pos[self.a][2]
            want = min(max(z, self.min + 0.005), self.max - 0.005)
            return s.tcp + np.array([0, 0, want - z])
        return None

    def evaluate(self, s: State) -> Eval:
        t = self.type
        if t == 'reach':
            if not s.known[self.a] or (self.b and not s.known[self.b]):
                return Eval(None, 0.05, f'{self.a} or {self.b} not observed recently')
            goal = (self.xyz if self.b is None else s.pos[self.b]) + self.offset
            d = s.pos[self.a] - goal
            dxy, dz = float(np.hypot(d[0], d[1])), float(abs(d[2]))
            met = dxy <= self.tol and dz <= self.tol_z
            # Rise before travelling sideways to a higher target, so whatever is carried clears obstacles.
            low = max(0.0, -d[2] - self.tol_z) if dxy > 0.02 else 0.0
            return Eval(met, 0.0 if met else dxy + dz + low, f'{self.a} is {dxy * 100:.1f} cm sideways and '
                        f'{d[2] * 100:+.1f} cm vertically from {self.b or "the point"}{_off(self.offset)}')
        if t == 'distance':
            if not (s.known[self.a] and s.known[self.b]):
                return Eval(None, 0.05, f'{self.a} or {self.b} not observed recently')
            d = s.pos[self.a] - s.pos[self.b]
            dist = float(np.hypot(d[0], d[1]) if self.axes == 'xy' else np.linalg.norm(d))
            cost = max(0.0, dist - self.max) + max(0.0, self.min - dist)
            return Eval(cost == 0, cost, f'{self.a}-{self.b} {self.axes} distance {dist * 100:.1f} cm '
                        f'(want {self.min * 100:.1f}..{min(self.max, 9.99) * 100:.1f} cm)')
        if t == 'gripper':
            if self.state == 'open':
                return Eval(s.jaw >= 0.9, max(0.0, 0.95 - s.jaw) * JAW_M, f'jaws {s.jaw:.2f} open (want >= 0.9)')
            if self.state == 'closed':
                done = s.jaw <= 0.1 or (s.jaw_cmd is not None and s.jaw_cmd <= 0.05)
                return Eval(done, 0.0 if done else s.jaw * JAW_M, f'jaws {s.jaw:.2f} open (want closed)')
            if holds_object(s, self.object):
                return Eval(True, 0.0, f'holding {self.object}; jaws {s.jaw:.2f}')
            if not s.known[self.object]:
                return Eval(None, 0.1, f'not holding {self.object}, which has not been observed recently')
            # Costs fall monotonically along approach -> in position -> closed: in position (at most
            # 0.9 JAW_M) is always cheaper than any point of the approach (at least 1.0 JAW_M, open).
            if between_jaws(s, self.object):  # in position: closing is what is left
                return Eval(False, 0.9 * JAW_M * s.jaw, f'{self.object} is between the jaws; jaws {s.jaw:.2f} open')
            _, path = approach(s, s.pos[self.object], self._grasp_z(s))
            return Eval(False, path + JAW_M * (2.0 - s.jaw),  # stay open until in position
                        f'not holding {self.object}: {path * 100:.1f} cm of approach left; jaws {s.jaw:.2f} open')
        if t == 'within':
            a, b = s.entities[self.a], s.entities[self.b]
            if not s.known[self.b]:
                return Eval(None, 0.05, f'{self.b} not observed recently')
            d = float(np.hypot(*(s.pos[self.a][:2] - s.pos[self.b][:2])))
            r = max(0.005, float(min(b.size)) / 2 - self.margin) if b.size.any() else 0.03
            held = self.a in s.holding
            # Held: carry it over the opening (above the rim), then let go.
            cost = max(0.0, d - r) + (0.02 + max(0.0, b.height + 0.03 - s.tcp[2]) if held else 0.0)
            met = cost == 0 and s.known[self.a]
            how = 'held' if held else ('seen' if a.visible else f'belief: {a.note or "last seen"}')
            if not held and not a.visible and a.release_doubtful:  # do not claim a drop we cannot vouch for
                return Eval(None, cost, f'{self.a}: {a.note}')
            return Eval(met if s.known[self.a] else None, cost,
                        f'{self.a} is {d * 100:.1f} cm from the centre of {self.b} (inside radius {r * 100:.1f} cm; {how})')
        if t == 'height':
            if not s.known[self.a]:
                return Eval(None, 0.05, f'{self.a} not observed recently')
            z = float(s.pos[self.a][2])
            cost = max(0.0, self.min - z) + max(0.0, z - self.max)
            return Eval(cost == 0, cost, f'{self.a} at z = {z * 100:.1f} cm')
        if t == 'yaw_aligned':
            yaw = s.entities[self.b].yaw
            if yaw is None:  # not "unobservable": it becomes known once the wrist camera sees it up close
                return Eval(False, 0.01, f'{self.b} orientation not measured yet (needs a close wrist-camera view)')
            err = jaw_yaw_error(s.jaw_yaw, yaw)
            return Eval(err <= self.tol_deg, 0.0 if err <= self.tol_deg else err * DEG_M,
                        f'jaw axis {err:.0f} deg off {self.b} faces')
        raise AssertionError(t)

    def _grasp_z(self, s: State) -> float:
        return max(GRASP_MIN_Z, float(s.pos[self.object][2]) + 0.005)


def _off(o):
    return '' if not np.any(o) else f' + [{o[0] * 100:.1f}, {o[1] * 100:.1f}, {o[2] * 100:.1f}] cm'


class Constraint:
    def __init__(self, spec: dict, names: set[str]):
        self.spec = dict(spec)
        self.type = spec.get('type')
        c = spec
        if self.type not in CONSTRAINTS:
            raise GoalError(f'unknown constraint type {self.type!r}; use one of {sorted(CONSTRAINTS)}')
        if self.type == 'min_height':
            self.z = _num(c, 'z', None, 0.0, 0.4)
            if self.z is None:
                raise GoalError('min_height needs z')
        elif self.type == 'region':
            self.box = {k: _vec(c.get(k, [-1, 1]), 2, k) for k in 'xyz'}
        elif self.type in ('avoid', 'keep_holding'):
            key = 'entity' if self.type == 'avoid' else 'object'
            self.entity = c.get(key)
            if self.entity not in names:
                raise GoalError(f'{self.type}: unknown entity {self.entity!r}')
            self.clearance = _num(c, 'clearance', 0.03, 0, 0.2)
        elif self.type == 'top_down':
            self.value = bool(c.get('value', True))
        elif self.type == 'max_step':
            self.m = _num(c, 'm', None, 0.003, 0.08)
            if self.m is None:
                raise GoalError('max_step needs m')

    def violated(self, before: State, after: State) -> str | None:
        """Reason a predicted (or observed) state breaks the constraint, else None."""
        p = after.tcp
        if self.type == 'min_height' and p[2] < self.z - 0.002 and p[2] < before.tcp[2] - 1e-4:
            return f'TCP would go below {self.z * 100:.0f} cm'
        if self.type == 'region':
            for i, k in enumerate('xyz'):
                lo, hi = self.box[k]
                if not lo <= p[i] <= hi and abs(p[i] - np.clip(p[i], lo, hi)) > abs(before.tcp[i] - np.clip(before.tcp[i], lo, hi)) - 1e-4:
                    return f'TCP would leave the region on {k}'
        if self.type == 'avoid':
            e = after.entities[self.entity]
            q = after.pos[self.entity]
            r = float(max(e.size)) / 2 + self.clearance if e.size.any() else self.clearance + 0.03
            inside = np.hypot(*(p[:2] - q[:2])) < r and p[2] < e.height + self.clearance
            if inside and after is not before:
                was = np.hypot(*(before.tcp[:2] - q[:2])) < r and before.tcp[2] < e.height + self.clearance
                if not was or np.hypot(*(p[:2] - q[:2])) < np.hypot(*(before.tcp[:2] - q[:2])) - 1e-4:
                    return f'TCP would enter the clearance zone of {self.entity}'
        if self.type == 'keep_holding' and self.entity in before.holding and self.entity not in after.holding:
            return f'would release {self.entity}'
        return None


@dataclass
class Goal:
    description: str
    conditions: list[Condition]
    constraints: list[Constraint]
    timeout_s: float = 90.0
    max_steps: int = 40
    settle: int = 2
    id: str = field(default_factory=lambda: f'g{next(_ids)}')
    created: float = field(default_factory=time.monotonic)
    status: str = 'pending'     # pending | running | succeeded | failed | cancelled
    reason: str = ''
    steps: int = 0
    history: list = field(default_factory=list)   # brief per-step records for System 2
    policy: str = ''

    @property
    def top_down(self) -> bool:
        return next((c.value for c in self.constraints if c.type == 'top_down'), True)

    @property
    def max_step(self) -> float:
        return next((c.m for c in self.constraints if c.type == 'max_step'), 0.04)

    def evaluate(self, s: State) -> list[Eval]:
        return [c.evaluate(s) for c in self.conditions]

    def cost(self, s: State) -> float:
        return sum(e.cost for e in self.evaluate(s))

    def violations(self, before: State, after: State) -> list[str]:
        return [r for c in self.constraints if (r := c.violated(before, after))]

    def terminal(self) -> bool:
        return self.status in ('succeeded', 'failed', 'cancelled')


def parse_goal(spec: dict, world: World) -> Goal:
    if not isinstance(spec, dict):
        raise GoalError('goal must be an object')
    names = set(world.entities)
    conds = spec.get('conditions')
    if not isinstance(conds, list) or not conds:
        raise GoalError('conditions must be a non-empty list')
    if len(conds) > 8:
        raise GoalError('at most 8 conditions per goal; split it into a sequence of goals')
    cons = spec.get('constraints') or []
    if not isinstance(cons, list):
        raise GoalError('constraints must be a list')
    g = Goal(str(spec.get('description') or 'unnamed goal')[:300],
             [Condition(c if isinstance(c, dict) else {}, names) for c in conds],
             [Constraint(c if isinstance(c, dict) else {}, names) for c in cons],
             timeout_s=_num(spec, 'timeout_s', 90, 5, 600), max_steps=int(_num(spec, 'max_steps', 40, 1, 200)))
    return g
