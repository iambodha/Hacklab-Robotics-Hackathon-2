"""System 1's action space: a small, changing set of short motions generated from the current
geometry (never a list of tasks), each with what the arm's kinematic model predicts it does to
every condition. A scorer (geometric or Open-Jev) picks one; the runner executes it and
observes what actually happened.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .geometry import wrap_deg
from .goals import Eval, Goal, State, between_jaws


def jaw_yaw_err(a: float, b: float) -> float:
    """Jaws are symmetric: yaw a and a + 180 are the same grip."""
    return abs(wrap_deg(a - b, 180.0))

MIN_TCP_Z = 0.015      # never put the TCP closer to the table than this
MAX_RADIUS = 0.40      # metres from the base axis
MAX_TCP_Z = 0.35
ENTRY_Z = 0.25         # from outside the working volume (e.g. the rest pose), come in to this height
ARM_LIMIT = 0.44       # move_tcp refuses targets beyond 0.45 m (height or radius)
JAW_STEP = float(os.environ.get('S2_JAW_STEP', 0.35))  # twin: max change per gripper call; real arm: 1 (one call)
GRIP_JAW = 0.25        # where closing jaws stall on a small object (approximate)


@dataclass
class Candidate:
    id: str
    kind: str                       # move | jaw | yaw | hold
    label: str
    delta: np.ndarray | None = None
    jaw: float | None = None
    yaw: float | None = None
    cost: float = 0.0               # predicted total condition cost afterwards
    evals: list[Eval] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return not self.violations

    def describe(self, cost_now: float) -> str:
        """Plain words for the model; measured: Open-Jev-2B ranks plain effects far better than numbers."""
        d = self.cost - cost_now
        effect = ('reaches the goal' if self.cost < 1e-4 else
                  f'gets {-d * 100:.1f} cm closer to the goal' if d < -1e-4 else
                  'makes no progress' if d <= 1e-4 else f'moves {d * 100:.1f} cm away from the goal')
        return f'{self.label}. This {effect}.'

    def record(self) -> dict:
        r = {'id': self.id, 'kind': self.kind, 'label': self.label, 'predicted_cost': round(self.cost, 4)}
        if self.delta is not None:
            r['delta_m'] = [round(float(v), 4) for v in self.delta]
        if self.jaw is not None:
            r['jaw'] = round(self.jaw, 3)
        if self.yaw is not None:
            r['yaw_deg'] = round(self.yaw, 1)
        if self.violations:
            r['violations'] = self.violations
        return r


def predict(goal: Goal, s: State, cand: Candidate) -> State:
    if cand.kind == 'move':
        return s.moved(delta=cand.delta)
    if cand.kind == 'yaw':
        return s.moved(yaw=cand.yaw)
    if cand.kind == 'jaw':
        after = s.moved(jaw_cmd=cand.jaw)
        if cand.jaw < s.jaw - 0.05:  # closing on something between the jaws grips it
            inside = [n for n, e in s.entities.items() if e.kind == 'object' and s.known.get(n) and between_jaws(s, n)]
            if inside:
                after.holding = frozenset(set(s.holding) | {inside[0]})
                after.jaw = max(cand.jaw, GRIP_JAW)
        else:
            after.holding = frozenset()
        return after
    return s


def _guard(s: State, after: State, goal: Goal) -> list[str]:
    """Hard limits. Each only forbids making things worse, so the arm can always move back in."""
    v = goal.violations(s, after)
    p, q = after.tcp, s.tcp
    if p[2] < MIN_TCP_Z - 1e-4 and p[2] < q[2] - 1e-4:
        v.append(f'TCP below the {MIN_TCP_Z * 100:.1f} cm table margin')
    if p[2] > MAX_TCP_Z and p[2] > q[2] + 1e-4:
        v.append('TCP above the working height')
    if np.hypot(p[0], p[1]) > MAX_RADIUS and np.hypot(p[0], p[1]) > np.hypot(q[0], q[1]) + 1e-4:
        v.append('outside the arm reach')
    if p[2] > ARM_LIMIT or np.hypot(p[0], p[1]) > ARM_LIMIT:
        v.append('outside the region the arm accepts targets in')
    return v


def generate(goal: Goal, s: State, blocked: list[np.ndarray] = (), no_grasp: bool = False) -> list[Candidate]:
    """blocked: unit directions in which a recent motion was stopped short (something in the way).
    no_grasp: the arm reported that the gripper is too tilted here to grasp."""
    raw: list[Candidate] = []
    max_step = goal.max_step
    targets = [t for c in goal.conditions if (t := c.target(s)) is not None]

    def move(label, delta):
        delta = np.asarray(delta, float)
        if np.linalg.norm(delta) < 0.0015:
            return
        z = s.tcp[2] + delta[2]
        if z < MIN_TCP_Z:  # clamp onto the safety margin rather than drop the direction
            delta[2] = MIN_TCP_Z - s.tcp[2]
            if np.linalg.norm(delta) < 0.0015:
                return
        raw.append(Candidate('', 'move', label, delta=delta))

    if s.tcp[2] > MAX_TCP_Z:  # e.g. the rest pose, jaws up: nothing is near, come into the working volume
        entry = targets[0] if targets else np.array([s.tcp[0], max(s.tcp[1], 0.12), ENTRY_Z])
        raw.append(Candidate('', 'move', 'come down into the working volume above the target area',
                             delta=np.array([entry[0], entry[1], max(ENTRY_Z, entry[2])]) - s.tcp))
    for k, tgt in enumerate(targets[:2]):
        d = tgt - s.tcp
        n = float(np.linalg.norm(d))
        if n < 0.002:
            continue
        sizes = sorted({min(n, max_step), min(n, 0.012), min(n, 0.004)}, reverse=True)
        for size in sizes:
            move('all the way to the target' if size >= n - 1e-6 else f'{size * 100:.1f} cm toward the target',
                 d / n * size)
        nxy = float(np.hypot(d[0], d[1]))
        if nxy > 0.003 and abs(d[2]) > 0.003:
            move('sideways toward the target at the current height', np.r_[d[:2] / nxy * min(nxy, max_step), 0])
            move('straight ' + ('down' if d[2] < 0 else 'up') + ' toward the target height',
                 [0, 0, np.sign(d[2]) * min(abs(d[2]), max_step)])
    step = float(np.clip(min((np.linalg.norm(t - s.tcp) for t in targets), default=0.02) / 2, 0.005, min(0.02, max_step)))
    for axis, name in enumerate('xyz'):
        for sign in (1, -1):
            dv = np.zeros(3)
            dv[axis] = sign * step
            move(f'{"+" if sign > 0 else "-"}{name} {step * 100:.1f} cm', dv)
    move('lift 3 cm (clearer view, away from the table)', [0, 0, min(0.03, max_step)])

    jaw_now = s.jaw if s.jaw_cmd is None else s.jaw_cmd
    if s.jaw < 0.95 or (s.jaw_cmd is not None and s.jaw_cmd < 0.95):
        raw.append(Candidate('', 'jaw', 'open the jaws fully', jaw=1.0))
    if jaw_now > 0.02:
        raw.append(Candidate('', 'jaw', 'close the jaws fully (stops when they grip something)', jaw=0.0))
        if jaw_now > JAW_STEP + 0.05:
            raw.append(Candidate('', 'jaw', f'close the jaws part way to {jaw_now - JAW_STEP:.2f}', jaw=jaw_now - JAW_STEP))

    for c in goal.conditions:
        if c.type == 'yaw_aligned' and s.entities[c.b].yaw is not None:
            obj, p = s.entities[c.b].yaw, s.pos[c.b]
            # A square-ish object can be gripped every 90 deg. Prefer jaws closing tangentially (across
            # the reach direction): when a later pose must tilt the gripper (far or high), the closing
            # axis then stays level and the object drops out cleanly on release instead of resting on
            # the lower jaw. Also offer the smallest turn, in case the preferred one is unreachable.
            tangential = np.degrees(np.arctan2(p[1], p[0])) + 90.0
            options = [obj + 90.0 * k for k in range(-3, 4)]
            for want, why in ((min(options, key=lambda a: abs(wrap_deg(a - tangential, 180.0))), 'across the reach direction'),
                              (s.jaw_yaw + wrap_deg(obj - s.jaw_yaw, 90.0), 'smallest turn')):
                if jaw_yaw_err(s.jaw_yaw, want) > 2:
                    raw.append(Candidate('', 'yaw', f'turn the jaws to {wrap_deg(want):.0f} deg to line up with {c.b} '
                                                    f'({why})', yaw=wrap_deg(want)))
    raw.append(Candidate('', 'hold', 'stay still and look again'))

    out, seen = [], set()
    for cand in raw:
        key = (cand.kind, None if cand.delta is None else tuple(np.round(cand.delta, 4)), cand.jaw, cand.yaw)
        if key in seen:
            continue
        seen.add(key)
        after = predict(goal, s, cand)
        cand.evals = goal.evaluate(after)
        cand.cost = sum(e.cost for e in cand.evals)
        cand.violations = [] if cand.kind == 'hold' else _guard(s, after, goal)
        if cand.kind == 'move' and any(cand.delta @ b > 0.85 * np.linalg.norm(cand.delta) for b in blocked):
            cand.violations.append('a recent motion this way was stopped short: something is in the way')
        if no_grasp and cand.kind == 'jaw' and cand.jaw < s.jaw - 0.05:
            cand.violations.append('the gripper is too tilted here to grasp (reported by the arm)')
        cand.id = f'c{len(out) + 1}'
        out.append(cand)
    return out
