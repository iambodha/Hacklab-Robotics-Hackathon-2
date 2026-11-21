"""The local goal runner (System 1's loop). One goal at a time:

  observe -> evaluate conditions -> generate candidates -> rank (geometric / Open-Jev)
          -> check freshness and guards -> execute ONE bounded motion -> observe ...

It ends the goal itself, with a reason System 2 can act on: succeeded (every condition met on
consecutive fresh observations), or failed because of a timeout, the step budget, a stall, a
condition that cannot be observed, no candidate that is predicted to help, a constraint
violation (e.g. a dropped object), a motor fault, or an internal error; or cancelled.
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .arm import ArmClient, ArmError
from .candidates import JAW_STEP, Candidate, generate
from .config import Config
from .goals import Goal, State
from .scorers import JevScorer, Ranking, geometric, jev_state
from .world import World

BLOCK_STEPS = 4   # how long a direction stays ruled out after a motion that way was stopped short
MAX_BIAS = 0.015  # largest steady-state (sag) correction folded into a motion command, per axis


class Trace:
    """Append-only JSONL decision log (human side; never shown to System 1)."""

    def __init__(self, root: Path):
        self.dir = root / time.strftime('%Y%m%d-%H%M%S')
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / 'decisions.jsonl'

    def write(self, **rec):
        rec['wall'] = time.strftime('%H:%M:%S')
        with self.path.open('a') as f:
            f.write(json.dumps(rec, default=_jsonable) + '\n')


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.round(4).tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


class Runner:
    def __init__(self, arm: ArmClient, world: World, cfg: Config, jev: JevScorer | None, trace: Trace):
        self.arm, self.world, self.cfg, self.jev, self.trace = arm, world, cfg, jev, trace
        self.lock = asyncio.Lock()
        self.goal: Goal | None = None
        self.task: asyncio.Task | None = None
        self.done = asyncio.Event()
        self.done.set()
        self.cancel_reason: str | None = None
        self.yaw: float | None = None
        self.blocked: list = []
        self.bias = np.zeros(3)
        self.no_grasp = False
        self.stats = {'decisions': 0, 'stale_discarded': 0, 'jev_failures': 0, 'agree': 0, 'compared': 0}

    # --- observation (shared with System 2's look/scene) ------------------------
    async def observe(self):
        async with self.lock:
            frame = await self.arm.frame()
            self.world.update(frame)
            return frame

    # --- goal lifecycle ------------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self.goal is not None and not self.goal.terminal()

    def submit(self, goal: Goal):
        if self.busy:
            raise RuntimeError(f'goal {self.goal.id} is still running; cancel_goal() first')
        goal.policy = self.cfg.policy
        self.goal, self.cancel_reason = goal, None
        self.done.clear()
        self.task = asyncio.create_task(self._run(goal))

    def cancel(self, reason: str = 'cancelled by System 2'):
        if self.busy:
            self.cancel_reason = reason

    async def wait(self, timeout: float) -> bool:
        try:
            await asyncio.wait_for(self.done.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def _run(self, goal: Goal):
        try:
            await self._loop(goal)
        except ArmError as e:
            self._end(goal, 'failed', f'arm error: {e}')
        except Exception as e:  # never leave a goal "running" behind a crashed task
            traceback.print_exc(file=sys.stderr)
            self._end(goal, 'failed', f'internal error: {type(e).__name__}: {e}')
        finally:
            if not goal.terminal():
                self._end(goal, 'failed', 'runner stopped')
            self.done.set()

    def _end(self, goal: Goal, status: str, reason: str):
        if goal.terminal():
            return
        goal.status, goal.reason = status, reason
        self.trace.write(event='goal_end', goal=goal.id, status=status, reason=reason, steps=goal.steps,
                         elapsed_s=round(time.monotonic() - goal.created, 2))

    # --- the loop --------------------------------------------------------------------
    async def _loop(self, goal: Goal):
        goal.status = 'running'
        self.blocked, self.bias, self.no_grasp = [], np.zeros(3), False
        self.trace.write(event='goal_start', goal=goal.id, description=goal.description, policy=goal.policy,
                         conditions=[c.spec for c in goal.conditions], constraints=[c.spec for c in goal.constraints])
        held_at_start = None
        best, since_best, unknown, settled = math.inf, 0, 0, 0
        while True:
            if self.cancel_reason:
                return self._end(goal, 'cancelled', self.cancel_reason)
            if time.monotonic() - goal.created > goal.timeout_s:
                return self._end(goal, 'failed', f'timeout after {goal.timeout_s:.0f} s ({self._status_text(goal)})')
            if goal.steps >= goal.max_steps:
                return self._end(goal, 'failed', f'step budget of {goal.max_steps} used up ({self._status_text(goal)})')

            frame = await self.observe()
            s = State.from_world(self.world, self.cfg.belief_ttl)
            if held_at_start is None:
                held_at_start = s.holding
            evals = goal.evaluate(s)
            cost = sum(e.cost for e in evals)
            self._last_evals = evals
            if goal.history and 'error_after_cm' not in goal.history[-1]:
                goal.history[-1]['error_after_cm'] = round(cost * 100, 1)

            if self.world.gripper.faults:
                return self._end(goal, 'failed', 'motor fault: ' + ', '.join(self.world.gripper.faults))
            for c in goal.constraints:
                if c.type == 'keep_holding' and c.entity in held_at_start and c.entity not in s.holding:
                    return self._end(goal, 'failed', f'lost the grip on {c.entity} ({self.world.entities[c.entity].note})')

            if all(e.met for e in evals):
                settled += 1
                if settled >= goal.settle:
                    return self._end(goal, 'succeeded', '; '.join(e.text for e in evals))
                continue
            settled = 0
            if any(e.met is None for e in evals):
                unknown += 1
                if unknown >= self.cfg.unknown_steps:
                    return self._end(goal, 'failed', 'cannot observe: ' + '; '.join(e.text for e in evals if e.met is None))
            else:
                unknown = 0
            if cost < best - 0.002:
                best, since_best = cost, 0
            else:
                since_best += 1
                if since_best >= self.cfg.stall_steps:
                    return self._end(goal, 'failed', f'stalled: no progress in {since_best} steps ({self._status_text(goal)})')

            self.blocked = [(d, k) for d, k in self.blocked if k > goal.steps - BLOCK_STEPS]
            cands = generate(goal, s, [d for d, _ in self.blocked], no_grasp=self.no_grasp)
            ranking = await self._rank(goal, s, evals, cands, cost)
            age = time.monotonic() - frame.t
            if age > self.cfg.max_decision_age:
                self.stats['stale_discarded'] += 1
                self.trace.write(event='stale', goal=goal.id, age_s=round(age, 2), scorer=ranking.scorer)
                continue
            if self.cancel_reason:
                return self._end(goal, 'cancelled', self.cancel_reason)
            by_id = {c.id: c for c in cands}
            ordered = [by_id[i] for i in ranking.order]
            if not ordered or min(c.cost for c in ordered) >= cost - 1e-4:
                if unknown == 0:
                    return self._end(goal, 'failed', 'no available motion is predicted to help: ' +
                                     '; '.join(e.text for e in evals if not e.met) +
                                     '. Restate the goal in terms of the gripper or a held object.')
                # Something is out of sight: look again, then lift for a clearer view.
                want = 'lift' if unknown >= 2 else 'stay still'
                ordered = [c for c in cands if c.label.startswith(want) and c.feasible] or \
                          [c for c in cands if c.kind == 'hold']
            await self._act(goal, s, cost, ordered, ranking, cands, frame.t)

    async def _rank(self, goal, s, evals, cands, cost) -> Ranking:
        geo = geometric(cands, cost)
        # The exact System 1 model input, kept in the trace for every policy (fine-tuning data).
        self._model_input = {'state': jev_state(goal, s, evals, self.world.summary()),
                             'options': {c.id: c.describe(cost) for c in cands if c.feasible}}
        if goal.policy == 'geometric' or self.jev is None:
            return geo
        try:
            jr = await self.jev.rank(cands, cost, self._model_input['state'])
        except Exception as e:
            self.stats['jev_failures'] += 1
            note = f'Open-Jev unavailable ({type(e).__name__}: {str(e)[:120]})'
            if goal.policy == 'jev' and self.cfg.jev_fallback == 'fail':
                raise RuntimeError(note)
            geo.note = note + '; used the geometric scorer'
            return geo
        self.stats['compared'] += 1
        self.stats['agree'] += int(bool(jr.order and geo.order and jr.order[0] == geo.order[0]))
        if goal.policy == 'shadow':
            geo.shadow = {'jev_choice': jr.order[0] if jr.order else None, 'probabilities': jr.probabilities,
                          'latency_s': round(jr.latency_s, 3)}
            return geo
        jr.shadow = {'geometric_choice': geo.order[0] if geo.order else None}
        return jr

    async def _act(self, goal: Goal, s: State, cost: float, ordered: list[Candidate], ranking: Ranking, cands, t_obs):
        self.stats['decisions'] += 1
        for attempt, c in enumerate(ordered[:3]):
            out = await self._execute(goal, s, c)
            first = out.splitlines()[0] if out else ''
            self.trace.write(event='decision', goal=goal.id, step=goal.steps, scorer=ranking.scorer,
                             error_cm=round(cost * 100, 2), chosen=c.id, attempt=attempt,
                             candidates=[x.record() for x in cands], probabilities=ranking.probabilities,
                             latency_s=round(ranking.latency_s, 3), shadow=ranking.shadow, note=ranking.note,
                             decision_age_s=round(time.monotonic() - t_obs, 2), result=first[:300],
                             model_input=getattr(self, '_model_input', None) if attempt == 0 else None)
            if first.startswith(('UNREACHABLE', 'REFUSED')):
                continue  # nothing moved: try the next-ranked candidate
            goal.steps += 1
            result = _brief(first)
            if c.kind == 'move':
                moved = _tcp_after(out)
                if moved is not None:
                    want, got = float(np.linalg.norm(c.delta)), float((moved - s.tcp) @ c.delta / np.linalg.norm(c.delta))
                    # Servo sag (no gravity compensation) leaves a steady offset: fold it into later commands.
                    self.bias = np.clip(self.bias + (s.tcp + c.delta - moved), -MAX_BIAS, MAX_BIAS)
                    if got < 0.4 * want and want >= 0.015:
                        result = f'blocked: commanded {want * 100:.1f} cm, moved {max(got, 0) * 100:.1f} cm that way'
                        self.blocked.append((c.delta / want, goal.steps))
            goal.history.append({'step': goal.steps, 'action': c.label, 'result': result,
                                 'error_before_cm': round(cost * 100, 1), 'predicted_cm': round(c.cost * 100, 1)})
            return
        goal.steps += 1
        goal.history.append({'step': goal.steps, 'action': 'none', 'result': 'top candidates unreachable',
                             'error_before_cm': round(cost * 100, 1)})

    async def _execute(self, goal: Goal, s: State, c: Candidate) -> str:
        try:
            return await self._command(goal, s, c)
        except ArmError as e:
            if 'Error executing tool' not in str(e):
                raise  # transport/protocol failure: not something another candidate fixes
            return f'REFUSED: {e}'  # the arm rejected the command before moving (limits, ranges)

    async def _command(self, goal: Goal, s: State, c: Candidate) -> str:
        if c.kind == 'hold':
            await asyncio.sleep(0.3)
            return 'held still'
        if c.kind == 'jaw':
            g = self.world.gripper
            cur = g.jaw if g.jaw_cmd is None else g.jaw_cmd
            if c.jaw < cur:
                cur = min(cur, g.jaw + 0.05)  # start from where the jaws really are
            out = ''
            while abs(c.jaw - cur) > 1e-3:
                cur = cur + float(np.clip(c.jaw - cur, -JAW_STEP + 0.01, JAW_STEP - 0.01))
                out = await self.arm.gripper(cur)
                g.jaw_cmd = cur
                if out.startswith('ABORTED'):
                    break
            return out
        if self.yaw is None:
            self.yaw = s.jaw_yaw if self.world.gripper.tilt_deg < 20 else 0.0
        yaw = c.yaw if c.kind == 'yaw' else self.yaw
        target = s.tcp + (c.delta if c.kind == 'move' else 0)
        cmd = target + self.bias
        if target[2] < 0.03:
            cmd[2] = max(cmd[2], target[2])  # near the table, never command lower than asked
        out = await self.arm.move_tcp(cmd, goal.top_down, yaw)
        if c.kind == 'yaw' and out.startswith('reached'):
            self.yaw = c.yaw
        # The arm says when it had to give up on a near-vertical pose; closing there grips badly (as pick_at knows).
        self.no_grasp = 'do not grasp' in out
        return out

    def _status_text(self, goal: Goal) -> str:
        evals = getattr(self, '_last_evals', None) or []
        return '; '.join(e.text for e in evals if not e.met)

    def status(self, goal: Goal | None = None) -> dict:
        goal = goal or self.goal
        if goal is None:
            return {'goal': None}
        out = {'goal': goal.id, 'description': goal.description, 'status': goal.status, 'reason': goal.reason,
               'policy': goal.policy, 'steps': goal.steps, 'elapsed_s': round(time.monotonic() - goal.created, 1),
               'recent': goal.history[-4:]}
        evals = getattr(self, '_last_evals', None)
        if evals and len(evals) == len(goal.conditions):
            out['conditions'] = [{'type': c.type, 'met': e.met, 'now': e.text} for c, e in zip(goal.conditions, evals)]
        return out


def _brief(text: str) -> str:
    if text.startswith('reached'):
        return 'moved as commanded'
    return text[:160]


_TCP_NOW = re.compile(r'tcp now \[(-?[\d.]+), (-?[\d.]+), (-?[\d.]+)\]')


def _tcp_after(text: str) -> np.ndarray | None:
    m = _TCP_NOW.search(text)
    return np.array([float(m[1]), float(m[2]), float(m[3])]) if m else None
