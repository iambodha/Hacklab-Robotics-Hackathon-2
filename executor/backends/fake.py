"""Deterministic fake arm for tests and demos: virtual clock, first-order servos, injectable faults.

Time only advances in sleep() (and advance()), so a test is fast and repeatable. Faults:
  stall={mid: norm}            the joint cannot move past that position (servo stalls short of its goal)
  fault_after={mid: (n, flag)} the motor reports status `flag` after n goal writes
  frozen={"wrist"}             that camera keeps delivering the same frame (same seq, old t_capture)
  table_z=0.01                 with kin: arm joints stop when the TCP would go below this world height (contact)
  object_norm=0.3              gripper closing stops at this opening (something between the jaws)
  images={"overview": jpeg}    camera content (bytes, or a callable returning bytes per frame); set_image() swaps it
Every write is logged in `log` as (t, kind, goals, tag) so tests can prove "zero motion" or "nothing after cancel".
"""
from __future__ import annotations

import threading
import time

import numpy as np

from .base import Backend, FrameSample, MotorRead, placeholder_jpeg

RANGES = {1: (1056, 3040), 2: (975, 3121), 3: (879, 3217), 4: (880, 3216), 5: (126, 3970), 6: (1005, 3091),
          7: (109, 3987), 8: (1225, 2871)}     # calibrated ranges of the follower (scene-contract.md, 22 Nov)


class FakeBackend(Backend):
    name = "fake"

    def __init__(self, start_norm: dict[int, float] | None = None, kin=None, speed: float = 0.6, fps: float = 30.0,
                 stall: dict | None = None, fault_after: dict | None = None, frozen: set | None = None,
                 table_z: float | None = None, object_norm: float | None = None, step_delay: float = 0.0,
                 ranges: dict | None = None, realtime: bool = False, images: dict | None = None):
        self.ranges = dict(ranges or RANGES)
        self.pos = {m: 0.5 for m in self.motor_ids}
        self.pos[8] = 0.9
        self.pos.update(start_norm or {})
        self.goal = dict(self.pos)
        self.kin, self.speed, self.fps = kin, speed, fps
        self.stall, self.fault_after = dict(stall or {}), dict(fault_after or {})
        self.frozen, self.table_z, self.object_norm = set(frozen or ()), table_z, object_norm
        self.step_delay = step_delay
        self.realtime = realtime              # console demo: sleeps take real time (tests: virtual only)
        self.t = 1000.0
        self.status = {m: 0 for m in self.motor_ids}
        self.load = {m: 40 for m in self.motor_ids}
        self.writes = 0
        self.log: list[tuple] = []
        self.trace: list[tuple[float, dict]] = []       # (t, motor norms) after every sleep
        self.tag: str | None = None
        self.lock = threading.RLock()
        self._frozen_at: dict[str, tuple[int, float]] = {}
        self._jpeg = {r: placeholder_jpeg(f"fake {r}") for r in self.cameras}
        self._jpeg.update(images or {})
        self.torque_limit: dict[int, float] = {}

    def set_image(self, role: str, jpeg) -> None:
        with self.lock:
            self._jpeg[role] = jpeg

    # --- clock -------------------------------------------------------------------
    def now(self) -> float:
        return time.monotonic() if self.realtime else self.t

    def advance(self, s: float) -> None:
        with self.lock:
            n = max(1, int(s / 0.01))
            for _ in range(n):
                self._physics(s / n)
                self.t += s / n
            if len(self.trace) < 20000:
                self.trace.append((self.t, dict(self.pos)))

    def sleep(self, s: float) -> None:
        if self.realtime:
            time.sleep(s)
            self.advance(s)
            self.t = time.monotonic()
            return
        self.advance(s)
        if self.step_delay:
            time.sleep(self.step_delay)

    def _norm_q(self, pos):
        return self.kin.q_from_norm(pos)

    def _physics(self, dt: float) -> None:
        new = dict(self.pos)
        for m in self.motor_ids:
            if self.status[m]:
                continue
            d = float(np.clip(self.goal[m] - self.pos[m], -self.speed * dt, self.speed * dt))
            x = self.pos[m] + d
            if m in self.stall:
                lim = self.stall[m]
                if (d > 0 and x > lim >= self.pos[m]) or (d < 0 and x < lim <= self.pos[m]):
                    x = lim
            if m == 8 and self.object_norm is not None and d < 0 and x < self.object_norm <= self.pos[m]:
                x = self.object_norm
            new[m] = x
        if self.kin is not None and self.table_z is not None:
            z = float(self.kin.fk(self._norm_q(new))[0][2])
            if z < self.table_z and z < float(self.kin.fk(self._norm_q(self.pos))[0][2]):
                for m in self.motor_ids[:-1]:
                    new[m] = self.pos[m]          # pressing on the table: the arm does not go further down
        for m in self.motor_ids:
            short = abs(self.goal[m] - new[m]) * (self.ranges[m][1] - self.ranges[m][0])
            self.load[m] = int(min(1000, 40 + (8 * short if short > 8 else 0)))
        self.pos = new

    # --- interface ------------------------------------------------------------------
    def _ticks(self, m, n):
        lo, hi = self.ranges[m]
        return int(round(lo + n * (hi - lo)))

    def read_state(self) -> MotorRead:
        with self.lock:
            out = {}
            for m in self.motor_ids:
                lo, hi = self.ranges[m]
                raw = self._ticks(m, self.pos[m])
                out[m] = {"raw": raw, "goal": self._ticks(m, self.goal[m]), "lo": lo, "hi": hi,
                          "norm": round((raw - lo) / (hi - lo), 4), "in_range": lo <= raw <= hi, "torque": 1,
                          "status": self.status[m], "temp": 31, "volt": 12.4, "load": self.load[m]}
            return MotorRead(self.now(), out)

    def frames(self) -> dict[str, FrameSample]:
        with self.lock:
            out = {}
            joints = [round(self.pos[m], 4) for m in self.motor_ids]
            now = self.now()
            for r in self.cameras:
                seq, t = int(now * self.fps), int(now * self.fps) / self.fps
                if r in self.frozen:
                    seq, t = self._frozen_at.setdefault(r, (seq, t))
                img = self._jpeg[r]
                out[r] = FrameSample(r, f"fake/{r}", seq, t, img() if callable(img) else img, joints)
            return out

    def write_goals(self, goals: dict[int, float]) -> None:
        with self.lock:
            self.writes += 1
            for m, (n, flag) in self.fault_after.items():
                if self.writes >= n:
                    self.status[m] = flag
            for m, n in goals.items():
                self.goal[m] = float(min(1.0, max(0.0, n)))
            self.log.append((self.t, "goals", dict(goals), self.tag))

    def hold(self) -> None:
        with self.lock:
            self.goal.update({m: self.pos[m] for m in self.motor_ids[:-1]})
            self.log.append((self.t, "hold", {}, self.tag))

    def set_torque_limit(self, mid: int, fraction: float) -> None:
        self.torque_limit[mid] = fraction
        self.log.append((self.t, "torque_limit", {mid: fraction}, self.tag))

    def motion_writes(self, tag=None) -> list[tuple]:
        return [e for e in self.log if e[1] == "goals" and (tag is None or e[3] == tag)]
