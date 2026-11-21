"""Motion primitives shared by every backend: creeping joint moves with tracking/contact/fault aborts and a
closed-loop settle, a force-limited gripper with touch, and a guarded descent. Ported from sim/real_arm_mcp.py
(_drive, gripper, _probe_hold, _descend_guarded) with three differences:

  * every result is structured and truthful: a move that ends outside tolerance is PARTIAL with per-joint
    residuals, never "reached";
  * cancel is checked before every single write; after cancel nothing but one hold is sent;
  * waypoints are given as joint vectors that the skill has already solved (whole path preflighted).
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import numpy as np

from .contracts import Status


class Cancelled(Exception):
    pass


@dataclass
class MotionResult:
    status: Status
    termination: str
    residual_ticks: dict = field(default_factory=dict)
    contact: bool = False

    @property
    def ok(self) -> bool:
        return self.status == Status.SUCCEEDED


@dataclass
class GripResult:
    status: Status
    termination: str
    norm: float
    load: int
    holding: bool | None
    contact: bool


class Motion:
    def __init__(self, backend, kin, cancel: threading.Event, log=None):
        self.b, self.kin, self.cancel = backend, kin, cancel
        self.P = kin.P
        self.MO, self.GR, self.TS = self.P.motion, self.P.gripper, self.P.touch
        self.log = log or (lambda *a, **k: None)
        self.track_tol = 250
        self.empty_closes: list[tuple[float, int]] = [(self.GR.empty_closed_norm, None)]

    # --- helpers ---------------------------------------------------------------------
    def read(self) -> dict:
        return self.b.read_state().motors

    def q_now(self, st=None):
        st = st or self.read()
        return self.kin.q_from_norm({m: st[m]["norm"] for m in self.kin.ids}, {m: st[m]["raw"] for m in self.kin.ids})

    def _check_cancel(self):
        if self.cancel.is_set():
            try:
                self.b.hold()
            except Exception as e:  # noqa: BLE001  (disabled backend: nothing was moving)
                self.log("hold after cancel failed", error=str(e))
            raise Cancelled()

    def _ticks(self, s, n):
        return s["lo"] + float(n) * (s["hi"] - s["lo"])

    def _faults(self, st, plan, grip=False):
        G = self.kin.grip_id
        return [(self.kin.name_of[m], hex(st[m]["status"])) for m in self.kin.ids
                if st[m]["status"] and (m != G or (G in plan and not grip))]

    # --- joint moves --------------------------------------------------------------------
    def drive(self, plan: dict[int, float], contact: bool = True, settle: bool = True, grip: bool = False,
              window: int | None = None, iters: int | None = None, dt: float | None = None) -> MotionResult:
        """plan {motor id: target norm}. Creep, then settle closed-loop. SUCCEEDED only if measured within settle_tol."""
        MO, G = self.MO, self.kin.grip_id
        dt = MO.dt if dt is None else dt
        st = self.read()
        if any(st[m]["torque"] == 0 for m in self.kin.ids):
            return MotionResult(Status.FAULT, "torque is off on some motors")
        for m in plan:
            if not st[m]["in_range"]:
                return MotionResult(Status.FAULT, f"{self.kin.name_of[m]} raw {st[m]['raw']} outside its calibrated range")
        f = self._faults(st, plan, grip)
        if f:
            return MotionResult(Status.FAULT, f"motor fault before moving {f}")
        target = {m: min(1.0, max(0.0, float(n))) for m, n in plan.items()}
        start = {m: st[m]["norm"] for m in plan}
        span = {m: st[m]["hi"] - st[m]["lo"] for m in plan}
        n = max([1] + [math.ceil(abs(target[m] - start[m]) * span[m] / min(MO.step * span[m], MO.max_ticks_per_step))
                       for m in plan])
        for k in range(1, n + 1):
            self._check_cancel()
            goals = {m: start[m] + (target[m] - start[m]) * k / n for m in plan}
            self.b.write_goals(goals)
            self.b.sleep(dt)
            now = self.read()
            bad = {self.kin.name_of[m]: int(round(self._ticks(now[m], goals[m]) - now[m]["raw"])) for m in plan
                   if abs(self._ticks(now[m], goals[m]) - now[m]["raw"]) > self.track_tol}
            faults = self._faults(now, plan, grip)
            hot = [(self.kin.name_of[m], now[m]["load"]) for m in plan if contact and m != G and now[m]["load"] > MO.load_abort]
            if bad or faults or hot:
                self._check_cancel()
                self.b.hold()
                res = self.residuals(now, target)
                if faults:
                    return MotionResult(Status.FAULT, f"motor fault {faults} at step {k}/{n}", res)
                if bad:
                    return MotionResult(Status.PARTIAL, f"stall: tracking error {bad} ticks at step {k}/{n}", res)
                return MotionResult(Status.PARTIAL, f"contact abort: load over {MO.load_abort} {hot} at step {k}/{n}", res, True)
        if not settle:
            return MotionResult(Status.SUCCEEDED, f"passed in {n} steps", self.residuals(self.read(), target))
        cmd = {m: self._ticks(st[m], target[m]) for m in plan}
        window = MO.settle_window if window is None else window
        now = self.read()
        for it in range(MO.settle_iters if iters is None else iters):
            self.b.sleep(MO.settle_wait_s)
            now = self.read()
            err = {m: self._ticks(now[m], target[m]) - now[m]["raw"] for m in plan}
            if all(abs(e) <= MO.settle_tol for e in err.values()):
                return MotionResult(Status.SUCCEEDED, f"reached in {n} steps (+{it} corrections)", self.residuals(now, target))
            hot = [(self.kin.name_of[m], now[m]["load"]) for m in plan if contact and m != G and now[m]["load"] > MO.load_abort]
            faults = [] if hot else self._faults(now, plan, grip)
            if hot or faults:
                self._check_cancel()
                self.b.hold()
                if faults:
                    return MotionResult(Status.FAULT, f"motor fault while settling {faults}", self.residuals(now, target))
                return MotionResult(Status.PARTIAL, f"contact abort while settling {hot}", self.residuals(now, target), True)
            for m, e in err.items():          # P-only servos stop short under load: push the goal a little past
                if abs(e) > MO.settle_tol:
                    goal_t = self._ticks(now[m], target[m])
                    c = max(goal_t - window, min(goal_t + window, cmd[m] + MO.settle_gain * e))
                    cmd[m] = max(now[m]["lo"] + MO.range_margin, min(now[m]["hi"] - MO.range_margin, c))
            self._check_cancel()
            self.b.write_goals({m: (cmd[m] - now[m]["lo"]) / (now[m]["hi"] - now[m]["lo"]) for m in plan})
        res = self.residuals(now, target)
        return MotionResult(Status.PARTIAL, f"short: stopped {res} ticks from the target after settling", res)

    def residuals(self, st, target: dict[int, float]) -> dict[str, int]:
        return {self.kin.name_of[m]: int(round(self._ticks(st[m], n) - st[m]["raw"])) for m, n in target.items()}

    def go_q(self, q, settle: bool = True, dt: float | None = None, contact: bool = True) -> MotionResult:
        """Arm joints to q (rad) in chunks of at most motion.chunk norm per joint; the last chunk settles."""
        MO = self.MO
        goal = self.kin.norm_from_q(q, self._ranges())
        res = MotionResult(Status.SUCCEEDED, "already there")
        for _ in range(MO.chunk_tries):
            st = self.read()
            now = {m: st[m]["norm"] for m in goal}
            gap = max(abs(goal[m] - now[m]) for m in goal)
            if gap < MO.arrive_tol and not settle:
                return MotionResult(Status.SUCCEEDED, "already there", self.residuals(st, goal))
            last = gap <= MO.chunk
            plan = {m: now[m] + float(np.clip(goal[m] - now[m], -MO.chunk, MO.chunk)) for m in goal}
            res = self.drive(plan, settle=settle and last, dt=dt, contact=contact)
            if res.status != Status.SUCCEEDED or last:
                return res
        return MotionResult(Status.PARTIAL, f"not there after {MO.chunk_tries} chunks", self.residuals(self.read(), goal))

    def _ranges(self):
        st = self.read()
        return {m: (st[m]["lo"], st[m]["hi"]) for m in self.kin.ids}

    # --- gripper ----------------------------------------------------------------------------
    def thin_min(self) -> float:
        return float(np.median([e[0] for e in self.empty_closes])) + self.GR.thin_margin

    def gripper(self, open_fraction: float, force: float | None = None) -> GripResult:
        GR, G = self.GR, self.kin.grip_id
        st = self.read()
        if st[G]["torque"] == 0 or not st[G]["in_range"]:
            return GripResult(Status.FAULT, "gripper torque off or outside its range", st[G]["norm"], st[G]["load"], None, False)
        force = float(min(1.0, max(GR.force_min, GR.force_default if force is None else force)))
        target = min(GR.open_max, max(0.0, float(open_fraction)))
        start = st[G]["norm"]
        closing = target < start
        self._check_cancel()
        self.b.set_torque_limit(G, max(GR.torque_min / GR.torque_full, force) if closing else 1.0)
        span = st[G]["hi"] - st[G]["lo"]
        n = max(1, math.ceil(abs(target - start) / (GR.close_step_fraction if closing else GR.step_fraction)))
        prev, still, contact = st[G]["raw"], 0, False
        for k in range(1, n + 1):
            self._check_cancel()
            goal = start + (target - start) * k / n
            self.b.write_goals({G: goal})
            self.b.sleep(GR.step_wait_s)
            now = self.read()[G]["raw"]
            still = still + 1 if abs(now - prev) < GR.stall_still else 0
            if closing and now - (st[G]["lo"] + goal * span) > GR.contact_lag and still >= GR.contact_reads:
                contact = True
                break
            prev = now
        self.b.sleep(GR.after_wait_s)
        st = self.read()
        g = st[G]
        if closing and (contact or g["raw"] - g["goal"] > GR.contact_lag) and g["norm"] > GR.closed_norm:
            squeeze = max(GR.squeeze_min_ticks, int(round(GR.squeeze_ticks * force)))
            self._check_cancel()
            self.b.write_goals({G: (g["raw"] - squeeze - g["lo"]) / span})
            self.b.sleep(GR.squeeze_wait_s)
            held, st = self.probe_hold()
            return GripResult(Status.SUCCEEDED, "jaws stalled on something: holding" if held else "touched but slipped out",
                              st[G]["norm"], st[G]["load"], held, True)
        if closing and g["norm"] <= GR.closed_norm:
            held, st = self.probe_hold()
            if not held:
                self.empty_closes.append((float(g["norm"]), int(g["load"])))
                del self.empty_closes[:-GR.empty_memory]
                self._check_cancel()
                self.b.write_goals({G: g["norm"]})    # closed on nothing: stop pushing
            return GripResult(Status.SUCCEEDED, "closed on something thin" if held else "closed fully: nothing between the jaws",
                              st[G]["norm"], st[G]["load"], held, held)
        reached = abs(g["norm"] - target) * span <= max(GR.open_stop_slack, self.MO.settle_tol) or (not closing and g["norm"] >= GR.open_enough)
        if not closing and g["goal"] - g["raw"] > GR.open_stop_slack:
            self._check_cancel()
            self.b.write_goals({G: g["norm"]})        # opened against the mechanical stop: stop pushing
        return GripResult(Status.SUCCEEDED if reached else Status.PARTIAL,
                          f"jaws at {g['norm']:.3f} (target {target:.3f})", g["norm"], g["load"], False if not closing else None, False)

    def probe_hold(self):
        """Close a little further: free jaws follow, held jaws do not (real_arm_mcp._probe_hold)."""
        GR, G = self.GR, self.kin.grip_id
        st = self.read()
        g = st[G]
        if g["norm"] >= GR.open_enough or g["norm"] < self.thin_min():
            return False, st
        span = g["hi"] - g["lo"]
        raw0 = g["raw"]
        self._check_cancel()
        self.b.write_goals({G: (raw0 - GR.probe_ticks - g["lo"]) / span})
        self.b.sleep(GR.probe_wait_s)
        st = self.read()
        moved = raw0 - st[G]["raw"]
        holding = moved < GR.probe_moved_fraction * GR.probe_ticks and st[G]["norm"] >= self.thin_min()
        if holding and st[G]["norm"] <= GR.closed_norm:
            loads = [e[1] for e in self.empty_closes if e[1] is not None]
            if loads and g["load"] < float(np.median(loads)) + GR.thin_load_margin:
                holding = False
        if holding:
            self._check_cancel()
            self.b.write_goals({G: (st[G]["raw"] - GR.squeeze_min_ticks - st[G]["lo"]) / span})
        return holding, st

    # --- guarded descent ------------------------------------------------------------------------
    def descend_guarded(self, path: list, dt: float | None = None):
        """path: preflighted joint vectors going straight down in small steps (skills.descent_path).
        Returns (MotionResult, contact_z_world or None). Contact = a joint stays short while the two most loaded
        arm motors carry more than touch.load_sum, on touch.reads steps in a row; then stop pushing and back off."""
        TS, A = self.TS, self.kin.arm_ids
        hits = 0
        for i, q in enumerate(path):
            norm = self.kin.norm_from_q(q, self._ranges())
            res = self.drive(norm, settle=True, contact=False, window=TS.window, iters=TS.iters, dt=TS.dt if dt is None else dt)
            if res.status in (Status.FAULT, Status.CANCELLED) or (res.status == Status.PARTIAL and not res.termination.startswith("short")):
                return res, None
            st = self.read()
            meas = float(self.kin.fk(self.q_now(st))[0][2])
            short = max(abs(self._ticks(st[m], norm[m]) - st[m]["raw"]) for m in A)
            loads = sorted((st[m]["load"] for m in A), reverse=True)
            hits = hits + 1 if (short >= TS.short_ticks and loads[0] + loads[1] >= TS.load_sum) else 0
            if hits >= TS.reads:
                self._check_cancel()
                self.b.hold()
                return MotionResult(Status.SUCCEEDED, f"touched at z={meas:.3f} (short {short:.0f} ticks, loads {loads[0]}+{loads[1]})",
                                    contact=True), meas
        return MotionResult(Status.SUCCEEDED, "reached the bottom of the descent without contact"), None
