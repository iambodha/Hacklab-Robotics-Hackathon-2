"""The cap-unscrewing skill: twist_cap, check_cap and extract_cap, written once against a small rig interface.

A Rig is what each MCP server provides (sim/real_arm_mcp.RealRig, sim/sim_arm_mcp.SimRig):

    async state() -> {joint name: {"raw", "goal", "load", "norm", "status", ...}}   live motor rows
    async q() -> joint angles (rad) [arm..., gripper]
    fk(q) -> (tcp xyz, tcp rotation; column 0 = jaw closing axis, column 1 = approach axis)
    solve(xyz, R, seed) -> q or None         position + orientation IK (synchronous; run in a thread)
    async move_down(x, y, z, yaw_deg) -> text    move_tcp with jaws down (rises first for long moves)
    async reachable(x, y, z, yaw_deg, seed) -> signed lean_deg or None (> 0: tilted away from the base; no motion)
    async go(q, dt, contact) -> text         drive the arm joints to q (one short step, gripper goal kept);
                                             text starts with "ABORTED" on a tracking error, fault or contact
    async descend(x, y, z_to, yaw_deg, top) -> (text, contact_z or None)    guarded descent from `top`
    async grip(open_fraction, force) -> (text, holding)    torque-limited close (or open)
    async holding() -> bool
    async frames() -> {"wrist": jpeg, "overview": jpeg}     raw camera frames (no grid) for the records
    j8 raw ticks fall as the jaws close (true on the ElRobot and in the twin)
    twist_sign: +1 if a positive rotation about +z in the model is counter-clockwise seen from above on this rig

Twist = rotation of the gripper about the VERTICAL axis through the TCP (the cap axis), solved by IK at each small
step: R_k = Rz(sign * theta_k) @ R_grasp, p_k = p_grasp + lift. With the jaws pointing straight down this is a pure
wrist roll (j7); with a leaning gripper the other wrist joints help. Safety limits from episodes.SAFETY are
enforced here and cannot be changed by the parameters.

Every strategy number is a skill-config key (learning/skillcfg.py; the learned defaults in
learning/profiles/bottle_cap.json). A Twister reads the active config once per call, so a new version applies from
the next call. The module constants below are the CODE BASELINES of those keys, kept for readers of r3/r4; what is
left as a plain constant is a physical fact of the gripper, a safety stop or a settle time (each named, with its
reason).
"""
from __future__ import annotations

import asyncio
import math
import time

import numpy as np

from . import capfree
from . import skillcfg as SC
from .episodes import SAFETY

_T, _L = SC.TWIST_DEFAULTS, SC.LIFT_DEFAULTS
# --- code baselines of skill-config keys (the Twister uses self.c[...], never these) ---
STEP_DEG = {"slow": _T["twist.step_deg.slow"], "medium": _T["twist.step_deg.medium"]}
STEP_DT = {"slow": _T["twist.step_dt.slow"], "medium": _T["twist.step_dt.medium"]}
HOVERS = tuple(_T["twist.hovers_m"])    # m above the cap top before the descent (highest reachable)
MAX_LEAN = _T["twist.max_lean_deg"]     # deg from vertical: steeper poses are not used for the cap
SIDE_LEAN = _T["twist.side_lean_deg"]   # auto approach: a grasp pose leaning more than this is approached
                                        # from the side (22 Nov: 90 = always from above; see skillcfg)
SIDE_BACK = _T["twist.side_back_m"]
SIDE_STEP = _T["twist.side_step_m"]
SIDE_CONTACT = _T["twist.side_contact_m"]
GUARD_ABOVE = _T["twist.guard_above_m"]
CONTACT_OK = _T["twist.contact_ok_m"]
GRASP_BELOW_TOP = _T["twist.grasp_below_top_m"]
LAG_STALL_DEG = _T["twist.lag_stall_deg"]
RELEASE_OPEN = _T["twist.release_open"]
RETREAT_UP = _T["twist.retreat_up_m"]
CHECK_LIFT = _L["check.lift_m"]
TILTS_DEG = tuple(_T["twist.ik_tilts_deg"])
# Physical geometry and immutable safety/continuity stops.
FINGERTIP_BELOW = 0.010      # fingertips extend 10 mm below TCP (table contact geometry)
CHECK_LOAD_STOP = 400        # maximum increase of the two largest arm loads during a lift
MAX_JOINT_JUMP_RAD = 0.5     # reject IK solutions on another arm branch

# Compatibility aliases for earlier callers; all strategy baselines have one source.
_B = SC.baseline()
OPEN_FULL = _B["twist.open_full"]
OPEN_ENOUGH = _B["twist.open_enough"]
RETREAT_EXTRA_OPEN = _B["twist.retreat_extra_open"]
TURN_BACK_MIN_DT = _B["twist.turn_back_min_dt_s"]
SIDE_STEP_DT = _B["twist.side_step_dt_s"]
SIDE_SETTLE_S = _B["twist.side_settle_s"]
STROKE_SETTLE_S = _B["twist.stroke_settle_s"]
LIFT_SETTLE_S = _B["lift.settle_s"]
NOTE_LEAN_DEG = _B["twist.note_lean_deg"]


def rz(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_of(R) -> float:
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


def lean_of(R) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, -float(R[2, 1])))))


def wrap(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


class Twister:
    """One call of the skill. `log(phase, status, measurements, note)` writes the episode phase records.
    prior_strokes: the strokes earlier twist_cap calls of the same attempt made (a resumed call continues them: the
    stroke count, the free signal and its reference load are per attempt)."""

    def __init__(self, rig, params: dict, log, j7: str = "j7", j8: str = "j8", arm_joints=None, clock=time.monotonic,
                 prior_strokes=None, resume_safe=True):
        self.rig, self.p, self.log = rig, dict(params), log
        self.j7, self.j8 = j7, j8
        self.arm_joints = arm_joints or [f"j{i}" for i in range(1, 8)]
        self.clock = clock
        self.trace: list[dict] = []
        self.t0 = clock()                       # the call's start: the time budget counts from here
        self.strokes: list[dict] = []
        self.prior = list(prior_strokes or [])
        self.resume_safe = resume_safe
        self.cap = None
        self.cfg = SC.active()                 # the skill-config version this call runs with (one snapshot)
        self.c = self.cfg["values"]

    # --- helpers -------------------------------------------------------------------------------------------
    async def _pose(self):
        q = await self.rig.q()
        pos, R = self.rig.fk(q)
        return q, np.asarray(pos, float), np.asarray(R, float)

    async def _sample(self, phase: str, stroke: int, cmd_deg: float, yaw0: float) -> dict:
        st = await self.rig.state()
        q, pos, R = await self._pose()
        row = {"t": round(self.clock() - self.t0, 3), "phase": phase, "stroke": stroke, "cmd_deg": round(cmd_deg, 2),
               "yaw_deg": round(self.rig.twist_sign * wrap(yaw_of(R) - yaw0), 2),
               "j7_raw": st[self.j7]["raw"], "j7_goal": st[self.j7]["goal"], "j7_load": st[self.j7]["load"],
               "j8_raw": st[self.j8]["raw"], "j8_load": st[self.j8]["load"], "j8_norm": st[self.j8]["norm"],
               "tcp_z": round(float(pos[2]), 4),
               "arm_load_max": max(st[j]["load"] for j in self.arm_joints if j != self.j7),
               "arm_load_sum": sum(sorted((st[j]["load"] for j in self.arm_joints), reverse=True)[:2]),
               "status": [st[j]["status"] for j in self.arm_joints + [self.j8]]}
        self.trace.append(row)
        return row

    def _solve(self, xyz, R, seed):
        """IK for (xyz, R); where that is just out of reach (jaws-down poses sit near a wrist joint limit on this arm),
        the same point with the gripper tilted a few degrees about the horizontal axis across the arm's reach."""
        sol = self.rig.solve(xyz, R, seed)
        if sol is not None:
            return sol
        r = np.asarray(xyz[:2], float)
        r = r / max(float(np.linalg.norm(r)), 1e-6)
        axis = np.array([-r[1], r[0], 0.0])
        for t in self.c["twist.ik_tilts_deg"]:
            a = math.radians(t)
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            sol = self.rig.solve(xyz, (np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K) @ R, seed)
            if sol is not None:
                return sol
        return None

    async def _plan(self, R0, p0, q0, deg_from: float, deg_to: float, step: float, lift_per_turn_mm: float):
        """IK waypoints rotating from deg_from to deg_to (about vertical, signed so that + = counter-clockwise) and
        rising with the thread. Truncated where IK fails; returns [(deg, q)]."""
        sign = self.rig.twist_sign
        n = max(1, int(math.ceil(abs(deg_to - deg_from) / step)))

        def run():
            out, seed = [], q0
            for k in range(1, n + 1):
                d = deg_from + (deg_to - deg_from) * k / n
                xyz = p0 + np.array([0.0, 0.0, lift_per_turn_mm / 1000.0 * (d - deg_from) / 360.0])
                sol = self._solve(xyz, rz(sign * (d - deg_from)) @ R0, seed)
                if sol is None or float(np.max(np.abs(np.asarray(sol)[:7] - np.asarray(seed)[:7]))) > MAX_JOINT_JUMP_RAD:
                    break
                out.append((d, sol))
                seed = sol
            return out
        return await asyncio.to_thread(run)

    async def _retreat(self, why: str) -> str:
        """Abort: open the jaws (they only touch the cap), then straight up."""
        out = []
        try:
            st = await self.rig.state()
            if st[self.j8]["norm"] < self.c["twist.open_enough"]:
                wide = min(self.c["twist.open_full"], st[self.j8]["norm"] + self.c["twist.release_open"] + self.c["twist.retreat_extra_open"])
                opened = (await self.rig.grip(wide, None))[0]
                if opened.startswith(("ABORTED", "FAULT")) or await self.rig.holding():
                    raise RuntimeError("jaw release not confirmed: " + opened.splitlines()[0])
                out.append("open: " + opened.splitlines()[0])
        except Exception as e:  # noqa: BLE001  (keep going: lifting away matters more)
            out.append(f"open failed: {e}; refusing to lift a possibly attached cap")
            self.log("retreat", "abort", {}, f"{why} -> " + "; ".join(out))
            return "; ".join(out)
        try:
            _, pos, R = await self._pose()
            out.append("up: " + (await self.rig.move_down(float(pos[0]), float(pos[1]), float(pos[2]) + self.c["twist.retreat_up_m"],
                                                          yaw_of(R))).splitlines()[0])
        except Exception as e:  # noqa: BLE001
            out.append(f"lift failed: {e}")
        self.log("retreat", "abort", {}, f"{why} -> " + "; ".join(out))
        return "; ".join(out)

    def speed(self) -> tuple[float, float]:
        """(deg per control step, s per control step) of the attempt's twist_speed, inside the safety caps."""
        sp = self.p["twist_speed"]
        return (min(self.c[f"twist.step_deg.{sp}"], SAFETY["max_step_deg"]),
                max(self.c[f"twist.step_dt.{sp}"], SAFETY["min_step_dt"]))

    async def _holding_at(self, x: float, y: float, z_grasp: float, cap_top_z: float) -> bool:
        """Resume test: earlier strokes in this attempt, the jaws closed on something, the TCP at the cap (within
        twist.resume_tol_m sideways, between the grasp height and the cap top)."""
        if not self.prior or not self.resume_safe:
            return False
        st = await self.rig.state()
        if st[self.j8]["norm"] >= self.c["twist.open_enough"]:
            return False
        _, pos, _ = await self._pose()
        tol = self.c["twist.resume_tol_m"]
        near = float(np.hypot(pos[0] - x, pos[1] - y)) <= tol and z_grasp - tol <= float(pos[2]) <= cap_top_z
        return near and bool(await self.rig.holding())

    # --- twist_cap -------------------------------------------------------------------------------------------
    async def _halt(self):
        if hasattr(self.rig, "halt"):
            # Bounded stop acknowledgement consumes at most 2 s of the profile's 10 s reply reserve.
            await asyncio.wait_for(self.rig.halt(), timeout=2.0)

    async def twist_cap(self, x: float, y: float, cap_top_z: float, cap_height: float | None = None) -> dict:
        budget = self.c["twist.time_budget_s"]
        try:
            async with asyncio.timeout(budget if budget > 0 else None):
                return await self._twist_cap(x, y, cap_top_z, cap_height)
        except TimeoutError:
            await self._halt()
            sig = capfree.free_signal(self.prior + self.strokes, self.c)
            holding = await self.rig.holding()
            done = len(self.prior) + len(self.strokes)
            note = "motion deadline reached; arm held at measured position; inspect grip before resuming"
            self.log("twist", "ok", {"stopped_by": "time budget", "deadline_interrupted": True, "elapsed_s": self.clock() - self.t0,
                                     "strokes_total": done, "holding": holding}, note)
            return dict(ok=True, strokes=self.strokes, abort_reason=None, notes=[note], resumed=bool(self.prior),
                        strokes_before=len(self.prior), strokes_commanded=int(self.p["strokes"]),
                        rotation_achieved_deg=sum(max(0, s["achieved_deg"]) for s in self.strokes),
                        rotation_attempt_deg=sig["rotation_deg"], z_change_mm=0, holding=holding,
                        stopped_by="time budget", strokes_total=done, strokes_needed=None, free_signal=sig,
                        elapsed_s=round(self.clock() - self.t0, 1), budget_s=budget, why_stopped=note)
        except asyncio.CancelledError:
            await self._halt()
            raise
        except Exception as error:
            await self._halt()
            why = f"{type(error).__name__}: {error}"
            await self._retreat(why)
            return dict(ok=False, strokes=self.strokes, abort_reason=why, notes=[], resumed=bool(self.prior))

    async def _twist_cap(self, x: float, y: float, cap_top_z: float, cap_height: float | None = None) -> dict:
        cap_height = self.c["bottle.cap_height_m"] if cap_height is None else cap_height
        if not all(math.isfinite(v) for v in (x, y, cap_top_z, cap_height)) or cap_height <= 0:
            return {"ok": False, "strokes": [], "abort_reason": "cap geometry must be finite with positive height", "notes": []}
        p = self.p
        total = int(p["strokes"])
        res = {"ok": False, "strokes": self.strokes, "abort_reason": None, "notes": [], "resumed": False,
               "strokes_before": len(self.prior), "strokes_commanded": total}
        z_floor = cap_top_z - cap_height - SAFETY["below_cap_edge_m"] + FINGERTIP_BELOW     # as a TCP height
        z_grasp = cap_top_z - self.c["twist.grasp_below_top_m"] + p["approach_z_offset_mm"] / 1000.0
        if z_grasp < z_floor - 1e-9:
            res["abort_reason"] = (f"grasp height {z_grasp:.3f} puts the fingertips below the safety floor (cap lower edge "
                                   f"- {SAFETY['below_cap_edge_m'] * 1000:.0f} mm; lowest TCP {z_floor:.3f}): raise "
                                   f"approach_z_offset_mm or check cap_height. Nothing moved.")
            self.log("approach", "fail", {"z_grasp": round(z_grasp, 4), "z_floor": round(z_floor, 4)}, res["abort_reason"])
            return res
        if len(self.prior) >= total:
            res["abort_reason"] = (f"this attempt already made its {total} strokes ({len(self.prior)}): extract_cap() or "
                                   f"check_cap() now, or end the attempt and begin one with more strokes. Nothing moved.")
            return res
        force = min(float(p["grip_torque_limit"]), SAFETY["max_grip_force"])
        close_to = 1.0 - float(p["grip_close"])
        step, dt = self.speed()
        yaw_start = float(p["yaw_start_deg"])

        if await self._holding_at(x, y, z_grasp, cap_top_z):
            res["resumed"] = True
            self.log("approach", "ok", {"mode": "resume", "strokes_before": len(self.prior)},
                     f"resumed: the jaws still hold the cap after {len(self.prior)} strokes (no re-approach)")
        else:
            self.log("locate", "ok", {"x": round(x, 4), "y": round(y, 4), "cap_top_z": round(cap_top_z, 4),
                                      "cap_height": cap_height}, "cap position given by the agent")
            # reachability first, without moving. Lean > 0: the approach axis tilts away from the base.
            q, _, _ = await self._pose()
            g_lean = await self.rig.reachable(x, y, z_grasp, yaw_start, q)
            if g_lean is None or abs(g_lean) > self.c["twist.max_lean_deg"]:
                res["abort_reason"] = (f"the grasp pose [{x:.3f}, {y:.3f}, {z_grasp:.3f}] is out of reach "
                                       f"({'no joint solution' if g_lean is None else f'the gripper would lean {abs(g_lean):.0f} deg'}). "
                                       f"Nothing moved.")
                self.log("approach", "fail", {"z_grasp": round(z_grasp, 4)}, res["abort_reason"])
                return res
            st = await self.rig.state()
            if st[self.j8]["norm"] < self.c["twist.open_enough"]:
                await self.rig.grip(self.c["twist.open_full"], None)
            approach = self.c["twist.approach"]
            side = approach == "side" or (approach == "auto" and abs(g_lean) > self.c["twist.side_lean_deg"])
            ok = await (self._approach_side(x, y, z_grasp, yaw_start, g_lean, res) if side else
                        self._approach_down(x, y, z_grasp, cap_top_z, yaw_start, q, res))
            if not ok:
                return res
            _, pos, R = await self._pose()
            if lean_of(R) > self.c["twist.note_lean_deg"]:
                res["notes"].append(f"the gripper leans {lean_of(R):.0f} deg from vertical at the cap ({'side' if side else 'top'} "
                                    f"approach): the twist is about the vertical cap axis, so several wrist joints move")
            if not await self._grasp("grasp", close_to, force, res):
                return res

        _, pos0, R0 = await self._pose()
        yaw0 = yaw_of(R0)
        if res["resumed"] and not capfree.free_signal(self.prior, self.c)["free"]:
            if not await self._release_backoff_regrasp(self.prior[-1], step, dt, yaw0, close_to, force, res):
                return res
        budget = float(self.c["twist.time_budget_s"])
        starts, stopped_by, sig = [], "strokes", capfree.free_signal(self.prior, self.c)
        for s in range(len(self.prior) + 1, total + 1):
            if sig["free"] and self.c["twist.stop_on_free"] == "on":
                stopped_by = "free"
                break
            cycle = self.c[f"twist.cycle_s.{p['twist_speed']}"]
            if budget > 0 and self.clock() - self.t0 + cycle > budget:
                stopped_by = "time budget"
                break
            starts.append(self.clock())
            stroke = await self._stroke(s, step, dt, yaw0)
            self.strokes.append(stroke)
            if stroke.get("budget_stop"):
                stopped_by = "time budget"
                break
            if stroke.get("fatal"):
                res["abort_reason"] = stroke["abort_reason"]
                await self._retreat(stroke["abort_reason"])
                return res
            sig = capfree.free_signal(self.prior + self.strokes, self.c)
            if sig["free"] and self.c["twist.stop_on_free"] == "on":
                stopped_by = "free"
                break                                      # keep the grip for extract_cap / check_cap
            if s == total:
                break                                      # keep the last grip for extract_cap / check_cap
            if budget > 0:
                cycle = ((starts[-1] - starts[0]) / (len(starts) - 1) if len(starts) > 1
                         else self.c[f"twist.cycle_s.{p['twist_speed']}"])
                if self.clock() - self.t0 + cycle > budget:
                    stopped_by = "time budget"
                    break                                  # the next cycle would end past the budget: resumable
            if not await self._release_backoff_regrasp(stroke, step, dt, yaw0, close_to, force, res):
                return res
        _, pos, _ = await self._pose()
        done = len(self.prior) + len(self.strokes)
        sig = capfree.free_signal(self.prior + self.strokes, self.c)
        res.update(ok=True, rotation_achieved_deg=round(sum(max(0.0, s["achieved_deg"]) for s in self.strokes), 1),
                   rotation_attempt_deg=sig["rotation_deg"], z_change_mm=round((float(pos[2]) - float(pos0[2])) * 1000, 1),
                   holding=await self.rig.holding(), stopped_by=stopped_by, strokes_total=done,
                   strokes_needed=sig["began_at"] if sig["free"] else None, free_signal=sig,
                   elapsed_s=round(self.clock() - self.t0, 1), budget_s=budget)
        why = {"free": f"stopped on free: {capfree.text(sig)}",
               "time budget": f"stopped at the time budget ({budget:.0f} s) after {done} of {total} strokes, cap still "
                              f"gripped: call twist_cap again with the same cap to resume",
               "strokes": f"all {total} strokes done; {capfree.text(sig)}"}[stopped_by]
        self.log("twist", "ok", {"summary": True, "stopped_by": stopped_by, "strokes_done": len(self.strokes),
                                 "strokes_total": done, "strokes_commanded": total, "strokes_needed": res["strokes_needed"],
                                 "free_signal": sig, "elapsed_s": res["elapsed_s"], "budget_s": budget,
                                 "resumed": res["resumed"]}, why)
        res["why_stopped"] = why
        return res

    async def _release_backoff_regrasp(self, stroke, step, dt, yaw0, close_to, force, res):
        s = stroke["stroke"]
        st = await self.rig.state()
        text, held = await self.rig.grip(min(self.c["twist.open_full"], st[self.j8]["norm"] + self.c["twist.release_open"]), None)
        if held or text.startswith(("ABORTED", "FAULT")):
            res["abort_reason"] = "release not confirmed before backoff: " + text.splitlines()[0]
            await self._retreat(res["abort_reason"])
            return False
        self.log("release", "ok", {"j8_norm": st[self.j8]["norm"]}, text.splitlines()[0])
        back = float(self.p["regrasp_backoff_deg"]) or max(stroke["achieved_deg"], 0.0)
        if back > 0:
            q, pos, R = await self._pose()
            path = await self._plan(R, pos, q, 0.0, -back, SAFETY["max_step_deg"], 0.0)
            stop = ""
            for _, sol in path:
                r = await self.rig.go(sol, max(dt, self.c["twist.turn_back_min_dt_s"]), True)
                if r.startswith("ABORTED"):
                    stop = r.splitlines()[0]
                    break
            row = await self._sample("turn_back", s, -back, yaw0)
            if stop:
                res["abort_reason"] = "turning back: " + stop
                self.log("release", "abort", {"turned_back_deg": row["yaw_deg"]}, res["abort_reason"])
                await self._retreat(res["abort_reason"])
                return False
            if len(path) * SAFETY["max_step_deg"] < back - 1:
                res["notes"].append(f"stroke {s}: could only turn back {len(path) * SAFETY['max_step_deg']:.0f} of {back:.0f} deg")
        if not await self._grasp("regrasp", close_to, force, res):
            return False
        return True

    async def _approach_down(self, x, y, z_grasp, cap_top_z, yaw_start, q, res) -> bool:
        """Jaws (nearly) down: hover at the highest reachable height above the cap, guarded descent to z_grasp."""
        hover_z = None
        hovers, max_lean = self.c["twist.hovers_m"], self.c["twist.max_lean_deg"]
        for h in hovers:
            lean_h = await self.rig.reachable(x, y, cap_top_z + h, yaw_start, q)
            if lean_h is not None and abs(lean_h) <= max_lean:
                hover_z = cap_top_z + h
                break
        if hover_z is None:
            res["abort_reason"] = (f"no reachable hover above the cap (tried {', '.join(f'+{h * 100:.1f}' for h in hovers)} cm "
                                   f"above z = {cap_top_z:.3f}). Nothing moved.")
            self.log("approach", "fail", {}, res["abort_reason"])
            return False
        out = await self.rig.move_down(x, y, hover_z, yaw_start)
        _, pos, R = await self._pose()
        hover = {"tcp": [round(float(v), 4) for v in pos], "lean_deg": round(lean_of(R), 1), "yaw_deg": round(yaw_of(R), 1)}
        if out.startswith(("ABORTED", "UNREACHABLE")) or "do not grasp" in out:
            res["abort_reason"] = "approach: " + out.splitlines()[0]
            self.log("approach", "abort", hover, res["abort_reason"])
            await self._retreat(res["abort_reason"])
            return False
        text, hit = await self.rig.descend(x, y, z_grasp, yaw_of(R), cap_top_z + self.c["twist.guard_above_m"])
        _, pos, R = await self._pose()
        meas = {"mode": "top", "tcp_z": round(float(pos[2]), 4), "z_grasp": round(z_grasp, 4), "lean_deg": round(lean_of(R), 1)}
        if hit is not None and hit - z_grasp <= self.c["twist.contact_ok_m"]:
            meas["contact_z"], hit = round(hit, 4), None          # touched right at the grasp height: close there
        if text.startswith("ABORTED") or hit is not None:
            why = ("unexpected contact at z = %.3f, %.0f mm above the grasp height: the jaws landed on the cap or the "
                   "bottle (misaligned)" % (hit, (hit - z_grasp) * 1000)) if hit is not None else "descent: " + text.splitlines()[0]
            res["abort_reason"], res["misaligned"] = why, hit is not None
            self.log("approach", "abort", meas, why)
            await self._retreat(why)
            return False
        self.log("approach", "ok", meas, text.splitlines()[0])
        return True

    async def _approach_side(self, x, y, z_grasp, yaw_start, g_lean, res) -> bool:
        """Steeply leaning gripper (the cap is high for this arm): start twist.side_back_m back along the jaws' approach axis
        from the grasp point, then advance along that axis in small guarded steps, so the cap slides between the open
        jaws (moving level instead would push the palm into the cap's top edge). A step that falls short = contact."""
        back, step_m = self.c["twist.side_back_m"], self.c["twist.side_step_m"]
        r = np.array([x, y], float) / max(float(np.hypot(x, y)), 1e-6)
        t = math.radians(g_lean)
        a = np.array([math.sin(t) * r[0], math.sin(t) * r[1], -math.cos(t)])      # approach axis (as move_tcp's IK)
        goal = np.array([x, y, z_grasp])
        pre = goal - back * a
        q, _, _ = await self._pose()
        lp = await self.rig.reachable(float(pre[0]), float(pre[1]), float(pre[2]), yaw_start, q)
        if lp is None or abs(lp) > self.c["twist.max_lean_deg"]:
            res["abort_reason"] = f"the pre-grasp point {np.round(pre, 3).tolist()} is out of reach. Nothing moved."
            self.log("approach", "fail", {}, res["abort_reason"])
            return False
        out = await self.rig.move_down(float(pre[0]), float(pre[1]), float(pre[2]), yaw_start)
        q, pos, R = await self._pose()
        meas = {"mode": "side", "pre": [round(float(v), 4) for v in pos], "lean_deg": round(lean_of(R), 1)}
        if out.startswith(("ABORTED", "UNREACHABLE")) or "do not grasp" in out:
            res["abort_reason"] = "approach (to the pre-grasp point): " + out.splitlines()[0]
            self.log("approach", "abort", meas, res["abort_reason"])
            await self._retreat(res["abort_reason"])
            return False
        a = np.asarray(R, float)[:, 1]                       # the pose's real approach axis
        start = pos.copy()
        goal = start + back * a
        n = max(1, int(math.ceil(back / step_m)))
        now = pos
        for k in range(1, n + 1):
            want = start + (goal - start) * k / n
            sol = await asyncio.to_thread(self._solve, want, R, q)
            if sol is None:
                why = f"approach: no joint solution {np.linalg.norm(goal - want) * 1000:.0f} mm before the cap"
                res["abort_reason"] = why
                self.log("approach", "abort", meas, why)
                await self._retreat(why)
                return False
            r_go = await self.rig.go(sol, self.c["twist.side_step_dt_s"], True)
            await asyncio.sleep(self.c["twist.side_settle_s"])
            q, now, _ = await self._pose()
            lag = float((want - now) @ a)
            if r_go.startswith("ABORTED") or lag > self.c["twist.side_contact_m"]:
                left = float((goal - now) @ a) * 1000
                why = ("unexpected contact %.0f mm before the grasp point (the jaws met the cap or the bottle: misaligned)"
                       % left) if not r_go.startswith("ABORTED") else "approach: " + r_go.splitlines()[0]
                meas.update(contact_mm_before=round(left, 1))
                res["abort_reason"], res["misaligned"] = why, True
                self.log("approach", "abort", meas, why)
                await self._retreat(why)
                return False
        meas.update(tcp=[round(float(v), 4) for v in now], z_grasp=round(z_grasp, 4))
        self.log("approach", "ok", meas, f"side approach: advanced {back * 1000:.0f} mm along the jaws onto the cap")
        return True

    async def _grasp(self, phase: str, close_to: float, force: float, res: dict) -> bool:
        text, held = await self.rig.grip(close_to, force)
        st = await self.rig.state()
        meas = {"j8_norm": st[self.j8]["norm"], "j8_load": st[self.j8]["load"], "force": force}
        touch = next((l for l in text.splitlines() if l.startswith("touch:")), text.splitlines()[0])
        if not held:
            res["abort_reason"] = f"{phase}: the jaws did not hold the cap ({touch})"
            res["misaligned"] = phase == "grasp"
            self.log(phase, "fail", meas, res["abort_reason"])
            await self._retreat(res["abort_reason"])
            return False
        self.log(phase, "ok", meas, touch)
        return True

    async def _stroke(self, s: int, step: float, dt: float, yaw0: float) -> dict:
        p = self.p
        q, pos, R = await self._pose()
        start = await self._sample("twist", s, 0.0, yaw0)
        base_yaw = start["yaw_deg"]
        path = await self._plan(R, pos, q, 0.0, float(p["twist_deg_per_stroke"]), step, float(p["lift_per_turn_mm"]))
        reachable = path[-1][0] if path else 0.0
        reason, fatal, rows, cmd = None, False, [start], 0.0
        if reachable < float(p["twist_deg_per_stroke"]) - 0.5:
            reason = f"IK: only {reachable:.0f} of {p['twist_deg_per_stroke']:.0f} deg reachable from here"
        budget_stop = False
        for d, sol in path:
            budget = self.c["twist.time_budget_s"]
            if budget > 0 and self.clock() - self.t0 + dt + self.c["twist.stroke_settle_s"] >= budget:
                budget_stop = True
                reason = "time budget reached during stroke"
                break
            cmd = d
            r = await self.rig.go(sol, dt, False)
            row = await self._sample("twist", s, d, yaw0)
            rows.append(row)
            achieved = row["yaw_deg"] - base_yaw
            if r.startswith("ABORTED"):
                reason = "stall: " + r.splitlines()[0]
                fatal = "fault" in r and "tracking" not in r
                break
            if any(row["status"]):
                reason, fatal = f"motor status flags {row['status']}", True
                break
            if row["j7_load"] > SAFETY["j7_load_hard"] or row["arm_load_max"] > SAFETY["arm_load_hard"]:
                reason, fatal = (f"hard load limit: j7 {row['j7_load']} (limit {SAFETY['j7_load_hard']}), other arm joints "
                                 f"up to {row['arm_load_max']} (limit {SAFETY['arm_load_hard']})"), True
                break
            if row["j7_load"] > float(p["j7_load_abort"]):
                reason = f"j7 load {row['j7_load']} over j7_load_abort {p['j7_load_abort']:.0f}: the cap resists"
                break
            if start["j8_raw"] - row["j8_raw"] > SAFETY["slip_ticks"]:          # j8 raw falls as the jaws close
                reason = f"slip: the jaws closed {start['j8_raw'] - row['j8_raw']} ticks further during the twist"
                break
            if d - achieved > self.c["twist.lag_stall_deg"]:
                reason = f"stall: the gripper is {d - achieved:.0f} deg behind the command"
                break
        if rows[1:]:
            await asyncio.sleep(self.c["twist.stroke_settle_s"])
            rows.append(await self._sample("twist_end", s, cmd, yaw0))
        end = rows[-1]
        twist_rows = rows[1:] or rows
        loads = [r["j7_load"] for r in twist_rows]
        m = {"stroke": s, "commanded_deg": round(cmd, 1), "achieved_deg": round(end["yaw_deg"] - base_yaw, 1),
             "peak_j7_load": max(loads), "mean_j7_load": round(sum(loads) / len(loads), 1),
             "j8_change_ticks": end["j8_raw"] - start["j8_raw"],
             "z_change_mm": round((end["tcp_z"] - start["tcp_z"]) * 1000, 2),
             "slip": bool(reason and reason.startswith("slip")), "abort_reason": reason, "fatal": fatal,
             "budget_stop": budget_stop}
        self.log("twist", "abort" if fatal else ("ok" if not reason else "fail"), {**m, "strokes": [m]},
                 reason or f"stroke {s}: {m['achieved_deg']:.0f} of {m['commanded_deg']:.0f} deg")
        return m

    # --- check_cap / extract_cap: the judging lifts ------------------------------------------------------------
    async def check_cap(self) -> dict:
        """The short lift (check.lift_m, r3: 15 mm). Returns {"cap_free", "cap_lift_mm", "resisted", "holding_after",
        ...}. On the real arm it could not tell a free cap from jaws sliding up a cap still on the thread (attempts
        1, 2, 7): use extract_cap for the decision."""
        return await self.lift_judge(self.c["check.lift_m"], "check")

    async def extract_cap(self) -> dict:
        """The extraction lift (extract.lift_m; learned: 6 cm, attempts 9-12). Longer than the ~16 mm the jaws hold
        of the cap, so the outcome is decisive: a cap still on the thread is held back (the lift stops and lets go,
        never yanks) or slides out of the jaws; a free cap comes up in them, clear of the bottle on camera."""
        return await self.lift_judge(self.c["extract.lift_m"], "extract")

    async def lift_judge(self, lift_m: float, label: str) -> dict:
        """Lift the closed gripper lift_m straight up in lift.step_m steps and judge whether the cap came free.
        Guarded: the safety load stop (CHECK_LOAD_STOP) and, before the last step and from lift.judge_after_m on, a
        gripper that rose less than lift.follow_fraction x the commanded lift (the thread holds it back) end the lift
        at once. The last step is judged after settling (the r3 check_cap rule, unchanged for its 15 mm)."""
        c = self.c
        held = bool(await self.rig.holding())        # never closes the jaws itself: they may not be at the cap
        q, pos0, R = await self._pose()
        base = await self._sample(label, 0, 0.0, yaw_of(R))
        out = {"holding_before": held, "resisted": False, "lift_steps": 0, "lift_m": round(float(lift_m), 4)}
        follow, after = float(c["lift.follow_fraction"]), float(c["lift.judge_after_m"])
        if held:
            n = max(1, int(math.ceil(float(lift_m) / float(c["lift.step_m"]) - 1e-9)))
            seed = q

            def plan():
                pts, sd = [], seed
                for k in range(1, n + 1):
                    sol = self._solve(pos0 + np.array([0, 0, lift_m * k / n]), R, sd)
                    if sol is None or np.max(np.abs(np.asarray(sol)[:7] - np.asarray(sd)[:7])) > MAX_JOINT_JUMP_RAD:
                        break
                    pts.append(sol)
                    sd = sol
                return pts
            lift = await asyncio.to_thread(plan)
            out["lift_planned_mm"] = round(lift_m * 1000 * len(lift) / n, 1)
            for k, sol in enumerate(lift, 1):
                r = await self.rig.go(sol, float(c["lift.step_dt_s"]), False)
                row = await self._sample(label, 0, 0.0, yaw_of(R))
                out["lift_steps"] += 1
                out["load_rise"] = row["arm_load_sum"] - base["arm_load_sum"]
                cmd, rose_now = lift_m * k / n, row["tcp_z"] - base["tcp_z"]
                if any(row["status"]) or row["j7_load"] > SAFETY["j7_load_hard"] or row["arm_load_max"] > SAFETY["arm_load_hard"]:
                    out["stopped"] = "motor status or immutable hard load limit during lift"
                    break
                if r.startswith("ABORTED") or out["load_rise"] > CHECK_LOAD_STOP:
                    out["stopped"] = r.splitlines()[0] if r.startswith("ABORTED") else \
                        f"arm loads rose by {out['load_rise']} (stop at {CHECK_LOAD_STOP})"
                    break
                if k < len(lift) and cmd >= after - 1e-9 and rose_now < follow * cmd:
                    out["stopped"] = (f"held back: the gripper rose {rose_now * 1000:.0f} of {cmd * 1000:.0f} mm "
                                      f"commanded (the thread holds the cap): stopped, not pulling further")
                    break
            await asyncio.sleep(self.c["lift.settle_s"])
        _, pos1, _ = await self._pose()
        out["holding_after"] = bool(await self.rig.holding())
        rose = round((float(pos1[2]) - float(pos0[2])) * 1000, 1)
        out["gripper_rose_mm"] = rose
        # Held back = the gripper stayed well short of the lift it was commanded (the thread holds the cap and the
        # jaws hold the cap), or a stop fired. Motor load alone is no evidence: it swings with the motion.
        out["resisted"] = bool(held and (out.get("stopped") or rose < follow * out.get("lift_planned_mm", 0)))
        out["cap_free"] = bool(held and out["holding_after"] and not out["resisted"]
                               and rose >= lift_m * 1000 * float(c["lift.free_rise_fraction"]))
        out["cap_lift_mm"] = rose if out["cap_free"] else 0.0
        return out
