"""The cap-unscrewing skill: twist_cap and check_cap, written once against a small rig interface.

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
"""
from __future__ import annotations

import asyncio
import math
import time

import numpy as np

from . import skillcfg as SC
from .episodes import SAFETY

STEP_DEG = {"slow": 3.0, "medium": 6.0}
STEP_DT = {"slow": 0.15, "medium": 0.10}
# Strategy settings: these module values are the code defaults; a Twister reads the active skill config
# (learning/skillcfg.py, keys twist.*) once per attempt, so a new config version applies from the next attempt.
HOVERS = tuple(SC.TWIST_DEFAULTS["twist.hovers_m"])    # m above the cap top before the descent (highest reachable)
MAX_LEAN = SC.TWIST_DEFAULTS["twist.max_lean_deg"]     # deg from vertical: steeper poses are not used for the cap
SIDE_LEAN = SC.TWIST_DEFAULTS["twist.side_lean_deg"]   # auto approach: a grasp pose leaning more than this is approached
                                                       # from the side (22 Nov: 90 = always from above; see skillcfg)
SIDE_BACK = 0.04             # side approach: start this far back along the jaws' approach axis
SIDE_STEP = 0.005            # advance in steps of this
SIDE_CONTACT = 0.006         # a step that ends this far short = contact
GUARD_ABOVE = 0.02           # the descent feels for contact from this height above the cap top (fingertips ~1 cm below TCP)
CONTACT_OK = 0.004           # a touch this close to the grasp height is accepted as arrival
GRASP_BELOW_TOP = SC.TWIST_DEFAULTS["twist.grasp_below_top_m"]   # nominal TCP height below the cap top
FINGERTIP_BELOW = 0.010      # the fingertips reach this far below the TCP (table contact at TCP z ~ 0.01)
LAG_STALL_DEG = 15.0         # achieved rotation this far behind the command = the cap does not turn
RELEASE_OPEN = SC.TWIST_DEFAULTS["twist.release_open"]   # jaw opening (norm) to release the cap between strokes
RETREAT_UP = 0.04            # m straight up after an abort
CHECK_LIFT = 0.015           # check_cap: lift this far with the cap gripped
TILTS_DEG = (4, -4, 8, -8, 12, -12)   # IK fallback tilts (see Twister._solve)
CHECK_LOAD_STOP = 400        # check_cap: safety stop on the rise of the two largest arm loads while lifting


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
    """One attempt's skill run. `log(phase, status, measurements, note)` writes the episode phase records."""

    def __init__(self, rig, params: dict, log, j7: str = "j7", j8: str = "j8", arm_joints=None, clock=time.monotonic):
        self.rig, self.p, self.log = rig, dict(params), log
        self.j7, self.j8 = j7, j8
        self.arm_joints = arm_joints or [f"j{i}" for i in range(1, 8)]
        self.clock = clock
        self.trace: list[dict] = []
        self.t0 = clock()
        self.strokes: list[dict] = []
        self.cap = None
        self.cfg = SC.active()                 # the skill-config version this attempt runs with (one snapshot)
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
        for t in TILTS_DEG:
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
                if sol is None or float(np.max(np.abs(np.asarray(sol)[:7] - np.asarray(seed)[:7]))) > 0.5:
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
            if st[self.j8]["norm"] < 0.85:
                out.append("open: " + (await self.rig.grip(min(0.96, st[self.j8]["norm"] + self.c["twist.release_open"] + 0.2), None))[0].splitlines()[0])
        except Exception as e:  # noqa: BLE001  (keep going: lifting away matters more)
            out.append(f"open failed: {e}")
        try:
            _, pos, R = await self._pose()
            out.append("up: " + (await self.rig.move_down(float(pos[0]), float(pos[1]), float(pos[2]) + RETREAT_UP,
                                                          yaw_of(R))).splitlines()[0])
        except Exception as e:  # noqa: BLE001
            out.append(f"lift failed: {e}")
        self.log("retreat", "abort", {}, f"{why} -> " + "; ".join(out))
        return "; ".join(out)

    # --- twist_cap -------------------------------------------------------------------------------------------
    async def twist_cap(self, x: float, y: float, cap_top_z: float, cap_height: float = 0.02) -> dict:
        p = self.p
        res = {"ok": False, "strokes": self.strokes, "abort_reason": None, "notes": []}
        z_floor = cap_top_z - cap_height - SAFETY["below_cap_edge_m"] + FINGERTIP_BELOW     # as a TCP height
        z_grasp = cap_top_z - self.c["twist.grasp_below_top_m"] + p["approach_z_offset_mm"] / 1000.0
        if z_grasp < z_floor - 1e-9:
            res["abort_reason"] = (f"grasp height {z_grasp:.3f} puts the fingertips below the safety floor (cap lower edge "
                                   f"- {SAFETY['below_cap_edge_m'] * 1000:.0f} mm; lowest TCP {z_floor:.3f}): raise "
                                   f"approach_z_offset_mm or check cap_height. Nothing moved.")
            self.log("approach", "fail", {"z_grasp": round(z_grasp, 4), "z_floor": round(z_floor, 4)}, res["abort_reason"])
            return res
        self.log("locate", "ok", {"x": round(x, 4), "y": round(y, 4), "cap_top_z": round(cap_top_z, 4),
                                  "cap_height": cap_height}, "cap position given by the agent")
        force = min(float(p["grip_torque_limit"]), SAFETY["max_grip_force"])
        close_to = 1.0 - float(p["grip_close"])
        step = min(STEP_DEG[p["twist_speed"]], SAFETY["max_step_deg"])
        dt = max(STEP_DT[p["twist_speed"]], SAFETY["min_step_dt"])
        yaw_start = float(p["yaw_start_deg"])

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
        if st[self.j8]["norm"] < 0.85:
            await self.rig.grip(0.96, None)
        approach = self.c["twist.approach"]
        side = approach == "side" or (approach == "auto" and abs(g_lean) > self.c["twist.side_lean_deg"])
        ok = await (self._approach_side(x, y, z_grasp, yaw_start, g_lean, res) if side else
                    self._approach_down(x, y, z_grasp, cap_top_z, yaw_start, q, res))
        if not ok:
            return res
        _, pos, R = await self._pose()
        if lean_of(R) > 25:
            res["notes"].append(f"the gripper leans {lean_of(R):.0f} deg from vertical at the cap ({'side' if side else 'top'} "
                                f"approach): the twist is about the vertical cap axis, so several wrist joints move")

        # grasp
        if not await self._grasp("grasp", close_to, force, res):
            return res

        _, pos0, R0 = await self._pose()
        yaw0 = yaw_of(R0)
        for s in range(1, int(p["strokes"]) + 1):
            stroke = await self._stroke(s, step, dt, yaw0)
            self.strokes.append(stroke)
            if stroke.get("fatal"):
                res["abort_reason"] = stroke["abort_reason"]
                await self._retreat(stroke["abort_reason"])
                return res
            if s == int(p["strokes"]):
                break                                      # keep the last grip for check_cap
            # release, turn back with open jaws, regrasp
            st = await self.rig.state()
            text, _ = await self.rig.grip(min(0.96, st[self.j8]["norm"] + self.c["twist.release_open"]), None)
            self.log("release", "ok", {"j8_norm": st[self.j8]["norm"]}, text.splitlines()[0])
            back = float(p["regrasp_backoff_deg"]) or max(stroke["achieved_deg"], 0.0)
            if back > 0:
                q, pos, R = await self._pose()
                path = await self._plan(R, pos, q, 0.0, -back, SAFETY["max_step_deg"], 0.0)
                stop = ""
                for _, sol in path:
                    r = await self.rig.go(sol, max(dt, 0.1), True)
                    if r.startswith("ABORTED"):
                        stop = r.splitlines()[0]
                        break
                row = await self._sample("turn_back", s, -back, yaw0)
                if stop:
                    res["abort_reason"] = "turning back: " + stop
                    self.log("release", "abort", {"turned_back_deg": row["yaw_deg"]}, res["abort_reason"])
                    await self._retreat(res["abort_reason"])
                    return res
                if len(path) * SAFETY["max_step_deg"] < back - 1:
                    res["notes"].append(f"stroke {s}: could only turn back {len(path) * SAFETY['max_step_deg']:.0f} of {back:.0f} deg")
            if not await self._grasp("regrasp", close_to, force, res):
                return res
        _, pos, _ = await self._pose()
        res.update(ok=True, rotation_achieved_deg=round(sum(max(0.0, s["achieved_deg"]) for s in self.strokes), 1),
                   z_change_mm=round((float(pos[2]) - float(pos0[2])) * 1000, 1), holding=await self.rig.holding())
        return res

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
        text, hit = await self.rig.descend(x, y, z_grasp, yaw_of(R), cap_top_z + GUARD_ABOVE)
        _, pos, R = await self._pose()
        meas = {"mode": "top", "tcp_z": round(float(pos[2]), 4), "z_grasp": round(z_grasp, 4), "lean_deg": round(lean_of(R), 1)}
        if hit is not None and hit - z_grasp <= CONTACT_OK:
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
        """Steeply leaning gripper (the cap is high for this arm): start SIDE_BACK back along the jaws' approach axis
        from the grasp point, then advance along that axis in small guarded steps, so the cap slides between the open
        jaws (moving level instead would push the palm into the cap's top edge). A step that falls short = contact."""
        r = np.array([x, y], float) / max(float(np.hypot(x, y)), 1e-6)
        t = math.radians(g_lean)
        a = np.array([math.sin(t) * r[0], math.sin(t) * r[1], -math.cos(t)])      # approach axis (as move_tcp's IK)
        goal = np.array([x, y, z_grasp])
        pre = goal - SIDE_BACK * a
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
        goal = start + SIDE_BACK * a
        n = max(1, int(math.ceil(SIDE_BACK / SIDE_STEP)))
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
            r_go = await self.rig.go(sol, 0.15, True)
            await asyncio.sleep(0.1)
            q, now, _ = await self._pose()
            lag = float((want - now) @ a)
            if r_go.startswith("ABORTED") or lag > SIDE_CONTACT:
                left = float((goal - now) @ a) * 1000
                why = ("unexpected contact %.0f mm before the grasp point (the jaws met the cap or the bottle: misaligned)"
                       % left) if not r_go.startswith("ABORTED") else "approach: " + r_go.splitlines()[0]
                meas.update(contact_mm_before=round(left, 1))
                res["abort_reason"], res["misaligned"] = why, True
                self.log("approach", "abort", meas, why)
                await self._retreat(why)
                return False
        meas.update(tcp=[round(float(v), 4) for v in now], z_grasp=round(z_grasp, 4))
        self.log("approach", "ok", meas, f"side approach: advanced {SIDE_BACK * 1000:.0f} mm along the jaws onto the cap")
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
        for d, sol in path:
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
            if d - achieved > LAG_STALL_DEG:
                reason = f"stall: the gripper is {d - achieved:.0f} deg behind the command"
                break
        if rows[1:]:
            await asyncio.sleep(0.2)
            rows.append(await self._sample("twist_end", s, cmd, yaw0))
        end = rows[-1]
        twist_rows = rows[1:] or rows
        loads = [r["j7_load"] for r in twist_rows]
        m = {"stroke": s, "commanded_deg": round(cmd, 1), "achieved_deg": round(end["yaw_deg"] - base_yaw, 1),
             "peak_j7_load": max(loads), "mean_j7_load": round(sum(loads) / len(loads), 1),
             "j8_change_ticks": end["j8_raw"] - start["j8_raw"],
             "z_change_mm": round((end["tcp_z"] - start["tcp_z"]) * 1000, 2),
             "slip": bool(reason and reason.startswith("slip")), "abort_reason": reason, "fatal": fatal}
        self.log("twist", "abort" if fatal else ("ok" if not reason else "fail"), {**m, "strokes": [m]},
                 reason or f"stroke {s}: {m['achieved_deg']:.0f} of {m['commanded_deg']:.0f} deg")
        return m

    # --- check_cap --------------------------------------------------------------------------------------------
    async def check_cap(self) -> dict:
        """Lift the closed gripper CHECK_LIFT straight up and judge from the loads and the jaws whether the cap came
        free. Returns {"cap_free", "cap_lift_mm", "resisted", "holding_after", ...}."""
        held = bool(await self.rig.holding())        # never closes the jaws itself: they may not be at the cap
        q, pos0, R = await self._pose()
        base = await self._sample("check", 0, 0.0, yaw_of(R))
        out = {"holding_before": held, "resisted": False, "lift_steps": 0}
        if held:
            n = 5
            seed = q

            def plan():
                pts, sd = [], seed
                for k in range(1, n + 1):
                    sol = self._solve(pos0 + np.array([0, 0, CHECK_LIFT * k / n]), R, sd)
                    if sol is None:
                        break
                    pts.append(sol)
                    sd = sol
                return pts
            lift = await asyncio.to_thread(plan)
            out["lift_planned_mm"] = round(CHECK_LIFT * 1000 * len(lift) / n, 1)
            for sol in lift:
                r = await self.rig.go(sol, 0.15, False)
                row = await self._sample("check", 0, 0.0, yaw_of(R))
                out["lift_steps"] += 1
                out["load_rise"] = row["arm_load_sum"] - base["arm_load_sum"]
                if r.startswith("ABORTED") or out["load_rise"] > CHECK_LOAD_STOP:
                    out["stopped"] = r.splitlines()[0] if r.startswith("ABORTED") else \
                        f"arm loads rose by {out['load_rise']} (stop at {CHECK_LOAD_STOP})"
                    break
            await asyncio.sleep(0.3)
        _, pos1, _ = await self._pose()
        out["holding_after"] = bool(await self.rig.holding())
        rose = round((float(pos1[2]) - float(pos0[2])) * 1000, 1)
        out["gripper_rose_mm"] = rose
        # Held back = the gripper stayed well short of the lift it was commanded (the thread holds the cap and the
        # jaws hold the cap), or the safety load stop fired. Motor load alone is no evidence: it swings with the motion.
        out["resisted"] = bool(held and (out.get("stopped") or rose < 0.5 * out.get("lift_planned_mm", 0)))
        out["cap_free"] = bool(held and out["holding_after"] and not out["resisted"] and rose >= CHECK_LIFT * 1000 * 0.6)
        out["cap_lift_mm"] = rose if out["cap_free"] else 0.0
        return out
