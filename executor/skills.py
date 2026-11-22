"""Skills wrapped as phases: plan (every waypoint solved with the IK before anything moves), execute,
verify (measured, not commanded), recover. Parameters are in the WORLD frame (metres, degrees for yaw).

plan() is pure: it only uses the kinematic model, the seed joints and the table memory, so validation can run
it and a plan that fails anywhere (including the last stir waypoint) means zero motion.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .contracts import Status
from .motion import GripResult, MotionResult


class PlanError(Exception):
    def __init__(self, reason: str, checks: dict | None = None):
        super().__init__(reason)
        self.reason, self.checks = reason, checks or {}


@dataclass
class Plan:
    paths: dict[str, list]                 # label -> list of joint vectors (rad)
    checks: dict = field(default_factory=dict)
    target: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)


@dataclass
class Verdict:
    status: Status
    termination: str
    achieved: dict
    residuals: dict


def _vec(p, n=3, name="xyz_world"):
    try:
        v = np.array([float(x) for x in p], float)
    except (TypeError, ValueError):
        raise PlanError(f"{name} must be a list of {n} numbers (metres, world frame)")
    if v.shape != (n,) or not np.all(np.isfinite(v)):
        raise PlanError(f"{name} must be a list of {n} finite numbers (metres, world frame)")
    return v


class Skill:
    name = ""
    version = "1"
    freshness_s = 2.0                      # max age of the observation a proposal is based on
    cameras: tuple[str, ...] = ("overview", "wrist")   # must be fresh for this skill (those the backend has)
    moves = True
    params_doc: dict[str, str] = {}
    doc = ""

    def describe(self) -> dict:
        return {"name": self.name, "version": self.version, "freshness_s": self.freshness_s, "cameras": list(self.cameras),
                "params": self.params_doc, "doc": self.doc}

    # helpers ------------------------------------------------------------------------------
    def workspace(self, ctx, p, label, min_z=None) -> dict:
        WS = ctx.P.workspace
        min_z = ctx.min_z if min_z is None else min_z
        b = ctx.kin.to_base(p)
        r = float(np.hypot(b[0], b[1]))
        ok = min_z - 1e-9 <= p[2] <= WS.max_tcp_z and r <= WS.max_radius
        c = {"result": "pass" if ok else "fail", "z": round(float(p[2]), 4), "z_range": [min_z, WS.max_tcp_z],
             "radius": round(r, 4), "max_radius": WS.max_radius}
        if not ok:
            raise PlanError(f"{label} {np.round(p, 3).tolist()} is outside the workspace (z {min_z}..{WS.max_tcp_z} m, "
                            f"radius <= {WS.max_radius} m; this point: z {p[2]:.3f}, radius {r:.3f})", {"workspace": c})
        return c

    def solve_chain(self, ctx, points, seed, yaw, label, top_down=True, tip_length=None, min_z=None):
        """IK for every point in order, each seeded by the previous solution. PlanError names the first failure."""
        out, leans = [], []
        q = np.asarray(seed, float)
        for i, p in enumerate(points):
            if tip_length is None:
                self.workspace(ctx, p, f"{label} waypoint {i + 1}/{len(points)}", min_z)
                sol, lean = ctx.kin.solve(p, q, yaw, top_down)
            else:
                sol, lean = ctx.kin.solve_tip(p, tip_length, q, yaw), 0.0
                if sol is not None:
                    self.workspace(ctx, ctx.kin.fk(sol)[0], f"{label} waypoint {i + 1}/{len(points)} (gripper)")
            if sol is None:
                raise PlanError(f"unreachable: no joint solution for {label} waypoint {i + 1}/{len(points)} "
                                f"{np.round(p, 3).tolist()}; nothing will move",
                                {"reachability": {"result": "fail", "label": label, "index": i + 1, "of": len(points),
                                                  "point": np.round(p, 4).tolist()}})
            if not ctx.kin.in_limits(sol):
                raise PlanError(f"{label} waypoint {i + 1} needs joints outside their limits",
                                {"limits": {"result": "fail", "index": i + 1}})
            out.append(sol); leans.append(abs(lean or 0.0) if tip_length is None else ctx.kin.tilt_deg(sol))
            q = sol
        return out, max(leans) if leans else 0.0

    def travel(self, ctx, target, seed, yaw, top_down=True, label="move"):
        """Waypoints like real_arm_mcp.move_tcp: far sideways moves lift to travel height first."""
        WS = ctx.P.workspace
        cur = ctx.kin.fk(seed)[0]
        pts = []
        if np.hypot(*(target[:2] - cur[:2])) > WS.direct_move_within:
            safe = max(target[2], WS.travel_height)
            if cur[2] < safe - WS.travel_slack:
                pts.append(np.array([cur[0], cur[1], safe]))
            if target[2] < safe:
                pts.append(np.array([target[0], target[1], safe]))
        pts.append(np.asarray(target, float))
        return self.solve_chain(ctx, pts, seed, yaw, label, top_down) + (pts,)

    def descent(self, ctx, x, y, z_from, z_to, yaw, seed):
        """Preflighted straight-down path: one free step to the guard zone, then touch.step increments."""
        TS = ctx.P.touch
        z_guard = max(z_to, ctx.table_z(x, y) + TS.guard_from)
        free = [z_guard] if z_from > z_guard + TS.step else []
        start = z_guard if free else z_from
        zs = free + [float(z) for z in np.arange(start - TS.step, z_to, -TS.step)] + [z_to]
        floor = TS.nominal_table_tcp_z - TS.max_below        # a guarded descent may press below min_tcp_z (it feels the table)
        sols, _ = self.solve_chain(ctx, [np.array([x, y, z]) for z in zs], seed, yaw, "descent", min_z=floor)
        return sols[:len(free)], sols[len(free):]

    def tcp_verdict(self, ctx, target, tol_m=None, label="tcp") -> Verdict:
        q = ctx.motion.q_now()
        p = ctx.kin.fk(q)[0]
        err = float(np.linalg.norm(p - target))
        tol = ctx.P.motion.off_target_note if tol_m is None else tol_m
        ok = err <= tol
        return Verdict(Status.SUCCEEDED if ok else Status.PARTIAL,
                       "reached (measured within tolerance)" if ok else f"PARTIAL: {label} is {err * 1000:.1f} mm from the target",
                       {"tcp_world": np.round(p, 4).tolist(), "joints_rad": np.round(q, 4).tolist()},
                       {"tcp_mm": round(err * 1000, 1), "tcp_xyz_mm": np.round((target - p) * 1000, 1).tolist(), "tol_mm": tol * 1000})

    def plan(self, ctx, params) -> Plan:
        raise NotImplementedError

    def execute(self, ctx, plan, run) -> MotionResult:
        raise NotImplementedError

    def verify(self, ctx, plan, run, res) -> Verdict:
        raise NotImplementedError

    def recover(self, ctx, plan, run, res) -> str | None:
        return None


def go_path(ctx, run, label, path, settle_last=True, **kw) -> MotionResult:
    res = MotionResult(Status.SUCCEEDED, "nothing to do")
    for i, q in enumerate(path):
        res = ctx.motion.go_q(q, settle=settle_last or i < len(path) - 1, **kw)
        run.note(f"{label} {i + 1}/{len(path)}: {res.status.value} {res.termination}")
        if res.status != Status.SUCCEEDED:
            return res
    return res


class MoveTcp(Skill):
    name, version = "move_tcp", "1"
    params_doc = {"xyz_world": "[x, y, z] metres, world frame (TCP = between the fingertip pads)",
                  "yaw_deg": "optional: world yaw of the jaw closing axis (omit = keep current)",
                  "top_down": "bool, default true: jaws as close to straight down as reachable"}
    doc = "Move the TCP; far moves go via travel height. Every waypoint is solved before moving."

    def plan(self, ctx, params):
        target = _vec(params.get("xyz_world"))
        yaw = params.get("yaw_deg")
        path, lean, pts = self.travel(ctx, target, ctx.q0, None if yaw is None else float(yaw), bool(params.get("top_down", True)))
        return Plan({"move": path}, {"reachability": {"result": "pass", "waypoints": len(path), "max_lean_deg": lean}},
                    {"xyz_world": target.tolist()}, {"points": [p.tolist() for p in pts]})

    def execute(self, ctx, plan, run):
        return go_path(ctx, run, "move", plan.paths["move"])

    def verify(self, ctx, plan, run, res):
        return self.tcp_verdict(ctx, np.array(plan.target["xyz_world"]))


class Park(MoveTcp):
    name = "park"
    params_doc = {}
    doc = "Move to the profile's park pose."

    def plan(self, ctx, params):
        return super().plan(ctx, {"xyz_world": ctx.kin.to_world(ctx.P.poses.park).tolist()})


class Gripper(Skill):
    name, version, freshness_s, cameras = "gripper", "1", 5.0, ()
    params_doc = {"open_fraction": "0 closed .. 1 open", "force": "optional 0.1..1 squeeze limit"}
    doc = "Open or close the jaws with a torque limit; reports touch (held / nothing)."

    def plan(self, ctx, params):
        try:
            f = float(params.get("open_fraction"))
        except (TypeError, ValueError):
            raise PlanError("open_fraction must be a number 0..1")
        if not 0.0 <= f <= 1.0:
            raise PlanError(f"open_fraction {f} outside 0..1", {"limits": {"result": "fail", "open_fraction": f}})
        return Plan({}, {"limits": {"result": "pass", "open_fraction": f}}, {"open_fraction": f, "force": params.get("force")})

    def execute(self, ctx, plan, run):
        g = ctx.motion.gripper(plan.target["open_fraction"], plan.target["force"])
        run.data["grip"] = g
        run.note(f"gripper: {g.termination}; jaws {g.norm:.3f}, load {g.load}, holding {g.holding}")
        return MotionResult(g.status, g.termination)

    def verify(self, ctx, plan, run, res):
        g: GripResult = run.data["grip"]
        G = ctx.kin.grip_id
        st = ctx.motion.read()[G]
        return Verdict(g.status, g.termination, {"gripper_norm": st["norm"], "load": st["load"], "holding": g.holding},
                       {"gripper_norm": round(plan.target["open_fraction"] - st["norm"], 4)} if g.holding is not True else {})


class Pick(Skill):
    name, version, freshness_s = "pick", "1", 1.5
    params_doc = {"xyz_world": "[x, y, z] object centre on the table, metres, world frame (z ignored: table)",
                  "object_height": "metres, default 0.02", "yaw_deg": "world yaw of the jaw closing axis (omit = keep current)",
                  "grip_force": "optional 0.1..1"}
    doc = "Open, hover, guarded descent, close with touch, lift to carry height. Uses the table height felt for this setup."

    def plan(self, ctx, params):
        P, PK = ctx.P, ctx.P.pick
        x, y, _ = _vec(params.get("xyz_world"))
        h = float(params.get("object_height", PK.default_height))
        yaw = float(params["yaw_deg"]) if params.get("yaw_deg") is not None else ctx.kin.yaw_now(ctx.q0)
        hover = np.array([x, y, max(h + PK.hover_above, PK.hover_min)])
        approach, lean, _ = self.travel(ctx, hover, ctx.q0, yaw, label="hover")
        table = ctx.table_z(x, y)
        flat = h < PK.flat_below
        goal_z = table - P.touch.max_below if flat else max(table + PK.table_clearance,
                                                            table - P.touch.nominal_table_tcp_z + h / 2 + PK.grasp_above_mid)
        goal_z = max(goal_z, ctx.min_z) if not flat else goal_z
        free, steps = self.descent(ctx, x, y, hover[2], goal_z, yaw, approach[-1])
        lift, _ = self.solve_chain(ctx, [np.array([x, y, P.poses.carry_z])], (steps or free or approach)[-1], yaw, "lift")
        back, _ = self.solve_chain(ctx, [np.array([x, y, PK.backoff_z])], (steps or free or approach)[-1], yaw, "back off")
        return Plan({"approach": approach, "free": free, "descent": steps, "lift": lift, "back": back},
                    {"reachability": {"result": "pass", "waypoints": len(approach) + len(free) + len(steps) + 2, "max_lean_deg": lean},
                     "table": {"result": "pass", "table_tcp_z": round(table, 4), "felt": ctx.memory_hit(x, y) is not None,
                               "goal_z": round(goal_z, 4), "flat": flat}},
                    {"xy": [x, y], "goal_z": goal_z, "flat": flat, "force": params.get("grip_force")},
                    {"preflight_from": {"back": "descent" if steps else "free" if free else "approach"}})

    def execute(self, ctx, plan, run):
        m = ctx.motion
        if m.read()[ctx.kin.grip_id]["norm"] < ctx.P.gripper.open_enough:
            g = m.gripper(1.0)
            run.note("open: " + g.termination)
        res = go_path(ctx, run, "hover", plan.paths["approach"])
        if res.status != Status.SUCCEEDED:
            return res
        res = go_path(ctx, run, "descend (free)", plan.paths["free"])
        if res.status != Status.SUCCEEDED:
            return res
        res, hit = m.descend_guarded(plan.paths["descent"])
        run.note("descend: " + res.termination)
        if res.status != Status.SUCCEEDED:
            return res
        if hit is not None and plan.target["flat"]:
            ctx.remember_touch(*plan.target["xy"], hit)
        g = m.gripper(0.0, plan.target["force"])
        run.data["grip"] = g
        run.note(f"close: {g.termination} (jaws {g.norm:.3f}, load {g.load})")
        if not g.holding:
            run.data["empty"] = True
            return MotionResult(Status.BLOCKED, "nothing held after closing")
        res = go_path(ctx, run, "lift", plan.paths["lift"])
        held, _ = m.probe_hold()
        run.data["held_after_lift"] = held
        return res

    def verify(self, ctx, plan, run, res):
        st = ctx.motion.read()[ctx.kin.grip_id]
        held = run.data.get("held_after_lift")
        achieved = {"gripper_norm": st["norm"], "gripper_load": st["load"], "holding": held,
                    "tcp_world": np.round(ctx.kin.fk(ctx.motion.q_now())[0], 4).tolist()}
        if res.status != Status.SUCCEEDED:
            return Verdict(res.status, res.termination, achieved, res.residual_ticks and {"joints_ticks": res.residual_ticks})
        if not held:
            return Verdict(Status.BLOCKED, "lifted but nothing is held (slipped out)", achieved, {})
        return Verdict(Status.SUCCEEDED, "holding after the lift (touch probe)", achieved, {})

    def recover(self, ctx, plan, run, res):
        if run.data.get("empty"):
            ctx.motion.gripper(1.0)
            r = go_path(ctx, run, "back off", plan.paths["back"])
            return f"opened and backed off: {r.status.value}"
        return None


class Place(Skill):
    name, version = "place", "1"
    params_doc = {"xyz_world": "[x, y, z] release point (gripper height z, at least 6 cm above a box rim)",
                  "yaw_deg": "optional world yaw for the release"}
    doc = "Carry to the release point, open, retreat to above the park spot."

    def plan(self, ctx, params):
        target = _vec(params.get("xyz_world"))
        yaw = params.get("yaw_deg")
        carry, lean, _ = self.travel(ctx, target, ctx.q0, None if yaw is None else float(yaw), label="carry")
        park = ctx.kin.to_world(ctx.P.poses.park)
        retreat, _, _ = self.travel(ctx, np.array([park[0], park[1], max(target[2], ctx.P.poses.retreat_min_z)]), carry[-1], None,
                                    label="retreat")
        return Plan({"carry": carry, "retreat": retreat}, {"reachability": {"result": "pass", "waypoints": len(carry) + len(retreat),
                                                                            "max_lean_deg": lean}}, {"xyz_world": target.tolist()})

    def execute(self, ctx, plan, run):
        res = go_path(ctx, run, "carry", plan.paths["carry"])
        if res.status != Status.SUCCEEDED:
            return res
        run.data["release_tcp"] = ctx.kin.fk(ctx.motion.q_now())[0].tolist()
        g = ctx.motion.gripper(1.0)
        run.note("release: " + g.termination)
        if g.status != Status.SUCCEEDED:
            return MotionResult(g.status, "release: " + g.termination)
        return go_path(ctx, run, "retreat", plan.paths["retreat"])

    def verify(self, ctx, plan, run, res):
        rel = run.data.get("release_tcp")
        st = ctx.motion.read()[ctx.kin.grip_id]
        ach = {"release_tcp_world": rel and np.round(rel, 4).tolist(), "gripper_norm": st["norm"]}
        if rel is None:
            return Verdict(res.status, res.termination, ach, {"joints_ticks": res.residual_ticks})
        err = float(np.linalg.norm(np.array(rel) - plan.target["xyz_world"]))
        ok = res.status == Status.SUCCEEDED and err <= ctx.P.motion.off_target_note
        return Verdict(Status.SUCCEEDED if ok else Status.PARTIAL,
                       "released at the target" if ok else f"released {err * 1000:.0f} mm from the target ({res.termination})",
                       ach, {"release_mm": round(err * 1000, 1)})


class Stir(Skill):
    name, version = "stir", "2"
    params_doc = {"xyz_world": "[x, y, z] cup centre, z = tool TIP height for stirring (metres, world)",
                  "radius": "metres (default profile)", "turns": "int", "tool_length": "metres below the fingertips",
                  "clearance": "metres above z for the entry/exit point (must clear the rim)"}
    doc = "Tip above the cup, straight down, circles, straight up. The WHOLE path is solved first: any unreachable point = no motion."

    def plan(self, ctx, params):
        ST = ctx.P.stir
        x, y, z = _vec(params.get("xyz_world"))
        radius = float(min(max(float(params.get("radius", ST.default_radius)), ST.min_radius), ST.max_radius))
        turns = int(min(max(int(params.get("turns", ST.default_turns)), 1), ST.max_turns))
        L = float(min(max(float(params.get("tool_length", 0.0)), 0.0), ST.max_tool_length))
        clearance = float(params.get("clearance", ST.default_clearance))
        yaw = ctx.kin.yaw_now(ctx.q0)
        first = ctx.kin.solve_tip([x, y, z + clearance], L, ctx.q0, yaw)
        if first is None:
            raise PlanError(f"unreachable: the tool tip cannot reach {np.round([x, y, z + clearance], 3).tolist()} above the cup; "
                            f"nothing will move", {"reachability": {"result": "fail", "label": "stir entry", "index": 1}})
        entry, _, _ = self.travel(ctx, ctx.kin.fk(first)[0], ctx.q0, yaw, label="approach")
        seed = entry[-1]
        n = ST.points_per_turn
        pts = [np.array([x, y, z + clearance]), np.array([x, y, z])]
        pts += [np.array([x + radius * np.cos(2 * np.pi * k / n), y + radius * np.sin(2 * np.pi * k / n), z]) for k in range(n + 1)]
        sols, tilt = self.solve_chain(ctx, pts, seed, yaw, "stir", tip_length=L)
        above, down, ring = sols[0], sols[1], sols[2:]
        return Plan({"approach": entry, "above": [above], "down": [down], "ring": ring, "out": [down, above]},
                    {"reachability": {"result": "pass", "waypoints": len(pts) + len(entry), "max_tilt_deg": round(tilt, 1)}},
                    {"centre": [x, y, z], "radius": radius, "turns": turns, "tool_length": L, "clearance": clearance})

    def execute(self, ctx, plan, run):
        ST, m = ctx.P.stir, ctx.motion
        res = go_path(ctx, run, "approach", plan.paths["approach"] + plan.paths["above"])
        if res.status != Status.SUCCEEDED:
            return res
        run.data["entered"] = True
        res = go_path(ctx, run, "lower", plan.paths["down"])
        if res.status != Status.SUCCEEDED:
            return res
        A = ctx.kin.arm_ids
        base = {a: v["load"] for a, v in m.read().items() if a in A}
        ring, turns = plan.paths["ring"], plan.target["turns"]
        path = ring + ring[1:] * (turns - 1)
        over, worst = 0, 0
        for k, q in enumerate(path):
            res = m.go_q(q, settle=False, dt=ST.dt)
            if res.status != Status.SUCCEEDED:
                return res
            now = m.read()
            push = max(abs(now[a]["load"] - base[a]) for a in A)
            worst = max(worst, push)
            over = over + 1 if push > ST.contact_load_delta else 0
            if over >= ST.contact_points:
                m.b.hold()
                return MotionResult(Status.PARTIAL, f"contact abort after {k / ST.points_per_turn:.1f} turns: hard resistance "
                                                    f"(arm load changed by {push})", contact=True)
        run.note(f"stirred {turns} turns, largest load change {worst}")
        res = go_path(ctx, run, "lift out", plan.paths["out"])
        run.data["lifted"] = res.status == Status.SUCCEEDED
        return res

    def verify(self, ctx, plan, run, res):
        q = ctx.motion.q_now()
        tip = ctx.kin.tip_of(q, plan.target["tool_length"])
        c = np.array(plan.target["centre"])
        above = c + np.r_[0, 0, plan.target["clearance"]]
        err = float(np.linalg.norm(tip - above))
        ach = {"tip_world": np.round(tip, 4).tolist()}
        if res.status != Status.SUCCEEDED:
            return Verdict(res.status, res.termination, ach, {"tip_to_exit_mm": round(err * 1000, 1), "joints_ticks": res.residual_ticks})
        ok = err <= max(ctx.P.motion.off_target_note, 0.008)
        return Verdict(Status.SUCCEEDED if ok else Status.PARTIAL, "stirred and lifted out" if ok else
                       f"stirred; the tip ended {err * 1000:.0f} mm from the exit point", ach, {"tip_to_exit_mm": round(err * 1000, 1)})

    def recover(self, ctx, plan, run, res):
        """Abort inside the cup: lift STRAIGHT UP from where the tip is now (never via the low centre point)."""
        if not run.data.get("entered") or run.data.get("lifted") or res.status == Status.CANCELLED:
            return None
        m = ctx.motion
        q = m.q_now()
        L = plan.target["tool_length"]
        tip = ctx.kin.tip_of(q, L)
        top = plan.target["centre"][2] + plan.target["clearance"]
        if tip[2] >= top - 0.005:
            return "tip already above the exit height"
        steps = [np.array([tip[0], tip[1], z]) for z in np.linspace(tip[2], top, 4)[1:]]
        seed, path = q, []
        for p in steps:
            sol = ctx.kin.solve_tip(p, L, seed, ctx.kin.yaw_now(seed))
            if sol is None:
                m.b.hold()
                return f"could not plan a vertical lift from {np.round(tip, 3).tolist()}: holding"
            path.append(sol); seed = sol
        r = go_path(ctx, run, "recover: lift straight up", path)
        return f"lifted straight up to z={top:.3f}: {r.status.value} {r.termination}"


class TouchTable(Skill):
    name, version = "touch_table", "1"
    params_doc = {"xy_world": "[x, y] metres, a FREE patch of table"}
    doc = "Guarded descent until the fingertips touch; the height is remembered for this setup_id only."

    def plan(self, ctx, params):
        TS = ctx.P.touch
        x, y = _vec(params.get("xy_world"), 2, "xy_world")
        yaw = ctx.kin.yaw_now(ctx.q0)
        approach, lean, _ = self.travel(ctx, np.array([x, y, TS.probe_from]), ctx.q0, yaw, label="approach")
        free, steps = self.descent(ctx, x, y, TS.probe_from, TS.nominal_table_tcp_z - TS.max_below, yaw, approach[-1])
        back, _ = self.solve_chain(ctx, [np.array([x, y, TS.probe_from])], (steps or free)[-1], yaw, "back")
        return Plan({"approach": approach, "free": free, "descent": steps, "back": back},
                    {"reachability": {"result": "pass", "waypoints": len(approach) + len(free) + len(steps) + 1}}, {"xy": [x, y]})

    def execute(self, ctx, plan, run):
        res = go_path(ctx, run, "approach", plan.paths["approach"] + plan.paths["free"])
        if res.status != Status.SUCCEEDED:
            return res
        res, z = ctx.motion.descend_guarded(plan.paths["descent"])
        run.note("descend: " + res.termination)
        run.data["z"] = z
        if z is not None:
            run.data["remembered"] = ctx.remember_touch(*plan.target["xy"], z)
        back = go_path(ctx, run, "back up", plan.paths["back"])
        return res if res.status != Status.SUCCEEDED else back

    def verify(self, ctx, plan, run, res):
        z = run.data.get("z")
        ach = {"contact_tcp_z": z, "remembered": run.data.get("remembered", False)}
        if res.status != Status.SUCCEEDED:
            return Verdict(res.status, res.termination, ach, {})
        if z is None:
            return Verdict(Status.PARTIAL, "no contact felt down to the lowest allowed height", ach, {})
        return Verdict(Status.SUCCEEDED, f"table felt at tcp z={z:.3f}", ach,
                       {"vs_nominal_mm": round((ctx.P.touch.nominal_table_tcp_z - z) * 1000, 1)})


SKILLS: dict[str, Skill] = {s.name: s for s in (MoveTcp(), Park(), Gripper(), Pick(), Place(), Stir(), TouchTable())}
