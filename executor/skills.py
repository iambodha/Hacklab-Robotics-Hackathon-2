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


BIN_WORDS = ("bin", "bucket", "box", "basket", "bowl")


def _est_dict(o) -> dict:
    return {"name": o.name, "xyz_world": list(o.xyz_world), "size": o.size, "yaw": o.yaw, "uncertainty_mm": o.uncertainty_mm,
            "visible": o.visible, "source": o.source}


def resolve_object(ctx, query: str | None, color: str | None = None, pixel=None, camera: str = "overview",
                   height_m: float | None = None, bin_like: bool = False):
    """World estimate of a named / coloured / pointed-at object from the proposal's observation, in this order:
    1. an object estimate in the observation whose name matches (or shares the colour word, or is bin-like),
    2. perception.locate on the observation's overview frame with the colour hint ("blue" works),
    3. a pixel on the overview frame (e.g. from Gemini), ray to the table plane.
    -> (ObjectEstimate, how). PlanError if it cannot be localised."""
    from .contracts import ObjectEstimate
    from .perception import Hint, colour_in
    obs = ctx.obs
    if obs is None:
        raise PlanError("no observation to localise from (based_on_obs unknown)", {"perception": {"result": "fail"}})
    q = (query or "").lower().strip()
    color = color or colour_in(q)
    seen = [o for o in obs.objects]
    for o in seen:
        n = o.name.lower()
        if (q and n == q) or (color and colour_in(n) == color) or (bin_like and any(w in n for w in BIN_WORDS)) or \
                (q and any(w in n for w in BIN_WORDS if w in q)):
            if not o.visible:
                raise PlanError(f"{o.name} is not visible in observation {obs.obs_id} ({o.source}); look again",
                                {"perception": {"result": "fail", "object": _est_dict(o)}})
            return o, f"observation {obs.obs_id} estimate"
    jpeg = ctx.frame("overview")
    f = obs.frames.get("overview")
    if f is not None and f.stale:
        raise PlanError(f"overview frame of {obs.obs_id} is stale: cannot localise {query or color}", {"perception": {"result": "fail"}})
    if color:
        h = Hint(query or f"{color} object", color, height_m if height_m is not None else 0.02)
        est, why = ctx.locator.locate_hint(jpeg, h, ctx.camera_info(), f.seq if f else None, f.t_capture if f else None)
        if est is not None:
            return est, f"perception colour {color} on {obs.obs_id}"
        if pixel is None:
            raise PlanError(f"{query or color} not found in {obs.obs_id}: {why}", {"perception": {"result": "fail", "why": why}})
    if pixel is not None:
        z = 0.0
        r = ctx.locator.pixel_to_world(camera, pixel[0], pixel[1], z, ctx.camera_info())
        if not r.get("ok"):
            raise PlanError(f"cannot localise {camera} pixel {pixel}: {r.get('reason')}", {"perception": {"result": "fail", **r}})
        hh = 0.02 if height_m is None else height_m
        return ObjectEstimate(query or "pointed object", [r["xyz"][0], r["xyz"][1], hh / 2], None, None, r.get("sigma_mm"),
                              f"{camera} pixel {list(pixel)} -> table ({r.get('mm_per_px')} mm/px)", True,
                              f.t_capture if f else None), f"pixel {list(pixel)} on {obs.obs_id}"
    raise PlanError(f"cannot find {query!r}: give xyz_world, an object name / colour seen in the observation, or a pixel",
                    {"perception": {"result": "fail", "objects_seen": [o.name for o in seen]}})


class PickAt(Pick):
    """Pick an object located by perception: by name or colour hint (resolve_object), a pixel, or xyz_world.
    An advisor's observation may be seconds old by the time its answer arrives (Gemini ~3-10 s): pick_at accepts
    observations up to `freshness_s`, and if the proposal's observation is older than `relocalise_after`, the object is
    localised again on a FRESH observation; moved more than max(20 mm, 3 sigma) since the proposal = rejected."""
    name, version, freshness_s = "pick_at", "1", 20.0
    relocalise_after = 1.5
    params_doc = {"object": "name or colour hint, e.g. 'blue' / 'blue object' (from the observation's estimates or "
                            "perception.locate on its overview frame)",
                  "xyz_world": "optional [x, y, z] metres, world: skips localisation",
                  "camera / pixel": "optional: a pointed pixel [u, v] on the overview frame of based_on_obs",
                  "object_height": "metres (default: the estimate's height)", "max_uncertainty_mm": "default 25",
                  "yaw_deg": "optional jaw yaw (default: across the object's short side, or keep current if round)",
                  "grip_force": "optional 0.1..1"}
    doc = ("Pick at a perceived object: localise it (perception, with uncertainty), then open, hover, guarded descent, "
           "close with touch, lift. Rejected if the estimate is not visible, stale or too uncertain.")

    def plan(self, ctx, params):
        from .perception import colour_in
        given = None
        if params.get("xyz_world") is not None and not params.get("needs_localisation"):
            from .contracts import ObjectEstimate
            given = ObjectEstimate(params.get("object") or "given point", list(_vec(params["xyz_world"])), None, None,
                                   params.get("sigma_mm"), params.get("source", "proposal xyz_world"))
        relocal = None
        if ctx.obs is not None and ctx.ex.now() - ctx.obs.t_mono > self.relocalise_after:
            old = ctx.obs.obs_id
            ctx.obs = ctx.ex.observe()                     # read-only: a fresh look before planning, nothing moves
            relocal = {"from_obs": old, "to_obs": ctx.obs.obs_id}
        if given is None:
            est, how = resolve_object(ctx, params.get("object"), params.get("color"), params.get("pixel"),
                                      params.get("camera", "overview"), params.get("object_height"))
            if relocal:
                how += f" (re-localised: {relocal['from_obs']} was older than {self.relocalise_after} s)"
        elif relocal and (params.get("color") or colour_in(params.get("object"))):
            est, how = resolve_object(ctx, params.get("object"), params.get("color"), None, "overview", params.get("object_height"))
            moved = float(np.linalg.norm(np.array(est.xyz_world[:2]) - np.array(given.xyz_world[:2]))) * 1000
            tol = max(20.0, 3 * float(given.uncertainty_mm or 0))
            relocal.update(moved_mm=round(moved, 1), tol_mm=round(tol, 1))
            if moved > tol:
                raise PlanError(f"{given.name} moved {moved:.0f} mm since the proposal's observation (> {tol:.0f} mm): propose again",
                                {"perception": {"result": "fail", "object": _est_dict(est), "relocalised": relocal}})
            how = f"re-localised on {ctx.obs.obs_id} ({moved:.0f} mm from the proposal's xyz_world)"
        else:
            est, how = given, "proposal xyz_world"
        limit = float(params.get("max_uncertainty_mm", 25.0))
        if est.uncertainty_mm is not None and est.uncertainty_mm > limit:
            raise PlanError(f"{est.name} position too uncertain: {est.uncertainty_mm} mm > {limit} mm",
                            {"perception": {"result": "fail", "object": _est_dict(est), "limit_mm": limit}})
        h = params.get("object_height")
        h = float(h) if h is not None else float(est.size[2]) if est.size and len(est.size) > 2 else ctx.P.pick.default_height
        yaw = params.get("yaw_deg")
        if yaw is None and est.yaw is not None and est.size and est.size[1] and est.size[0] / est.size[1] > 1.3:
            yaw = (np.degrees(est.yaw) + 90 + 90) % 180 - 90       # jaws close across the short side
        plan = super().plan(ctx, {"xyz_world": est.xyz_world, "object_height": h, "yaw_deg": yaw,
                                  "grip_force": params.get("grip_force")})
        plan.checks["perception"] = {"result": "pass", "how": how, "object": _est_dict(est), "limit_mm": limit,
                                     **({"relocalised": relocal} if relocal else {})}
        plan.target["object"] = _est_dict(est)
        return plan

    def verify(self, ctx, plan, run, res):
        v = super().verify(ctx, plan, run, res)
        v.achieved["object_estimate"] = plan.target["object"]
        return v


class DropAt(Place):
    """Release above a target region (the bin): from the observation's bin estimate, a region, a bin dict or xyz_world."""
    name, version = "drop_at", "1"
    params_doc = {"target": "name of the place target in the observation (default: anything bin-like)",
                  "region_world_xy": "optional [[x0, y0], [x1, y1]] metres, world: the opening to drop into",
                  "bin": "optional {xyz_world, size [w, d, h], yaw_deg}", "xyz_world": "optional [x, y, z]: centre on the table",
                  "camera / pixel": "optional: a pointed pixel on the overview frame of based_on_obs",
                  "rim_height_m": "default: the bin's height, else 0.08", "clearance_m": "TCP above the rim, default 0.06",
                  "object": "optional: what is being dropped (the evaluator looks for it inside the region afterwards)"}
    doc = "Carry above the centre of the target region (rim + clearance), open, retreat. Twin preflight includes the bin walls."

    def plan(self, ctx, params):
        bin_ = None
        if params.get("region_world_xy"):
            (x0, y0), (x1, y1) = params["region_world_xy"]
            x0, x1, y0, y1 = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
            rim = float(params.get("rim_height_m", 0.08))
            bin_ = {"xyz_world": [(x0 + x1) / 2, (y0 + y1) / 2, 0.0], "size": [x1 - x0, y1 - y0, rim], "yaw_deg": 0.0,
                    "wall": 0.004, "from": "region_world_xy"}
            how = "region_world_xy"
        elif isinstance(params.get("bin"), dict):
            b = params["bin"]
            bin_ = {"xyz_world": list(_vec(b["xyz_world"])), "size": list(b.get("size") or [0.2, 0.15, 0.08]),
                    "yaw_deg": float(b.get("yaw_deg") or 0.0), "wall": float(b.get("wall") or 0.004), "from": "proposal bin"}
            how = "bin"
        elif params.get("xyz_world") is not None and not params.get("needs_localisation"):
            c = _vec(params["xyz_world"])
            rim = float(params.get("rim_height_m", 0.08))
            bin_, how = {"xyz_world": c.tolist(), "size": [0.0, 0.0, rim], "yaw_deg": 0.0, "wall": 0.0, "from": "xyz_world"}, "xyz_world"
        else:
            est, how = resolve_object(ctx, params.get("target"), params.get("color"), params.get("pixel"),
                                      params.get("camera", "overview"), params.get("rim_height_m"), bin_like=True)
            s = list(est.size or [0.0, 0.0])
            rim = float(params.get("rim_height_m") or (s[2] if len(s) > 2 else 0.08))
            bin_ = {"xyz_world": list(est.xyz_world), "size": [s[0], s[1], rim], "yaw_deg": float(np.degrees(est.yaw or 0.0)),
                    "wall": 0.004, "from": f"{how}: {est.source}", "uncertainty_mm": est.uncertainty_mm}
        rim = float(bin_["size"][2])
        z = rim + float(params.get("clearance_m", 0.06))
        x, y = bin_["xyz_world"][:2]
        plan = super().plan(ctx, {"xyz_world": [x, y, z], "yaw_deg": params.get("yaw_deg")})
        w, d = bin_["size"][:2]
        unc = bin_.get("uncertainty_mm")
        plan.checks["target"] = {"result": "pass", "how": how, "centre_xy": [round(x, 4), round(y, 4)], "release_z": round(z, 4),
                                 "rim_height_m": rim, "opening_m": [w, d], "uncertainty_mm": unc}
        if w and d and unc is not None and 2 * unc / 1000 >= min(w, d) / 2:
            raise PlanError(f"bin position too uncertain ({unc} mm) for an opening of {min(w, d) * 100:.0f} cm",
                            {"target": {**plan.checks["target"], "result": "fail"}})
        plan.target.update(region=bin_, object=params.get("object"))
        if w and d:
            plan.data["bin"] = bin_                       # twin preflight adds its walls
        return plan

    def verify(self, ctx, plan, run, res):
        v = super().verify(ctx, plan, run, res)
        b, rel = plan.target["region"], run.data.get("release_tcp")
        v.achieved["target_region"] = b
        if rel is not None and b["size"][0] and b["size"][1]:
            local = np.array(rel[:2]) - np.array(b["xyz_world"][:2])
            a = np.radians(b.get("yaw_deg") or 0.0)
            u = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]]) @ local
            inside = bool(abs(u[0]) <= b["size"][0] / 2 - b["wall"] and abs(u[1]) <= b["size"][1] / 2 - b["wall"])
            v.achieved["release_over_opening"] = inside
            if not inside and v.status == Status.SUCCEEDED:
                v.status, v.termination = Status.PARTIAL, "released outside the bin opening"
        return v


def _wipe_limits() -> dict:
    from skills.wipe import limits as wl
    return wl.load()


class Wipe(Skill):
    """skills/wipe proposals executed through the executor's phases. One proposal = one stroke (from_xy_world ->
    to_xy_world, `passes` times there and back) with the FIXED contact settings of the hardware-validated wipe tool
    (sim/real_arm_mcp.wipe, values from the robot profile via skills/wipe/limits.py). The proposal's attached `limits`
    must equal the executor's own; a mismatch is rejected (nothing is adapted here)."""
    name, version, freshness_s = "wipe", "1", 30.0
    cameras = ()          # one plan's strokes share the inspection observation; the board is re-imaged after the park
    params_doc = {"from_xy_world": "[x, y] metres, world", "to_xy_world": "[x, y] metres, world", "passes": "1..max (there/back)",
                  "limits": "the fixed contact settings (skills/wipe/limits.py), must match the executor's",
                  "plan_id": "groups the strokes of one coverage plan (the evaluator scores them together)",
                  "lines_world_xy": "alternative: several strokes [[x0, y0, x1, y1], ...] in one execution"}
    doc = ("Wipe with a held tissue: above the stroke start, guarded descent until the fingertips feel the table, light "
           "pressure (contact - backoff), straight stroke(s), lift. Hard resistance aborts. The whole path is solved first.")
    contact_label = "stroke"

    def strokes_of(self, ctx, params) -> list[tuple[np.ndarray, np.ndarray]]:
        if params.get("lines_world_xy"):
            return [(_vec(l[:2], 2, "lines_world_xy"), _vec(l[2:4], 2, "lines_world_xy")) for l in params["lines_world_xy"]]
        return [(_vec(params.get("from_xy_world"), 2, "from_xy_world"), _vec(params.get("to_xy_world"), 2, "to_xy_world"))]

    def plan(self, ctx, params):
        L = _wipe_limits()
        given = params.get("limits")
        if given:
            diff = {k: (given[k], L[k]) for k in L if k in given and given[k] != L[k]}
            if diff:
                raise PlanError(f"wipe limits in the proposal differ from the executor's (proposal, executor): {diff}",
                                {"limits": {"result": "fail", "differs": diff}})
        strokes = self.strokes_of(ctx, params)
        passes = int(params.get("passes", 1))
        if not 1 <= passes <= L["max_passes"]:
            raise PlanError(f"passes {passes} outside 1..{L['max_passes']}", {"limits": {"result": "fail", "passes": passes}})
        TS, WS = ctx.P.touch, ctx.P.workspace
        floor = TS.nominal_table_tcp_z - TS.max_below
        step = float(ctx.P.wipe.waypoint_every)
        yaw = ctx.kin.yaw_now(ctx.q0)
        paths, info, seed = {}, [], ctx.q0
        for i, (a, b) in enumerate(strokes, 1):
            length = float(np.linalg.norm(b - a))
            if not L["min_stroke_m"] - 1e-9 <= length <= L["max_stroke_m"] + 1e-9:
                raise PlanError(f"stroke {i} is {length * 100:.1f} cm; must be {L['min_stroke_m'] * 100:.0f}..{L['max_stroke_m'] * 100:.0f} cm",
                                {"limits": {"result": "fail", "stroke": i, "length_m": round(length, 4)}})
            approach, _, _ = self.travel(ctx, np.r_[a, L["start_height_m"]], seed, yaw, label=f"stroke {i} approach")
            free, steps = self.descent(ctx, a[0], a[1], L["start_height_m"], floor, yaw, approach[-1])
            zc = max(floor, ctx.table_z(a[0], a[1]) - L["backoff_m"])
            n = max(2, int(length / step) + 1)
            line, _ = self.solve_chain(ctx, [np.r_[a + (b - a) * k / n, zc] for k in range(n + 1)], (steps or free or approach)[-1],
                                       yaw, f"stroke {i}", min_z=floor)
            end = b if passes % 2 else a
            lift, _ = self.solve_chain(ctx, [np.r_[end, L["lift_after_m"]]], line[-1] if passes % 2 else line[0], yaw, f"stroke {i} lift")
            paths.update({f"approach {i}": approach, f"free {i}": free, f"descent {i}": steps, f"{self.contact_label} {i}": line,
                          f"lift {i}": lift})
            info.append({"from": a.tolist(), "to": b.tolist(), "length_m": round(length, 4), "planned_contact_z": round(zc, 4),
                         "waypoints": len(line)})
            seed = lift[-1]
        n_wp = sum(len(v) for v in paths.values())
        return Plan(paths, {"reachability": {"result": "pass", "waypoints": n_wp},
                            "limits": {"result": "pass", "hard_load_sum": L["hard_load_sum"], "backoff_m": L["backoff_m"],
                                       "passes": passes, "strokes": len(strokes)}},
                    {"strokes": info, "passes": passes, "yaw": yaw, "plan_id": params.get("plan_id")},
                    {"contact_paths": [k for k in paths if k.startswith(("descent", self.contact_label))], "limits": L})

    def execute(self, ctx, plan, run):
        m, L = ctx.motion, plan.data["limits"]
        A, TS = ctx.kin.arm_ids, ctx.P.touch
        held0, _ = m.probe_hold()
        run.data.update(held_before=held0, strokes_world=[], contact_z=[], passes_done=0, lifted=False)
        run.note("holding something (the tissue)" if held0 else "nothing is held: wiping with the bare fingertips")
        for i, s in enumerate(plan.target["strokes"], 1):
            res = go_path(ctx, run, f"stroke {i} approach", plan.paths[f"approach {i}"] + plan.paths[f"free {i}"])
            if res.status != Status.SUCCEEDED:
                return res
            run.data["in_contact"] = True
            res, hit = m.descend_guarded(plan.paths[f"descent {i}"])
            run.note(f"stroke {i} down: {res.termination}")
            if res.status != Status.SUCCEEDED:
                return res
            run.data["contact_z"].append(hit)
            if hit is None:
                run.data["no_contact"] = True
                return MotionResult(Status.PARTIAL, f"stroke {i}: the table was not felt down to the lowest allowed height")
            ctx.remember_touch(*s["from"], hit)
            line = plan.paths[f"{self.contact_label} {i}"]
            z = max(TS.nominal_table_tcp_z - TS.max_below, hit - L["backoff_m"])
            if abs(z - s["planned_contact_z"]) > 0.003:            # felt height differs from the plan: same line, at the felt height
                a, b = np.array(s["from"]), np.array(s["to"])
                pts = [np.r_[a + (b - a) * k / (len(line) - 1), z] for k in range(len(line))]
                try:
                    line, _ = self.solve_chain(ctx, pts, m.q_now(), plan.target["yaw"], f"stroke {i} (felt height)",
                                               min_z=TS.nominal_table_tcp_z - TS.max_below)
                except PlanError as e:
                    return MotionResult(Status.PARTIAL, f"stroke {i}: {e.reason}")
                run.note(f"stroke {i}: table felt at z={hit:.3f}, stroke re-solved at z={z:.3f} (planned {s['planned_contact_z']:.3f})")
            start = ctx.kin.fk(m.q_now())[0]
            hard = 0
            for p in range(plan.target["passes"]):
                for q in (line[1:] if p % 2 == 0 else line[-2::-1]):
                    res = m.go_q(q, settle=False, dt=ctx.P.wipe.dt)
                    if res.status != Status.SUCCEEDED:
                        run.data["strokes_world"].append([start[:2].tolist(), ctx.kin.fk(m.q_now())[0][:2].tolist()])
                        return res
                    loads = sorted((m.read()[a]["load"] for a in A), reverse=True)
                    hard = hard + 1 if loads[0] + loads[1] >= L["hard_load_sum"] else 0
                    if hard >= L["hard_reads"]:
                        m.b.hold()
                        run.data["strokes_world"].append([start[:2].tolist(), ctx.kin.fk(m.q_now())[0][:2].tolist()])
                        return MotionResult(Status.PARTIAL, f"stroke {i} pass {p + 1}: hard resistance (motor loads "
                                                            f"{loads[0]}+{loads[1]} >= {L['hard_load_sum']})", contact=True)
                end = ctx.kin.fk(m.q_now())[0]
                run.data["strokes_world"].append([start[:2].tolist(), end[:2].tolist()])
                start = end
                run.data["passes_done"] += 1
            lift = plan.paths[f"lift {i}"]
            if abs(z - s["planned_contact_z"]) > 0.003:
                lift, _ = self.solve_chain(ctx, [ctx.kin.fk(lift[-1])[0]], m.q_now(), plan.target["yaw"], f"stroke {i} lift")
            res = go_path(ctx, run, f"stroke {i} lift", lift)
            run.data["in_contact"] = False
            if res.status != Status.SUCCEEDED:
                return res
        run.data["lifted"] = True
        held1, _ = m.probe_hold()
        run.data["held_after"] = held1
        if held0 and not held1:
            run.note("the tissue is no longer in the jaws: it stayed on the table")
        return MotionResult(Status.SUCCEEDED, f"wiped {len(plan.target['strokes'])} stroke(s) x {plan.target['passes']} pass(es)")

    def verify(self, ctx, plan, run, res):
        d = run.data
        p_tcp = ctx.kin.fk(ctx.motion.q_now())[0]
        ach = {"strokes_world": d.get("strokes_world", []), "contact_z": d.get("contact_z", []), "passes_done": d.get("passes_done", 0),
               "holding_before": d.get("held_before"), "holding_after": d.get("held_after"), "tcp_world": np.round(p_tcp, 4).tolist()}
        want = len(plan.target["strokes"]) * plan.target["passes"]
        resid = {"passes_missing": want - ach["passes_done"]}
        if res.status != Status.SUCCEEDED:
            return Verdict(res.status, res.termination, ach, resid)
        if d.get("held_before") and not d.get("held_after"):
            return Verdict(Status.PARTIAL, "wiped, but the tissue was lost from the jaws", ach, resid)
        return Verdict(Status.SUCCEEDED, f"{res.termination}; table felt at z={[round(z, 3) for z in ach['contact_z']]}", ach, resid)

    def recover(self, ctx, plan, run, res):
        """Stopped on the table: lift straight up from where the TCP is now."""
        if not run.data.get("in_contact") or res.status == Status.CANCELLED:
            return None
        m = ctx.motion
        p = ctx.kin.fk(m.q_now())[0]
        top = plan.data["limits"]["lift_after_m"]
        try:
            path, _ = self.solve_chain(ctx, [np.r_[p[:2], z] for z in np.linspace(p[2], top, 4)[1:]], m.q_now(), plan.target["yaw"],
                                       "recover lift", min_z=ctx.P.touch.nominal_table_tcp_z - ctx.P.touch.max_below)
        except PlanError as e:
            m.b.hold()
            return f"could not plan a vertical lift: {e.reason}; holding"
        r = go_path(ctx, run, "recover: lift straight up", path)
        return f"lifted straight up to z={top:.3f}: {r.status.value} {r.termination}"


class WipeRegion(Wipe):
    """A world rectangle (e.g. Gemini's wipe_region) covered by skills/wipe's boustrophedon planner, all strokes in one
    execution (incumbent skill version unless `w_hat` / `beta` are given)."""
    name = "wipe_region"
    params_doc = {"region_world_xy": "[[x0, y0], [x1, y1]] metres, world", "budget": "max strokes (default 6)",
                  "w_hat": "optional effective width m", "lines_world_xy": "optional explicit strokes (overrides the planner)"}
    doc = "Cover a table rectangle with wipe strokes (skills/wipe planner), guarded contact on each, then lift."

    def strokes_of(self, ctx, params):
        if params.get("lines_world_xy"):
            return super().strokes_of(ctx, params)
        from skills.wipe import Region, WipeSkill
        (x0, y0), (x1, y1) = params["region_world_xy"]
        sk = WipeSkill()
        v = sk.seed_version(**{k: params[k] for k in ("w_hat", "beta") if k in params})
        wp = sk.plan(v, {"region": Region(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), "budget": int(params.get("budget", 6))})
        return [(np.array(s.start), np.array(s.end)) for s in wp.strokes]


class Inspect(Skill):
    name, version, freshness_s, cameras, moves = "inspect", "1", 30.0, (), False
    params_doc = {"camera": "overview | wrist (informational)"}
    doc = "No motion: acknowledges a request for a fresh look; observe() returns the images."

    def plan(self, ctx, params):
        return Plan({}, {"motion": {"result": "pass", "moves": False}}, {"camera": params.get("camera", "overview")})

    def execute(self, ctx, plan, run):
        return MotionResult(Status.SUCCEEDED, "no motion: observe() for fresh images")

    def verify(self, ctx, plan, run, res):
        return Verdict(res.status, res.termination, {}, {})


class Stop(Inspect):
    name = "stop"
    params_doc = {"reason": "text"}
    doc = "Hold position (an advisor's 'stop'). To stop a running execution, the operator cancels it instead."

    def plan(self, ctx, params):
        return Plan({}, {"motion": {"result": "pass", "moves": False}}, {"reason": str(params.get("reason", ""))})

    def execute(self, ctx, plan, run):
        ctx.motion.b.hold()
        return MotionResult(Status.SUCCEEDED, f"holding position: {plan.target['reason'] or 'stop requested'}")


SKILLS: dict[str, Skill] = {s.name: s for s in (MoveTcp(), Park(), Gripper(), Pick(), Place(), Stir(), TouchTable(), PickAt(), DropAt(),
                                                Wipe(), WipeRegion(), Inspect(), Stop())}
