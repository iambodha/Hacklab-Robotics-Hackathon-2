"""Adapter skill for the executor (registered with install(ex)): the executor's `pick` bundles hover + guarded
descent + close + lift in one proposal; the task wants each as its own proposal, and the existing skills cover all
of them except a guarded descent that does NOT close. Built only from executor pieces (Skill.descent,
Motion.descend_guarded, go_path), so it obeys the same preflight / validation / PARTIAL rules.
"""
from __future__ import annotations

import numpy as np

from executor.contracts import Status
from executor.skills import Plan, PlanError, Skill, Verdict, _vec, go_path


class DescendGuarded(Skill):
    name, version, freshness_s = "descend_guarded", "1", 2.0
    params_doc = {"xy_world": "[x, y] metres, world: where to go straight down (the TCP must already be above it, within 1.5 cm)",
                  "z_world": "TCP height to stop at (metres); contact on the way stops earlier",
                  "yaw_deg": "world yaw of the jaw closing axis to hold (omit = keep current)"}
    doc = ("Straight down from the current height: free to the guard zone, then small effort-limited steps that stop on "
           "contact. Does not touch the gripper. SUCCEEDED = reached z without contact; PARTIAL = stopped by contact or short.")
    xy_tol = 0.015

    def plan(self, ctx, params):
        x, y = _vec(params.get("xy_world"), 2, "xy_world")
        z = float(params.get("z_world"))
        p0 = ctx.kin.fk(ctx.q0)[0]
        off = float(np.hypot(p0[0] - x, p0[1] - y))
        if off > self.xy_tol:
            raise PlanError(f"descend_guarded: the TCP is {off * 1000:.0f} mm from above {np.round([x, y], 3).tolist()} "
                            f"(must be within {self.xy_tol * 1000:.0f} mm: approach first)",
                            {"precondition": {"result": "fail", "xy_offset_mm": round(off * 1000, 1)}})
        if z >= p0[2]:
            raise PlanError(f"descend_guarded: target z {z:.3f} is not below the TCP ({p0[2]:.3f})")
        yaw = float(params["yaw_deg"]) if params.get("yaw_deg") is not None else ctx.kin.yaw_now(ctx.q0)
        free, steps = self.descent(ctx, x, y, float(p0[2]), z, yaw, ctx.q0)
        return Plan({"free": free, "descent": steps},
                    {"reachability": {"result": "pass", "waypoints": len(free) + len(steps)},
                     "precondition": {"result": "pass", "xy_offset_mm": round(off * 1000, 1)}},
                    {"xyz_world": [x, y, z]})

    def execute(self, ctx, plan, run):
        res = go_path(ctx, run, "descend (free)", plan.paths["free"])
        if res.status != Status.SUCCEEDED:
            return res
        res, hit = ctx.motion.descend_guarded(plan.paths["descent"])
        run.note("descend: " + res.termination)
        run.data["contact_z"] = hit
        return res

    def verify(self, ctx, plan, run, res):
        v = self.tcp_verdict(ctx, np.array(plan.target["xyz_world"]), tol_m=0.006, label="descent")
        hit = run.data.get("contact_z")
        v.achieved["contact_z"] = hit
        if res.status != Status.SUCCEEDED:
            return Verdict(res.status, res.termination, v.achieved, v.residuals)
        if hit is not None:
            return Verdict(Status.PARTIAL, f"contact at z={hit:.3f} before the target z={plan.target['xyz_world'][2]:.3f}",
                           v.achieved, v.residuals)
        return v


ADAPTER_SKILLS = (DescendGuarded(),)


def install(ex) -> list[str]:
    """Register the adapter skills on an Executor instance (no change to executor/)."""
    for s in ADAPTER_SKILLS:
        ex.skills[s.name] = s
    return [s.name for s in ADAPTER_SKILLS]
