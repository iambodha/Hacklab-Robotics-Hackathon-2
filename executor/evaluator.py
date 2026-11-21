"""Independent outcome evaluation: measured from a NEW observation taken after the execution (and simulator ground
truth when the backend is the twin), never from the executor's own result. Unknown stays unknown (success=None)."""
from __future__ import annotations

import numpy as np

from .contracts import Outcome, Status


def evaluate(ex, eid: str) -> Outcome:
    r, p, plan = ex.status(eid), ex.proposals.get(eid), ex.plans.get(eid)
    if r is None or p is None:
        return Outcome(eid, "unknown", float("nan"), "", None, [f"no execution {eid}"])
    if r.status in (Status.REJECTED, Status.QUEUED, Status.RUNNING):
        return Outcome(eid, "not_executed", 0.0, "", None, [f"status {r.status.value}"])
    obs = ex.observe()
    ev = [f"observation {obs.obs_id} (freshness {obs.freshness_s:.2f} s)"] + [f"{role} frame seq {f.seq}" + (" STALE" if f.stale else "")
                                                                            for role, f in obs.frames.items()]
    if r.recording_id:
        ev.append(f"recording {r.recording_id}")
    joints_fresh = ex.now() - obs.joints.t_mono < 1.0
    q = np.array(obs.joints.rad, float)
    tcp = ex.kin.fk(q)[0]
    truth = ex.backend.truth() if hasattr(ex.backend, "truth") else None
    if truth:
        ev.append(f"sim ground truth at t={truth.get('sim_time')}")

    if p.skill in ("move_tcp", "park"):
        target = np.array(plan.target["xyz_world"]) if plan else None
        if target is None or not joints_fresh:
            return Outcome(eid, "tcp_error_mm", float("nan"), "mm", None, ev + ["joints not fresh"])
        err = float(np.linalg.norm(tcp - target)) * 1000
        return Outcome(eid, "tcp_error_mm", round(err, 1), "mm", err <= ex.kin.P.motion.off_target_note * 1000, ev)

    if p.skill == "gripper":
        want = float(p.params.get("open_fraction", 0))
        got = obs.joints.gripper_open
        err = abs(want - got) if got is not None else float("nan")
        held = obs.joints.load[-1] >= ex.kin.P.gripper.holding_load and got is not None and got > ex.kin.P.gripper.closed_norm
        if want < got and held:
            return Outcome(eid, "jaw_error", round(err, 3), "norm", None, ev + ["jaws stopped on something: closed as far as possible"])
        return Outcome(eid, "jaw_error", round(err, 3), "norm", (err <= 0.03) if joints_fresh else None, ev)

    if p.skill == "pick":
        if truth:
            xy = np.array(plan.target["xy"]) if plan else tcp[:2]
            best = None
            for name, o in truth.get("objects", {}).items():
                pw = ex.kin.to_world(o["xyz"])
                d = float(np.linalg.norm(pw[:2] - xy))
                if best is None or d < best[0]:
                    best = (d, name, pw)
            if best:
                lifted = float(best[2][2]) > 0.04 and float(np.linalg.norm(best[2][:2] - tcp[:2])) < 0.05
                return Outcome(eid, "object_lifted_m", round(float(best[2][2]), 3), "m", lifted, ev + [f"nearest object {best[1]}"])
        g = obs.joints.gripper_open
        load = obs.joints.load[-1]
        GR = ex.kin.P.gripper
        if g is None or not joints_fresh:
            return Outcome(eid, "held", float("nan"), "bool", None, ev)
        held = g > GR.closed_norm and load >= GR.holding_load and g < GR.open_enough
        return Outcome(eid, "held", float(held), "bool", True if held else None,
                       ev + [f"jaws {g:.3f}, load {load}" + ("" if held else ": cannot tell from touch alone without motion")])

    if p.skill == "stir":
        before = ex.observation(p.based_on_obs)
        c = np.array(plan.target["centre"]) if plan else None
        if c is None or before is None:
            return Outcome(eid, "cup_displacement_mm", float("nan"), "mm", None, ev)

        def near(objs):
            vis = [o for o in objs if o.visible and np.linalg.norm(np.array(o.xyz_world[:2]) - c[:2]) < 0.08]
            return min(vis, key=lambda o: np.linalg.norm(np.array(o.xyz_world[:2]) - c[:2])) if vis else None
        a, b = near(before.objects), near(obs.objects)
        if a is None or b is None:
            return Outcome(eid, "cup_displacement_mm", float("nan"), "mm", None, ev + ["cup not visible before and after"])
        d = float(np.linalg.norm(np.array(a.xyz_world[:2]) - np.array(b.xyz_world[:2]))) * 1000
        return Outcome(eid, "cup_displacement_mm", round(d, 1), "mm", d < 15 if r.status == Status.SUCCEEDED else False, ev)

    if p.skill == "place":
        rel = r.achieved.get("release_tcp_world")
        if not rel or not plan:
            return Outcome(eid, "release_error_mm", float("nan"), "mm", None, ev)
        if truth:
            objs = [(float(np.linalg.norm(ex.kin.to_world(o["xyz"])[:2] - np.array(rel[:2]))), n, o) for n, o in truth["objects"].items()]
            d, name, o = min(objs)
            return Outcome(eid, "object_to_release_xy_m", round(d, 3), "m", bool(o.get("in_bin")) if d < 0.1 else False,
                           ev + [f"{name} in_bin={o.get('in_bin')} (sim truth)"])
        err = float(np.linalg.norm(np.array(rel) - np.array(plan.target["xyz_world"]))) * 1000
        return Outcome(eid, "release_error_mm", round(err, 1), "mm", None if not joints_fresh else err < 10, ev +
                       ["release point from the measured joints at release; whether the object landed needs a camera check"])

    return Outcome(eid, "status", 0.0, "", None, ev + [f"no evaluator for {p.skill}"])
