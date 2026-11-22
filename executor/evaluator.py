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

    if p.skill in ("wipe", "wipe_region") or (p.skill == "park" and p.params.get("plan_id") in ex.wipe_plans):
        return _wipe_outcome(ex, eid, p, obs, ev)

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

    if p.skill in ("pick", "pick_at"):
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

    if p.skill in ("place", "drop_at"):
        rel = r.achieved.get("release_tcp_world")
        if not rel or not plan:
            return Outcome(eid, "release_error_mm", float("nan"), "mm", None, ev)
        if truth and truth.get("objects"):
            objs = [(float(np.linalg.norm(ex.kin.to_world(o["xyz"])[:2] - np.array(rel[:2]))), n, o) for n, o in truth["objects"].items()]
            d, name, o = min(objs)
            return Outcome(eid, "object_to_release_xy_m", round(d, 3), "m", bool(o.get("in_bin")) if d < 0.1 else False,
                           ev + [f"{name} in_bin={o.get('in_bin')} (sim truth)"])
        region, what = plan.target.get("region"), plan.target.get("object")
        if p.skill == "drop_at" and what and region and region["size"][0]:
            return _object_in_region(ex, eid, obs, what, region, ev)
        err = float(np.linalg.norm(np.array(rel) - np.array(plan.target["xyz_world"]))) * 1000
        return Outcome(eid, "release_error_mm", round(err, 1), "mm", None if not joints_fresh else err < 10, ev +
                       ["release point from the measured joints at release; whether the object landed needs a camera check"])

    return Outcome(eid, "status", 0.0, "", None, ev + [f"no evaluator for {p.skill}"])


def _camera(info: dict):
    """skills.eval Camera (K, T_world_cam OpenCV axes) from the executor's overview camera info."""
    from skills.eval import Camera
    K = np.array([[info["fx"], 0, info["cx"]], [0, info["fy"], info["cy"]], [0, 0, 1]], float)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.array(info["R_base_cam"], float), np.array(info["t_base_cam"], float)
    dist = np.array(info.get("dist") or [], float)[:5]
    return Camera(K, T, (int(info["width"]), int(info["height"])), dist if np.any(np.abs(dist) > 1e-9) else None)


def arm_over_board(ex, q, board, cam, margin: float = 0.01) -> bool:
    """Does any part of the arm (the body origins on the base -> TCP chain, the TCP itself, and the segments between
    them) project onto the board in the overview camera? Rays from the camera centre through the arm points,
    intersected with the board plane. The TCP matters: the wrist body origin can sit outside the board's shadow while
    the jaws below it cover the board."""
    K = ex.kin.kin
    m = K.model
    chain, b = [], int(m.site_bodyid[K.site])
    while b > 0:
        chain.append(b)
        b = int(m.body_parentid[b])
    with ex.kin.lock:
        ex.kin.fk(q)
        pts = np.array([ex.kin.to_world(K.data.xpos[b]) for b in chain[::-1]] + [ex.kin.to_world(K.data.site_xpos[K.site])])
    pts = np.concatenate([pts] + [pts[:-1] + (pts[1:] - pts[:-1]) * t for t in (0.25, 0.5, 0.75)])
    c = cam.T_world_cam[:3, 3]
    for p in pts:
        if p[2] >= c[2] - 1e-6:
            continue
        x = c + (p - c) * (c[2] - board.z) / (c[2] - p[2])
        if board.x0 - margin <= x[0] <= board.x1 + margin and board.y0 - margin <= x[1] <= board.y1 + margin:
            return True
    return False


def _wipe_outcome(ex, eid, p, obs, ev) -> Outcome:
    """skills/eval/wipe_eval on the plan's BEFORE overview image (first stroke's observation) and a NEW one, with the
    strokes the executor MEASURED (TCP at contact and at the end of each pass), never the commanded ones."""
    from skills.eval import Board, evaluate as wipe_evaluate
    from skills.eval.synth import from_jpeg
    pid = p.params.get("plan_id") or eid
    wp = ex.wipe_plans.get(pid)
    metric = "contamination_reduction"
    if wp is None:
        return Outcome(eid, metric, float("nan"), "fraction", None, ev + [f"no wipe plan {pid}"])
    strokes = [tuple(map(tuple, s)) for e in wp["eids"] if ex.status(e) for s in ex.status(e).achieved.get("strokes_world", [])]
    if not strokes:
        return Outcome(eid, metric, float("nan"), "fraction", None, ev + [f"plan {pid}: no executed strokes"])
    if wp["board_world_xy"]:
        (x0, y0), (x1, y1) = wp["board_world_xy"]
        board = Board(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    else:
        xy = np.array([pt for s in strokes for pt in s])
        board = Board(*(xy.min(0) - 0.05), *(xy.max(0) + 0.05))
    info = ex.camera_info().get("overview")
    after = ex.frame_jpeg(obs.obs_id, "overview")
    if info is None or not wp["before"] or not after:
        return Outcome(eid, metric, float("nan"), "fraction", None, ev + ["no calibrated overview camera / images"])
    cam = _camera(info)
    f = obs.frames.get("overview")
    in_view = arm_over_board(ex, np.array(obs.joints.rad, float), board, cam)
    res = wipe_evaluate(from_jpeg(wp["before"]), from_jpeg(after), H=cam.H_img_world(), board=board, execution_id=eid,
                        strokes=strokes, evidence=ev + [f"before: observation {wp['before_obs']}", f"plan {pid}: {len(strokes)} measured strokes",
                                                        f"board {[round(v, 3) for v in (board.x0, board.y0, board.x1, board.y1)]}"],
                        arm_in_view=in_view, stale=bool(wp["before_stale"] or (f is not None and f.stale)))
    ex.wipe_plans[pid]["last_evaluation"] = {"execution_id": eid, "stats": res.stats}
    o = res.outcome
    o.evidence.append("evaluator: skills/eval/wipe_eval (independent of the executor's result)")
    return o


def _object_in_region(ex, eid, obs, what: str, region: dict, ev: list) -> Outcome:
    """Camera check after a drop: is the object seen inside the bin footprint? Not seen = unknown (it may be hidden
    by the rim), never success."""
    from .perception import colour_in
    col = colour_in(what)
    seen = [o for o in obs.objects if o.visible and (o.name.lower() == what.lower() or (col and colour_in(o.name) == col))]
    if not seen:
        return Outcome(eid, "object_in_region", float("nan"), "bool", None,
                       ev + [f"{what} not visible after the drop (inside the bin below the rim, or occluded): unknown"])
    o = seen[0]
    local = np.array(o.xyz_world[:2]) - np.array(region["xyz_world"][:2])
    a = np.radians(region.get("yaw_deg") or 0.0)
    u = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]]) @ local
    inside = bool(abs(u[0]) <= region["size"][0] / 2 and abs(u[1]) <= region["size"][1] / 2)
    return Outcome(eid, "object_in_region", float(inside), "bool", inside,
                   ev + [f"{o.name} seen at {o.xyz_world} (sigma {o.uncertainty_mm} mm, {o.source}): "
                         + ("inside" if inside else "outside") + " the target footprint"])
