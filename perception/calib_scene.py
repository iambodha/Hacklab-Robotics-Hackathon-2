"""ArUco scene calibration -> calibration/scene_calib.json (scene-contract.md).

  python -m perception.calib_scene --live              # grab frames from Station, calibrate, write the json
  python -m perception.calib_scene --frames-dir DIR    # offline, from overview_*.jpg / wrist_*.jpg + joints.json

Steps
 1. Median marker corners over many overview frames (dictionary auto-detected, edge-refined corners).
 2. Joint fit: overview camera pose + every marker's (x, y, yaw) on one shared table plane (+ focal length
    when no calibrated intrinsics exist; see perception/calib_intrinsics.py for the board procedure).
 3. Arm base on the table (defines the world frame; perception/arm_base.py). Sources, in the order "auto" tries:
      tape         tape distances foot front corners -> marker corners (scene_config.json arm_base.tape_mm; README)
      foot+wrist   overview: the arm foot's front edge (position); wrist: a marker through FK (mostly yaw)
      foot         foot only, yaw assumed square to the base-marker line
      wrist        the wrist camera sees a calibrated marker; FK (URDF + CAD mount) closes the loop
      midpoint     midpoint of the base markers, facing the far markers (rough fallback)
 4. Hold-one-out accuracy on the markers, wrist consistency, annotated image, json.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import math
from pathlib import Path

import cv2
import numpy as np

from . import accuracy, arm_base, aruco, geometry as g, scene_fit, wrist
from .scene import CALIB_DIR, CONFIG_PATH, SCENE_PATH, intrinsics_file, load_intrinsics
from .station_io import QUEUES, ROOT

W, H = 299, 224


# ---- capture ------------------------------------------------------------------------------------------
async def _capture(n: int) -> dict:
    from . import station_io as sio
    cli = await sio.connect()
    j0 = await sio.read_joints(cli)
    ov, wr = await asyncio.gather(sio.grab_jpegs(cli, "overview", n), sio.grab_jpegs(cli, "wrist", n))
    j1 = await sio.read_joints(cli)
    return {"overview": ov, "wrist": wr, "joints": j0, "joints_after": j1}


async def _capture_wrist_poses(n_poses: int, timeout_s: float = 240, still_s: float = 1.5, min_ticks: int = 60):
    """The user moves the limp arm BY HAND (torque stays off; nothing is commanded). A pose is recorded
    when the joints have been still for still_s and differ from every recorded pose by > min_ticks."""
    import time
    from . import station_io as sio
    cli = await sio.connect()
    poses, last, still_since, t0 = [], None, None, time.time()
    print(f"Wrist poses: move the arm by hand so the wrist camera sees one or more markers from {n_poses} different "
          f"poses (vary base rotation and wrist tilt), and hold each still for ~2 s.")
    while len(poses) < n_poses and time.time() - t0 < timeout_s:
        j = await sio.read_joints(cli)
        if last is not None and _joints_moved(last, j, tol=4) is None:
            still_since = still_since or time.time()
        else:
            still_since = None
        last = j
        new = all(_joints_moved(p["joints"], j, tol=min_ticks) for p in poses)
        if still_since and time.time() - still_since > still_s and new:
            frames = await sio.grab_jpegs(cli, "wrist", 15)
            if _joints_moved(j, await sio.read_joints(cli), tol=4) is None:
                poses.append({"joints": j, "jpegs": frames})
                print(f"  pose {len(poses)}/{n_poses} recorded")
        await asyncio.sleep(0.3)
    return poses


def capture_live(n: int, save_dir: Path | None, wrist_poses: int = 0) -> dict:
    cap = asyncio.run(_capture(n))
    extra = asyncio.run(_capture_wrist_poses(wrist_poses)) if wrist_poses else []
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        for cam in ("overview", "wrist"):
            for k, b in enumerate(cap[cam][:: max(1, n // 10)]):      # keep at most ~10 per camera
                (save_dir / f"{cam}_{k:02d}.jpg").write_bytes(b)
        for p, pose in enumerate(extra):
            for k, b in enumerate(pose["jpegs"][::3]):
                (save_dir / f"pose{p + 1:02d}_wrist_{k:02d}.jpg").write_bytes(b)
        (save_dir / "joints.json").write_text(json.dumps({"joints": cap["joints"], "joints_after": cap["joints_after"],
                                                          "poses": [p["joints"] for p in extra]}))
    return {"overview": [aruco_decode(b) for b in cap["overview"]], "wrist": [aruco_decode(b) for b in cap["wrist"]],
            "joints": cap["joints"], "joints_after": cap["joints_after"],
            "wrist_poses": [{"joints": p["joints"], "frames": [aruco_decode(b) for b in p["jpegs"]]} for p in extra]}


def aruco_decode(b: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)


def load_dir(d: Path) -> dict:
    j = json.loads((d / "joints.json").read_text()) if (d / "joints.json").exists() else {}
    poses = [{"joints": pj, "frames": [cv2.imread(str(p)) for p in sorted(d.glob(f"pose{k + 1:02d}_wrist_*.jpg"))]}
             for k, pj in enumerate(j.get("poses", []))]
    return {"overview": [cv2.imread(str(p)) for p in sorted(d.glob("overview_*.jpg"))],
            "wrist": [cv2.imread(str(p)) for p in sorted(d.glob("wrist_*.jpg"))],
            "joints": j.get("joints"), "joints_after": j.get("joints_after"), "wrist_poses": poses}


# ---- arm base / world frame ----------------------------------------------------------------------------
def _base_axes(fit: scene_fit.SceneFit, base_ids: list[int]):
    """Midpoint of the base markers and the in-plane unit vector perpendicular to their line,
    pointing towards the other markers (the arm faces into the table)."""
    c = {i: np.array(fit.markers[i][:2]) for i in fit.markers}
    a, b = c[base_ids[0]], c[base_ids[1]]
    mid, line = (a + b) / 2, (b - a) / np.linalg.norm(b - a)
    perp = np.array([-line[1], line[0]])
    far = [c[i] for i in c if i not in base_ids]
    if far and (np.mean(far, 0) - mid) @ perp < 0:
        perp = -perp
    return mid, perp, c


def T_table_base_from(p, fwd_yaw: float, base_z: float) -> np.ndarray:
    """Base frame on the table whose +y (URDF reach direction) points along fwd_yaw."""
    return g.T_from(g.rot_z(fwd_yaw - math.pi / 2), [p[0], p[1], base_z])


# ---- main calibration -----------------------------------------------------------------------------------
def calibrate(data: dict, cfg: dict, fit_k1: bool = False, log=print) -> dict:
    size = float(cfg["marker_size_m"])
    ov = [f for f in data["overview"] if f is not None]
    if not ov:
        raise SystemExit("no overview frames")
    name = cfg.get("dictionary", "auto")
    seen = {}
    if name == "auto":
        name, seen = aruco.autodetect_dictionary(ov[:: max(1, len(ov) // 8)])
        log(f"dictionary candidates: { {k: sorted(v) for k, v in seen.items()} } -> {name}")
    if not name:
        raise SystemExit("no ArUco markers visible in the overview camera")
    med, std, cnt = aruco.average_corners(ov, name)
    log(f"overview: {len(ov)} frames; markers " + ", ".join(f"{i} (in {cnt[i]} frames, corner jitter "
                                                            f"{std[i].max():.2f} px)" for i in sorted(med)))
    if len(med) < 2:
        raise SystemExit(f"need >= 2 markers in the overview, saw {sorted(med)}")
    sig = {i: float(np.sqrt(np.mean(std[i] ** 2))) for i in med}

    intr_ov = load_intrinsics("overview")
    base_ids = [i for i in cfg.get("base_marker_ids", []) if i in med]
    ref_id = base_ids[0] if base_ids else min(med)
    if intr_ov is not None and len(med) >= 3:
        # cross-check: board intrinsics must explain the markers about as well as the marker-only estimate
        f_board = scene_fit.fit(med, size, W, H, intr=intr_ov, sigma_px=sig, ref_id=ref_id)
        f_free = scene_fit.fit(med, size, W, H, intr=None, sigma_px=sig, ref_id=ref_id)
        log(f"board intrinsics check: markers reproject at {f_board.rms_px():.2f} px with them, "
            f"{f_free.rms_px():.2f} px with f fitted to the markers")
        if f_board.rms_px() > 2 * f_free.rms_px() + 0.3:
            log("  -> board intrinsics REJECTED (inconsistent with the markers); recapture them. Using the marker estimate.")
            intr_ov = None
    fit = scene_fit.fit(med, size, W, H, intr=intr_ov, sigma_px=sig, ref_id=ref_id,
                        free_f=intr_ov is None, free_k1=fit_k1)
    if intr_ov is None:
        fit.intr["source"] = "estimated from the coplanar markers (fx=fy, centred principal point" + \
                             (", k1 fitted" if fit_k1 else ", no distortion") + ")"
        fit.intr["sigma_f_rel"] = fit.sigma.get("f_rel")
    log(f"overview fit: rms {fit.rms_px():.3f} px; f = {fit.intr['fx'] * W:.1f} px"
        + (f" +- {fit.sigma['f_px']:.1f}" if "f_px" in fit.sigma else " (calibrated)"))

    # --- arm base (world frame) ---
    ab = cfg.get("arm_base", {})
    base_z = float(ab.get("base_z_m", 0.0))
    src = ab.get("source", "auto")
    T_table_base, base_info = None, {}
    wr_frames = [f for f in data.get("wrist", []) if f is not None]
    joints = data.get("joints")
    wr_med = {}
    if wr_frames:
        wr_med, wr_std, wr_cnt = aruco.average_corners(wr_frames, name)
        log(f"wrist: {len(wr_frames)} frames; markers {sorted(wr_med)}")
    moved = _joints_moved(joints, data.get("joints_after"))
    T_base_wcam = wrist.T_base_wristcam(joints) if joints else None
    intr_wr = load_intrinsics("wrist")
    wrist_est = None
    T_tm = {i: fit.T_table_marker(i) for i in fit.markers}
    w_obs, w_T, w_src = [], [], []
    if T_base_wcam is not None and any(i in T_tm for i in wr_med) and not moved:
        w_obs.append({i: wr_med[i] for i in wr_med if i in T_tm})
        w_T.append(T_base_wcam)
        w_src.append(f"current pose, markers {sorted(w_obs[-1])}")
    for k, pose in enumerate(data.get("wrist_poses") or []):
        pm, _, _ = aruco.average_corners([f for f in pose["frames"] if f is not None], name)
        pm = {i: c for i, c in pm.items() if i in T_tm}
        if pm:
            w_obs.append(pm)
            w_T.append(wrist.T_base_wristcam(pose["joints"]))
            w_src.append(f"{pose.get('source', f'hand pose {k + 1}')}, markers {sorted(pm)}")
    if w_obs:
        wrist_est = wrist.solve_base_from_wrist(w_obs, T_tm, w_T, size, W, H, intr=intr_wr, base_z=base_z)
        if intr_wr is None:
            wrist_est["intr"]["source"] = "estimated jointly with the arm base from FK + markers (rough)"
        log(f"wrist-only estimate of the arm base from {len(w_obs)} pose(s): "
            f"{np.round(wrist_est['T_table_base'][:3, 3], 4).tolist()} (table frame), "
            f"rms {wrist_est['rms_px']:.2f} px, sigma {wrist_est['sigma_mm']:.1f} mm / {wrist_est['sigma_yaw_deg']:.2f} deg")
    # the foot's front edge in the overview
    foot, foot_pt = None, None
    if len(base_ids) == 2:
        gray = np.median(np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in ov]), 0).astype(np.float32)
        foot = arm_base.detect_foot(gray, fit, base_ids)
        if foot:
            foot_pt = arm_base.foot_front_point(foot, fit)
            log(f"foot: front edge at pixel ({foot['uv_centre'][0]:.1f}, {foot['v_edge']:.2f}), columns "
                f"{foot['u_left']:.1f}-{foot['u_right']:.1f}, measured width {foot_pt[2] * 1000:.0f} mm (CAD 100), "
                f"front-centre sigma {np.sqrt(np.linalg.eigvalsh(foot_pt[1])).round(4) * 1000} mm")
        else:
            log("foot: not found between the base markers")
    if foot_pt is not None and not 0.07 < foot_pt[2] < 0.14:
        log(f"  -> foot width {foot_pt[2] * 1000:.0f} mm is implausible (CAD 100 mm): foot detection ignored")
        foot_pt = None
    tape = {k: v for k, v in (ab.get("tape_mm") or {}).items() if v is not None}
    if src in ("tape", "auto") and len(tape) >= 3 and len(base_ids) == 2:
        mid, perp, _ = _base_axes(fit, base_ids)     # start (and pick the marker corners) like tape_guide.jpg did
        T0 = T_table_base_from(mid, math.atan2(perp[1], perp[0]), base_z)
        if foot_pt is not None:
            yaw0 = math.atan2(perp[1], perp[0]) - math.pi / 2
            T0 = g.T_from(g.rot_z(yaw0), [*arm_base.base_from_foot(foot_pt[0], yaw0), base_z])
        elif wrist_est is not None:
            T0 = wrist_est["T_table_base"]
        segs = arm_base.tape_segments(T0, T_tm, size, base_ids)
        t = arm_base.solve_tape(segs, tape, T0, base_z)
        T_table_base = t["T_table_base"]
        base_info = {"source": f"tape ({', '.join(t['used'])}: foot front corners -> marker corners, scene_config.json)",
                     "sigma_mm": t["sigma_mm"], "sigma_yaw_deg": t["sigma_yaw_deg"], "tape_residual_mm": t["residual_mm"]}
        if w_obs:                 # the 100 mm foot baseline leaves yaw at ~1 deg: let the wrist view tighten it
            f_tape, _ = arm_base.tape_residuals(segs, tape)
            est = wrist.solve_base_from_wrist(w_obs, T_tm, w_T, size, W, H, intr=intr_wr or wrist_est["intr"],
                                              base_z=base_z, sigma_px=max(1.0, wrist_est["rms_px"]),
                                              extra_residuals=f_tape)
            T_table_base = est["T_table_base"]
            base_info.update({"source": base_info["source"] + " + wrist (yaw)", "sigma_mm": est["sigma_mm"],
                              "sigma_yaw_deg": est["sigma_yaw_deg"], "tape_only": {k: t[k] for k in ("sigma_mm", "sigma_yaw_deg")},
                              "tape_residual_mm": (f_tape(T_table_base) * arm_base.TAPE_SIGMA_M * 1000).round(2).tolist(),
                              "wrist_rms_px": est["rms_px"], "wrist_poses": w_src})
    elif src in ("foot", "auto") and foot_pt is not None and w_obs:
        it_w = intr_wr or wrist_est["intr"]
        sigma_px = max(1.0, wrist_est["rms_px"])
        est = wrist.solve_base_from_wrist(w_obs, T_tm, w_T, size, W, H, intr=it_w, base_z=base_z,
                                          point_priors=[(arm_base.FOOT_FRONT_CENTRE, foot_pt[0], foot_pt[1])],
                                          sigma_px=sigma_px)
        T_table_base = est["T_table_base"]
        R_wt = (wrist.T_world_base(base_z) @ g.inv(T_table_base))[:2, :2]
        cov = R_wt @ est["cov_xyyaw"][:2, :2] @ R_wt.T                     # world x/y
        base_info = {"source": "foot + wrist (overview: front edge of the arm foot -> position; wrist camera: marker "
                               f"seen through FK at {len(w_obs)} arm pose(s) -> mostly yaw)",
                     "wrist_poses": w_src,
                     "sigma_mm": est["sigma_mm"], "sigma_yaw_deg": est["sigma_yaw_deg"],
                     "sigma_world_x_mm": float(np.sqrt(cov[0, 0]) * 1000), "sigma_world_y_mm": float(np.sqrt(cov[1, 1]) * 1000),
                     "wrist_rms_px": est["rms_px"], "wrist_sigma_px_used": sigma_px,
                     "caveat": "sigma assumes the foot edge to 0.5 px and the wrist model error = its own rms; joint "
                               "signs of j1/j5/j7 unverified. The tape procedure (README) pins it to ~2 mm."}
    elif src in ("foot", "auto") and foot_pt is not None:
        mid, perp, _ = _base_axes(fit, base_ids)
        yaw, s_yaw = math.atan2(perp[1], perp[0]) - math.pi / 2, math.radians(3.0)
        p = arm_base.base_from_foot(foot_pt[0], yaw)
        T_table_base = g.T_from(g.rot_z(yaw), [p[0], p[1], base_z])
        s = math.sqrt(max(np.linalg.eigvalsh(foot_pt[1]))) + arm_base.FOOT_FRONT_Y * s_yaw
        base_info = {"source": "foot (overview: front edge of the arm foot); yaw ASSUMED square to the base-marker line",
                     "sigma_mm": s * 1000, "sigma_yaw_deg": 3.0}
    elif src in ("wrist", "auto") and wrist_est is not None:
        T_table_base = wrist_est["T_table_base"]
        base_info = {"source": "wrist (FK from Station joint ticks + CAD camera mount + markers seen by the wrist)",
                     "sigma_mm": wrist_est["sigma_mm"], "sigma_yaw_deg": wrist_est["sigma_yaw_deg"],
                     "caveat": "statistical sigma only; joint zero offsets/signs and wrist intrinsics are not "
                               "calibrated, so treat as +-2 cm / +-3 deg until checked"}
    elif len(base_ids) == 2:
        mid, perp, _ = _base_axes(fit, base_ids)
        T_table_base = T_table_base_from(mid, math.atan2(perp[1], perp[0]), base_z)
        base_info = {"source": "midpoint of the base markers, facing the far markers (ROUGH: not measured)",
                     "sigma_mm": 50.0, "sigma_yaw_deg": 5.0}
    else:
        T_table_base = g.T_from(g.rot_z(-math.pi / 2), [0, 0, base_z])
        base_info = {"source": f"none: world frame = marker {ref_id} (no base markers, no wrist view)",
                     "sigma_mm": float("nan"), "sigma_yaw_deg": float("nan")}
    if moved:
        base_info["warning"] = f"arm moved during capture ({moved}); wrist estimate skipped"
    if foot_pt is not None:
        yaw_b = math.atan2(T_table_base[1, 0], T_table_base[0, 0])
        p_foot = arm_base.base_from_foot(foot_pt[0], yaw_b)
        base_info["foot"] = {"pixel": np.round(foot["uv_centre"], 2).tolist(), "columns": [foot["u_left"], foot["u_right"]],
                             "width_mm": round(foot_pt[2] * 1000, 1), "row_scatter_px": round(foot["row_scatter_px"], 2),
                             "base_from_foot_alone_mm_off": round(float(np.linalg.norm(p_foot - T_table_base[:2, 3])) * 1000, 1)}
    if len(base_ids) == 2:
        _, perp, _ = _base_axes(fit, base_ids)
        base_info["yaw_vs_base_marker_line_deg"] = round(math.degrees(math.remainder(
            wrist.planar_of(T_table_base)[2] + math.pi / 2 - math.atan2(perp[1], perp[0]), 2 * math.pi)), 2)
    if wrist_est is not None:
        base_info["wrist_only"] = {"mm_off": round(float(np.linalg.norm(wrist_est["T_table_base"][:2, 3] - T_table_base[:2, 3])) * 1000, 1),
                                   "yaw_off_deg": round(math.degrees(math.remainder(
                                       wrist.planar_of(wrist_est["T_table_base"])[2] - wrist.planar_of(T_table_base)[2], 2 * math.pi)), 2),
                                   "rms_px": round(wrist_est["rms_px"], 2), "sigma_mm": round(wrist_est["sigma_mm"], 1)}
    log(f"arm base source: {base_info['source']}  (sigma {base_info['sigma_mm']:.1f} mm, {base_info['sigma_yaw_deg']:.2f} deg)")

    T_world_base = wrist.T_world_base(base_z)
    T_world_table = T_world_base @ g.inv(T_table_base)
    T_world_cam = T_world_table @ fit.T_table_cam

    # --- quality ---
    holdout = accuracy.held_out(med, size, W, H, intr_ov, sig, full=fit)
    holdout_fixed = accuracy.held_out(med, size, W, H, fit.intr, sig, full=fit) if intr_ov is None else None
    per_marker = {}
    for i in sorted(med):
        Pw = g.apply(T_world_table, accuracy.backproject_table(fit, med[i]))      # observed corners on the table
        Pm = g.apply(T_world_table, fit.corners_table(i))
        per_marker[i] = {"reproj_px": fit.rms_px(i), "rms_mm": float(np.sqrt(np.mean(np.sum((Pw - Pm) ** 2, 1))) * 1000),
                         "jitter_px": sig[i], "frames": cnt[i], "sigma_centre_mm": fit.marker_sigma_mm.get(i, 0.0)}
    wrist_check = None
    if T_base_wcam is not None and wr_med:
        it_w = intr_wr or (wrist_est or {}).get("intr")
        if it_w is not None:
            T_world_wcam = T_world_base @ T_base_wcam
            rows = {}
            for i in wr_med:
                if i not in fit.markers:
                    continue
                Pm = g.apply(T_world_table, fit.corners_table(i))
                uv, _ = g.project(T_world_wcam, it_w, W, H, Pm)
                Pw = g.pixel_to_plane(T_world_wcam, it_w, W, H, wr_med[i], 0.0)
                rows[i] = {"px": float(np.sqrt(np.mean(np.sum((uv - wr_med[i]) ** 2, 1)))),
                           "mm": float(np.sqrt(np.mean(np.sum((Pw - Pm) ** 2, 1))) * 1000)}
            wrist_check = {"markers": rows, "note": "marker seen by both cameras: overview-fit corners vs wrist "
                           "pixel->table through FK. Only an independent check when the base is measured or the "
                           "arm pose differs from the one used for the wrist estimate."}

    # --- table extent ---
    if cfg.get("table_extent_xy"):
        extent, extent_src = cfg["table_extent_xy"], "measured (scene_config.json)"
    else:
        allc = np.vstack([g.apply(T_world_table, fit.corners_table(i)) for i in fit.markers])
        lo, hi = allc[:, :2].min(0) - 0.03, allc[:, :2].max(0) + 0.03
        lo[0] = min(lo[0], -0.05)
        extent, extent_src = [lo.round(4).tolist(), hi.round(4).tolist()], "bounding box of the markers + 3 cm (not the real table edge)"

    roles = {i: "base" if i in base_ids else "top" for i in med}
    wcam_cfg = json.loads((ROOT / "camera_pose/wrist_camera.json").read_text())
    intr_w_out = intr_wr or (wrist_est or {}).get("intr")
    calib = {
        "version": 1,
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "world_frame": "follower base_link origin on table plane, z up, x forward at j1=0",
        "table": {"plane_world": [0, 0, 1, 0], "extent_xy": extent, "extent_source": extent_src, "height_m": 0.0},
        "markers": {"dictionary": name, "size_m": size,
                    "items": [{"id": i, "role": roles[i], "T_world_marker": _m(T_world_table @ fit.T_table_marker(i)),
                               "rms_mm": round(per_marker[i]["rms_mm"], 3), "reproj_px": round(per_marker[i]["reproj_px"], 3),
                               "sigma_centre_mm": round(per_marker[i]["sigma_centre_mm"], 2)} for i in sorted(med)],
                    "seen_by_wrist": sorted(wr_med)},
        "cameras": {
            "overview": {"queue": QUEUES["overview"], "image_size": [W, H], "K_normalized": _kn(fit.intr),
                         "dist": g.dist_vec(fit.intr).round(6).tolist(), "intrinsics_source": fit.intr.get("source", "charuco"),
                         "sigma_f_rel": fit.intr.get("sigma_f_rel"),
                         "T_world_cam": _m(T_world_cam), "reproj_px": round(fit.rms_px(), 3),
                         "fovy_deg": round(g.fovy_deg(fit.intr, H), 2)},
            "wrist": {"queue": QUEUES["wrist"], "image_size": [W, H],
                      "K_normalized": _kn(intr_w_out) if intr_w_out else None,
                      "dist": g.dist_vec(intr_w_out).round(6).tolist() if intr_w_out else None,
                      "intrinsics_source": (intr_w_out or {}).get("source", "charuco" if intr_wr else None),
                      "sigma_f_rel": None if intr_wr or not wrist_est else wrist_est.get("sigma_f_rel"),
                      "T_link_cam": _m(np.array(wcam_cfg["T_link_camera"])), "link": "Gripper_Base_v1_1",
                      "source": "CAD (camera_pose/wrist_camera.json)"},
        },
        "arm": {"bus": "5B61033301", "urdf": "norma-core/hardware/elrobot/simulation/elrobot_follower.urdf",
                "joint_mapping": f"{wrist.arm_mapping()} (robot profile sim/robots/elrobot.json, as the executor and "
                                 "twin): q = lower + n*(upper - lower), n = fraction of the calibrated range; "
                                 "signs of j1/j5/j7 unverified",
                "T_world_base": _m(T_world_base), "base_on_table": base_info,
                "j1_axis_world": g.apply(T_world_base, [[0, arm_base.J1_AXIS_Y, 0]])[0].round(4).tolist(),
                "tape_segments": arm_base.tape_segments(T_table_base, T_tm, size, base_ids) if len(base_ids) == 2 else [],
                "joints_at_calibration": joints},
        "quality": {"overview_reproj_px": round(fit.rms_px(), 3), "per_marker": per_marker,
                    "held_out": holdout,
                    "held_out_fixed_intrinsics": holdout_fixed,
                    "held_out_note": "held_out: camera (and f, if not board-calibrated) refitted without the marker, then "
                                     "its sides/diagonals measured on the table. held_out_fixed_intrinsics: same with f "
                                     "fixed to the full-fit value (optimistic: f saw every marker).",
                    "wrist_check": wrist_check,
                    "wrist_base_estimate": None if wrist_est is None else {
                        "T_world_base_est": _m(T_world_table @ wrist_est["T_table_base"]), "rms_px": wrist_est["rms_px"],
                        "sigma_mm": wrist_est["sigma_mm"], "sigma_yaw_deg": wrist_est["sigma_yaw_deg"],
                        "sigma_f_rel": wrist_est.get("sigma_f_rel")},
                    "pixel_footprint_mm": _footprint_mm(T_world_cam, fit.intr, med)},
        "T_world_table_internal": _m(T_world_table),
    }
    return {"calib": calib, "fit": fit, "frame": ov[len(ov) // 2], "wrist_frame": wr_frames[len(wr_frames) // 2] if wr_frames else None,
            "obs": med, "wr_obs": wr_med, "foot": foot, "T_cam_base": fit.T_cam_table @ T_table_base}


def _joints_moved(a, b, tol: int = 8):
    if not a or not b:
        return None
    m = {j: b[j]["raw"] - a[j]["raw"] for j in a if j != "j8" and abs(b[j]["raw"] - a[j]["raw"]) > tol}
    return m or None


def _footprint_mm(T_world_cam, intr, med):
    """Size of one pixel on the table (mm) at each marker: the resolution limit of everything else."""
    out = {}
    for i, c in med.items():
        J = g.pixel_footprint_jacobian(T_world_cam, intr, W, H, c.mean(0))
        out[i] = round(float(np.sqrt(abs(np.linalg.det(J)))) * 1000, 2)
    return out


def _m(T) -> list:
    return np.round(np.asarray(T, float), 6).tolist()


def _kn(intr) -> dict:
    return {k: round(float(intr[k]), 6) for k in ("fx", "fy", "cx", "cy")}


def main():
    ap = argparse.ArgumentParser(description="ArUco scene calibration -> calibration/scene_calib.json")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--live", action="store_true", help="grab frames from Station")
    src.add_argument("--frames-dir", type=Path)
    ap.add_argument("--frames", type=int, default=60, help="frames per camera (live)")
    ap.add_argument("--wrist-poses", type=int, default=0,
                    help="also record N extra hand-placed arm poses with markers in the wrist view (live)")
    ap.add_argument("--extra-wrist-dir", type=Path, action="append", default=[],
                    help="earlier capture dir (frames/calib-*) whose wrist frames + joints.json are added as an extra arm "
                         "pose; only valid if the arm base has not moved since (the foot edge check shows it)")
    ap.add_argument("--fit-k1", action="store_true", help="also fit radial distortion k1 (needs markers spread over the image)")
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--out", type=Path, default=SCENE_PATH)
    ap.add_argument("--dry-run", action="store_true", help="don't write the json")
    ap.add_argument("--force", action="store_true", help="write even if the arm-base source gets weaker")
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text())
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    data = capture_live(a.frames, ROOT / "frames" / f"calib-{stamp}", a.wrist_poses) if a.live else load_dir(a.frames_dir)
    for d in a.extra_wrist_dir:
        old = load_dir(d)
        data["wrist_poses"] = data.get("wrist_poses", []) + [{"joints": old["joints"], "frames": old["wrist"], "source": str(d)}]
    res = calibrate(data, cfg, fit_k1=a.fit_k1)
    c = res["calib"]
    from .annotate import annotate_overview, annotate_wrist
    from .scene import Scene
    sc = Scene.from_dict(c)
    print_report(c)
    good = ("tape", "foot")
    prev = json.loads(a.out.read_text())["arm"]["base_on_table"]["source"] if a.out.exists() else ""
    new = c["arm"]["base_on_table"]["source"]
    if not a.dry_run and not a.force and prev.startswith(good) and not new.startswith(good):
        raise SystemExit(f"NOT written: the arm base would degrade from '{prev[:40]}...' to '{new[:40]}...'. Is something "
                         "standing in front of the arm foot, or did the base move? Clear it and rerun, or pass --force.")
    if not a.dry_run:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(c, indent=1))
        cv2.imwrite(str(CALIB_DIR / "overview_annotated.jpg"), annotate_overview(sc, res["frame"], res["obs"]),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        if res["wrist_frame"] is not None and data.get("joints"):
            cv2.imwrite(str(CALIB_DIR / "wrist_annotated.jpg"),
                        annotate_wrist(sc, res["wrist_frame"], data["joints"], res["wr_obs"]), [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(CALIB_DIR / "arm_base_overview.jpg"),
                    arm_base.annotate_foot(res["frame"], res["foot"], res["T_cam_base"], res["fit"].intr),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        if c["arm"]["tape_segments"]:
            cv2.imwrite(str(CALIB_DIR / "tape_guide.jpg"), arm_base.annotate_tape(sc, res["frame"], c["arm"]["tape_segments"]),
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(f"wrote {a.out} and {CALIB_DIR}/{{overview_annotated,wrist_annotated,arm_base_overview,tape_guide}}.jpg")


def print_report(c: dict):
    q = c["quality"]
    print("\n== scene calibration ==")
    print(f"dictionary {c['markers']['dictionary']}, markers {[m['id'] for m in c['markers']['items']]}, "
          f"wrist sees {c['markers']['seen_by_wrist']}")
    for m in c["markers"]["items"]:
        pm = q["per_marker"][m["id"]] if m["id"] in q["per_marker"] else q["per_marker"][str(m["id"])]
        print(f"  marker {m['id']:>3} ({m['role']}): centre {np.round(np.array(m['T_world_marker'])[:2, 3], 4).tolist()} m, "
              f"reproj {pm['reproj_px']:.3f} px, {pm['rms_mm']:.2f} mm on table, 1 px = {q['pixel_footprint_mm'][m['id']]} mm")
    ov = c["cameras"]["overview"]
    print(f"overview: f = {ov['K_normalized']['fx'] * 299:.1f} px ({ov['intrinsics_source']}), fovy {ov['fovy_deg']} deg, "
          f"position {np.round(np.array(ov['T_world_cam'])[:3, 3], 4).tolist()} m")
    print(f"arm base: {c['arm']['base_on_table']['source']}")
    for key, title in (("held_out", "held-out"), ("held_out_fixed_intrinsics", "held-out, f fixed")):
        for r in q.get(key) or []:
            print(f"  {title} marker {r['held_out']}: sides {r['side_err_mm']} mm, diagonals {r['diag_err_mm']} mm, "
                  f"rms {r['rms_mm']:.2f} mm" + (f", centre shift {r['centre_shift_mm']:.1f} mm" if "centre_shift_mm" in r else ""))
    if q["wrist_check"]:
        print(f"wrist check: {q['wrist_check']['markers']}")


if __name__ == "__main__":
    main()
