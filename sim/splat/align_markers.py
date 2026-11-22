"""Put a COLMAP reconstruction into the twin's frame (metric) with the ArUco markers on the table.

  python sim/splat/align_markers.py <DATA_ROOT/splat/<name>> [--layout home]

1. Detect the layout's markers (DICT_4X4_50, ids 20-23, 10 cm) in every registered frame (sub-pixel corners).
2. Triangulate each marker corner from all frames that see it (DLT on undistorted rays from COLMAP's camera, then
   outlier views dropped by reprojection error).
3. Robust Sim3 (LO-RANSAC, pycolmap.estimate_sim3d_robust) from those points onto the same corners in the twin: the
   corner sites aruco_<id>_c<k> of the MuJoCo model built from sim/layouts/<layout>.json (the frame MuJoCo renders
   in, p_sim = T_sim_world @ p_world; the corner order is OpenCV's, as sim/selftest.py checks).
4. Transform the reconstruction and write it to dataset/sparse/0 (what Brush trains on), so the exported splat is
   already in the twin's frame; align.json holds sim_from_colmap, world_from_colmap and per-corner residuals, and
   twin_cameras.json the twin's camera poses (OpenCV convention, sim frame) for opening the same views in a viewer.
--check-ply <ply> afterwards: confirms a trained splat sits in that frame (centres vs sparse points, table plane).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pycolmap

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))


def twin_corners(layout_name: str) -> tuple[dict, np.ndarray, dict, dict]:
    """{(id, k): xyz in the sim frame}, T_sim_world, marker spec, twin camera poses, from the twin's MuJoCo model."""
    import build_scene  # noqa: E402  (sets MUJOCO_GL)
    import compare_real
    import mujoco
    model, layout = compare_real.load_model(build_scene.layout_path(layout_name))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("rest").id)
    mujoco.mj_forward(model, data)
    mk = layout["markers"]
    pts = {(it["id"], k): data.site_xpos[model.site(f"aruco_{it['id']}_c{k}").id].copy()
           for it in mk["items"] for k in range(4)}
    T = np.asarray(layout.get("T_sim_world", {}).get("value", np.eye(4)), float)
    W, H = layout["image_size"]
    cams = {}
    for c in range(model.ncam):                   # same frame as the splat: a viewer can open these views directly
        R = data.cam_xmat[c].reshape(3, 3) @ build_scene.CV_TO_MJ
        Tc = np.eye(4); Tc[:3, :3], Tc[:3, 3] = R, data.cam_xpos[c]
        cams[model.camera(c).name] = {"T_sim_cam_opencv": np.round(Tc, 6).tolist(), "fovy_deg": float(model.cam_fovy[c]),
                                      "image_size": [W, H], "note": "wrist_cam at the rest pose" if model.camera(c).name == "wrist_cam" else ""}
    return pts, T, mk, cams


def ply_xyz(path: Path) -> np.ndarray:
    """Splat centres of a binary little-endian float32 3DGS .ply (Brush export)."""
    b = path.read_bytes()
    head = b[: b.index(b"end_header\n") + 11].decode()
    lines = head.splitlines()
    props = [l.split()[-1] for l in lines if l.startswith("property")]
    n = int(next(l for l in lines if l.startswith("element vertex")).split()[-1])
    a = np.frombuffer(b[len(head):], dtype="<f4", count=n * len(props)).reshape(n, len(props))
    return a[:, [props.index(k) for k in "xyz"]]


def check_ply(work: Path, ply: Path) -> dict:
    """Is the splat in the twin frame? Distance of its centres to the aligned sparse points, share on the table plane."""
    from scipy.spatial import cKDTree
    X = ply_xyz(ply).astype(float)
    P = np.array([p.xyz for p in pycolmap.Reconstruction(work / "dataset" / "sparse" / "0").points3D.values()])
    d, _ = cKDTree(P).query(X)
    return {"splats": len(X), "median_mm_to_sparse": round(float(np.median(d)) * 1000, 1),
            "share_within_1cm_of_table_plane": round(float((np.abs(X[:, 2]) < 0.01).mean()), 3),
            "median_xyz": np.round(np.median(X, 0), 3).tolist()}


def detect(path: Path, dictionary: str, ids: set[int]) -> dict:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)), p)
    c, found, _ = det.detectMarkers(img)
    return {} if found is None else {int(i): cc.reshape(4, 2).astype(float) for i, cc in zip(found.ravel(), c) if int(i) in ids}


def triangulate(obs: list[tuple[np.ndarray, np.ndarray]], max_px_ray: float) -> tuple[np.ndarray | None, int, float]:
    """obs = [(P 3x4 cam_from_world, normalized xy)]. DLT, then drop rays off by > max_px_ray (normalized units)."""
    use = list(range(len(obs)))
    X = None
    for _ in range(3):
        if len(use) < 2:
            return None, 0, 0.0
        A = np.concatenate([[x[0] * P[2] - P[0], x[1] * P[2] - P[1]] for P, x in (obs[i] for i in use)])
        X = np.linalg.svd(A)[2][-1]
        X = X[:3] / X[3]
        err = []
        for P, x in obs:
            pc = P @ np.r_[X, 1]
            err.append(np.inf if pc[2] <= 0 else np.linalg.norm(pc[:2] / pc[2] - x))
        keep = [i for i in range(len(obs)) if err[i] < max_px_ray]
        if keep == use:
            break
        use = keep
    if X is None or len(use) < 2:
        return None, 0, 0.0
    centres = np.array([-obs[i][0][:, :3].T @ obs[i][0][:, 3] for i in use])
    rays = centres - X
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    angle = float(np.degrees(np.arccos(np.clip(rays @ rays.T, -1, 1))).max())
    return X, len(use), angle


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work", type=Path)
    ap.add_argument("--layout", default="home")
    ap.add_argument("--min-views", type=int, default=3)
    ap.add_argument("--min-angle", type=float, default=3.0, help="min triangulation angle (deg)")
    ap.add_argument("--max-error-m", type=float, default=0.01, help="RANSAC inlier threshold in the twin (m)")
    ap.add_argument("--check-ply", type=Path, help="only check that a trained splat is in the twin frame")
    a = ap.parse_args()
    if a.check_ply:
        print(json.dumps(check_ply(a.work, a.check_ply)))
        return
    rec = pycolmap.Reconstruction(a.work / "sfm" / "sparse" / "0")
    target, T_sim_world, mk, twin_cams = twin_corners(a.layout)
    ids = {it["id"] for it in mk["items"]}
    images = a.work / "dataset" / "images"
    obs: dict[tuple, list] = {}
    seen_per_image = {}
    tol = 2.0 / np.mean([c.mean_focal_length() for c in rec.cameras.values()])   # ~2 px in normalized image units
    for img in rec.images.values():
        if not img.has_pose:
            continue
        found = detect(images / img.name, mk["dictionary"], ids)
        seen_per_image[img.name] = sorted(found)
        cam = rec.cameras[img.camera_id]
        P = img.cam_from_world().matrix()
        for mid, c in found.items():
            xy = cam.cam_from_img(c)
            for k in range(4):
                obs.setdefault((mid, k), []).append((P, xy[k]))
    src, tgt, keys, info = [], [], [], {}
    for key, o in sorted(obs.items()):
        X, n, ang = triangulate(o, tol)
        info[f"{key[0]}_c{key[1]}"] = {"views": len(o), "used": n, "angle_deg": round(ang, 1)}
        if X is not None and n >= a.min_views and ang >= a.min_angle and key in target:
            src.append(X); tgt.append(target[key]); keys.append(key)
    if len(src) < 4:
        raise SystemExit(f"only {len(src)} well-triangulated marker corners (need >= 4; markers seen: "
                         f"{sorted({k[0] for k in obs})}); film the markers from more angles")
    src, tgt = np.array(src), np.array(tgt)
    ro = pycolmap.RANSACOptions()
    ro.max_error = a.max_error_m                              # in the target (twin) frame: metres
    est = pycolmap.estimate_sim3d_robust(src, tgt, ro)
    if est is None:
        raise SystemExit("Sim3 estimation failed")
    sim = est["tgt_from_src"]
    M = sim.matrix()
    S = np.eye(4); S[:3, :] = M
    res = np.linalg.norm((src @ M[:, :3].T + M[:, 3]) - tgt, axis=1)
    per_marker = {}
    for (mid, k), r in zip(keys, res):
        per_marker.setdefault(mid, []).append(r)
    rec.transform(sim)
    out = a.work / "dataset" / "sparse" / "0"
    out.mkdir(parents=True, exist_ok=True)
    rec.write(out)
    world_from_colmap = np.linalg.inv(T_sim_world) @ S
    result = {"layout": a.layout, "frame": "twin sim frame (MuJoCo world = URDF base, +y forward, z up, metres)",
              "sim_from_colmap": np.round(S, 6).tolist(), "scale": float(np.cbrt(np.linalg.det(S[:3, :3]))),
              "world_from_colmap": np.round(world_from_colmap, 6).tolist(), "T_sim_world": T_sim_world.tolist(),
              "corners_used": len(src), "ransac_inliers": int(est["num_inliers"]), "residual_mm": {"mean": round(float(res.mean() * 1000), 2),
                                                         "max": round(float(res.max() * 1000), 2)},
              "residual_mm_per_marker": {m: round(float(np.mean(v) * 1000), 2) for m, v in per_marker.items()},
              "corners": info, "frames_with_markers": sum(1 for v in seen_per_image.values() if v),
              "frames_registered": len(seen_per_image)}
    (a.work / "align.json").write_text(json.dumps(result, indent=1))
    (a.work / "twin_cameras.json").write_text(json.dumps({"frame": result["frame"], "cameras": twin_cams}, indent=1))
    print(json.dumps({k: result[k] for k in ("corners_used", "scale", "residual_mm", "residual_mm_per_marker",
                                               "frames_with_markers", "frames_registered")}))


if __name__ == "__main__":
    main()
