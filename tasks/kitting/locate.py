"""Find the kit sheet, its cells and its start spots in an overview image, from the sheet's own ArUco markers.

The overview camera is calibrated against the table (sim/layouts/home.json: T_cv, K, dist from scene_calib), so every
detected marker corner is a ray that meets the table plane at a known point. The sheet pose (x, y, yaw on the table)
is the 2D rigid fit of those points to the same corners in the sheet frame (kit_a4.json). One marker (4 corners) is
enough; all visible ones are used. No PnP: the camera pose is the calibrated one, only the paper moves.

    python -m tasks.kitting.locate IMAGE.jpg [--twin] [--out overlay.jpg]      # real overview frame (299x224)

Used by the sim training loop on every episode (rendered twin image, randomized lighting and camera error) and by
tasks/kitting/run_real.py on the real overview frame, with the same code.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .spec import Pose2D, Spec  # noqa: E402

CORNER_SIGMA_PX = 0.3     # corner noise of perception/aruco.detect with edge refinement on 299x224 frames
SHEET_Z = 0.0004          # paper top above the table (sim/kitting_scene.SHEET_Z; a real sheet is ~0.1 mm)
MIN_MARKERS = 1
MAX_RESIDUAL_MM = 6.0     # a fit worse than this means a wrong id, a bent sheet or a bad calibration: refuse


@dataclass
class CameraModel:
    """A calibrated camera: T (4x4) camera -> table frame, OpenCV axes; intr = perception.geometry intrinsics dict."""
    T: np.ndarray
    intr: dict
    W: int
    H: int
    source: str = ""


def camera_model(layout: dict, name: str = "overview_cam", twin: bool = False) -> CameraModel:
    """twin=True: the MuJoCo render's pinhole (fovy, no distortion, principal point at the centre); False: the real
    camera's calibrated K and distortion from the layout (scene_calib)."""
    from perception.geometry import intr_from_K
    cam = next(c for c in layout["cameras"] if c["name"] == name)
    W, H = layout["image_size"]
    T = np.asarray(cam["T_cv"], float)
    if twin or "K_normalized" not in cam:
        f = (H / 2) / math.tan(math.radians(cam["fovy"]) / 2)
        K = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1.0]])
        return CameraModel(T, intr_from_K(K, [0] * 5, W, H), W, H, f"twin pinhole fovy {cam['fovy']}")
    return CameraModel(T, {**cam["K_normalized"], "dist": list(cam.get("dist", [0] * 5))}, W, H,
                       f"calibrated {cam.get('src', '?')}")


def fit_rigid_2d(src: np.ndarray, dst: np.ndarray) -> tuple[Pose2D, np.ndarray]:
    """Least-squares rotation + translation with dst ~ R src + t (Kabsch in 2D). -> (pose, residuals (N,))."""
    cs, cd = src.mean(0), dst.mean(0)
    Hm = (src - cs).T @ (dst - cd)
    a = math.atan2(Hm[0, 1] - Hm[1, 0], Hm[0, 0] + Hm[1, 1])
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    t = cd - R @ cs
    res = np.linalg.norm(src @ R.T + t - dst, axis=1)
    return Pose2D(float(t[0]), float(t[1]), math.degrees(a)), res


# The sheet's 50 mm markers are 8-15 px wide in a 299x224 overview frame: one upscaled decode (perception/aruco.detect)
# finds 1-3 of the 4 and which ones flips with the interpolation. Decoding at several upscales/interpolations and taking
# the union finds 2-4 (measured on twin renders, docs/kitting/JOURNAL.md). Corners: the median over the decodes, then
# perception's sub-pixel edge refinement at the original resolution.
DECODES = [(3, "cubic"), (3, "linear"), (3, "lanczos"), (4, "cubic"), (4, "linear"), (4, "lanczos"), (6, "cubic"),
           (6, "lanczos")]
PAD_PX = 4


def detect_sheet_markers(img: np.ndarray, spec: Spec) -> dict[int, np.ndarray]:
    """{marker id: (4, 2) corners (OpenCV order, pixel centres at integers)} of the sheet's markers only."""
    import cv2
    from perception.aruco import _at_border, detector, refine_edges
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    interp = {"cubic": cv2.INTER_CUBIC, "linear": cv2.INTER_LINEAR, "lanczos": cv2.INTER_LANCZOS4}
    det = detector(spec.dictionary)
    seen: dict[int, list[np.ndarray]] = {}
    padded = cv2.copyMakeBorder(gray, PAD_PX, PAD_PX, PAD_PX, PAD_PX, cv2.BORDER_CONSTANT, value=255)
    for up, how in DECODES:
        corners, ids, _ = det.detectMarkers(cv2.resize(padded, None, fx=up, fy=up, interpolation=interp[how]))
        for i, c in zip([] if ids is None else ids.ravel(), corners):
            if int(i) in spec.markers():
                seen.setdefault(int(i), []).append((c.reshape(4, 2).astype(float) + 0.5) / up - 0.5 - PAD_PX)
    g32, (H, W) = gray.astype(np.float32), gray.shape[:2]
    out = {}
    for i, cs in seen.items():
        c = np.median(np.stack(cs), axis=0)
        if not _at_border(c, W, H):
            out[i] = refine_edges(g32, c)
    return out


def locate_sheet(img: np.ndarray, spec: Spec, cam: CameraModel) -> dict:
    """img: RGB (H, W, 3). -> {"ok", "pose" (Pose2D, table frame), "markers": ids used, "residual_mm" (rms),
    "sigma_mm" (expected position error from corner noise), "cells"/"spots" (table xy), "why"}."""
    from perception.geometry import pixel_footprint_jacobian, pixel_to_plane
    found = detect_sheet_markers(img, spec)
    if len(found) < MIN_MARKERS:
        return {"ok": False, "why": f"no sheet marker ({sorted(spec.markers())}) found in the image", "markers": []}
    def fit(ids):
        px = np.vstack([found[i] for i in ids])
        table = pixel_to_plane(cam.T, cam.intr, cam.W, cam.H, px, SHEET_Z)[:, :2]
        sheet = np.vstack([spec.marker_corners_mm(i) for i in ids]) / 1000.0
        rows = np.isfinite(table).all(1)
        pose, res = fit_rigid_2d(sheet[rows], table[rows])
        return pose, res, px, rows
    ids = sorted(found)
    pose, res, px, ok_rows = fit(ids)
    # a mis-decoded id (a marker read as another sheet id) shows up as one marker with a large residual: drop it
    while len(ids) > 1 and np.sqrt(np.mean(res ** 2)) * 1000 > MAX_RESIDUAL_MM / 2:
        worst = min(ids, key=lambda i: fit([j for j in ids if j != i])[1].mean())   # removing it fits best
        ids = [j for j in ids if j != worst]
        pose, res, px, ok_rows = fit(ids)
    found = {i: found[i] for i in ids}
    J = pixel_footprint_jacobian(cam.T, cam.intr, cam.W, cam.H, px.mean(0), SHEET_Z)
    sigma = float(np.sqrt((J @ J.T).trace()) * CORNER_SIGMA_PX / math.sqrt(max(1, ok_rows.sum())) * 1000)
    rms = float(np.sqrt(np.mean(res ** 2)) * 1000)
    out = {"ok": rms <= MAX_RESIDUAL_MM, "pose": pose, "markers": sorted(found), "residual_mm": round(rms, 2),
           "sigma_mm": round(sigma, 2), "mm_per_px": round(float(np.sqrt(abs(np.linalg.det(J)))) * 1000, 2),
           "cells": {c: spec.cell_pose(pose, c).as_list()[:2] for c in spec.cells()},
           "spots": {s: spec.spot_pose(pose, s).as_list() for s in spec.spots()},
           "corners_px": {i: found[i].round(2).tolist() for i in sorted(found)}, "camera": cam.source}
    if not out["ok"]:
        out["why"] = f"marker fit residual {rms:.1f} mm > {MAX_RESIDUAL_MM} mm"
    return out


def overlay(img: np.ndarray, spec: Spec, cam: CameraModel, pose: Pose2D, scale: int = 3) -> np.ndarray:
    """The located sheet drawn back into the image (upscaled): cells green, spots yellow, markers magenta."""
    import cv2
    from perception.geometry import project
    big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC).copy()

    def poly(pts_mm, colour, closed=True):
        xyz = np.c_[spec.to_table(pose, pts_mm), np.full(len(pts_mm), SHEET_Z)]
        uv, _ = project(cam.T, cam.intr, cam.W, cam.H, xyz)
        cv2.polylines(big, [np.round((uv + 0.5) * scale - 0.5).astype(np.int32)], closed, colour, 1, cv2.LINE_AA)
        return uv
    W, H = spec.paper_mm
    poly(np.array([[-W / 2, H / 2], [W / 2, H / 2], [W / 2, -H / 2], [-W / 2, -H / 2]]), (255, 255, 255))
    for i in spec.markers():
        poly(spec.marker_corners_mm(i), (255, 0, 255))
    for c in spec.cells():
        uv = poly(spec.cell_corners_mm(c), (0, 220, 0))
        cv2.putText(big, c, tuple(np.round(uv.mean(0) * scale).astype(int) - [8, -4]), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (0, 255, 0), 1, cv2.LINE_AA)
    for s, sp in spec.spots().items():
        d = 12.0
        poly(sp.apply([[-d, 0], [d, 0]]), (255, 220, 0), False)
        poly(sp.apply([[0, -d], [0, d]]), (255, 220, 0), False)
    return big


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--spec", default=None)
    ap.add_argument("--layout", default=str(ROOT / "sim" / "layouts" / "home.json"))
    ap.add_argument("--twin", action="store_true", help="the image is a twin render (pinhole, no distortion)")
    ap.add_argument("--out", default=None, help="overlay image (default: next to the input, *_kit.jpg)")
    a = ap.parse_args()
    from PIL import Image
    spec = Spec.load(a.spec)
    img = np.asarray(Image.open(a.image).convert("RGB"))
    cam = camera_model(json.loads(Path(a.layout).read_text()), twin=a.twin)
    r = locate_sheet(img, spec, cam)
    if r.get("pose") is not None:
        out = Path(a.out or Path(a.image).with_name(Path(a.image).stem + "_kit.jpg"))
        Image.fromarray(overlay(img, spec, cam, r["pose"])).save(out, quality=90)
        r["overlay"] = str(out)
        r["pose"] = r["pose"].as_list()
    print(json.dumps(r, indent=1, default=str))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
