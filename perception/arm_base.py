"""Arm base on the table (= the world frame origin) without moving the arm.

Three observations, combined in perception/calib_scene.py:
  foot   the overview camera sees the front face of the black base foot; its bottom edge (on the table) is found
         column by column, back-projected through the overview calibration -> the foot's front-centre point
  wrist  the wrist camera sees a calibrated marker at the current pose; FK (URDF, range_fraction) + the CAD camera
         mount close the loop (perception/wrist.py). Mostly constrains yaw once the foot pins the position.
  tape   four tape distances from the foot's front corners to marker corners (README: 2 minutes, +-2 mm)

  python -m perception.arm_base tape-guide     # draws the four tape segments A-D on a live overview frame
                                               # -> calibration/tape_guide.jpg, and prints what to measure

ElRobot base_link geometry (norma-core hardware/elrobot/simulation/assets/base_link.stl, URDF base_link frame:
+y = reach direction, z up): the foot is 100 mm wide (x in [-50, 50] mm) and its front face is at y = +56.6 mm.
The j1 axis is at y = +42.5 mm (URDF joints "Rigid 1" + "rev_motor_01"), i.e. NOT at the base_link origin.
Robot's left = base -x = world +y.
"""
from __future__ import annotations

import argparse
import asyncio
import math

import cv2
import numpy as np

from . import geometry as g

FOOT_FRONT_Y = 0.05658
FOOT_HALF_W = 0.050
J1_AXIS_Y = 0.007585 + 0.034904
FOOT_CORNERS = {"FL": (-FOOT_HALF_W, FOOT_FRONT_Y), "FR": (FOOT_HALF_W, FOOT_FRONT_Y)}   # base_link xy
FOOT_FRONT_CENTRE = (0.0, FOOT_FRONT_Y)
TAPE_SIGMA_M = 0.001


# ---- foot in the overview ------------------------------------------------------------------------------
def detect_foot(gray: np.ndarray, fit, base_ids: list[int], min_step_px: float = 3.0, min_width_px: int = 8):
    """Front-bottom edge of the dark arm foot in a (median) overview grey image.

    Searches the columns between the two base markers. For each column, scanning up from the table, the first
    table -> dark crossing (half the local table brightness, sub-pixel). Most columns end at the table's far edge;
    the foot stands on the table, so its columns end lower in the image (closer to the camera). The foot = the
    widest run of columns at least min_step_px below the median edge row.
    -> dict(uv_centre, v_edge, u_left, u_right, columns, edge_rows) or None."""
    H, W = gray.shape[:2]
    c = {i: g.apply(fit.T_cam_table, [[*fit.markers[i][:2], 0.0]])[0] for i in base_ids}
    uv = {i: g.project(np.eye(4), fit.intr, W, H, p)[0][0] for i, p in c.items()}
    half = fit.size / 2 * fit.intr["fx"] * W / min(p[2] for p in c.values())     # marker half-width in px
    u0 = int(max(0, min(p[0] for p in uv.values()) + 1.2 * half))
    u1 = int(min(W - 1, max(p[0] for p in uv.values()) - 1.2 * half))
    vm = float(np.mean([p[1] for p in uv.values()]))
    v_top, v_bot = int(max(1, vm - 25)), int(min(H - 2, vm + 18))
    cols, rows = [], []
    for u in range(u0, u1 + 1):
        col = gray[:, u].astype(float)
        table = float(np.median(col[v_bot - 4:v_bot + 1]))
        thr = 0.5 * table
        edge = np.nan
        for v in range(v_bot, v_top, -1):
            if col[v - 1] < thr <= col[v]:
                edge = (v - 1) + (thr - col[v - 1]) / (col[v] - col[v - 1])
                break
        cols.append(u)
        rows.append(edge)
    cols, rows = np.array(cols), np.array(rows)
    if np.isfinite(rows).sum() < min_width_px:
        return None
    base = float(np.nanmedian(rows))
    low = np.isfinite(rows) & (rows > base + min_step_px)
    runs, start = [], None
    for k, f in enumerate(np.r_[low, False]):
        if f and start is None:
            start = k
        elif not f and start is not None:
            runs.append((start, k))
            start = None
    if not runs:
        return None
    a, b = max(runs, key=lambda r: r[1] - r[0])
    if b - a < min_width_px:
        return None
    inner = rows[a + 2:b - 2] if b - a > 6 else rows[a:b]
    v = float(np.median(inner))
    u_l, u_r = cols[a] - 0.5, cols[b - 1] + 0.5
    return {"uv_centre": [(u_l + u_r) / 2, v], "v_edge": v, "u_left": float(u_l), "u_right": float(u_r),
            "far_edge_row": base, "columns": cols[a:b].tolist(), "edge_rows": rows[a:b].round(2).tolist(),
            "row_scatter_px": float(1.4826 * np.median(np.abs(inner - v)))}


def foot_front_point(det: dict, fit, sigma_px: float = 0.5, sigma_extra_m: float = 0.002):
    """Foot detection -> (front-centre point xy in the table frame, 2x2 covariance m^2, measured width m).
    sigma_px: edge-row / edge-column uncertainty (threshold, blur, shadow); sigma_extra_m: overview calibration
    error near the base markers (held-out ~2-3 mm) and the CAD foot dimensions."""
    W, H = fit.W, fit.H
    u, v = det["uv_centre"]
    P = g.pixel_to_plane(fit.T_table_cam, fit.intr, W, H, [[u, v], [det["u_left"], v], [det["u_right"], v]], 0.0)
    J = g.pixel_footprint_jacobian(fit.T_table_cam, fit.intr, W, H, (u, v))
    cov = J @ np.diag([(sigma_px / math.sqrt(2)) ** 2, sigma_px ** 2]) @ J.T + np.eye(2) * sigma_extra_m ** 2
    return P[0, :2], cov, float(np.linalg.norm(P[2, :2] - P[1, :2]))


def base_from_foot(F_xy, yaw: float) -> np.ndarray:
    """Base origin (table xy) from the foot's front-centre point and the base yaw (of base +x in the table)."""
    R = g.rot_z(yaw)[:2, :2]
    return np.asarray(F_xy) - R @ np.array(FOOT_FRONT_CENTRE)


# ---- tape ----------------------------------------------------------------------------------------------
def tape_segments(T_table_base: np.ndarray, T_table_marker: dict, size: float, base_ids: list[int]) -> list[dict]:
    """The four tape distances, chosen to be well conditioned: from each front corner of the foot to the nearest
    corner of the black square of (a) the nearest base marker (sideways) and (b) the nearest far marker (forward).
    Each foot corner is then fixed by two near-perpendicular circles; the 100 mm foot width ties them together."""
    from .aruco import marker_object_points
    far_ids = [i for i in T_table_marker if i not in base_ids]
    segs, labels = [], iter("ABCD")
    for name in ("FL", "FR"):
        p = g.apply(T_table_base, [[*FOOT_CORNERS[name], 0.0]])[0, :2]
        for group in (base_ids, far_ids):
            ids = [i for i in group if i in T_table_marker]
            if not ids:
                continue
            corners = {i: g.apply(T_table_marker[i], marker_object_points(size))[:, :2] for i in ids}
            i = min(ids, key=lambda m: np.linalg.norm(corners[m] - p, axis=1).min())
            k = int(np.argmin(np.linalg.norm(corners[i] - p, axis=1)))
            segs.append({"label": next(labels), "foot_corner": name, "marker": int(i), "marker_corner": k,
                         "marker_corner_xy": corners[i][k].tolist(),
                         "predicted_m": float(np.linalg.norm(corners[i][k] - p))})
    return segs


def tape_residuals(segs: list[dict], tape_mm: dict):
    """-> (f(T_table_base) -> residuals / TAPE_SIGMA_M, labels used)."""
    used = [s for s in segs if tape_mm.get(s["label"]) is not None]
    if len(used) < 3:
        raise ValueError(f"need >= 3 tape distances, got {[s['label'] for s in used]}")

    def f(T):
        return np.array([(np.linalg.norm(g.apply(T, [[*FOOT_CORNERS[s["foot_corner"]], 0]])[0, :2]
                                         - s["marker_corner_xy"]) - tape_mm[s["label"]] / 1000) / TAPE_SIGMA_M
                         for s in used])
    return f, [s["label"] for s in used]


def solve_tape(segs: list[dict], tape_mm: dict, T0: np.ndarray, base_z: float = 0.0) -> dict:
    """Base (x, y, yaw) on the table from tape distances (mm) keyed by segment label."""
    f, used = tape_residuals(segs, tape_mm)

    def res(x):
        return f(g.planar_T(*x))

    x0 = np.array([T0[0, 3], T0[1, 3], math.atan2(T0[1, 0], T0[0, 0])])
    x, r, J = g.levmar(res, x0)
    s = np.sqrt(np.maximum(np.diag(g.covariance(J, r, 1.0)), 0))
    return {"T_table_base": g.T_from(g.rot_z(x[2]), [x[0], x[1], base_z]),
            "sigma_mm": float(np.hypot(s[0], s[1]) * 1000), "sigma_yaw_deg": float(math.degrees(s[2])),
            "residual_mm": (r * TAPE_SIGMA_M * 1000).round(2).tolist(), "used": used}


# ---- evidence images -----------------------------------------------------------------------------------
def annotate_foot(img: np.ndarray, det: dict | None, T_cam_base: np.ndarray, intr: dict, scale: int = 4,
                  pad: int = 28) -> np.ndarray:
    """Crop around the foot, upscaled: detected bottom-edge points (green), the CAD foot outline at table level
    projected through the final base estimate (magenta), the j1 axis foot point (cyan)."""
    H, W = img.shape[:2]
    outline = g.apply(T_cam_base, [[-FOOT_HALF_W, FOOT_FRONT_Y, 0], [FOOT_HALF_W, FOOT_FRONT_Y, 0],
                                   [FOOT_HALF_W, -0.0034, 0], [-FOOT_HALF_W, -0.0034, 0]])
    uv, _ = g.project(np.eye(4), intr, W, H, outline)
    j1, _ = g.project(np.eye(4), intr, W, H, g.apply(T_cam_base, [[0, J1_AXIS_Y, 0]]))
    cu, cv_ = (det["uv_centre"] if det else uv.mean(0))
    u0, v0 = int(max(0, cu - pad * 1.6)), int(max(0, cv_ - pad))
    u1, v1 = int(min(W, cu + pad * 1.6)), int(min(H, cv_ + pad * 0.8))
    crop = cv2.resize(img[v0:v1, u0:u1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    def P(p):
        return (int(round((p[0] - u0 + 0.5) * scale - 0.5)), int(round((p[1] - v0 + 0.5) * scale - 0.5)))
    cv2.polylines(crop, [np.array([P(p) for p in uv], np.int32)], True, (255, 0, 255), 1, cv2.LINE_AA)
    cv2.circle(crop, P(j1[0]), 4, (255, 255, 0), 1, cv2.LINE_AA)
    if det:
        for u, v in zip(det["columns"], det["edge_rows"]):
            cv2.circle(crop, P((u, v)), 2, (0, 255, 0), -1)
        cv2.circle(crop, P(det["uv_centre"]), 5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(crop, "green: detected foot edge  magenta: CAD foot at final base  cyan: j1 axis", (4, crop.shape[0] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return crop


def annotate_tape(sc, img: np.ndarray, segs: list[dict], scale: int = 3) -> np.ndarray:
    """Overview frame with the tape segments A-D drawn from the foot corners to the marker corners."""
    H, W = img.shape[:2]
    out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    T_wc = sc.T_world_overview
    T_wt = np.array(sc.d["T_world_table_internal"])

    def P(pw):
        uv, _ = g.project(T_wc, sc.intr("overview"), W, H, [pw])
        return (int(round((uv[0, 0] + 0.5) * scale - 0.5)), int(round((uv[0, 1] + 0.5) * scale - 0.5)))
    for s in segs:
        a = g.apply(sc.T_world_base, [[*FOOT_CORNERS[s["foot_corner"]], 0]])[0]
        b = g.apply(T_wt, [[*s["marker_corner_xy"], 0]])[0]
        pa, pb = P(a), P(b)
        cv2.line(out, pa, pb, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(out, pa, 5, (255, 0, 255), -1)
        cv2.circle(out, pb, 5, (0, 0, 255), -1)
        mid = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2 - 6)
        cv2.putText(out, s["label"], mid, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, s["label"], mid, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, "tape: magenta = front corner of the black foot (at the table), red = corner of the black square",
                (6, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def describe_tape(segs: list[dict]) -> list[str]:
    side = {"FL": "front-LEFT", "FR": "front-RIGHT"}
    return [f"{s['label']}: {side[s['foot_corner']]} corner of the black arm foot (robot's view, at table level) -> the "
            f"nearest corner of the black square of marker {s['marker']}   (expected ~{s['predicted_m'] * 1000:.0f} mm)"
            for s in segs]


def main():
    ap = argparse.ArgumentParser(description="arm base helpers")
    ap.add_argument("cmd", choices=["tape-guide"])
    a = ap.parse_args()
    from . import station_io as sio
    from .scene import CALIB_DIR, CONFIG_PATH, Scene
    sc = Scene.load()
    segs = sc.d["arm"].get("tape_segments")
    if not segs:
        raise SystemExit("scene_calib.json has no tape_segments: rerun python -m perception.calib_scene --live")

    async def frame():
        return sio.decode_jpeg((await sio.grab_jpegs(await sio.connect(), "overview", 1))[0])
    img = asyncio.run(frame())
    p = CALIB_DIR / "tape_guide.jpg"
    cv2.imwrite(str(p), annotate_tape(sc, img, segs), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("\n".join(describe_tape(segs)))
    print(f'wrote {p}. Put the four distances (mm) in {CONFIG_PATH} -> arm_base.tape_mm '
          '{"A": .., "B": .., "C": .., "D": ..} and rerun python -m perception.calib_scene --live')


if __name__ == "__main__":
    main()
