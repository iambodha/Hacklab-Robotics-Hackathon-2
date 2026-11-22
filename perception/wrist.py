"""Wrist camera in 3D: Station joint ticks -> ElRobot URDF FK -> CAD camera mount -> T_base_wristcam.

Joint mapping (camera_pose/HANDOFF.md finding 3): q = mid(URDF limits) + (raw - mid(calibrated range)) * 2pi/4096.
That assumes the calibrated range is centred on the URDF zero. The signs of j1/j5/j7 are unverified on
hardware (HANDOFF task 3). Every number derived from FK is therefore labelled with its source.

Frames: URDF base_link reaches along its +y axis, z up. The scene world frame (scene-contract.md) has
x = the arm's forward direction at j1 = 0, so T_world_base = Rz(-90 deg) with the origin on the table.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from . import geometry as g
from .aruco import marker_object_points
from .scene_fit import ippe

ROOT = Path(__file__).resolve().parent.parent
R_WORLD_BASE = g.rot_z(-math.pi / 2)        # base +y (reach) -> world +x; base +x -> world -y
_WCAM = None


def wrist_camera():
    """camera_pose.elrobot_camera.WristCamera (URDF FK + CAD T_link_camera), cached."""
    global _WCAM
    if _WCAM is None:
        from .station_io import NORMA_CORE
        import os
        if NORMA_CORE and "NORMA_CORE_REPO" not in os.environ:
            os.environ["NORMA_CORE_REPO"] = str(NORMA_CORE)
        sys.path.insert(0, str(ROOT))
        from camera_pose.elrobot_camera import WristCamera
        _WCAM = WristCamera()
    return _WCAM


RAW_CENTRE = 2048       # every calibrated follower range on this arm is centred on raw 2048 (22 Nov)


def sane_joints(joints: dict) -> dict:
    """The tick mapping only uses the calibrated range's centre. If Station's calibration is missing or
    reset (range_min == range_max), use the servo centre 2048, which is what the calibrated ranges are
    centred on, so FK keeps working uncalibrated."""
    out = {}
    for j, v in joints.items():
        if isinstance(v, dict) and abs(v["hi"] - v["lo"]) < 200:
            v = {**v, "lo": RAW_CENTRE - 1000, "hi": RAW_CENTRE + 1000}
        out[j] = v
    return out


def joints_to_q(joints: dict, mapping: dict | None = None) -> dict:
    return wrist_camera().joints_to_q(sane_joints(joints), mapping)


def T_base_wristcam(joints: dict, T_link_cam: np.ndarray | None = None) -> np.ndarray:
    cam = wrist_camera()
    q = cam.joints_to_q(sane_joints(joints))
    return cam.link_pose(q) @ (cam.T_link_cam if T_link_cam is None else T_link_cam)


def tcp_base(joints: dict) -> np.ndarray:
    cam = wrist_camera()
    return cam.tcp(cam.joints_to_q(sane_joints(joints)))


def T_link_cam_cad() -> np.ndarray:
    return wrist_camera().T_link_cam.copy()


def T_world_base(base_z: float = 0.0) -> np.ndarray:
    return g.T_from(R_WORLD_BASE, [0, 0, base_z])


def planar_of(T: np.ndarray) -> tuple[float, float, float]:
    """Project a 4x4 onto the table: (x, y, yaw of its x axis)."""
    return float(T[0, 3]), float(T[1, 3]), math.atan2(T[1, 0], T[0, 0])


def solve_base_from_wrist(obs, T_table_marker: dict[int, np.ndarray], T_base_cam, size: float, W: int, H: int,
                          intr: dict | None = None, base_z: float = 0.0, f_scan_hfov=range(40, 131, 5)):
    """Arm base pose on the table from markers seen by the wrist camera, from one or several arm poses.

    obs / T_base_cam: one view ({id: corners}, 4x4) or lists of them (one per arm pose).
    Unknowns: base (x, y, yaw) in the table frame (the base stands flat on the table, z = base_z) and,
    if intr is None, the wrist focal length f (fx = fy, centred principal point, no distortion).
    Known: each marker's pose on the table (from the overview fit), T_base_cam from FK + CAD.
    -> dict(T_table_base, intr, rms_px, sigma_mm, sigma_yaw_deg, sigma_f_rel)."""
    if isinstance(obs, dict):
        obs, T_base_cam = [obs], [T_base_cam]
    obj = marker_object_points(size)
    views = [(i, o[i], Tb) for o, Tb in zip(obs, T_base_cam) for i in sorted(o) if i in T_table_marker]
    if not views:
        raise ValueError("wrist sees no marker with a known table pose")
    ids = sorted({v[0] for v in views})
    cx, cy = (W - 1) / 2, (H - 1) / 2

    def intr_of(x, f=None):
        f = x[3] if len(x) > 3 else f
        return intr if f is None else g.intr_from_K(np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]]), [0] * 5, W, H)

    def T_tb(x):
        return g.T_from(g.rot_z(x[2]), [x[0], x[1], base_z])

    def residuals(x, f=None):
        it = intr_of(x, f)
        Tb = T_tb(x)
        r = [(g.project(Tb @ Tc, it, W, H, g.apply(T_table_marker[i], obj))[0] - c).ravel() for i, c, Tc in views]
        return np.concatenate(r)

    def init(f=None):
        it = intr_of([], f)
        K, d = g.K_matrix(it, W, H), g.dist_vec(it)
        cands = [np.array(planar_of(T_table_marker[i] @ g.inv(T_cam_m) @ g.inv(Tc)))
                 for i, c, Tc in views for T_cam_m, _ in ippe(c, K, d, size)]
        return min(cands, key=lambda p: float(residuals(p, f) @ residuals(p, f)))

    if intr is None:
        best = None
        for hf in f_scan_hfov:
            f0 = g.intr_from_fov(hf, W, H)["fx"] * W
            x, r, _ = g.levmar(lambda x: residuals(x, f0), init(f0))
            if best is None or r @ r < best[0]:
                best = (r @ r, np.r_[x, f0])
        x, r, J = g.levmar(residuals, best[1])
    else:
        x, r, J = g.levmar(residuals, init())
    it = intr_of(x)
    s = np.sqrt(np.maximum(np.diag(g.covariance(J, r)), 0))
    out = {"T_table_base": T_tb(x), "intr": it, "rms_px": float(np.sqrt(np.mean(r.reshape(-1, 2) ** 2) * 2)),
           "sigma_mm": float(np.hypot(s[0], s[1]) * 1000), "sigma_yaw_deg": float(math.degrees(s[2])),
           "markers": ids, "n_obs": int(len(r)), "n_poses": len(obs)}
    if intr is None:
        out["sigma_f_rel"] = float(s[3] / x[3])
    return out
