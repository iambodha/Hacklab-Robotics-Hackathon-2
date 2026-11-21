"""grasp_model — learned IK by bilinear interpolation over a taught 3x3 grid.

grasp_pose(u, v) -> {j1..j8} where (u,v) is the sheet position
(u: left 0 .. right 1 ; v: top 0 .. bottom 1).

This is the "IK" for the hackathon demo: instead of analytic inverse
kinematics, we interpolate between grasp poses taught by teleop. The seed
grid (data/grid_grasp.json) came from last year's identical arm — RE-TEACH
it on this year's setup with the teach_grid_point tool; overrides are saved
to data/grid_grasp.local.json and take precedence.
"""
from __future__ import annotations

import json

from . import config

JOINTS = [f"j{i}" for i in range(1, 9)]
_US = [0.0, 0.5, 1.0]
_VS = [0.0, 0.5, 1.0]
_COL = {0: "L", 1: "M", 2: "R"}
_ROW = {0: "T", 1: "M", 2: "B"}

LABEL_UV = {r + c: (u, v) for r, v in zip("TMB", _VS) for c, u in zip("LMR", _US)}
# -> {'TL': (0,0), 'TM': (0.5,0), ..., 'BR': (1,1)}

SEED_GRID = config.DATA_DIR / "grid_grasp.json"
LOCAL_GRID = config.DATA_DIR / "grid_grasp.local.json"


def load_grid() -> dict:
    """label -> grasp_pose dict. Local re-taught points override the seed."""
    grid = json.loads(SEED_GRID.read_text()) if SEED_GRID.exists() else {}
    if LOCAL_GRID.exists():
        grid.update(json.loads(LOCAL_GRID.read_text()))
    return {lab: p["grasp_pose"] for lab, p in grid.items()}


def teach_point(label: str, u: float, v: float, pose: dict) -> None:
    """Record/replace one grid point in the local override file."""
    label = label.strip().upper()
    data = json.loads(LOCAL_GRID.read_text()) if LOCAL_GRID.exists() else {}
    data[label] = {"u": u, "v": v, "grasp_pose": pose}
    LOCAL_GRID.write_text(json.dumps(data, indent=2))


def _bracket(x, axis):
    for i in range(len(axis) - 1):
        if axis[i] <= x <= axis[i + 1]:
            return i, (x - axis[i]) / (axis[i + 1] - axis[i])
    if x < axis[0]:
        return 0, 0.0
    return len(axis) - 2, 1.0


def grasp_pose(u: float, v: float, poses: dict | None = None) -> dict:
    poses = poses or load_grid()
    u = max(0.0, min(1.0, u))
    v = max(0.0, min(1.0, v))
    missing = [r + c for r in "TMB" for c in "LMR" if r + c not in poses]
    if missing:
        raise RuntimeError(f"grasp grid missing points {missing} — teach them first")
    ci, fu = _bracket(u, _US)
    ri, fv = _bracket(v, _VS)
    c00 = _ROW[ri] + _COL[ci]
    c10 = _ROW[ri] + _COL[ci + 1]
    c01 = _ROW[ri + 1] + _COL[ci]
    c11 = _ROW[ri + 1] + _COL[ci + 1]
    out = {}
    for j in JOINTS:
        top = poses[c00][j] * (1 - fu) + poses[c10][j] * fu
        bot = poses[c01][j] * (1 - fu) + poses[c11][j] * fu
        out[j] = round(top * (1 - fv) + bot * fv, 4)
    return out
