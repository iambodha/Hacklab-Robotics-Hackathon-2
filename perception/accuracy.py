"""Accuracy from the markers themselves (no extra hardware).

Hold-one-out: fit the camera with every marker except k, then back-project marker k's detected corners
onto the table with that fit and measure the result against what we know for certain about marker k:
its four sides are size_m and its diagonals size_m*sqrt(2). Errors here are what pixel_to_table would make
on an object at that spot (scale, perspective and focal errors all show up). The held-out marker's
centre is also compared with where the full fit puts it (stability of the layout, in mm).
"""
from __future__ import annotations

import math

import numpy as np

from . import geometry as g, scene_fit


def backproject_table(fit: scene_fit.SceneFit, uv) -> np.ndarray:
    """Pixels -> table-frame points (z=0) for a SceneFit."""
    return g.pixel_to_plane(fit.T_table_cam, fit.intr, fit.W, fit.H, uv, 0.0)


def shape_errors(P: np.ndarray, size: float) -> tuple[np.ndarray, np.ndarray]:
    sides = np.linalg.norm(P - np.roll(P, -1, 0), axis=1)
    diags = np.array([np.linalg.norm(P[0] - P[2]), np.linalg.norm(P[1] - P[3])])
    return (sides - size) * 1000, (diags - size * math.sqrt(2)) * 1000


def held_out(obs: dict[int, np.ndarray], size: float, W: int, H: int, intr: dict | None,
             sigma_px: dict[int, float] | None = None, full: scene_fit.SceneFit | None = None) -> list[dict]:
    """One row per marker: side/diagonal errors (mm) of the held-out marker measured through the others' fit."""
    rows = []
    min_others = 1 if intr is not None else 2
    if len(obs) - 1 < min_others:
        return rows
    for k in sorted(obs):
        rest = {i: c for i, c in obs.items() if i != k}
        f = scene_fit.fit(rest, size, W, H, intr=intr, sigma_px=sigma_px)
        P = backproject_table(f, obs[k])
        se, de = shape_errors(P, size)
        row = {"held_out": k, "fit_with": sorted(rest), "side_err_mm": se.round(2).tolist(),
               "diag_err_mm": de.round(2).tolist(), "side_err_mm_max": float(np.abs(se).max()),
               "diag_err_mm_max": float(np.abs(de).max()),
               "rms_mm": float(np.sqrt(np.mean(np.r_[se, de] ** 2)))}
        if full is not None and f.ref_id in full.markers:
            # compare the held-out centre in the shared reference-marker frame
            T_full = g.inv(full.T_table_marker(f.ref_id))
            c_full = g.apply(T_full @ full.T_table_marker(k), [[0, 0, 0]])[0]
            row["centre_shift_mm"] = float(np.linalg.norm(P.mean(0)[:2] - c_full[:2]) * 1000)
        rows.append(row)
    return rows
