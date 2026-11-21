"""Joint fit of one camera looking at several square markers lying on one plane (the table).

Model (all markers share the table plane, which is physically true and is what makes a 299x224
camera usable for millimetre work):
  T_cam_table            6 dof   (rvec, tvec)
  marker m on the table  3 dof   (x, y, yaw) in the table frame; the reference marker is the gauge (0, 0, 0)
  optional intrinsics    f (fx = fy, principal point kept) and k1
Residuals are corner reprojection errors in pixels, weighted by each marker's per-corner frame jitter.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import geometry as g
from .aruco import marker_object_points


@dataclass
class SceneFit:
    T_cam_table: np.ndarray
    markers: dict[int, tuple[float, float, float]]      # id -> (x, y, yaw) in the table frame
    intr: dict
    W: int
    H: int
    size: float
    ref_id: int
    residual_px: dict[int, np.ndarray] = field(default_factory=dict)   # id -> (4,2)
    sigma: dict[str, float] = field(default_factory=dict)              # 1-sigma of fitted quantities
    marker_sigma_mm: dict[int, float] = field(default_factory=dict)    # 1-sigma marker centre position
    free_f: bool = False
    free_k1: bool = False

    @property
    def T_table_cam(self) -> np.ndarray:
        return g.inv(self.T_cam_table)

    def T_table_marker(self, i: int) -> np.ndarray:
        return g.planar_T(*self.markers[i])

    def corners_table(self, i: int) -> np.ndarray:
        return g.apply(self.T_table_marker(i), marker_object_points(self.size))

    def rms_px(self, i: int | None = None) -> float:
        r = np.concatenate([v.reshape(-1, 2) for k, v in self.residual_px.items() if i is None or k == i])
        return float(np.sqrt(np.mean(np.sum(r ** 2, 1))))     # per-corner 2D RMS


def ippe(corners: np.ndarray, K, dist, size: float):
    """Both IPPE solutions for a square marker -> list of (T_cam_marker, reprojection error)."""
    _, rv, tv, err = cv2.solvePnPGeneric(marker_object_points(size), corners.reshape(-1, 1, 2), K, dist,
                                         flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return [(g.T_from_rvec(r, t), float(e)) for r, t, e in zip(rv, tv, np.ravel(err))]


def _init(obs, K, dist, size, ref_id):
    """Pick the IPPE branch per marker so all normals agree, then express markers in the ref frame."""
    sols = {i: ippe(c, K, dist, size) for i, c in obs.items()}
    best, best_score = None, math.inf
    for ref_T, _ in sols[ref_id]:
        n_ref = ref_T[:3, 2]
        choice, score = {ref_id: ref_T}, 0.0
        for i, s in sols.items():
            if i != ref_id:
                choice[i] = min(s, key=lambda te: 1 - te[0][:3, 2] @ n_ref)[0]
                score += 1 - choice[i][:3, 2] @ n_ref
        if score < best_score:
            best, best_score = choice, score
    T_cam_table = best[ref_id]
    markers = {}
    for i, T in best.items():
        Tm = g.inv(T_cam_table) @ T
        markers[i] = (float(Tm[0, 3]), float(Tm[1, 3]), math.atan2(Tm[1, 0], Tm[0, 0]))
    markers[ref_id] = (0.0, 0.0, 0.0)
    return T_cam_table, markers


class _Problem:
    def __init__(self, obs, size, W, H, sig, ref_id, base_intr, use_f, use_k1):
        self.obs, self.size, self.W, self.H, self.sig = obs, size, W, H, sig
        self.ids = sorted(obs)
        self.ref_id, self.others = ref_id, [i for i in self.ids if i != ref_id]
        self.base, self.use_f, self.use_k1 = base_intr, use_f, use_k1
        self.obj = marker_object_points(size)
        self.k = 6 + 3 * len(self.others)

    def pack(self, T, mk, intr):
        rv, tv = g.rvec_tvec(T)
        x = [rv, tv, *[mk[i] for i in self.others]]
        if self.use_f:
            x.append([intr["fx"] * self.W])
        if self.use_k1:
            x.append([g.dist_vec(intr)[0]])
        return np.concatenate([np.ravel(a) for a in x])

    def unpack(self, x):
        T = g.T_from_rvec(x[0:3], x[3:6])
        mk = {self.ref_id: (0.0, 0.0, 0.0)}
        for j, i in enumerate(self.others):
            mk[i] = tuple(float(v) for v in x[6 + 3 * j: 9 + 3 * j])
        it = dict(self.base)
        k = self.k
        if self.use_f:
            f = x[k]
            it["fx"], it["fy"] = f / self.W, f / self.H
            k += 1
        if self.use_k1:
            it["dist"] = [float(x[k])] + list(g.dist_vec(self.base)[1:])
        return T, mk, it

    def residuals(self, x, scaled=True):
        T, mk, it = self.unpack(x)
        K, d = g.K_matrix(it, self.W, self.H), g.dist_vec(it)
        rv, tv = g.rvec_tvec(T)
        out = []
        for i in self.ids:
            P = g.apply(g.planar_T(*mk[i]), self.obj)
            uv, _ = cv2.projectPoints(P.reshape(-1, 1, 3), rv, tv, K, d)
            e = uv.reshape(4, 2) - self.obs[i]
            out.append((e / self.sig[i] if scaled else e).ravel())
        return np.concatenate(out)


def fit(obs: dict[int, np.ndarray], size: float, W: int, H: int, intr: dict | None = None,
        sigma_px: dict[int, float] | None = None, ref_id: int | None = None,
        free_f: bool | None = None, free_k1: bool = False, f_scan_hfov=range(40, 131, 5)) -> SceneFit:
    """obs: {marker id: (4,2) pixel corners}. intr None -> estimate f (fx=fy, centred principal point)."""
    ids = sorted(obs)
    if not ids:
        raise ValueError("no markers")
    ref_id = ids[0] if ref_id is None else ref_id
    free_f = intr is None if free_f is None else free_f
    sig = {i: max(0.15, float((sigma_px or {}).get(i, 0.3))) for i in ids}

    def run(base, use_f, use_k1, T0=None, mk0=None):
        p = _Problem(obs, size, W, H, sig, ref_id, base, use_f, use_k1)
        if T0 is None:
            T0, mk0 = _init(obs, g.K_matrix(base, W, H), g.dist_vec(base), size, ref_id)
        x, r, J = g.levmar(p.residuals, p.pack(T0, mk0, base))
        return p, x, r, J

    if free_f:
        scans = []
        for hf in (intr and [None]) or f_scan_hfov:
            base = intr if hf is None else g.intr_from_fov(hf, W, H)
            p, x, r, _ = run(base, False, False)
            scans.append((r @ r, base))
        base = min(scans, key=lambda s: s[0])[1]
        p, x, r, J = run(base, True, False)
        if free_k1:
            T, mk, it = p.unpack(x)
            p, x, r, J = run(it, True, True, T, mk)
    else:
        p, x, r, J = run(intr, False, free_k1)

    T, mk, it = p.unpack(x)
    out = SceneFit(T, mk, it, W, H, size, ref_id, free_f=free_f, free_k1=free_k1)
    res = p.residuals(x, scaled=False).reshape(-1, 4, 2)
    out.residual_px = {i: res[k] for k, i in enumerate(p.ids)}
    s = np.sqrt(np.maximum(np.diag(g.covariance(J, r)), 0))
    k = p.k
    if free_f:
        out.sigma["f_px"], out.sigma["f_rel"] = float(s[k]), float(s[k] / x[k])
        k += 1
    if p.use_k1:
        out.sigma["k1"] = float(s[k])
    out.sigma["cam_pos_mm_in_cam"] = float(np.linalg.norm(s[3:6]) * 1000)
    out.marker_sigma_mm = {ref_id: 0.0, **{i: float(np.hypot(s[6 + 3 * j], s[7 + 3 * j]) * 1000)
                                           for j, i in enumerate(p.others)}}
    return out
