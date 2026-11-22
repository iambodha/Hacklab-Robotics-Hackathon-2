"""Camera and rigid-body maths (numpy + cv2). World metres, OpenCV camera axes, pixel centres at integers.

Intrinsics are passed around as a dict in the scene-contract form:
  {"fx": fx/W, "fy": fy/H, "cx": cx/W, "cy": cy/H}  (K_normalized)  plus "dist": [k1,k2,p1,p2,k3]
and converted to a pixel K for a given image size with `K_matrix`.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


# ---- rigid transforms -------------------------------------------------------------------------------
def T_from(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.asarray(t, float).ravel()
    return T


def T_from_rvec(rvec, tvec) -> np.ndarray:
    return T_from(cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0], tvec)


def rvec_tvec(T: np.ndarray):
    return cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3].copy()


def inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return T_from(R.T, -R.T @ t)


def apply(T: np.ndarray, P) -> np.ndarray:
    P = np.atleast_2d(np.asarray(P, float))
    return P @ T[:3, :3].T + T[:3, 3]


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def planar_T(x: float, y: float, yaw: float) -> np.ndarray:
    return T_from(rot_z(yaw), [x, y, 0.0])


# ---- intrinsics ---------------------------------------------------------------------------------------
def K_matrix(intr: dict, W: int, H: int) -> np.ndarray:
    return np.array([[intr["fx"] * W, 0, intr["cx"] * W], [0, intr["fy"] * H, intr["cy"] * H], [0, 0, 1.0]])


def dist_vec(intr: dict) -> np.ndarray:
    return np.array((list(intr.get("dist") or []) + [0.0] * 5)[:5], float)


def intr_from_K(K: np.ndarray, dist, W: int, H: int, **extra) -> dict:
    return {"fx": K[0, 0] / W, "fy": K[1, 1] / H, "cx": K[0, 2] / W, "cy": K[1, 2] / H,
            "dist": [float(d) for d in np.ravel(dist)[:5]], **extra}


def intr_from_fov(hfov_deg: float, W: int, H: int, **extra) -> dict:
    f = (W / 2) / math.tan(math.radians(hfov_deg) / 2)
    return intr_from_K(np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1]]), [0] * 5, W, H, **extra)


def fovy_deg(intr: dict, H: int) -> float:
    return math.degrees(2 * math.atan((H / 2) / (intr["fy"] * H)))


# ---- projection -------------------------------------------------------------------------------------
def project(T_world_cam: np.ndarray, intr: dict, W: int, H: int, P_world):
    """World points (N,3) -> pixels (N,2), depth (N,). Depth <= 0 means behind the camera."""
    P = np.atleast_2d(np.asarray(P_world, float))
    Pc = apply(inv(T_world_cam), P)
    z = Pc[:, 2]
    uv, _ = cv2.projectPoints(Pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K_matrix(intr, W, H), dist_vec(intr))
    return uv.reshape(-1, 2), z


def in_image(uv, depth, W, H, margin: float = 0.5) -> np.ndarray:
    uv = np.atleast_2d(uv)
    return (depth > 0) & (uv[:, 0] >= -margin) & (uv[:, 0] <= W - 1 + margin) & \
        (uv[:, 1] >= -margin) & (uv[:, 1] <= H - 1 + margin)


def pixel_rays(T_world_cam: np.ndarray, intr: dict, W: int, H: int, uv):
    """Pixels (N,2) -> ray origin (3,), unit directions (N,3) in world."""
    uv = np.atleast_2d(np.asarray(uv, float)).reshape(-1, 1, 2)
    xy = cv2.undistortPoints(uv, K_matrix(intr, W, H), dist_vec(intr),
                             criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-10)).reshape(-1, 2)
    d = np.c_[xy, np.ones(len(xy))] @ T_world_cam[:3, :3].T
    return T_world_cam[:3, 3].copy(), d / np.linalg.norm(d, axis=1, keepdims=True)


def ray_plane(o, d, plane=(0, 0, 1, 0)):
    """Intersect rays o + t d with plane n.x + c = 0. -> points (N,3), NaN where parallel or behind."""
    n, c = np.asarray(plane[:3], float), float(plane[3])
    d = np.atleast_2d(d)
    den = d @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -(o @ n + c) / den
        P = o + t[:, None] * d
    P[(np.abs(den) < 1e-12) | (t <= 0)] = np.nan
    return P


def pixel_to_plane(T_world_cam, intr, W, H, uv, z: float = 0.0):
    o, d = pixel_rays(T_world_cam, intr, W, H, uv)
    return ray_plane(o, d, (0, 0, 1, -z))


def pixel_footprint_jacobian(T_world_cam, intr, W, H, uv, z: float = 0.0, eps: float = 0.5) -> np.ndarray:
    """d(table x,y)/d(u,v) at a pixel (2x2, metres per pixel), by central differences."""
    u, v = float(uv[0]), float(uv[1])
    pts = pixel_to_plane(T_world_cam, intr, W, H, [[u + eps, v], [u - eps, v], [u, v + eps], [u, v - eps]], z)
    return np.c_[(pts[0, :2] - pts[1, :2]) / (2 * eps), (pts[2, :2] - pts[3, :2]) / (2 * eps)]


def triangulate_rays(o1, d1, o2, d2):
    """Closest approach of two rays. -> (midpoint, gap between rays in m, t1, t2)."""
    w0 = o1 - o2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d, e = d1 @ w0, d2 @ w0
    den = a * c - b * b
    if den < 1e-12:
        return None, float("inf"), None, None
    t1, t2 = (b * e - c * d) / den, (a * e - b * d) / den
    p1, p2 = o1 + t1 * d1, o2 + t2 * d2
    return (p1 + p2) / 2, float(np.linalg.norm(p1 - p2)), float(t1), float(t2)


# ---- small Levenberg-Marquardt (numpy only; problems here have < 40 parameters) ---------------------
def levmar(fun, x0, iters: int = 200, eps: float = 1e-7, tol: float = 1e-12):
    """Minimise ||fun(x)||^2. -> (x, residuals, J at the solution)."""
    x = np.asarray(x0, float).copy()
    lam = 1e-3
    r = fun(x)
    cost = r @ r

    def jac(x, r):
        J = np.empty((len(r), len(x)))
        for i in range(len(x)):
            h = eps * max(1.0, abs(x[i]))
            xp = x.copy()
            xp[i] += h
            J[:, i] = (fun(xp) - r) / h
        return J

    J = jac(x, r)
    for _ in range(iters):
        A, g = J.T @ J, J.T @ r
        step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
        xn = x + step
        rn = fun(xn)
        cn = rn @ rn
        if cn < cost:
            done = cost - cn < tol * max(cost, 1e-30) or np.linalg.norm(step) < 1e-12
            x, r, cost, lam = xn, rn, cn, max(lam / 3, 1e-12)
            J = jac(x, r)
            if done:
                break
        else:
            lam *= 4
            if lam > 1e12:
                break
    return x, r, J


def covariance(J: np.ndarray, r: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """Parameter covariance. With sigma None, scale by the residual variance (a posteriori)."""
    dof = max(1, len(r) - J.shape[1])
    s2 = (r @ r) / dof if sigma is None else sigma ** 2
    return np.linalg.pinv(J.T @ J) * s2
