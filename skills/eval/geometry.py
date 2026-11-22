"""Board rectification: table-plane homography from the calibrated overview camera (a site's scene_calib.json,
sites.get_site().scene_calib, written by perception) or from >= 4 image<->world point pairs (e.g. marker corners). Pure numpy, deterministic.

Rectified board image convention: row i <-> world y = y0 + (i + 0.5) / px_per_m, column j <-> x = x0 + (j + 0.5) / px_per_m.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Board:
    """The whole board in world coordinates (metres, axis-aligned on the table plane)."""
    x0: float
    y0: float
    x1: float
    y1: float
    z: float = 0.0

    def shape(self, px_per_m: float) -> tuple[int, int]:
        return int(round((self.y1 - self.y0) * px_per_m)), int(round((self.x1 - self.x0) * px_per_m))

    def grid(self, px_per_m: float):
        """World x, y of every rectified pixel centre (two arrays of the rectified shape)."""
        h, w = self.shape(px_per_m)
        xs = self.x0 + (np.arange(w) + 0.5) / px_per_m
        ys = self.y0 + (np.arange(h) + 0.5) / px_per_m
        return np.meshgrid(xs, ys)

    def corners(self):
        return np.array([[self.x0, self.y0], [self.x1, self.y0], [self.x1, self.y1], [self.x0, self.y1]], float)


def _distort(xn: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """OpenCV radial/tangential model [k1, k2, p1, p2, k3] on normalised image coords (..., 2)."""
    k1, k2, p1, p2, k3 = (list(dist) + [0.0] * 5)[:5]
    x, y = xn[..., 0], xn[..., 1]
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([xd, yd], -1)


def _undistort(xd: np.ndarray, dist: np.ndarray, iters: int = 20) -> np.ndarray:
    """Inverse of _distort by fixed-point iteration (enough for the mild UVC lenses here)."""
    x = xd.copy()
    for _ in range(iters):
        x = x + (xd - _distort(x, dist))
    return x


@dataclass(frozen=True)
class PlaneMap:
    """Image <- world-plane mapping with lens distortion: the undistorted homography H, then K^-1, distortion, K.
    apply_H() accepts it wherever a plain 3x3 homography is accepted."""
    H: np.ndarray
    K: np.ndarray
    dist: np.ndarray

    def apply(self, xy) -> np.ndarray:
        xy = np.asarray(xy, float)
        p = np.stack([xy[..., 0], xy[..., 1], np.ones(xy.shape[:-1])], -1) @ self.H.T
        uv = p[..., :2] / p[..., 2:3]
        xn = np.stack([(uv[..., 0] - self.K[0, 2]) / self.K[0, 0], (uv[..., 1] - self.K[1, 2]) / self.K[1, 1]], -1)
        xd = _distort(xn, self.dist)
        return np.stack([xd[..., 0] * self.K[0, 0] + self.K[0, 2], xd[..., 1] * self.K[1, 1] + self.K[1, 2]], -1)


@dataclass
class Camera:
    K: np.ndarray            # 3x3 pixel intrinsics for `size`
    T_world_cam: np.ndarray  # 4x4, OpenCV camera axes
    size: tuple[int, int]    # (width, height)
    dist: np.ndarray | None = None  # OpenCV [k1, k2, p1, p2, k3]; None/zeros = pinhole

    @classmethod
    def from_scene_calib(cls, calib: dict | str | Path, role: str = "overview") -> "Camera":
        if not isinstance(calib, dict):
            calib = json.loads(Path(calib).read_text())
        c = calib["cameras"][role]
        W, H = c["image_size"]
        k = c["K_normalized"]
        K = np.array([[k["fx"] * W, 0, k["cx"] * W], [0, k["fy"] * H, k["cy"] * H], [0, 0, 1]], float)
        dist = np.array([float(d) for d in c.get("dist", [])][:5], float)
        return cls(K, np.array(c["T_world_cam"], float), (int(W), int(H)),
                   dist if np.any(np.abs(dist) > 1e-9) else None)

    def _H_pinhole(self, z: float = 0.0) -> np.ndarray:
        T = np.linalg.inv(self.T_world_cam)
        R, t = T[:3, :3], T[:3, 3]
        M = np.column_stack([R[:, 0], R[:, 1], R[:, 2] * z + t])
        return self.K @ M

    def H_img_world(self, z: float = 0.0):
        """Image <- world-plane (height z) mapping: a 3x3 homography [u, v, 1] ~ H @ [x, y, 1] for a pinhole camera,
        or a PlaneMap (homography + lens distortion) when the calibration has distortion. Use via apply_H/rectify."""
        H = self._H_pinhole(z)
        return H if self.dist is None else PlaneMap(H, self.K, self.dist)

    def project(self, xyz) -> np.ndarray:
        P = np.atleast_2d(np.asarray(xyz, float))
        T = np.linalg.inv(self.T_world_cam)
        pc = P @ T[:3, :3].T + T[:3, 3]
        xn = pc[:, :2] / pc[:, 2:3]
        if self.dist is not None:
            xn = _distort(xn, self.dist)
        return np.column_stack([xn[:, 0] * self.K[0, 0] + self.K[0, 2], xn[:, 1] * self.K[1, 1] + self.K[1, 2]])

    def pixel_to_plane(self, uv, z: float = 0.0) -> np.ndarray:
        uv = np.atleast_2d(np.asarray(uv, float))
        if self.dist is not None:
            xd = np.column_stack([(uv[:, 0] - self.K[0, 2]) / self.K[0, 0], (uv[:, 1] - self.K[1, 2]) / self.K[1, 1]])
            xn = _undistort(xd, self.dist)
            uv = np.column_stack([xn[:, 0] * self.K[0, 0] + self.K[0, 2], xn[:, 1] * self.K[1, 1] + self.K[1, 2]])
        rays = np.column_stack([uv, np.ones(len(uv))]) @ np.linalg.inv(self.K).T
        d = rays @ self.T_world_cam[:3, :3].T
        o = self.T_world_cam[:3, 3]
        s = (z - o[2]) / d[:, 2]
        return o + d * s[:, None]


def homography_from_points(img_uv, world_xy) -> np.ndarray:
    """DLT homography [u, v, 1] ~ H @ [x, y, 1] from >= 4 correspondences (e.g. ArUco corners on the table)."""
    uv, xy = np.asarray(img_uv, float), np.asarray(world_xy, float)
    if len(uv) < 4 or len(uv) != len(xy):
        raise ValueError("need >= 4 matching image/world points")
    A = []
    for (u, v), (x, y) in zip(uv, xy):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.array(A))
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def apply_H(H, xy) -> np.ndarray:
    if isinstance(H, PlaneMap):
        return H.apply(xy)
    xy = np.asarray(xy, float)
    p = np.stack([xy[..., 0], xy[..., 1], np.ones(xy.shape[:-1])], -1) @ H.T
    return p[..., :2] / p[..., 2:3]


def rectify(image: np.ndarray, H: np.ndarray, board: Board, px_per_m: float):
    """(rectified float image HxWx3 in 0..1, mask of pixels that came from inside the camera image)."""
    img = image.astype(np.float32) / (255.0 if image.dtype == np.uint8 else 1.0)
    X, Y = board.grid(px_per_m)
    uv = apply_H(H, np.stack([X, Y], -1))
    u, v = uv[..., 0] - 0.5, uv[..., 1] - 0.5           # pixel centres at integer + 0.5
    h, w = img.shape[:2]
    inside = (u >= 0) & (v >= 0) & (u <= w - 1) & (v <= h - 1)
    u, v = np.clip(u, 0, w - 1), np.clip(v, 0, h - 1)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    u1, v1 = np.minimum(u0 + 1, w - 1), np.minimum(v0 + 1, h - 1)
    fu, fv = (u - u0)[..., None], (v - v0)[..., None]
    out = (img[v0, u0] * (1 - fu) * (1 - fv) + img[v0, u1] * fu * (1 - fv) +
           img[v1, u0] * (1 - fu) * fv + img[v1, u1] * fu * fv)
    return out, inside


def segment_distance(X, Y, segments) -> np.ndarray:
    """Distance (metres) of every point (X, Y) to the nearest of the given segments [((x0, y0), (x1, y1)), ...]."""
    d = np.full(X.shape, np.inf)
    for a, b in segments:
        a, b = np.asarray(a, float), np.asarray(b, float)
        ab = b - a
        L2 = float(ab @ ab) or 1e-12
        t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / L2, 0, 1)
        d = np.minimum(d, np.hypot(X - (a[0] + t * ab[0]), Y - (a[1] + t * ab[1])))
    return d
