"""Pinhole camera model (optional OpenCV lens distortion) in the arm base frame (OpenCV axes: x right, y down, z forward)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R: np.ndarray  # base <- camera rotation
    t: np.ndarray  # camera centre in the base frame
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))  # OpenCV [k1, k2, p1, p2, k3]

    @classmethod
    def from_info(cls, info: dict) -> 'Camera':
        dist = np.zeros(5)
        d = np.asarray(info.get('dist') or [], float)[:5]
        dist[:len(d)] = d
        return cls(int(info['width']), int(info['height']), float(info['fx']), float(info['fy']),
                   float(info['cx']), float(info['cy']), np.asarray(info['R_base_cam'], float),
                   np.asarray(info['t_base_cam'], float), dist)

    def _distort(self, x: np.ndarray, y: np.ndarray):
        k1, k2, p1, p2, k3 = self.dist
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        return x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y

    def ray(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        xd, yd = (u - self.cx) / self.fx, (v - self.cy) / self.fy
        x, y = xd, yd
        if np.any(self.dist):
            for _ in range(20):                       # fixed-point inverse of the (mild) distortion
                dx, dy = self._distort(x, y)
                x, y = x + xd - dx, y + yd - dy
        d = self.R @ np.array([x, y, 1.0])
        return self.t.copy(), d / np.linalg.norm(d)

    def to_plane(self, u: float, v: float, z: float) -> np.ndarray | None:
        """Where the viewing ray through pixel (u, v) meets the horizontal plane at height z."""
        o, d = self.ray(u, v)
        if abs(d[2]) < 1e-6:
            return None
        s = (z - o[2]) / d[2]
        return o + s * d if s > 0 else None

    def project(self, points) -> tuple[np.ndarray, np.ndarray]:
        """Base-frame points (N, 3) -> pixels (N, 2) and an in-front-of-camera mask."""
        p = (np.atleast_2d(np.asarray(points, float)) - self.t) @ self.R  # rows: R^T (p - t)
        front = p[:, 2] > 1e-6
        z = np.where(front, p[:, 2], 1.0)
        x, y = self._distort(p[:, 0] / z, p[:, 1] / z)
        return np.c_[self.fx * x + self.cx, self.fy * y + self.cy], front


def wrap_deg(a: float, period: float = 360.0) -> float:
    """Wrap an angle to [-period/2, period/2)."""
    return (a + period / 2) % period - period / 2
