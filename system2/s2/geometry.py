"""Pinhole camera model in the arm base frame (OpenCV axes: x right, y down, z forward)."""
from __future__ import annotations

from dataclasses import dataclass

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

    @classmethod
    def from_info(cls, info: dict) -> 'Camera':
        return cls(int(info['width']), int(info['height']), float(info['fx']), float(info['fy']),
                   float(info['cx']), float(info['cy']), np.asarray(info['R_base_cam'], float),
                   np.asarray(info['t_base_cam'], float))

    def ray(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        d = self.R @ np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])
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
        return np.c_[self.fx * p[:, 0] / z + self.cx, self.fy * p[:, 1] / z + self.cy], front


def wrap_deg(a: float, period: float = 360.0) -> float:
    """Wrap an angle to [-period/2, period/2)."""
    return (a + period / 2) % period - period / 2
