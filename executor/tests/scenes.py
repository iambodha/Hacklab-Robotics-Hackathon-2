"""Synthetic overview images through the calibrated camera (the site's scene_calib.json), for fake-backend tests."""
from __future__ import annotations

import cv2
import numpy as np

from perception import geometry as g
from perception.locate import scene


def render_overview(objects=(), table_bgr=(150, 160, 170), ss: int = 8) -> bytes:
    """JPEG of the table with upright cylinders: objects = [(centre_xy, radius_m, height_m, bgr), ...]."""
    sc = scene()
    T, intr, (W, H) = sc.T_world_overview, sc.intr("overview"), sc.size("overview")
    big = np.zeros((H * ss, W * ss, 3), np.uint8)
    big[:] = table_bgr
    for centre, r, h, bgr in objects:
        a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        ring = np.c_[centre[0] + r * np.cos(a), centre[1] + r * np.sin(a)]
        V = np.r_[np.c_[ring, np.zeros(len(a))], np.c_[ring, np.full(len(a), h)]]
        uv, _ = g.project(T, intr, W, H, V)
        hull = cv2.convexHull(np.float32((uv + 0.5) * ss)).reshape(-1, 2)
        cv2.fillConvexPoly(big, np.int32(np.round(hull)), bgr, cv2.LINE_AA)
    img = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


BLUE = (200, 60, 20)        # BGR
