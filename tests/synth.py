"""Synthetic scenes: a camera above a table with ArUco markers, rendered by planar homography."""
from __future__ import annotations

import math

import cv2
import numpy as np

from perception import aruco, geometry as g

W, H = 299, 224
TRUE_INTR = g.intr_from_fov(75.0, W, H)


def look_at(eye, target, up=(0, 0, 1)) -> np.ndarray:
    """T_world_cam for an OpenCV camera at eye looking at target."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return g.T_from(np.c_[x, y, z], eye)


def scene_markers():
    """id -> T_world_marker (marker lies on z=0, facing up), like the real table (base pair + far pair)."""
    return {22: g.planar_T(0.10, -0.20, 0.05), 23: g.planar_T(0.10, 0.20, -0.03),
            20: g.planar_T(0.75, 0.30, 0.10), 21: g.planar_T(0.75, -0.30, 1.6)}


def overview_cam():
    return look_at([1.3, 0.05, 1.0], [0.4, 0.0, 0.0])


def render(T_world_cam, intr, markers, size=0.10, dict_name="DICT_4X4_50", W=W, H=H, ss=4):
    """Render the markers (white border included) into a grey table image. Supersampled for antialiasing."""
    big = np.full((H * ss, W * ss), 175, np.uint8)
    K = g.K_matrix(intr, W, H)
    Kb = K.copy()
    Kb[:2] *= ss
    Kb[0, 2] += (ss - 1) / 2
    Kb[1, 2] += (ss - 1) / 2
    d = aruco.dictionary(dict_name)
    px = 240
    tag = cv2.aruco.generateImageMarker(d, 0, px)
    for mid, T in markers.items():
        tag = cv2.aruco.generateImageMarker(d, mid, px)
        border = px // 6
        tile = cv2.copyMakeBorder(tag, border, border, border, border, cv2.BORDER_CONSTANT, value=255)
        n = tile.shape[0]
        s_tile = size * n / px
        h = s_tile / 2
        # tile pixel (col,row) -> marker frame (x right, y up)
        src = np.float32([[0, 0], [n, 0], [n, n], [0, n]]) - 0.5
        obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
        uv, _ = g.project(T_world_cam, intr, W, H, g.apply(T, obj))
        dst = (uv * ss + (ss - 1) / 2).astype(np.float32)
        Hm = cv2.getPerspectiveTransform(src.astype(np.float32), dst)
        warped = cv2.warpPerspective(tile, Hm, (W * ss, H * ss), flags=cv2.INTER_LINEAR, borderValue=0)
        mask = cv2.warpPerspective(np.full_like(tile, 255), Hm, (W * ss, H * ss), flags=cv2.INTER_NEAREST)
        big[mask > 0] = warped[mask > 0]
    img = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def true_corners(T_world_cam, intr, markers, size=0.10):
    obj = aruco.marker_object_points(size)
    return {i: g.project(T_world_cam, intr, W, H, g.apply(T, obj))[0] for i, T in markers.items()}


def yaw_of(T) -> float:
    return math.atan2(T[1, 0], T[0, 0])
