"""Object detection on a light work surface: everything that is clearly darker or more saturated
than the table becomes a connected-component detection. No object classes and no simulator
state: System 2 decides what a detection is (track), System 1 only follows it.

Overview camera: components that cover the arm (its base, the wrist camera or the TCP, all
projected with the calibration) are the arm. Wrist camera: components touching the bottom
edge are the jaws (or whatever they hold).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import Camera

COLOR_NAMES = [('red', 0), ('orange', 15), ('yellow', 30), ('green', 60), ('cyan', 90), ('blue', 115),
               ('purple', 140), ('pink', 165), ('red', 180)]


@dataclass
class Detection:
    id: str
    camera: str
    u: float
    v: float
    bbox: tuple[int, int, int, int]   # x0, y0, x1, y1 (pixels, inclusive-exclusive)
    area: int
    color: str
    rgb: tuple[int, int, int]
    contour: np.ndarray = field(repr=False)  # (N, 2) pixel outline
    touches_edge: bool = False
    base_u: tuple | None = None    # (u_left, u_right, v) of its lowest row: where it stands on the table

    def at_height(self, cam: Camera, z: float) -> np.ndarray | None:
        return cam.to_plane(self.u, self.v, z)

    def container_footprint(self, cam: Camera, depth: float | None = None) -> tuple[np.ndarray, np.ndarray] | None:
        """Centre xy and [width, depth] of a container seen from the side: the lowest row of its blob is
        where the near wall stands on the table, which gives the width and the near edge reliably. Its
        depth cannot be measured from one side view, so it extends `depth` (default: = width) away
        from the camera."""
        if self.base_u is None:
            return None
        left, right = cam.to_plane(self.base_u[0], self.base_u[2], 0.0), cam.to_plane(self.base_u[1], self.base_u[2], 0.0)
        if left is None or right is None:
            return None
        near = (left[:2] + right[:2]) / 2
        edge = right[:2] - left[:2]
        width = float(np.linalg.norm(edge))
        away = np.array([-edge[1], edge[0]]) / max(width, 1e-6)  # perpendicular to the near wall...
        if away @ (near - cam.t[:2]) < 0:                          # ...pointing away from the camera
            away = -away
        depth = width if depth is None else depth
        return near + away * depth / 2, np.array([width, depth])

    def footprint(self, cam: Camera, z: float) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Outline projected onto the plane z: (centre xy, size [long, short] m, yaw deg of the long side)."""
        pts = [cam.to_plane(float(u), float(v), z) for u, v in self.contour]
        pts = np.array([p[:2] for p in pts if p is not None], np.float32)
        if len(pts) < 3:
            return None
        (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
        if w < h:
            w, h, ang = h, w, ang + 90
        return np.array([cx, cy]), np.array([w, h]), float((ang + 90) % 180 - 90)


def color_name(rgb) -> str:
    hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0, 0]
    h, s, v = int(hsv[0]) * 2 % 360 / 2, int(hsv[1]), int(hsv[2])  # OpenCV hue is 0..179
    if s < 60 or v < 50:
        return 'black' if v < 60 else 'dark grey' if v < 120 else 'grey' if v < 200 else 'white'
    return min(COLOR_NAMES, key=lambda c: abs(c[1] - h))[0]


def object_mask(rgb: np.ndarray, dark_v: int = 110, min_sat: int = 90) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = ((hsv[..., 2] < dark_v) | ((hsv[..., 1] > min_sat) & (hsv[..., 2] > 60))).astype(np.uint8)
    # An odd, symmetric kernel: an even one shifts the mask by a pixel (a few mm on the table).
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _components(rgb, mask, camera, min_area, prefix):
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = mask.shape
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        comp = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outline = max(contours, key=cv2.contourArea)
        contour = outline.reshape(-1, 2).astype(float) + 0.5  # pixel i spans [i, i+1) in camera coordinates
        rgb_mean = tuple(int(c) for c in rgb[labels == i].mean(0))
        det = Detection(f'{prefix}{len(out) + 1}', camera, float(cents[i][0]) + 0.5, float(cents[i][1]) + 0.5,
                        (int(x), int(y), int(x + bw), int(y + bh)), int(area), color_name(rgb_mean), rgb_mean,
                        contour, x <= 0 or y <= 0 or x + bw >= w or y + bh >= h)
        if area > 400:  # big enough to be a container: where does its near wall stand on the table?
            cols = np.flatnonzero(comp.any(0))
            bottom = np.array([np.flatnonzero(comp[:, u]).max() for u in cols])
            near = cols[bottom >= bottom.max() - max(4, bh // 8)]  # the bottom edge, allowing a slight tilt
            det.base_u = (float(near.min()), float(near.max()) + 1.0, float(np.median(bottom[np.isin(cols, near)])) + 1.0)
        out.append(det)
    return out, labels


def detect_overview(rgb: np.ndarray, cam: Camera, arm_points, workspace, min_area: int = 12) -> list[Detection]:
    """arm_points: base-frame points on the arm (base, wrist camera, TCP); workspace: (x0, x1, y0, y1) on z = 0."""
    mask = object_mask(rgb)
    dets, labels = _components(rgb, mask, 'overview', min_area, 'o')
    uv, front = cam.project(np.asarray(arm_points, float))
    h, w = labels.shape
    arm_labels = set()
    for (u, v), ok in zip(uv, front):
        if not ok:
            continue
        x0, x1, y0, y1 = int(u) - 4, int(u) + 5, int(v) - 4, int(v) + 5
        patch = labels[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
        arm_labels.update(int(k) for k in np.unique(patch) if k)
    keep = []
    for d in dets:
        if _label_of(labels, d) in arm_labels:
            continue
        foot = cam.to_plane(d.u, float(d.bbox[3] - 1), 0.0)  # lowest image point: where it meets the table
        if foot is None or not (workspace[0] <= foot[0] <= workspace[1] and workspace[2] <= foot[1] <= workspace[3]):
            continue
        keep.append(d)
    return _renumber(keep, 'o')


def detect_wrist(rgb: np.ndarray, min_area: int = 20) -> list[Detection]:
    mask = object_mask(rgb)
    dets, _ = _components(rgb, mask, 'wrist', min_area, 'w')
    h = rgb.shape[0]
    return _renumber([d for d in dets if d.bbox[3] < h - 1], 'w')  # touching the bottom edge = jaws


def _label_of(labels, d: Detection) -> int:
    """Component label of a detection, robust to non-convex shapes whose centroid falls outside."""
    x, y = d.contour[0].astype(int)
    return int(labels[min(y, labels.shape[0] - 1), min(x, labels.shape[1] - 1)])


def _renumber(dets, prefix):
    for i, d in enumerate(dets, 1):
        d.id = f'{prefix}{i}'
    return dets
