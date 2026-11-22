"""Synthetic wipe world for tests, the experiment runner's sim backend and the loop's sim benchmark.

A board with a dirt-density map (1 mm grid, world frame) is wiped by strokes of a KNOWN true width, then rendered
through a pinhole camera (default: the calibrated overview camera) with optional shadows, an occluder, noise and
JPEG compression. Seeded, so every image is reproducible. Lives next to the evaluator because, like it, the
improvement loop must not be able to change it.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np

from .geometry import Board, Camera, segment_distance

# Frozen fixture, deliberately not site-selected (seeded images must not change with SITE): home's
# calibration/scene_calib.json (perception, 2026-11-22 00:20) overview camera, 299x224
OVERVIEW_CALIB = {"cameras": {"overview": {
    "image_size": [299, 224], "K_normalized": {"fx": 0.485183, "fy": 0.647633, "cx": 0.498328, "cy": 0.497768},
    "dist": [0, 0, 0, 0, 0],
    "T_world_cam": [[-0.03043, 0.712648, -0.700861, 0.643111], [0.999456, 0.012781, -0.030399, -0.019168],
                    [-0.012706, -0.701405, -0.712649, 0.384288], [0, 0, 0, 1]]}}}

BOARD = Board(0.14, -0.16, 0.44, 0.14)             # 30 x 30 cm white board in front of the arm
TARGET = (0.22, -0.07, 0.34, 0.03)                 # 12 x 10 cm stain area (x0, y0, x1, y1)


def default_camera(scale: int = 1) -> Camera:
    cam = Camera.from_scene_calib(OVERVIEW_CALIB)
    if scale != 1:
        cam = Camera(cam.K @ np.diag([scale, scale, 1.0]), cam.T_world_cam, (cam.size[0] * scale, cam.size[1] * scale))
    return cam


@dataclass
class SynthWorld:
    board: Board = BOARD
    res: float = 1000.0                                      # dirt grid, px per metre
    board_rgb: tuple = (0.93, 0.93, 0.91)
    stain_rgb: tuple = (0.50, 0.32, 0.18)                    # cocoa / coffee
    table_rgb: tuple = (0.60, 0.44, 0.30)
    dirt: np.ndarray = field(default=None, repr=False)
    log: list = field(default_factory=list)

    def __post_init__(self):
        if self.dirt is None:
            self.dirt = np.zeros(self.board.shape(self.res), np.float32)
        self.X, self.Y = self.board.grid(self.res)

    def copy(self) -> "SynthWorld":
        return SynthWorld(self.board, self.res, self.board_rgb, self.stain_rgb, self.table_rgb, self.dirt.copy(), list(self.log))

    # stains ------------------------------------------------------------------------------------
    def add_stain_rect(self, rect, seed: int = 0, ragged: float = 0.004, density: float = 1.0):
        """A stain filling `rect` (x0, y0, x1, y1) with a ragged edge (+- `ragged` m)."""
        x0, y0, x1, y1 = rect
        rng = np.random.default_rng(seed)
        k = np.arange(1, 6)
        ph = rng.uniform(0, 2 * np.pi, (4, 5))
        def wob(t, i):
            return ragged * np.sum(np.sin(np.outer(t, k) * 40 + ph[i]) / k, -1) / 1.5
        X, Y = self.X, self.Y
        inside = ((X >= x0 + wob(Y.ravel(), 0).reshape(X.shape)) & (X <= x1 + wob(Y.ravel(), 1).reshape(X.shape)) &
                  (Y >= y0 + wob(X.ravel(), 2).reshape(X.shape)) & (Y <= y1 + wob(X.ravel(), 3).reshape(X.shape)))
        self.dirt = np.maximum(self.dirt, inside.astype(np.float32) * density)
        self.log.append({"stain_rect": list(rect), "seed": seed})
        return self

    def add_stain_blobs(self, centre, spread: float, n: int = 6, r=(0.008, 0.02), seed: int = 0):
        rng = np.random.default_rng(seed)
        for _ in range(n):
            c = np.asarray(centre) + rng.normal(0, spread, 2)
            rad = rng.uniform(*r)
            self.dirt = np.maximum(self.dirt, (np.hypot(self.X - c[0], self.Y - c[1]) <= rad).astype(np.float32))
        self.log.append({"stain_blobs": list(map(float, centre)), "spread": spread, "n": n, "seed": seed})
        return self

    # wiping ------------------------------------------------------------------------------------
    def wipe(self, strokes, w_true: float, efficiency: float = 0.97, push_frac: float = 0.03, offset=(0.0, 0.0),
             edge: float = 0.001):
        """Each stroke ((x0, y0), (x1, y1)) removes `efficiency` of the dirt within w_true/2 of the (offset) path and
        pushes `push_frac` of what it removed into a pile just past the stroke's end."""
        o = np.asarray(offset, float)
        for a, b in strokes:
            a, b = np.asarray(a, float) + o, np.asarray(b, float) + o
            d = segment_distance(self.X, self.Y, [(a, b)])
            r = efficiency * np.clip((w_true / 2 - d) / edge + 0.5, 0, 1)
            removed = float((self.dirt * r).sum())
            self.dirt *= (1 - r).astype(np.float32)
            u = (b - a) / max(np.linalg.norm(b - a), 1e-9)
            pile = (np.hypot(self.X - (b[0] + u[0] * w_true / 2), self.Y - (b[1] + u[1] * w_true / 2)) <= w_true / 2) & \
                   (((self.X - b[0]) * u[0] + (self.Y - b[1]) * u[1]) > 0)
            if pile.any() and push_frac > 0:
                self.dirt[pile] = np.minimum(1.0, self.dirt[pile] + push_frac * removed / pile.sum())
        self.log.append({"wipe": [[list(map(float, a)), list(map(float, b))] for a, b in strokes], "w_true": w_true,
                         "efficiency": efficiency, "push_frac": push_frac, "offset": list(map(float, o))})
        return self

    # metrics on the ground truth ------------------------------------------------------------
    def dirt_area(self, thresh: float = 0.5) -> float:
        return float((self.dirt >= thresh).sum()) / self.res ** 2

    # rendering ----------------------------------------------------------------------------------
    def render(self, cam: Camera | None = None, *, shadow=None, occluder=None, occluder_rgb=(0.18, 0.18, 0.2),
               noise: float = 0.01, seed: int = 0, jpeg_quality: int | None = 90, ss: int = 2) -> np.ndarray:
        """uint8 RGB image. shadow: (x, y, sigma_m, depth) soft multiplicative shadow on the plane.
        occluder: list of image-pixel polygons [(u, v), ...] drawn in occluder_rgb (e.g. the arm)."""
        cam = cam or default_camera()
        W, H = cam.size
        uu, vv = np.meshgrid((np.arange(W * ss) + 0.5) / ss, (np.arange(H * ss) + 0.5) / ss)
        P = cam.pixel_to_plane(np.stack([uu.ravel(), vv.ravel()], 1), self.board.z)
        x, y = P[:, 0], P[:, 1]
        bd = self.board
        on = (x >= bd.x0) & (x < bd.x1) & (y >= bd.y0) & (y < bd.y1)
        j = np.clip(((x - bd.x0) * self.res).astype(int), 0, self.dirt.shape[1] - 1)
        i = np.clip(((y - bd.y0) * self.res).astype(int), 0, self.dirt.shape[0] - 1)
        dd = np.where(on, self.dirt[i, j], 0.0)[:, None]
        rgb = np.where(on[:, None], np.asarray(self.board_rgb) * (1 - dd) + np.asarray(self.stain_rgb) * dd,
                       np.asarray(self.table_rgb))
        if shadow is not None:
            sx, sy, sig, depth = shadow
            rgb = rgb * (1 - depth * np.exp(-((x - sx) ** 2 + (y - sy) ** 2) / (2 * sig ** 2)))[:, None]
        img = rgb.reshape(H * ss, W * ss, 3)
        if occluder:
            from PIL import Image, ImageDraw
            m = Image.new("L", (W * ss, H * ss), 0)
            for poly in occluder:
                ImageDraw.Draw(m).polygon([(u * ss, v * ss) for u, v in poly], fill=255)
            mk = np.asarray(m) > 0
            img[mk] = occluder_rgb
        img = img.reshape(H, ss, W, ss, 3).mean((1, 3))
        if noise:
            img = img + np.random.default_rng(seed).normal(0, noise, img.shape)
        out = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
        if jpeg_quality:
            from PIL import Image
            buf = io.BytesIO()
            Image.fromarray(out).save(buf, "JPEG", quality=jpeg_quality)
            out = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
        return out


def to_jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def from_jpeg(data: bytes) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
