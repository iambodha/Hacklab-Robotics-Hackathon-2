"""Draw what the maths sees: markers, world axes, a 5 cm table grid, the arm base, the wrist camera's view.

Images are upscaled (default x3) so lines and labels are legible; the ruler ticks along the border are in
ORIGINAL frame pixels (299x224), which is what locate()/pixel_to_table() expect.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import geometry as g

GRID_STEP = 0.05
COL = {"grid": (90, 200, 255), "grid10": (0, 140, 255), "det": (0, 255, 0), "model": (0, 0, 255),
       "x": (0, 0, 255), "y": (0, 200, 0), "z": (255, 80, 0), "text": (255, 255, 255), "wrist": (255, 0, 255)}


def _poly(img, T, intr, W, H, P, color, s, thick=1, closed=False):
    uv, z = g.project(T, intr, W, H, P)
    ok = g.in_image(uv, z, W, H, margin=60)
    pts = []
    for p, k in zip(uv, ok):
        if k:
            pts.append(p)
        elif len(pts) > 1:
            cv2.polylines(img, [np.int32((np.array(pts) + 0.5) * s)], False, color, thick, cv2.LINE_AA)
            pts = []
        else:
            pts = []
    if len(pts) > 1:
        cv2.polylines(img, [np.int32((np.array(pts) + 0.5) * s)], closed and len(pts) == len(uv), color, thick, cv2.LINE_AA)


def _text(img, txt, xy, s=0.45, color=COL["text"]):
    xy = tuple(int(v) for v in xy)
    cv2.putText(img, txt, xy, cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, xy, cv2.FONT_HERSHEY_SIMPLEX, s, color, 1, cv2.LINE_AA)


def _ruler(img, W, H, s, step=25):
    for u in range(0, W, step):
        x = int((u + 0.5) * s)
        cv2.line(img, (x, 0), (x, 6 if u % 50 else 12), COL["text"], 1)
        if u % 50 == 0:
            _text(img, str(u), (x + 2, 22), 0.4)
    for v in range(0, H, step):
        y = int((v + 0.5) * s)
        cv2.line(img, (0, y), (6 if v % 50 else 12, y), COL["text"], 1)
        if v % 50 == 0 and v:
            _text(img, str(v), (14, y + 4), 0.4)


def draw_scene(img, scene, T_world_cam, intr, obs=None, label="", s=3, grid=True):
    H, W = img.shape[:2]
    out = cv2.resize(img, (W * s, H * s), interpolation=cv2.INTER_CUBIC)
    if grid:
        (x0, y0), (x1, y1) = scene.table_extent
        xs = np.arange(np.ceil(x0 / GRID_STEP) * GRID_STEP, x1 + 1e-9, GRID_STEP)
        ys = np.arange(np.ceil(y0 / GRID_STEP) * GRID_STEP, y1 + 1e-9, GRID_STEP)
        for x in xs:
            col, th = (COL["grid10"], 2) if abs(round(x / 0.1) * 0.1 - x) < 1e-6 else (COL["grid"], 1)
            _poly(out, T_world_cam, intr, W, H, np.c_[np.full(60, x), np.linspace(y0, y1, 60), np.zeros(60)], col, s, th)
        for y in ys:
            col, th = (COL["grid10"], 2) if abs(round(y / 0.1) * 0.1 - y) < 1e-6 else (COL["grid"], 1)
            _poly(out, T_world_cam, intr, W, H, np.c_[np.linspace(x0, x1, 60), np.full(60, y), np.zeros(60)], col, s, th)
        for x in xs:
            if abs(round(x / 0.1) * 0.1 - x) < 1e-6:
                uv, z = g.project(T_world_cam, intr, W, H, [[x, ys[0] if len(ys) else y0, 0]])
                if g.in_image(uv, z, W, H)[0]:
                    _text(out, f"x{x:+.1f}", (uv[0] + 0.5) * s + (3, -3), 0.4, COL["grid10"])
    # markers: model outline (red) and detection (green)
    for i, T in scene.markers.items():
        P = g.apply(T, _sq(scene.marker_size))
        _poly(out, T_world_cam, intr, W, H, np.vstack([P, P[:1]]), COL["model"], s, 1)
        uv, z = g.project(T_world_cam, intr, W, H, T[:3, 3])
        if g.in_image(uv, z, W, H)[0]:
            _text(out, f"id{i}", (uv[0] + 0.5) * s + (-12, 4), 0.5)
    for i, c in (obs or {}).items():
        cv2.polylines(out, [np.int32((c + 0.5) * s)], True, COL["det"], 1, cv2.LINE_AA)
    # world axes at the arm base origin (10 cm)
    o = np.zeros(3)
    for k, name in enumerate("xyz"):
        e = np.zeros(3)
        e[k] = 0.10
        _poly(out, T_world_cam, intr, W, H, np.linspace(o, e, 20), COL[name], s, 3)
        uv, z = g.project(T_world_cam, intr, W, H, e)
        if g.in_image(uv, z, W, H)[0]:
            _text(out, name, (uv[0] + 0.5) * s + 4, 0.6, COL[name])
    _ruler(out, W, H, s)
    if label:
        _text(out, label, (10, H * s - 10), 0.5)
    return out


def _sq(size):
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])


def annotate_overview(scene, frame, obs=None, joints=None, extra=None):
    T, intr = scene.T_world_overview, scene.intr("overview")
    out = draw_scene(frame, scene, T, intr, obs,
                     "overview | grid 5 cm (orange = 10 cm) | red = calibrated marker | green = detected | "
                     "ruler = 299x224 px")
    s = out.shape[1] // frame.shape[1]
    H, W = frame.shape[:2]
    if joints is not None and scene.intr("wrist") is not None:
        # wrist camera's view footprint on the table
        Tw = scene.T_world_cam("wrist", joints)
        iw = scene.intr("wrist")
        Ww, Hw = scene.size("wrist")
        edge = np.r_[np.c_[np.linspace(0, Ww - 1, 20), np.zeros(20)], np.c_[np.full(20, Ww - 1), np.linspace(0, Hw - 1, 20)],
                     np.c_[np.linspace(Ww - 1, 0, 20), np.full(20, Hw - 1)], np.c_[np.zeros(20), np.linspace(Hw - 1, 0, 20)]]
        P = g.pixel_to_plane(Tw, iw, Ww, Hw, edge, 0.0)
        P = P[~np.isnan(P).any(1)]
        if len(P) > 2:
            _poly(out, T, intr, W, H, P, COL["wrist"], s, 1)
        from . import wrist as wr
        tcp = (scene.T_world_base @ np.r_[wr.tcp_base(joints), 1])[:3]
        uv, z = g.project(T, intr, W, H, tcp)
        if g.in_image(uv, z, W, H)[0]:
            cv2.drawMarker(out, tuple(int(v) for v in (uv[0] + 0.5) * s), COL["wrist"], cv2.MARKER_CROSS, 14, 2)
            _text(out, "TCP", (uv[0] + 0.5) * s + 6, 0.45, COL["wrist"])
    for item in extra or []:
        _draw_item(out, item, T, intr, W, H, s)
    return out


def annotate_wrist(scene, frame, joints, obs=None, extra=None):
    intr = scene.intr("wrist")
    if intr is None:
        return frame
    T = scene.T_world_cam("wrist", joints)
    out = draw_scene(frame, scene, T, intr, obs, "wrist (pose from FK) | red = marker from the overview calibration | green = detected")
    s = out.shape[1] // frame.shape[1]
    H, W = frame.shape[:2]
    for item in extra or []:
        _draw_item(out, item, T, intr, W, H, s)
    return out


def _draw_item(out, item, T, intr, W, H, s):
    """item: {"polygon": (N,3) world} or {"point": (3,) world, "label": str}."""
    col = item.get("color", (0, 255, 255))
    if "polygon" in item:
        P = np.asarray(item["polygon"], float)
        _poly(out, T, intr, W, H, np.vstack([P, P[:1]]), col, s, 2)
    if "point" in item:
        uv, z = g.project(T, intr, W, H, item["point"])
        if g.in_image(uv, z, W, H)[0]:
            cv2.drawMarker(out, tuple(int(v) for v in (uv[0] + 0.5) * s), col, cv2.MARKER_TILTED_CROSS, 12, 2)
            if item.get("label"):
                _text(out, item["label"], (uv[0] + 0.5) * s + 6, 0.45, col)
