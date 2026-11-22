"""Localisation API, all in world metres (scene-contract.md world frame).

Claude recognises (gives a pixel, a bbox or a colour); this module measures:
  pixel_to_table(cam, u, v, z=0)            ray / plane intersection, with mm-per-pixel and a 1-sigma in mm
  world_to_pixel(cam, xyz)                  projection (+ in-view flag)
  triangulate(obs_overview, obs_wrist)      two-camera ray intersection with gap and 1-sigma
  locate_object(cam, image, bbox|pixel, color, height_m)  footprint centre, size, yaw on the table
  scene_map()                               table, markers, arm base, camera frustums, located objects

`cam` is "overview" or "wrist". The wrist camera pose comes from FK on the live joint readings, so wrist
calls need `joints` ({"j1": {"raw", "lo", "hi"}, ...} as read from Station).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from . import geometry as g
from .scene import SCENE_PATH, Scene

DEFAULT_PIXEL_SIGMA = 1.0          # px: how well a pixel is usually pointed at (Claude, a click, a bbox centre)
_SCENE: Scene | None = None


def scene() -> Scene:
    global _SCENE
    if _SCENE is None:
        _SCENE = Scene.load()
    return _SCENE


def scene_file_exists() -> bool:
    return SCENE_PATH.exists()


def reload() -> Scene:
    global _SCENE
    _SCENE = None
    return scene()


class Locator:
    def __init__(self, sc: Scene | None = None):
        self.sc = sc or scene()

    # ---- cameras ---------------------------------------------------------------------------------------
    def camera(self, cam: str, joints: dict | None = None):
        intr = self.sc.intr(cam)
        if intr is None:
            raise ValueError(f"{cam} camera has no intrinsics yet (run perception.calib_intrinsics or calib_scene)")
        W, H = self.sc.size(cam)
        return self.sc.T_world_cam(cam, joints), intr, W, H

    def calib_sigma_mm(self, cam: str, xyz=None) -> float:
        """Calibration part of the error budget at a world point (1-sigma, mm)."""
        q = self.sc.d.get("quality", {})
        ho = [r["rms_mm"] for r in q.get("held_out") or []]
        s_ov = float(np.median(ho)) if ho else 5.0
        if cam == "overview":
            return s_ov
        b = self.sc.d["arm"].get("base_on_table", {})
        s_b, s_yaw = float(b.get("sigma_mm") or 20.0), math.radians(float(b.get("sigma_yaw_deg") or 3.0))
        r = 0.3 if xyz is None else float(np.linalg.norm(np.asarray(xyz)[:2]))
        # wrist intrinsics not board-calibrated: a relative focal error f_rel scales off-axis distances
        f_rel = float(self.sc.d["cameras"]["wrist"].get("sigma_f_rel") or 0.0)
        s_f = 1000 * 0.15 * f_rel                                    # ~15 cm typical off-axis distance in view
        return float(math.sqrt(s_b ** 2 + (1000 * r * s_yaw) ** 2 + s_ov ** 2 + s_f ** 2))

    def base_sigma_mm(self, xyz) -> float:
        """Arm-base (world origin) uncertainty at a world point: what the arm sees on top of sigma_centre_mm."""
        b = self.sc.d["arm"].get("base_on_table", {})
        s_b, s_yaw = float(b.get("sigma_mm") or 20.0), math.radians(float(b.get("sigma_yaw_deg") or 3.0))
        return float(math.hypot(s_b, 1000 * float(np.linalg.norm(np.asarray(xyz)[:2])) * s_yaw))

    # ---- basic maps ------------------------------------------------------------------------------------
    def pixel_to_table(self, cam: str, u: float, v: float, z: float = 0.0, joints: dict | None = None,
                       pixel_sigma: float = DEFAULT_PIXEL_SIGMA) -> dict:
        T, intr, W, H = self.camera(cam, joints)
        P = g.pixel_to_plane(T, intr, W, H, [[u, v]], z)[0]
        if np.isnan(P).any():
            return {"ok": False, "reason": f"pixel ({u}, {v}) of the {cam} camera does not look at the plane z={z}"}
        J = g.pixel_footprint_jacobian(T, intr, W, H, (u, v), z)
        sv = np.linalg.svd(J, compute_uv=False) * 1000
        s_px = float(sv[0]) * pixel_sigma
        s_cal = self.calib_sigma_mm(cam, P)
        s = math.hypot(s_px, s_cal)
        return {"ok": True, "xyz": P.round(4).tolist(), "camera": cam, "pixel": [u, v], "z_plane": z,
                "mm_per_px": sv.round(2).tolist(), "sigma_mm": round(s, 1),
                "sigma_vs_arm_mm": round(math.hypot(s, self.base_sigma_mm(P)), 1) if cam == "overview" else round(s, 1),
                "sigma_parts_mm": {"pixel": round(s_px, 1), "calibration": round(s_cal, 1)},
                "range_m": round(float(np.linalg.norm(P - T[:3, 3])), 3)}

    def world_to_pixel(self, cam: str, xyz, joints: dict | None = None) -> dict:
        T, intr, W, H = self.camera(cam, joints)
        uv, depth = g.project(T, intr, W, H, [xyz])
        inside = bool(g.in_image(uv, depth, W, H)[0])
        return {"camera": cam, "pixel": uv[0].round(2).tolist(), "depth_m": round(float(depth[0]), 4), "in_view": inside}

    def triangulate(self, obs_overview, obs_wrist, joints: dict, pixel_sigma: float = DEFAULT_PIXEL_SIGMA) -> dict:
        """obs_*: (u, v) of the same physical point in each camera."""
        rays, sig = [], []
        for cam, uv in (("overview", obs_overview), ("wrist", obs_wrist)):
            T, intr, W, H = self.camera(cam, joints)
            o, d = g.pixel_rays(T, intr, W, H, [uv])
            rays.append((o, d[0]))
            sig.append(pixel_sigma / (intr["fx"] * W))           # angular sigma, rad
        p, gap, t1, t2 = g.triangulate_rays(rays[0][0], rays[0][1], rays[1][0], rays[1][1])
        if p is None or t1 <= 0 or t2 <= 0:
            return {"ok": False, "reason": "rays are parallel or the point is behind a camera"}
        A = np.zeros((3, 3))
        for (o, d), s, t in zip(rays, sig, (t1, t2)):
            A += (np.eye(3) - np.outer(d, d)) / (t * s) ** 2
        cov = np.linalg.pinv(A)
        s_geo = float(np.sqrt(np.trace(cov))) * 1000
        s_cal = math.hypot(self.calib_sigma_mm("overview"), self.calib_sigma_mm("wrist", p))
        angle = math.degrees(math.acos(abs(float(rays[0][1] @ rays[1][1]))))
        return {"ok": True, "xyz": p.round(4).tolist(), "ray_gap_mm": round(gap * 1000, 1),
                "sigma_mm": round(math.hypot(s_geo, s_cal / math.sqrt(2)), 1),
                "sigma_parts_mm": {"pixel": round(s_geo, 1), "calibration": round(s_cal / math.sqrt(2), 1)},
                "ray_angle_deg": round(angle, 1),
                "note": "ray_gap_mm much larger than sigma_mm means the two pixels are not the same point "
                        "or a calibration is off"}

    # ---- objects ---------------------------------------------------------------------------------------
    def locate_object(self, cam: str, image: np.ndarray, bbox=None, pixel=None, color: str | None = None,
                      height_m: float = 0.0, joints: dict | None = None, grid_mm: float = 2.0) -> dict:
        """Footprint of an object on the table from one view.

        Give a bbox (u0, v0, u1, v1) and/or a pixel on the object; optionally a colour name. The silhouette
        is segmented (colour threshold, else GrabCut in the bbox) and its footprint is taken as the table
        points whose projections at z=0 AND at z=height_m both fall inside the silhouette: exact for a
        vertical prism, and independent of where the camera looks from. height_m = 0 treats the object as
        flat; `centre_shift_mm_per_cm_height` says how much the centre would move per cm of real height.
        height_m = None estimates the height, assuming a round or square footprint (estimate_height).
        Colour without bbox/pixel searches only pixels that look at the table."""
        T, intr, W, H = self.camera(cam, joints)
        # colour only: search where the camera sees the table (not the window, the wall or the floor)
        roi = table_mask(T, intr, W, H, self.sc.table_extent) if bbox is None and pixel is None else None
        mask, how = segment(image, bbox=bbox, pixel=pixel, color=color, roi=roi)
        if mask is None or mask.sum() < 3:
            if pixel is not None:
                r = self.pixel_to_table(cam, pixel[0], pixel[1], height_m or 0.0, joints)
                r["method"] = f"point only ({how}); no silhouette"
                return r
            return {"ok": False, "reason": f"no object found ({how})"}
        auto = height_m is None
        if auto:
            height_m = estimate_height(mask, T, intr, W, H, grid_mm)
        fp = footprint(mask, T, intr, W, H, height_m, grid_mm)
        if fp is None:
            return {"ok": False, "reason": "silhouette does not intersect the table plane"}
        res = self._describe(fp, cam, T, intr, W, H, grid_mm)
        alt = footprint(mask, T, intr, W, H, height_m + 0.01, grid_mm)
        if alt is not None:
            res["centre_shift_mm_per_cm_height"] = round(float(np.linalg.norm(alt.mean(0) - fp.mean(0))) * 1000, 1)
            if auto:              # estimated height good to ~0.5 (round) - 1 cm (square): add its effect on the centre
                res["sigma_centre_mm"] = round(math.hypot(res["sigma_centre_mm"], 0.7 * res["centre_shift_mm_per_cm_height"]), 1)
        res.update({"camera": cam, "method": how, "height_m": round(height_m, 3), "silhouette_px": int(mask.sum())})
        # the wrist's sigma already contains the base (FK runs from it); the overview's does not
        res["sigma_vs_arm_mm"] = round(math.hypot(res["sigma_centre_mm"], self.base_sigma_mm(res["centre_xy"])), 1) \
            if cam == "overview" else res["sigma_centre_mm"]
        if height_m > 0:
            res["top"] = top_outline(mask, T, intr, W, H, height_m)
        if auto:
            res["height_source"] = ("estimated from the silhouette assuming a round (~+-5 mm) or square (~+-1 cm) "
                                    "footprint; pass height_m if known")
        return res

    def _describe(self, pts, cam, T, intr, W, H, grid_mm) -> dict:
        c = pts.mean(0)
        box = cv2.boxPoints(cv2.minAreaRect((pts * 1000).astype(np.float32))) / 1000
        e1, e2 = box[1] - box[0], box[2] - box[1]
        long_e, short_e = (e1, e2) if np.linalg.norm(e1) >= np.linalg.norm(e2) else (e2, e1)
        l, w = float(np.linalg.norm(long_e)) * 1000, float(np.linalg.norm(short_e)) * 1000
        yaw = (math.degrees(math.atan2(long_e[1], long_e[0])) + 90) % 180 - 90   # long side vs +x, [-90, 90)
        uv, _ = g.project(T, intr, W, H, [[c[0], c[1], 0.0]])
        J = g.pixel_footprint_jacobian(T, intr, W, H, uv[0], 0.0)
        mmpp = float(np.linalg.svd(J, compute_uv=False)[0] * 1000)
        s_cal = self.calib_sigma_mm(cam, [c[0], c[1], 0])
        s_c = math.hypot(0.5 * mmpp, s_cal)
        return {"ok": True, "centre_xy": c.round(4).tolist(), "size_m": [round(l / 1000, 4), round(w / 1000, 4)],
                "yaw_deg": round(float(yaw), 1), "area_cm2": round(len(pts) * (grid_mm / 10) ** 2, 1),
                "sigma_centre_mm": round(s_c, 1), "sigma_size_mm": round(mmpp, 1), "mm_per_px": round(mmpp, 2),
                "polygon_xy": np.round(cv2.convexHull((pts * 1000).astype(np.float32)).reshape(-1, 2) / 1000, 4).tolist(),
                "box_xy": np.round(box, 4).tolist()}

    # ---- scene map ---------------------------------------------------------------------------------------
    def scene_map(self, joints: dict | None = None, objects: list | None = None) -> dict:
        sc = self.sc
        out = {"frame": sc.d["world_frame"], "units": "metres, degrees",
               "table": {"extent_xy": sc.table_extent.round(4).tolist(), "source": sc.d["table"].get("extent_source"),
                         "z": 0.0},
               "markers": [{"id": i, "centre_xy": T[:2, 3].round(4).tolist(),
                            "yaw_deg": round(math.degrees(math.atan2(T[1, 0], T[0, 0])), 1),
                            "role": next(m["role"] for m in sc.d["markers"]["items"] if m["id"] == i)}
                           for i, T in sc.markers.items()],
               "arm_base": {"origin_xyz": sc.T_world_base[:3, 3].round(4).tolist(), "forward": "+x",
                            "source": sc.d["arm"].get("base_on_table", {}).get("source")},
               "cameras": {"overview": self._cam_summary("overview")}}
        if joints is not None and sc.intr("wrist") is not None:
            out["cameras"]["wrist"] = self._cam_summary("wrist", joints)
            from . import wrist
            out["tcp_xyz"] = (sc.T_world_base @ np.r_[wrist.tcp_base(joints), 1])[:3].round(4).tolist()
        out["objects"] = objects or []
        return out

    def _cam_summary(self, cam, joints=None) -> dict:
        T, intr, W, H = self.camera(cam, joints)
        z = T[:3, 2]
        edge = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], float)
        P = g.pixel_to_plane(T, intr, W, H, edge, 0.0)
        return {"position_xyz": T[:3, 3].round(4).tolist(), "view_dir": z.round(3).tolist(),
                "pitch_down_deg": round(math.degrees(math.asin(max(-1, min(1, -z[2])))), 1),
                "fovy_deg": round(g.fovy_deg(intr, H), 1),
                "table_footprint_corners_xy": [None if np.isnan(p).any() else p[:2].round(3).tolist() for p in P],
                "intrinsics": intr.get("source")}


# ---- segmentation + footprint (module level so they're testable on their own) -------------------------------
HSV_RANGES = {   # OpenCV HSV: H 0-180
    "red": [((0, 90, 60), (10, 255, 255)), ((170, 90, 60), (180, 255, 255))],
    "orange": [((10, 100, 80), (22, 255, 255))], "yellow": [((22, 80, 80), (35, 255, 255))],
    "green": [((35, 60, 40), (85, 255, 255))], "blue": [((90, 70, 40), (130, 255, 255))],
    "purple": [((130, 50, 40), (170, 255, 255))], "black": [((0, 0, 0), (180, 255, 60))],
    "white": [((0, 0, 200), (180, 40, 255))],
}


def table_mask(T, intr, W: int, H: int, extent_xy, margin_m: float = 0.05) -> np.ndarray:
    """uint8 0/1: pixels whose ray meets the table plane inside extent_xy (+ margin)."""
    vs, us = np.mgrid[0:H, 0:W]
    P = g.pixel_to_plane(T, intr, W, H, np.c_[us.ravel(), vs.ravel()].astype(float), 0.0)[:, :2]
    (x0, y0), (x1, y1) = np.asarray(extent_xy) + [[-margin_m, -margin_m], [margin_m, margin_m]]
    with np.errstate(invalid="ignore"):
        ok = (P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)
    return ok.reshape(H, W).astype(np.uint8)


def segment(image: np.ndarray, bbox=None, pixel=None, color: str | None = None, roi: np.ndarray | None = None):
    """-> (mask uint8 0/1 or None, description).
    roi (no bbox/pixel only): 0/1 mask of where objects stand. The largest component whose lowest pixel (its
    contact with the table, nearest the camera) is inside roi wins; the component itself is not clipped, so a tall
    object whose top projects beyond the table still keeps its whole silhouette."""
    H, W = image.shape[:2]
    stand = None
    if roi is not None and bbox is None and pixel is None:
        stand, roi = roi, None
    if bbox is not None:
        roi = np.zeros((H, W), np.uint8)
        u0, v0, u1, v1 = [int(round(x)) for x in bbox]
        u0, u1 = sorted((max(0, u0), min(W - 1, u1)))
        v0, v1 = sorted((max(0, v0), min(H - 1, v1)))
        pad_u, pad_v = max(2, (u1 - u0) // 8), max(2, (v1 - v0) // 8)
        roi[max(0, v0 - pad_v):v1 + pad_v + 1, max(0, u0 - pad_u):u1 + pad_u + 1] = 1
    elif pixel is not None:
        roi = np.zeros((H, W), np.uint8)
        u, v = int(round(pixel[0])), int(round(pixel[1]))
        roi[max(0, v - 30):v + 31, max(0, u - 30):u + 31] = 1
    elif roi is None:
        roi = np.ones((H, W), np.uint8)
    if color:
        if color not in HSV_RANGES:
            return None, f"unknown colour {color!r}; known {sorted(HSV_RANGES)}"
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        m = np.zeros((H, W), np.uint8)
        for lo, hi in HSV_RANGES[color]:
            m |= cv2.inRange(hsv, np.array(lo), np.array(hi)) // 255
        m &= roi
        how = f"colour {color}"
    elif bbox is not None:
        u0, v0, u1, v1 = [int(round(x)) for x in bbox]
        rect = (max(0, min(u0, u1)), max(0, min(v0, v1)), max(2, abs(u1 - u0)), max(2, abs(v1 - v0)))
        gc = np.zeros((H, W), np.uint8)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(image, gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        except cv2.error:
            return None, "grabcut failed"
        m = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
        how = "grabcut in bbox"
    else:
        return None, "need a bbox or a colour to segment"
    # symmetric kernel: OpenCV's opening with an even (2x2) kernel shifts the mask by one pixel
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None, how + ": empty"
    if pixel is not None and lab[int(round(pixel[1])), int(round(pixel[0]))] > 0:
        k = lab[int(round(pixel[1])), int(round(pixel[0]))]
    else:
        cand = list(range(1, n))
        if stand is not None:
            def on_table(c):
                ys, xs = np.nonzero(lab == c)
                low = ys == ys.max()
                return bool(stand[ys[low], xs[low]].any())
            cand = [c for c in cand if on_table(c)]
            if not cand:
                return None, how + ": nothing standing on the table"
            how += " (on the table)"
        k = max(cand, key=lambda c: stats[c, cv2.CC_STAT_AREA])
    return (lab == k).astype(np.uint8), how


def footprint(mask: np.ndarray, T, intr, W, H, height_m: float = 0.0, grid_mm: float = 2.0):
    """Table points (N,2) whose projections at z=0 and z=height_m both land in the silhouette mask."""
    ys, xs = np.nonzero(mask)
    pix = np.c_[xs, ys].astype(float)
    hull = cv2.convexHull(pix.astype(np.float32)).reshape(-1, 2)
    corners = np.r_[hull, hull + [0.5, 0.5], hull - [0.5, 0.5]]
    P0 = g.pixel_to_plane(T, intr, W, H, corners, 0.0)
    Ph = g.pixel_to_plane(T, intr, W, H, corners, height_m) if height_m > 0 else P0
    P = np.r_[P0, Ph][:, :2]
    P = P[~np.isnan(P).any(1)]
    if not len(P):
        return None
    lo, hi = P.min(0), P.max(0)
    step = grid_mm / 1000
    if np.prod((hi - lo) / step) > 4e6:
        step = float(np.sqrt(np.prod(hi - lo) / 4e6))
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0] + step, step), np.arange(lo[1], hi[1] + step, step))
    G = np.c_[gx.ravel(), gy.ravel()]

    def inside(z):
        uv, d = g.project(T, intr, W, H, np.c_[G, np.full(len(G), z)])
        ui, vi = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        ok = (d > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        r = np.zeros(len(G), bool)
        r[ok] = mask[vi[ok], ui[ok]] > 0
        return r

    keep = inside(0.0) & (inside(height_m) if height_m > 0 else True)
    return G[keep] if keep.any() else None


def top_outline(mask: np.ndarray, T, intr, W, H, height_m: float) -> dict:
    """Top (rim) of an object of known height, for open containers that taper (a bin: wider at the top) where the
    prism footprint is the smaller base. The silhouette's far boundary is the far rim edge and its widest part is the
    rim width; assuming a round/square rim, its centre is half a rim width in front of the far edge.
    -> {"centre_xy", "size_m", "z"}."""
    ys, xs = np.nonzero(mask)
    hull = cv2.convexHull(np.c_[xs, ys].astype(np.float32)).reshape(-1, 2)
    P = g.pixel_to_plane(T, intr, W, H, np.r_[hull + [0.5, 0.5], hull - [0.5, 0.5]], height_m)[:, :2]
    P = P[~np.isnan(P).any(1)]
    d = P.mean(0) - T[:2, 3]
    d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    w = float(np.ptp(P @ n))
    c = d * (float((P @ d).max()) - w / 2) + n * float((P @ n).min() + w / 2)
    return {"centre_xy": c.round(4).tolist(), "size_m": round(w, 4), "z": round(height_m, 3),
            "note": "rim from the silhouette (far edge + widest extent), assuming a round or square top"}


def estimate_height(mask: np.ndarray, T, intr, W, H, grid_mm: float = 2.0, h_max: float = 0.25) -> float:
    """Height of an object with a round or square footprint from one silhouette.

    A tall object's silhouette reaches further back in the image than its footprint, so the z=0 back-projection is
    stretched along the viewing direction. The footprint (points inside the silhouette at z=0 AND z=h) shrinks
    along the viewing direction as h grows; for a circle or a square (any rotation) its extents along and across
    the viewing direction are equal. -> the h where depth extent = lateral extent (0 if already flatter)."""
    def ratio(h):
        fp = footprint(mask, T, intr, W, H, h, grid_mm)
        if fp is None or len(fp) < 4:
            return None
        d = fp.mean(0) - T[:2, 3]
        d /= np.linalg.norm(d)
        n = np.array([-d[1], d[0]])
        return float(np.ptp(fp @ d) / max(np.ptp(fp @ n), 1e-6))
    prev_h, prev_r = 0.0, ratio(0.0)
    if prev_r is None or prev_r <= 1.0:
        return 0.0
    for h in np.arange(0.005, h_max + 1e-9, 0.005):
        r = ratio(h)
        if r is None:
            break
        if r <= 1.0:
            return float(prev_h + (prev_r - 1.0) / (prev_r - r) * (h - prev_h))
        prev_h, prev_r = h, r
    return float(prev_h)


# ---- module-level shortcuts ------------------------------------------------------------------------------
def pixel_to_table(cam, u, v, z=0.0, joints=None):
    return Locator().pixel_to_table(cam, u, v, z, joints)


def world_to_pixel(cam, xyz, joints=None):
    return Locator().world_to_pixel(cam, xyz, joints)


def triangulate(obs_overview, obs_wrist, joints):
    return Locator().triangulate(obs_overview, obs_wrist, joints)


def locate_object(cam, image, **kw):
    return Locator().locate_object(cam, image, **kw)


def scene_map(joints=None, objects=None):
    return Locator().scene_map(joints, objects)


def annotate_result(img: np.ndarray, r: dict, sc: Scene, mask: np.ndarray | None = None, scale: int = 4,
                    pad: int = 30) -> np.ndarray:
    """Evidence crop around a located object, upscaled: silhouette (yellow), footprint at z=0 (green) and at the
    object's height (cyan), centre (red cross), world centre +- sigma as text."""
    H, W = img.shape[:2]
    T, intr = sc.T_world_overview, sc.intr("overview")
    poly = np.array(r["polygon_xy"])
    h = float(r.get("height_m") or 0.0)
    uv0, _ = g.project(T, intr, W, H, np.c_[poly, np.zeros(len(poly))])
    uvh, _ = g.project(T, intr, W, H, np.c_[poly, np.full(len(poly), h)])
    c, _ = g.project(T, intr, W, H, [[*r["centre_xy"], 0.0]])
    allp = np.r_[uv0, uvh]
    u0, v0 = int(max(0, allp[:, 0].min() - pad)), int(max(0, allp[:, 1].min() - pad))
    u1, v1 = int(min(W, allp[:, 0].max() + pad)), int(min(H, allp[:, 1].max() + pad))
    out = cv2.resize(img[v0:v1, u0:u1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    def P(p):
        return np.round((np.atleast_2d(p) - [u0, v0] + 0.5) * scale - 0.5).astype(np.int32)
    if mask is not None:
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for k in cs:
            cv2.polylines(out, [P(k.reshape(-1, 2).astype(float))], True, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.polylines(out, [P(uv0)], True, (0, 255, 0), 1, cv2.LINE_AA)
    if h > 0:
        cv2.polylines(out, [P(uvh)], True, (255, 255, 0), 1, cv2.LINE_AA)
    if r.get("top"):                       # rim: square of the rim width, aligned with the viewing direction, at z=h
        tc, s = np.array(r["top"]["centre_xy"]), r["top"]["size_m"] / 2
        d = tc - T[:2, 3]
        d /= np.linalg.norm(d)
        n = np.array([-d[1], d[0]])
        sq = [tc + s * (a * d + b * n) for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        uvt, _ = g.project(T, intr, W, H, np.c_[sq, np.full(4, h)])
        cv2.polylines(out, [P(uvt)], True, (255, 0, 255), 1, cv2.LINE_AA)
        txt_rim = f"rim centre ({tc[0]:.3f}, {tc[1]:.3f}) m, {2 * s * 1000:.0f} mm wide (magenta)"
    else:
        txt_rim = None
    cv2.drawMarker(out, tuple(int(x) for x in P(c[0])[0]), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
    txt = [f"centre ({r['centre_xy'][0]:.3f}, {r['centre_xy'][1]:.3f}) m  sigma {r['sigma_centre_mm']} mm",
           f"size {r['size_m'][0] * 1000:.0f} x {r['size_m'][1] * 1000:.0f} mm  height {h * 100:.1f} cm",
           "yellow silhouette, green footprint z=0, cyan at height"] + ([txt_rim] if txt_rim else [])
    wpx = max(out.shape[1], 440)
    canvas = np.zeros((out.shape[0] + 8 + 16 * len(txt), wpx, 3), np.uint8)
    canvas[:out.shape[0], :out.shape[1]] = out
    for k, t in enumerate(txt):
        cv2.putText(canvas, t, (4, out.shape[0] + 16 + 16 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    """python -m perception.locate --color blue [--frame F.jpg] [--bbox u0 v0 u1 v1] [--height auto|M] [--out X.jpg]
    Overview camera; live frame from Station unless --frame is given."""
    import argparse
    import asyncio
    import json
    ap = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    ap.add_argument("--frame")
    ap.add_argument("--color")
    ap.add_argument("--bbox", type=float, nargs=4)
    ap.add_argument("--pixel", type=float, nargs=2)
    ap.add_argument("--height", default="0", help="metres, or 'auto' (round/square footprint)")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.frame:
        img = cv2.imread(a.frame)
    else:
        from . import station_io as sio

        async def grab():
            return sio.decode_jpeg((await sio.grab_jpegs(await sio.connect(), "overview", 1))[0])
        img = asyncio.run(grab())
    loc = Locator(reload())
    h = None if a.height == "auto" else float(a.height)
    r = loc.locate_object("overview", img, bbox=a.bbox, pixel=a.pixel, color=a.color, height_m=h)
    print(json.dumps({k: v for k, v in r.items() if k not in ("polygon_xy", "box_xy")}, indent=1))
    if a.out and r.get("ok"):
        T, intr, W, H = loc.camera("overview")
        roi = table_mask(T, intr, W, H, loc.sc.table_extent) if a.bbox is None and a.pixel is None else None
        mask, _ = segment(img, bbox=a.bbox, pixel=a.pixel, color=a.color, roi=roi)
        cv2.imwrite(a.out, annotate_result(img, r, loc.sc, mask), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print("wrote", a.out)


if __name__ == "__main__":
    main()
