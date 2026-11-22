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
        return {"ok": True, "xyz": P.round(4).tolist(), "camera": cam, "pixel": [u, v], "z_plane": z,
                "mm_per_px": sv.round(2).tolist(), "sigma_mm": round(math.hypot(s_px, s_cal), 1),
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
        flat; `centre_shift_mm_per_cm_height` says how much the centre would move per cm of real height."""
        T, intr, W, H = self.camera(cam, joints)
        mask, how = segment(image, bbox=bbox, pixel=pixel, color=color)
        if mask is None or mask.sum() < 3:
            if pixel is not None:
                r = self.pixel_to_table(cam, pixel[0], pixel[1], height_m, joints)
                r["method"] = f"point only ({how}); no silhouette"
                return r
            return {"ok": False, "reason": f"no object found ({how})"}
        fp = footprint(mask, T, intr, W, H, height_m, grid_mm)
        if fp is None:
            return {"ok": False, "reason": "silhouette does not intersect the table plane"}
        res = self._describe(fp, cam, T, intr, W, H, grid_mm)
        alt = footprint(mask, T, intr, W, H, height_m + 0.01, grid_mm)
        if alt is not None:
            res["centre_shift_mm_per_cm_height"] = round(float(np.linalg.norm(alt.mean(0) - fp.mean(0))) * 1000, 1)
        res.update({"camera": cam, "method": how, "height_m": height_m, "silhouette_px": int(mask.sum())})
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


def segment(image: np.ndarray, bbox=None, pixel=None, color: str | None = None):
    """-> (mask uint8 0/1 or None, description)."""
    H, W = image.shape[:2]
    roi = np.zeros((H, W), np.uint8)
    if bbox is not None:
        u0, v0, u1, v1 = [int(round(x)) for x in bbox]
        u0, u1 = sorted((max(0, u0), min(W - 1, u1)))
        v0, v1 = sorted((max(0, v0), min(H - 1, v1)))
        pad_u, pad_v = max(2, (u1 - u0) // 8), max(2, (v1 - v0) // 8)
        roi[max(0, v0 - pad_v):v1 + pad_v + 1, max(0, u0 - pad_u):u1 + pad_u + 1] = 1
    elif pixel is not None:
        u, v = int(round(pixel[0])), int(round(pixel[1]))
        roi[max(0, v - 30):v + 31, max(0, u - 30):u + 31] = 1
    else:
        roi[:] = 1
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
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
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
