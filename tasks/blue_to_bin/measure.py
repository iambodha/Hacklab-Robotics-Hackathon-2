"""Measure the demo objects in the real scene from the overview camera (read only, nothing is commanded).

    uv run --no-project --with numpy --with scipy --with opencv-contrib-python \
        python -m tasks.blue_to_bin.measure --tail 10          # newest 10 stored frames (works while video is down)
    ... python -m tasks.blue_to_bin.measure --live 10          # next 10 live frames (one subscription)
    ... python -m tasks.blue_to_bin.measure --frames DIR       # overview*.jpg in DIR (or one JPEG file)

Blue cap: colour silhouette (perception.locate.segment, "blue") -> an upright cylinder (centre x, y, radius, height)
whose projection has the same extreme pixels (left, right, top, bottom) as the silhouette.
Bin: a transparent tub has no reliable colour, so its rim and base are given as pixels (recognised by looking at
the image, `bin_pixels` in measured.json, re-pick them when the bin moves). A tapered upright cylinder (centre,
rim radius, rim height, base radius) is fitted to them. Everything is world metres (scene-contract frame) with
1-sigma from the pixel sigma (Jacobian) and the calibration (perception's held-out marker error).

Writes tasks/blue_to_bin/measured.json and measured_overview.jpg (model outlines over the frame).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from perception import geometry as g
from perception.locate import Locator, segment

HERE = Path(__file__).resolve().parent
OUT = HERE / "measured.json"
PIXEL_SIGMA = 0.7          # px, a silhouette extreme / a picked edge pixel

# Pixels picked on the 00:58 frame (Station's last stored overview entry before its restart). (u, v) in 299x224.
DEFAULT_BIN_PIXELS = {"rim_left": [179.0, 97.0], "rim_right": [214.0, 97.0], "rim_far": [195.0, 84.5],
                      "rim_near": [199.0, 103.5], "base_left": [182.0, 115.0], "base_right": [205.5, 115.0],
                      "base_near": [196.0, 119.5]}


def circle(c, r, z, n=96):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a), np.full(n, z)]


class Cam:
    def __init__(self, loc: Locator):
        self.T, self.intr, self.W, self.H = loc.camera("overview")

    def px(self, P):
        return g.project(self.T, self.intr, self.W, self.H, P)[0]

    def extremes(self, P):
        """(u_min, u_max, v_min, v_max) of projected world points."""
        uv = self.px(P)
        return np.array([uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max()])

    def cylinder(self, c, r, h):
        return np.r_[circle(c, r, 0.0), circle(c, r, h)]


def _fit(fun, x0, lo, hi):
    from scipy.optimize import least_squares
    r = least_squares(fun, x0, bounds=(lo, hi), x_scale="jac", diff_step=1e-4)
    J = r.jac
    cov = np.linalg.pinv(J.T @ J) * PIXEL_SIGMA ** 2
    return r.x, np.sqrt(np.clip(np.diag(cov), 0, None)), float(np.sqrt(np.mean(r.fun ** 2)))


def table_roi(loc: Locator, cam: Cam, margin=0.0) -> np.ndarray:
    """Pixels whose ray meets the table plane inside scene_calib's table extent (keeps the background out)."""
    vv, uu = np.mgrid[0:cam.H, 0:cam.W]
    P = g.pixel_to_plane(cam.T, cam.intr, cam.W, cam.H, np.c_[uu.ravel(), vv.ravel()].astype(float), 0.0)
    (x0, y0), (x1, y1) = loc.sc.table_extent
    ok = ~np.isnan(P).any(1) & (P[:, 0] >= x0 - margin) & (P[:, 0] <= x1 + margin) & (P[:, 1] >= y0 - margin) & (P[:, 1] <= y1 + margin)
    return ok.reshape(cam.H, cam.W)


def table_blob(img, roi: np.ndarray, color="blue", min_px=8):
    """(mask of the largest `color` blob whose pixels see the table, description). The ROI is applied BEFORE
    picking the largest component (perception.locate.segment alone would pick a blue patch in the background)."""
    vs, us = np.nonzero(roi)
    if not len(us):
        return None, "table not in view"
    big = cv2.dilate(roi.astype(np.uint8), np.ones((15, 15), np.uint8))   # objects stand up out of the table area
    masked = img.copy()
    masked[big == 0] = 0
    mask, how = segment(masked, color=color)
    if mask is None or mask.sum() < min_px:
        return None, f"{how} inside the table extent"
    return mask, f"{how} inside the table extent"


def measure_cap(img, loc: Locator, cam: Cam, color="blue") -> dict:
    mask, how = table_blob(img, table_roi(loc, cam), color)
    if mask is None:
        return {"ok": False, "reason": f"no {color} silhouette ({how})"}
    ys, xs = np.nonzero(mask)
    ext = np.array([xs.min() - 0.5, xs.max() + 0.5, ys.min() - 0.5, ys.max() + 0.5])
    centre_px = [float(xs.mean()), float(ys.mean())]
    p0 = loc.pixel_to_table("overview", *centre_px, 0.0)["xyz"]

    def res(x):
        return cam.extremes(cam.cylinder(x[:2], x[2], x[3])) - ext
    x, sd, rms = _fit(res, [p0[0], p0[1], 0.015, 0.01], [p0[0] - 0.1, p0[1] - 0.1, 0.005, 0.002], [p0[0] + 0.1, p0[1] + 0.1, 0.06, 0.08])
    model = np.zeros_like(mask)
    for P in (circle(x[:2], x[2], 0.0), circle(x[:2], x[2], x[3])):
        cv2.fillPoly(model, [np.int32(np.round(cam.px(P)))], 1)
    hull = cv2.convexHull(np.int32(np.round(cam.px(cam.cylinder(x[:2], x[2], x[3])))))
    cv2.fillPoly(model, [hull], 1)
    iou = float((model & mask).sum() / max(1, (model | mask).sum()))
    s_cal = loc.calib_sigma_mm("overview")
    return {"ok": True, "method": f"{how} silhouette -> upright cylinder fit to its extreme pixels",
            "centre_xy": np.round(x[:2], 4).tolist(), "diameter_m": round(2 * x[2], 4), "height_m": round(x[3], 4),
            "sigma_mm": {"centre_pixel": round(1000 * float(np.hypot(*sd[:2])), 1), "calibration": round(s_cal, 1),
                         "centre": round(float(np.hypot(1000 * np.hypot(*sd[:2]), s_cal)), 1),
                         "diameter": round(2000 * sd[2], 1), "height": round(1000 * sd[3], 1)},
            "silhouette_px": int(mask.sum()), "silhouette_bbox_px": ext.tolist(), "centre_px": centre_px,
            "fit_rms_px": round(rms, 2), "model_iou": round(iou, 3)}


def measure_bin(loc: Locator, cam: Cam, pix: dict) -> dict:
    p = {k: np.asarray(v, float) for k, v in pix.items()}
    rim_mid = (p["rim_left"] + p["rim_right"]) / 2
    c0 = loc.pixel_to_table("overview", *((p["base_left"] + p["base_right"]) / 2), 0.0)["xyz"]

    def res(x):
        c, rr, H, rb = x[:2], x[2], x[3], x[4]
        top, bot = cam.extremes(circle(c, rr, H)), cam.extremes(circle(c, rb, 0.0))
        return np.r_[top[0] - p["rim_left"][0], top[1] - p["rim_right"][0], top[2] - p["rim_far"][1], top[3] - p["rim_near"][1],
                     (top[0] + top[1]) / 2 - rim_mid[0], bot[0] - p["base_left"][0], bot[1] - p["base_right"][0],
                     bot[3] - p["base_near"][1]]
    x, sd, rms = _fit(res, [c0[0], c0[1], 0.05, 0.08, 0.04], [c0[0] - 0.15, c0[1] - 0.15, 0.02, 0.02, 0.015],
                      [c0[0] + 0.15, c0[1] + 0.15, 0.15, 0.3, 0.15])
    s_cal = loc.calib_sigma_mm("overview")
    return {"ok": True, "method": "tapered upright cylinder fitted to picked rim/base pixels (transparent tub, no colour)",
            "centre_xy": np.round(x[:2], 4).tolist(), "rim_diameter_m": round(2 * x[2], 4), "rim_height_m": round(x[3], 4),
            "base_diameter_m": round(2 * x[4], 4),
            "sigma_mm": {"centre_pixel": round(1000 * float(np.hypot(*sd[:2])), 1), "calibration": round(s_cal, 1),
                         "centre": round(float(np.hypot(1000 * np.hypot(*sd[:2]), s_cal)), 1),
                         "rim_diameter": round(2000 * sd[2], 1), "rim_height": round(1000 * sd[3], 1),
                         "base_diameter": round(2000 * sd[4], 1)},
            "fit_rms_px": round(rms, 2), "pixels": pix}


def annotate(img, loc: Locator, cam: Cam, cap: dict, tub: dict, label: str) -> np.ndarray:
    from perception.annotate import annotate_overview
    extra = []
    if cap.get("ok"):
        c, r, h = cap["centre_xy"], cap["diameter_m"] / 2, cap["height_m"]
        extra += [{"polygon": circle(c, r, 0.0, 48), "color": (255, 255, 0)}, {"polygon": circle(c, r, h, 48), "color": (255, 255, 0)},
                  {"point": [*c, 0.0], "label": f"cap {c[0]:.3f},{c[1]:.3f} d{1000 * 2 * r:.0f} h{1000 * h:.0f}mm",
                   "color": (255, 255, 0)}]
    if tub.get("ok"):
        c = tub["centre_xy"]
        extra += [{"polygon": circle(c, tub["rim_diameter_m"] / 2, tub["rim_height_m"], 64), "color": (0, 255, 255)},
                  {"polygon": circle(c, tub["base_diameter_m"] / 2, 0.0, 64), "color": (0, 200, 255)},
                  {"point": [*c, 0.0], "label": f"bin {c[0]:.3f},{c[1]:.3f} rim d{1000 * tub['rim_diameter_m']:.0f} "
                                                f"h{1000 * tub['rim_height_m']:.0f}mm", "color": (0, 255, 255)}]
    out = annotate_overview(loc.sc, img, extra=extra)
    s = out.shape[1] // img.shape[1]
    for k, (u, v) in (tub.get("pixels") or {}).items():
        cv2.circle(out, (int((u + 0.5) * s), int((v + 0.5) * s)), 3, (255, 0, 255), -1)
    for k, (c, t) in enumerate([((0, 0, 0), 3), ((255, 255, 255), 1)]):
        cv2.putText(out, label[:110], (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, t, cv2.LINE_AA)
    return out


# ---- frame sources (read only) --------------------------------------------------------------------------
async def _station_frames(n: int, tail: bool) -> list[bytes]:
    from perception import station_io as s
    cli = await s.connect()
    if not tail:
        return await s.grab_jpegs(cli, "overview", n)
    _, usbvideo, fr = s._import_station()
    out = []
    for k in range(1, n + 1):                   # one read per stored entry; reads end on their own (no follow)
        qr = cli.read_from_tail(s.QUEUES["overview"], offset=bytes([k]), limit=1, step=1, buf_size=1)
        e = await asyncio.wait_for(qr.data.get(), 5)
        if e is None:
            continue
        pack = usbvideo.RxEnvelopeReader(memoryview(bytes(e.Data))).get_frames()
        data = (pack.get_frames_data() or []) if pack else []
        if data and pack.get_format().get_kind() == fr.FrameFormatKind.FF_JPEG:
            out.append(bytes(data[-1]))
    return out


def load_frames(args) -> tuple[list[np.ndarray], str]:
    if args.frames:
        p = Path(args.frames)
        files = [p] if p.is_file() else sorted(p.glob("overview*.jpg")) or sorted(p.glob("*.jpg"))
        return [cv2.imread(str(f)) for f in files], f"files {args.frames}"
    jpgs = asyncio.run(_station_frames(args.tail or args.live, bool(args.tail)))
    src = f"Station {'newest stored' if args.tail else 'live'} overview entries x{len(jpgs)}"
    return [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for b in jpgs], src


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tail", type=int, help="newest N stored overview frames")
    src.add_argument("--live", type=int, help="next N live overview frames")
    src.add_argument("--frames", help="directory of overview JPEGs, or one JPEG")
    ap.add_argument("--bin-pixels", help="JSON with rim_left/right/far/near and base_left/right/near pixels")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    frames, source = load_frames(args)
    if not frames:
        raise SystemExit("no overview frames")
    img = np.median(np.stack(frames), axis=0).astype(np.uint8)
    loc = Locator()
    cam = Cam(loc)
    pix = json.loads(Path(args.bin_pixels).read_text()) if args.bin_pixels else DEFAULT_BIN_PIXELS
    cap, tub = measure_cap(img, loc, cam), measure_bin(loc, cam, pix)
    out = Path(args.out)
    cv2.imwrite(str(out.with_name("measured_overview_raw.jpg")), img)
    cv2.imwrite(str(out.with_name("measured_overview.jpg")), annotate(img, loc, cam, cap, tub, source))
    rec = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "frame": "world (scene-contract), metres", "source": source,
           "scene_calib": loc.sc.d.get("created"), "blue_cap": cap, "bin": tub,
           "notes": ["read only: no command was sent", "calibration sigma = perception's held-out marker error (median)"]}
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
