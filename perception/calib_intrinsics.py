"""Camera intrinsics from a printed ChArUco board, through Station frames.

  python -m perception.calib_intrinsics board                 # writes calibration/board/charuco_A4.{png,pdf}
  python -m perception.calib_intrinsics live overview         # hold the board in front of the camera
  python -m perception.calib_intrinsics live wrist
  python -m perception.calib_intrinsics dir wrist frames/intrinsics-wrist-...   # recompute from saved frames

Board: 8 x 6 squares of 30.0 mm, ArUco markers 22.0 mm, DICT_4X4_100 ids 50..73 (no clash with the
scene markers, which are DICT_4X4_50 ids < 50). Print the PDF at 100 % ("actual size", no "fit to page")
and MEASURE one square with a ruler; pass --square-mm if it isn't 30.0.

Result: calibration/intrinsics_<cam>.json with K normalised by the image size (fx/W, fy/H, cx/W, cy/H) and
OpenCV distortion [k1,k2,p1,p2,k3]. For the wrist camera the same intrinsics are also written into
camera_pose/wrist_camera.json, which the existing wrist tools read. Rerun calib_scene afterwards.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .scene import CALIB_DIR, intrinsics_file

SQUARES = (8, 6)
SQUARE_MM, MARKER_MM = 30.0, 22.0
DICT = "DICT_4X4_100"
FIRST_ID = 50
BOARD_DIR = CALIB_DIR / "board"
ROOT = Path(__file__).resolve().parent.parent


def board(square_mm: float = SQUARE_MM) -> cv2.aruco.CharucoBoard:
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT))
    n = SQUARES[0] * SQUARES[1] // 2
    ids = np.arange(FIRST_ID, FIRST_ID + n, dtype=np.int32)
    return cv2.aruco.CharucoBoard(SQUARES, square_mm / 1000, square_mm * MARKER_MM / SQUARE_MM / 1000, d, ids)


def make_board(out_dir: Path = BOARD_DIR, dpi: int = 300) -> list[Path]:
    """A4 landscape page with the board centred at exact size, plus a label with the dimensions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mm = dpi / 25.4
    page = (int(round(297 * mm)), int(round(210 * mm)))
    bw, bh = int(round(SQUARES[0] * SQUARE_MM * mm)), int(round(SQUARES[1] * SQUARE_MM * mm))
    img = board().generateImage((bw, bh), marginSize=0, borderBits=1)
    canvas = np.full((page[1], page[0]), 255, np.uint8)
    x0, y0 = (page[0] - bw) // 2, (page[1] - bh) // 2 - int(4 * mm)
    canvas[y0:y0 + bh, x0:x0 + bw] = img
    label = (f"ChArUco {SQUARES[0]}x{SQUARES[1]}  square {SQUARE_MM:.1f} mm  marker {MARKER_MM:.1f} mm  {DICT} ids "
             f"{FIRST_ID}-{FIRST_ID + SQUARES[0] * SQUARES[1] // 2 - 1}  |  print at 100% (A4 landscape), "
             f"check: 4 squares = {4 * SQUARE_MM:.0f} mm")
    cv2.putText(canvas, label, (x0, y0 + bh + int(9 * mm)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 2, cv2.LINE_AA)
    # a 100 mm ruler bar to check the print scale
    ry = y0 + bh + int(13 * mm)
    cv2.rectangle(canvas, (x0, ry), (x0 + int(round(100 * mm)), ry + int(2 * mm)), 0, -1)
    cv2.putText(canvas, "100 mm", (x0 + int(round(100 * mm)) + 20, ry + int(2 * mm)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    png = out_dir / "charuco_A4.png"
    cv2.imwrite(str(png), canvas)
    paths = [png]
    try:
        from PIL import Image
        pdf = out_dir / "charuco_A4.pdf"
        Image.fromarray(canvas).convert("1").save(pdf, "PDF", resolution=dpi)
        paths.append(pdf)
    except ImportError:
        print("pillow not installed: PNG only (print it at 300 dpi = A4 landscape)")
    return paths


def detect_board(img: np.ndarray, b: cv2.aruco.CharucoBoard):
    """-> (object points (N,3), image points (N,2)) of the detected ChArUco corners, or None."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    params = cv2.aruco.DetectorParameters()
    params.minMarkerPerimeterRate = 0.02
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE      # ChArUco refines the chessboard corners itself
    cd = cv2.aruco.CharucoDetector(b, cv2.aruco.CharucoParameters(), params)
    cc, cid, _, _ = cd.detectBoard(gray)
    if cid is None or len(cid) < 6:
        return None
    obj, imgp = b.matchImagePoints(cc, cid)
    return obj.reshape(-1, 3), imgp.reshape(-1, 2)


def calibrate(views: list[tuple[np.ndarray, np.ndarray]], W: int, H: int, rational: bool = False):
    """views: [(obj (N,3), img (N,2))]. Rejects views with error > 3x the median and refits once."""
    flags = cv2.CALIB_RATIONAL_MODEL if rational else 0

    def run(vs):
        obj = [v[0].astype(np.float32) for v in vs]
        img = [v[1].astype(np.float32).reshape(-1, 1, 2) for v in vs]
        rms, K, dist, rv, tv, sdi, _, per = cv2.calibrateCameraExtended(obj, img, (W, H), None, None, flags=flags)
        return rms, K, dist.ravel(), per.ravel(), sdi.ravel()

    rms, K, dist, per, sdi = run(views)
    keep = [v for v, e in zip(views, per) if e <= 3 * np.median(per)]
    if len(keep) < len(views) and len(keep) >= 8:
        rms, K, dist, per, sdi = run(keep)
    tilts = np.array([tilt(v) for v in views])
    return {"rms_px": float(rms), "K": K, "dist": dist, "per_view_px": per,
            "n_views": len(keep) if len(keep) >= 8 else len(views),
            "sigma_f_rel": float(max(sdi[0] / K[0, 0], sdi[1] / K[1, 1])), "sigma_c_px": float(max(sdi[2], sdi[3])),
            "max_tilt_deg": np.degrees(np.abs(tilts).max(0)).round(1).tolist() if len(tilts) else [0, 0]}


def tilt(view, W: int = 299, H: int = 224) -> np.ndarray:
    """Board tilt (radians) about the camera x and y axes, 0 = facing the camera. From a planar PnP with
    a nominal focal length (tilt angles hardly depend on it)."""
    obj, img = view
    f = 0.5 * W
    K = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1]])
    ok, rv, _ = cv2.solvePnP(obj.astype(np.float64), img.astype(np.float64), K, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return np.zeros(2)
    n = cv2.Rodrigues(rv)[0][:, 2]                  # board normal in the camera frame
    n = n if n[2] < 0 else -n                       # pointing back at the camera
    return np.array([np.arctan2(n[1], -n[2]), np.arctan2(n[0], -n[2])])


def save(cam: str, res: dict, W: int, H: int, square_mm: float) -> Path:
    K, d = res["K"], res["dist"]
    out = {"camera": cam, "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "image_size": [W, H], "K_normalized": {"fx": K[0, 0] / W, "fy": K[1, 1] / H, "cx": K[0, 2] / W, "cy": K[1, 2] / H},
           "dist": [float(v) for v in d[:5]], "dist_full": [float(v) for v in d], "rms_px": round(res["rms_px"], 4),
           "n_views": res["n_views"], "sigma_f_rel": round(res["sigma_f_rel"], 4), "max_tilt_deg": res["max_tilt_deg"],
           "board": {"squares": SQUARES, "square_mm": square_mm, "marker_mm": square_mm * MARKER_MM / SQUARE_MM,
                                                 "dictionary": DICT, "first_id": FIRST_ID},
           "hfov_deg": float(np.degrees(2 * np.arctan(W / 2 / K[0, 0]))), "vfov_deg": float(np.degrees(2 * np.arctan(H / 2 / K[1, 1])))}
    p = intrinsics_file(cam)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    if cam == "wrist":                       # keep camera_pose/wrist_camera.json (read by arm_mcp's wrist tools) in sync
        wj = ROOT / "camera_pose/wrist_camera.json"
        w = json.loads(wj.read_text())
        w["intrinsics"] = {"fx_n": out["K_normalized"]["fx"], "fy_n": out["K_normalized"]["fy"], "cx_n": out["K_normalized"]["cx"],
                           "cy_n": out["K_normalized"]["cy"], "dist": out["dist"], "rms_px": out["rms_px"],
                           "source": f"perception/calib_intrinsics.py {out['created']}"}
        w["intrinsics_note"] = "ChArUco calibration; normalised by image width/height"
        wj.write_text(json.dumps(w, indent=2))
    return p


def _novel(view, kept: list, W: int) -> bool:
    """A view is new if its corner centroid moved, its apparent size changed, or its tilt changed.
    Tilt matters most: fronto-parallel views leave the focal length unconstrained."""
    obs = view[1]
    c, s, t = obs.mean(0), np.sqrt(np.prod(obs.std(0))), tilt(view)
    for k in kept:
        o = k[1]
        if np.linalg.norm(o.mean(0) - c) < 0.08 * W and abs(np.sqrt(np.prod(o.std(0))) / s - 1) < 0.15 \
                and np.abs(tilt(k) - t).max() < np.radians(10):
            return False
    return True


async def capture_live(cam: str, n_views: int, b, square_mm: float, min_corners: int = 12, timeout_s: float = 300):
    from . import station_io as sio
    cli = await sio.connect()
    save_dir = ROOT / "frames" / f"intrinsics-{cam}-{dt.datetime.now():%Y%m%d-%H%M%S}"
    save_dir.mkdir(parents=True, exist_ok=True)
    views, kept, W, H, t0, last = [], [], None, None, time.time(), 0.0
    print(f"Hold the board in front of the {cam} camera, 15-50 cm away. Move it slowly: centre, all four corners and "
          f"edges of the image, tilted ~30-45 deg left/right/up/down, near and far. Hold still when a view is taken.")
    while len(views) < n_views and time.time() - t0 < timeout_s:
        jpg = (await sio.grab_jpegs(cli, cam, 1))[0]
        img = sio.decode_jpeg(jpg)
        H, W = img.shape[:2]
        det = detect_board(img, b)
        if det is None or len(det[0]) < min_corners or time.time() - last < 0.7 or not _novel(det, kept, W):
            continue
        views.append(det)
        kept.append(det)
        last = time.time()
        (save_dir / f"{cam}_{len(views):02d}.jpg").write_bytes(jpg)
        tl = np.degrees(tilt(det))
        print(f"  view {len(views)}/{n_views}: {len(det[0])} corners at ({det[1].mean(0)[0]:.0f}, {det[1].mean(0)[1]:.0f}), "
              f"tilt ~({tl[0]:+.0f}, {tl[1]:+.0f}) deg" + ("   <- tilt the board more" if np.abs(tl).max() < 15 else ""))
    print(f"saved frames in {save_dir}")
    return views, W, H


def views_from_dir(d: Path, b):
    views, W, H = [], None, None
    for p in sorted(d.glob("*.jpg")):
        img = cv2.imread(str(p))
        H, W = img.shape[:2]
        det = detect_board(img, b)
        if det is not None and len(det[0]) >= 12:
            views.append(det)
    return views, W, H


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("board")
    for name in ("live", "dir"):
        p = sub.add_parser(name)
        p.add_argument("camera", choices=["overview", "wrist"])
        if name == "dir":
            p.add_argument("path", type=Path)
        p.add_argument("--views", type=int, default=25)
        p.add_argument("--square-mm", type=float, default=SQUARE_MM, help="measured printed square size")
        p.add_argument("--rational", action="store_true", help="8-coefficient distortion model (very wide lenses)")
        p.add_argument("--force", action="store_true", help="save even if the focal length is poorly constrained")
    a = ap.parse_args()
    if a.cmd == "board":
        for p in make_board():
            print("wrote", p)
        return
    b = board(a.square_mm)
    views, W, H = asyncio.run(capture_live(a.camera, a.views, b, a.square_mm)) if a.cmd == "live" else views_from_dir(a.path, b)
    if len(views) < 8:
        raise SystemExit(f"only {len(views)} usable views; need >= 8 (aim for 20+)")
    res = calibrate(views, W, H, a.rational)
    K = res["K"]
    print(f"{a.camera}: {res['n_views']} views, RMS {res['rms_px']:.3f} px (good: < 0.5 px at 299x224)")
    print(f"  fx {K[0, 0]:.2f} fy {K[1, 1]:.2f} cx {K[0, 2]:.2f} cy {K[1, 2]:.2f}  dist {np.round(res['dist'][:5], 4).tolist()}")
    print(f"  focal 1-sigma {100 * res['sigma_f_rel']:.1f} % (need < 3 %), principal point 1-sigma {res['sigma_c_px']:.1f} px, "
          f"max board tilt ~{res['max_tilt_deg']} deg (want >= 30)")
    if res["sigma_f_rel"] > 0.03 and not a.force:
        raise SystemExit("NOT saved: the focal length is poorly constrained. Recapture with the board clearly TILTED "
                         "(30-45 deg towards/away from the camera, left/right/up/down) and nearer the image corners; "
                         "or pass --force.")
    print(f"  wrote {save(a.camera, res, W, H, a.square_mm)}; now rerun: python -m perception.calib_scene --live")


if __name__ == "__main__":
    main()
