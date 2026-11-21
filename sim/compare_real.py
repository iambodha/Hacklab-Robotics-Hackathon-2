"""Sim-vs-real check: render the twin's cameras at the real arm's joint state, next to the real Station frames.

  python3 sim/compare_real.py [--layout home] [--frames DIR] [--out sim/real_compare/<timestamp>]

Reads one overview + wrist frame and the follower's raw ticks through Station (read-only), sets the twin's joints
with sim/real_joints.py, renders overview_cam and wrist_cam at the Station resolution and writes:
  compare.jpg    rows = cameras; columns = real | twin | overlay (real image, real edges green, twin edges magenta,
                 real marker corners green circles, twin model corners red crosses)
  metrics.json   per camera: marker corner px error (twin model corners projected vs corners detected in the real
                 frame; also corners detected in the twin render), table colour error (mean RGB difference of the
                 table area, and mean per-pixel CIE76 delta E)
  real_*.jpg, twin_*.png
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_scene  # noqa: E402  (sets MUJOCO_GL)
import mujoco  # noqa: E402
import real_joints  # noqa: E402

HERE = Path(__file__).resolve().parent
CAMS = {"overview": "overview_cam", "wrist": "wrist_cam"}


def load_model(layout_file: Path):
    out = build_scene.generated_dir(layout_file)
    xml = out / "workspace.xml"
    sources = [layout_file, HERE / "build_scene.py"]
    layout = json.loads(layout_file.read_text())
    tex = layout["table"].get("texture")
    if tex:
        sources.append(HERE / tex)
    if not xml.exists() or xml.stat().st_mtime < max(p.stat().st_mtime for p in sources):
        build_scene.build(layout_file, out, report=False)
    return mujoco.MjModel.from_xml_path(str(xml)), layout


def load_model_from(layout: dict, layout_file: Path):
    """Model of an (edited) layout dict without writing the layout or its generated scene."""
    import xml.etree.ElementTree as ET
    out = build_scene.generated_dir(layout_file) / "scratch"
    out.mkdir(parents=True, exist_ok=True)
    root = build_scene.build_xml(layout, out)
    build_scene.add_rest_keyframe(root, layout)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), layout


def camera_K(model, cam_id, W, H):
    f = H / 2 / math.tan(math.radians(model.cam_fovy[cam_id]) / 2)
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])


def project(model, data, cam_id, pts, W, H):
    """World points -> pixels of a MuJoCo camera (pinhole, principal point at the centre)."""
    R = data.cam_xmat[cam_id].reshape(3, 3) @ build_scene.CV_TO_MJ
    pc = (np.asarray(pts) - data.cam_xpos[cam_id]) @ R
    return (camera_K(model, cam_id, W, H) @ (pc / pc[:, 2:3]).T).T[:, :2], pc[:, 2]


def detect(img, dictionary):
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)),
                                  cv2.aruco.DetectorParameters())
    c, ids, _ = det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return {} if ids is None else {int(i): cc.reshape(4, 2).astype(float) for i, cc in zip(ids.ravel(), c)}


def undistort_px(px, cam_layout, W, H):
    """Real (distorted) pixels -> ideal pinhole pixels with the layout's intrinsics, when it has them."""
    k, d = cam_layout.get("K_normalized"), cam_layout.get("dist")
    if not k or not d or not any(d):
        return px
    K = np.array([[k["fx"] * W, 0, k["cx"] * W], [0, k["fy"] * H, k["cy"] * H], [0, 0, 1.0]])
    return cv2.undistortPoints(px.reshape(-1, 1, 2), K, np.array(d, float), P=K).reshape(-1, 2)


def table_mask(model, renderer, data, cam, layout):
    """Pixels whose nearest geom is the table top (marker papers, arm and everything else excluded), eroded."""
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=cam)
    seg = renderer.render()[..., 0]
    renderer.disable_segmentation_rendering()
    ids = [model.geom(n).id for n in ("table_top", "table_skin") if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) >= 0]
    return cv2.erode(np.isin(seg, ids).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def colour_error(real, twin, mask):
    if mask.sum() < 50:
        return None
    lab_r = cv2.cvtColor(real, cv2.COLOR_BGR2LAB).astype(float)[mask]
    lab_t = cv2.cvtColor(twin, cv2.COLOR_BGR2LAB).astype(float)[mask]
    to_cie = np.array([100 / 255, 1, 1]); off = np.array([0, 128, 128])
    de = np.linalg.norm((lab_r - off) * to_cie - (lab_t - off) * to_cie, axis=1)
    mr, mt = real[mask].mean(0)[::-1], twin[mask].mean(0)[::-1]
    return {"mean_rgb_real": np.round(mr, 1).tolist(), "mean_rgb_twin": np.round(mt, 1).tolist(),
            "mean_rgb_error": round(float(np.linalg.norm(mr - mt)), 2), "mean_delta_e": round(float(de.mean()), 2),
            "pixels": int(mask.sum())}


def overlay(real, twin, real_c, twin_c, scale=2):
    big = cv2.resize(real, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    er = cv2.resize(cv2.Canny(cv2.GaussianBlur(real, (3, 3), 0), 60, 140), None, fx=scale, fy=scale) > 0
    et = cv2.resize(cv2.Canny(cv2.GaussianBlur(twin, (3, 3), 0), 60, 140), None, fx=scale, fy=scale) > 0
    out = (big * 0.6).astype(np.uint8)
    out[er] = (60, 230, 60); out[et] = (230, 60, 230); out[er & et] = (255, 255, 255)
    for pts in real_c.values():
        for p in pts:
            cv2.circle(out, tuple(int(v) for v in p * scale), 5, (0, 255, 0), 1, cv2.LINE_AA)
    for pts in twin_c.values():
        for p in pts:
            x, y = (p * scale).astype(int)
            cv2.line(out, (x - 4, y), (x + 4, y), (0, 0, 255), 1); cv2.line(out, (x, y - 4), (x, y + 4), (0, 0, 255), 1)
    return out


async def grab_live():
    import station_io
    cli = await station_io.connect()
    jf = station_io.JointFollower(cli)
    ov, wr = await asyncio.gather(station_io.grab_jpegs(cli, "overview"), station_io.grab_jpegs(cli, "wrist"))
    raw, meta = await jf.read()
    dec = lambda b: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    return {"overview": dec(ov[0]), "wrist": dec(wr[0])}, raw, meta["ranges"]


def compare(layout_file, frames, raw, ranges, out: Path) -> dict:
    model, layout = load_model(layout_file)
    data = mujoco.MjData(model)
    W, H = layout["image_size"]
    q = real_joints.to_q(raw, ranges, *model.actuator_ctrlrange.T, real_joints.load_mapping(layout))
    real_joints.set_arm_q(model, data, q)
    mk = layout.get("markers") or {}
    dictionary = mk.get("dictionary", "DICT_4X4_50")
    metrics = {"layout": str(layout_file), "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "raw_ticks": raw,
               "q_rad": np.round(q, 4).tolist(), "joint_mapping": real_joints.load_mapping(layout),
               "sources": {"table": layout["table"].get("src"), "markers": sorted({i.get("src") for i in mk.get("items", [])} - {None}),
                           "overview_cam": next((c.get("src") for c in layout["cameras"] if c["name"] == "overview_cam"), None),
                           "wrist_camera": layout["wrist_camera"].get("src")}}
    rows = []
    opt = mujoco.MjvOption(); opt.geomgroup[1] = 0
    with mujoco.Renderer(model, H, W) as r:
        for name, cam in CAMS.items():
            cid = model.camera(cam).id
            r.update_scene(data, camera=cam, scene_option=opt)
            twin = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
            real = frames[name]
            if real.shape[:2] != (H, W):
                real = cv2.resize(real, (W, H))
            cam_layout = layout["wrist_camera"] if name == "wrist" else next(c for c in layout["cameras"] if c["name"] == cam)
            real_c = detect(real, dictionary)
            twin_det = detect(twin, dictionary)
            model_c = {}
            for it in mk.get("items", []):
                sites = [f"aruco_{it['id']}_c{k}" for k in range(4)]
                pts = np.array([data.site_xpos[model.site(s).id] for s in sites])
                px, z = project(model, data, cid, pts, W, H)
                if (z > 0).all() and (px > -20).all() and (px < [W + 20, H + 20]).all():
                    model_c[it["id"]] = px
            common = sorted(set(real_c) & set(model_c))
            real_u = {i: undistort_px(real_c[i], cam_layout, W, H) for i in real_c}
            per = {i: float(np.linalg.norm(real_u[i] - model_c[i], axis=1).mean()) for i in common}
            per_det = {i: float(np.linalg.norm(real_u[i] - twin_det[i], axis=1).mean()) for i in sorted(set(real_c) & set(twin_det))}
            m8 = table_mask(model, r, data, cam, layout).astype(np.uint8)
            for pts in real_c.values():                      # drop the real markers and their paper from the sample
                grown = (pts - pts.mean(0)) * 1.6 + pts.mean(0)
                cv2.fillConvexPoly(m8, cv2.convexHull(grown.astype(np.int32)), 0)
            mask = m8 > 0
            metrics[name] = {
                "markers_real": sorted(real_c), "markers_twin_render": sorted(twin_det), "markers_model_in_view": sorted(model_c),
                "corner_px_error_mean": round(float(np.mean(list(per.values()))), 2) if per else None,
                "corner_px_error_per_marker": {i: round(v, 2) for i, v in per.items()},
                "corner_px_error_render_detection": {i: round(v, 2) for i, v in per_det.items()},
                "table_colour": colour_error(real, twin, mask)}
            cv2.imwrite(str(out / f"real_{name}.jpg"), real)
            cv2.imwrite(str(out / f"twin_{name}.png"), twin)
            big = lambda im: cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            ov = overlay(real, twin, real_u, model_c)
            row = np.hstack([big(real), big(twin), ov])
            e = metrics[name]["corner_px_error_mean"]; c = metrics[name]["table_colour"]
            label = f"{name}: corner err {e if e is not None else '-'} px ({len(common)} markers)  table colour err " + \
                    (f"{c['mean_rgb_error']} rgb / dE {c['mean_delta_e']}" if c else "-")
            cv2.rectangle(row, (0, 0), (row.shape[1], 22), (0, 0, 0), -1)
            cv2.putText(row, label + "    [real | twin | overlay: real edges green, twin magenta]", (6, 16), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            rows.append(row)
    cv2.imwrite(str(out / "compare.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", default="home")
    ap.add_argument("--frames", type=Path, help="DIR with overview.jpg, wrist.jpg, joints.json instead of Station")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    layout_file = build_scene.layout_path(a.layout)
    if a.frames:
        frames = {k: cv2.imread(str(a.frames / f"{k}.jpg")) for k in CAMS}
        j = json.loads((a.frames / "joints.json").read_text())
        raw, ranges = {int(k): v for k, v in j["raw"].items()}, {int(k): tuple(v) for k, v in j["ranges"].items()}
    else:
        frames, raw, ranges = asyncio.run(grab_live())
    out = a.out or HERE / "real_compare" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    m = compare(layout_file, frames, raw, ranges, out)
    for name in CAMS:
        c = m[name]["table_colour"]
        print(f"{name:9s} markers real {m[name]['markers_real']} twin {m[name]['markers_twin_render']}  "
              f"corner px error {m[name]['corner_px_error_mean']} {m[name]['corner_px_error_per_marker']}  "
              f"table colour error {c and c['mean_rgb_error']} rgb, dE {c and c['mean_delta_e']}")
    print("wrote", out)


if __name__ == "__main__":
    main()
