"""Sim-vs-real check: render the twin's cameras at the real arm's joint state, next to the real Station frames.

  python3 sim/compare_real.py [--layout home] [--frames DIR | --capture frames/calib-*] [--out DIR] [--fit-offsets]

Reads one overview + wrist frame and the follower's raw ticks through Station (read-only; or from saved frames),
sets the twin's joints with sim/real_joints.py, renders overview_cam and wrist_cam at the Station resolution (the
twin cameras use the calibrated pinhole, build_scene.camera_lens) and writes:
  compare_<view>.jpg  real (undistorted to the same pinhole) | twin | overlay (real edges green, twin edges magenta,
                 real marker corners green circles, twin model corners red crosses; overview: arm outlines, real
                 orange, twin blue). Views: overview, wrist, and wrist_poseNN for a --capture's hand-placed poses.
  metrics.json   per view: marker corner px error (twin model corners projected vs undistorted corners detected in
                 the real frame; also corners detected in the twin render), overview arm silhouette (IoU, mean outline
                 distance real->twin / twin->real), table colour error (mean RGB difference of the table area, and
                 mean per-pixel CIE76 delta E)
--fit-offsets instead fits per-joint zero offsets (q = range_fraction + delta) plus the wrist camera mount pitch to
the same frames (wrist marker corners, wrist jaw edges, overview arm silhouette) and writes offset_fit.{json,jpg}.
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
    return build_scene.camera_K(model, cam_id, W, H)


def project(model, data, cam_id, pts, W, H):
    """World points -> pixels of a MuJoCo camera (the pinhole it renders with: calibrated K or centred fovy)."""
    R = data.cam_xmat[cam_id].reshape(3, 3) @ build_scene.CV_TO_MJ
    pc = (np.asarray(pts) - data.cam_xpos[cam_id]) @ R
    return (camera_K(model, cam_id, W, H) @ (pc / pc[:, 2:3]).T).T[:, :2], pc[:, 2]


def detect(img, dictionary, destripe=()):
    """Sub-pixel marker corners (perception.aruco: padded, upscaled, edge-refined; destripe for wrist close-ups)."""
    from perception import aruco
    return aruco.detect(img, dictionary, destripe=destripe)


def _layout_K(cam_layout, W, H):
    k = cam_layout.get("K_normalized")
    return None if not k else np.array([[k["fx"] * W, 0, k["cx"] * W], [0, k["fy"] * H, k["cy"] * H], [0, 0, 1.0]])


def undistort_px(px, cam_layout, W, H):
    """Real (distorted) pixels -> ideal pinhole pixels with the layout's intrinsics, when it has them."""
    K, d = _layout_K(cam_layout, W, H), cam_layout.get("dist")
    if K is None or not d or not any(d):
        return px
    return cv2.undistortPoints(px.reshape(-1, 1, 2), K, np.array(d, float), P=K).reshape(-1, 2)


def undistort_img(img, cam_layout):
    """Real frame resampled to the ideal pinhole (same K, no distortion): what the twin camera renders."""
    H, W = img.shape[:2]
    K, d = _layout_K(cam_layout, W, H), cam_layout.get("dist")
    if K is None or not d or not any(d):
        return img
    return cv2.undistort(img, K, np.array(d, float), None, K)


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


def load_capture(d: Path):
    """A perception.calib_scene capture (frames/calib-*: overview_*.jpg, wrist_*.jpg, poseNN_wrist_*.jpg, joints.json
    with Station joints {"j1": {"raw", "lo", "hi"}, ...}) -> per-camera median frames, raw ticks, ranges, and the
    hand-placed wrist poses [{"name", "wrist", "raw", "ranges"}]. The arm is still while each camera's frames are
    taken, so the median only removes noise and passers-by."""
    j = json.loads((d / "joints.json").read_text())

    def ticks(js):
        return {int(k[1:]): v["raw"] for k, v in js.items()}, {int(k[1:]): (v["lo"], v["hi"]) for k, v in js.items()}

    def median(imgs):
        return np.median(np.stack(imgs), 0).astype(np.uint8) if imgs else None

    def read(pattern):
        return [cv2.imread(str(p)) for p in sorted(d.glob(pattern))]

    ov, wr = read("overview_*.jpg"), read("wrist_*.jpg")
    frames = {"overview": median(ov), "wrist": median(wr), "overview_all": ov, "wrist_all": wr}
    raw, ranges = ticks(j["joints"])
    poses = []
    for k, pj in enumerate(j.get("poses") or []):
        imgs = read(f"pose{k + 1:02d}_wrist_*.jpg")
        if imgs:
            r, rg = ticks(pj)
            poses.append({"name": f"pose{k + 1:02d}", "wrist": median(imgs), "wrist_all": imgs, "raw": r, "ranges": rg})
    return frames, raw, ranges, poses


# Arm silhouette in the overview: the arm is black plastic on a light table. Real arm pixels are looked for only near
# the twin's arm (a gross pose error then shows as low IoU, not as a match to the chair behind the table).
ARM_DARK_GRAY = 80          # gray level below which a pixel is arm (table ~170-200, arm 20-60 in the office frames)
ARM_ROI_PX = 25             # search band around the twin's arm silhouette
MARKER_GROW = 1.6           # a marker quad scaled by this covers its white paper border too


def arm_geoms(model):
    root = model.body_rootid[model.site_bodyid[model.site("grasp_tcp").id]]
    return np.flatnonzero(model.body_rootid[model.geom_bodyid] == root)


def segment(renderer, data, cam, opt):
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=cam, scene_option=opt)
    s = renderer.render()
    renderer.disable_segmentation_rendering()
    return np.where(s[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), s[..., 0], -1)


def silhouette(real, twin_mask, support, real_markers):
    """-> (real arm mask, metrics): IoU inside the search band and the mean outline distance both ways (px).
    support: pixels where the twin sees the table or the arm; real dark pixels elsewhere (the chair behind the
    table's far edge) are not arm candidates."""
    roi = cv2.dilate(twin_mask.astype(np.uint8), np.ones((2 * ARM_ROI_PX + 1,) * 2, np.uint8)) > 0
    dark = (cv2.GaussianBlur(cv2.cvtColor(real, cv2.COLOR_BGR2GRAY), (3, 3), 0) < ARM_DARK_GRAY).astype(np.uint8)
    for pts in real_markers.values():
        cv2.fillConvexPoly(dark, cv2.convexHull(((pts - pts.mean(0)) * MARKER_GROW + pts.mean(0)).astype(np.int32)), 0)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    real_mask = (dark > 0) & roi & support
    n, lab = cv2.connectedComponents(real_mask.astype(np.uint8))
    if n > 1:                                   # keep the components that touch the twin arm (drops cable / shadow bits)
        real_mask = np.isin(lab, [k for k in range(1, n) if (twin_mask & (lab == k)).any()])
    inter, union = (real_mask & twin_mask).sum(), (real_mask | twin_mask).sum()
    out = {"iou": round(float(inter / union), 3) if union else None}
    er, et = _outline(real_mask, 2) & support, _outline(twin_mask, 2) & support
    if er.any() and et.any():
        dt_t = cv2.distanceTransform((~et).astype(np.uint8), cv2.DIST_L2, 5)
        dt_r = cv2.distanceTransform((~er).astype(np.uint8), cv2.DIST_L2, 5)
        out.update(real_to_twin_px=round(float(dt_t[er].mean()), 2), twin_to_real_px=round(float(dt_r[et].mean()), 2))
    return real_mask, out


def silhouette_panel(real, real_mask, twin_mask, scale=2):
    """Darkened real frame; arm pixels only real = orange, only twin = blue, both = white."""
    out = (real * 0.45).astype(np.uint8)
    out[real_mask & ~twin_mask] = (0, 160, 255)
    out[twin_mask & ~real_mask] = (255, 120, 0)
    out[real_mask & twin_mask] = (235, 235, 235)
    return cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def arm_support(seg, model, geoms):
    names = ("table_top", "table_skin")
    table = [model.geom(n).id for n in names if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) >= 0]
    return np.isin(seg, np.r_[table, geoms])


def compare_model(model, layout, frames, raw, ranges, out: Path, poses=(), undistort=True, tag="") -> dict:
    """Render the twin at the real joints (and at every wrist pose), compare with the real frames; writes
    compare_overview.jpg, compare_wrist.jpg, compare_wrist_<pose>.jpg and metrics.json into out."""
    data = mujoco.MjData(model)
    W, H = layout["image_size"]
    mapping = real_joints.load_mapping(layout)
    mk = layout.get("markers") or {}
    dictionary = mk.get("dictionary", "DICT_4X4_50")
    q = real_joints.to_q(raw, ranges, *model.actuator_ctrlrange.T, mapping)
    metrics = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "raw_ticks": raw, "q_rad": np.round(q, 4).tolist(),
               "joint_mapping": mapping, "undistorted_real": undistort,
               "sources": {"table": layout["table"].get("src"), "markers": sorted({i.get("src") for i in mk.get("items", [])} - {None}),
                           "overview_cam": next((c.get("src") for c in layout["cameras"] if c["name"] == "overview_cam"), None),
                           "wrist_camera": layout["wrist_camera"].get("src")}}
    views = [("overview", "overview", frames.get("overview"), frames.get("overview_all"), raw, ranges),
             ("wrist", "wrist", frames.get("wrist"), frames.get("wrist_all"), raw, ranges)]
    views += [(f"wrist_{p['name']}", "wrist", p["wrist"], p.get("wrist_all"), p["raw"], p["ranges"]) for p in poses]
    opt = mujoco.MjvOption(); opt.geomgroup[1] = 0
    geoms = arm_geoms(model)
    with mujoco.Renderer(model, H, W) as r:
        for key, name, real, real_all, rw, rg in views:
            if real is None:
                continue
            cam = CAMS[name]
            qv = real_joints.to_q(rw, rg, *model.actuator_ctrlrange.T, mapping)
            real_joints.set_arm_q(model, data, qv)
            cid = model.camera(cam).id
            r.update_scene(data, camera=cam, scene_option=opt)
            twin = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
            if real.shape[:2] != (H, W):
                real = cv2.resize(real, (W, H))
            cam_layout = layout["wrist_camera"] if name == "wrist" else next(c for c in layout["cameras"] if c["name"] == cam)
            real_c = real_corners(real, real_all, name, dictionary)
            real_u = {i: undistort_px(real_c[i], cam_layout, W, H) for i in real_c}
            shown = undistort_img(real, cam_layout) if undistort else real
            twin_det = detect(twin, dictionary)
            model_c = {}
            for it in mk.get("items", []):
                pts = np.array([data.site_xpos[model.site(f"aruco_{it['id']}_c{k}").id] for k in range(4)])
                px, z = project(model, data, cid, pts, W, H)
                if (z > 0).all() and (px > -20).all() and (px < [W + 20, H + 20]).all():
                    model_c[it["id"]] = px
            common = sorted(set(real_c) & set(model_c))
            per = {i: np.linalg.norm(real_u[i] - model_c[i], axis=1) for i in common}
            per_det = {i: float(np.linalg.norm(real_u[i] - twin_det[i], axis=1).mean()) for i in sorted(set(real_c) & set(twin_det))}
            m8 = table_mask(model, r, data, cam, layout).astype(np.uint8)
            for pts in real_u.values():
                cv2.fillConvexPoly(m8, cv2.convexHull(((pts - pts.mean(0)) * MARKER_GROW + pts.mean(0)).astype(np.int32)), 0)
            v = {"markers_real": sorted(real_c), "markers_twin_render": sorted(twin_det), "markers_model_in_view": sorted(model_c),
                 "corner_px_error_mean": round(float(np.mean([e.mean() for e in per.values()])), 2) if per else None,
                 "corner_px_error_max": round(float(max(e.max() for e in per.values())), 2) if per else None,
                 "corner_px_error_per_marker": {i: round(float(e.mean()), 2) for i, e in per.items()},
                 "centre_px_offset_per_marker": {i: np.round(model_c[i].mean(0) - real_u[i].mean(0), 2).tolist() for i in common},
                 "corner_px_error_render_detection": {i: round(v_, 2) for i, v_ in per_det.items()},
                 "table_colour": colour_error(shown, twin, m8 > 0)}
            if name == "wrist":
                v["q_rad"] = np.round(qv, 4).tolist()
            sil = None
            if name == "overview":
                seg = segment(r, data, cam, opt)
                twin_mask = np.isin(seg, geoms)
                real_mask, v["arm_silhouette"] = silhouette(shown, twin_mask, arm_support(seg, model, geoms), real_u)
                sil = (real_mask, twin_mask)
            metrics[key] = v
            big = lambda im: cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            tiles = [big(shown), big(twin), overlay(shown, twin, real_u, model_c)]
            if sil is not None:
                tiles.append(silhouette_panel(shown, *sil))
            row = np.hstack(tiles)
            e = v["corner_px_error_mean"]
            label = f"{tag}{key}: marker corners twin-real {e if e is not None else '-'} px mean, {v['corner_px_error_max'] or '-'} max " \
                    f"({len(common)} markers)"
            if sil is not None:
                s_ = v["arm_silhouette"]
                label += f"; arm IoU {s_['iou']}, outline {s_.get('real_to_twin_px')}/{s_.get('twin_to_real_px')} px"
            legend = "real" + (" (undistorted)" if undistort else "") + " | twin | overlay: real edges green, twin magenta, " \
                     "real corners o, twin x" + (" | arm: real only orange, twin only blue, both white" if sil is not None else "")
            cv2.rectangle(row, (0, 0), (row.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(row, label, (6, 16), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(row, legend, (6, 34), 0, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imwrite(str(out / f"compare_{key}.jpg"), row, [cv2.IMWRITE_JPEG_QUALITY, 85])
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return metrics


def real_corners(real, real_all, name, dictionary):
    """Marker corners in a real view: the per-corner median over the individual frames when there are several (as
    perception.calib_scene measures them; a far marker can fail to decode in the median image), else the frame's."""
    from perception import aruco
    destripe = aruco.DESTRIPE_WRIST_PX if name == "wrist" else aruco.DESTRIPE_OVERVIEW_PX
    if real_all and len(real_all) > 1:
        from perception.calib_scene import OVERVIEW_MIN_FRAC
        return aruco.average_corners(real_all, dictionary, min_frac=OVERVIEW_MIN_FRAC, destripe=destripe)[0]
    return detect(real, dictionary, destripe)


def compare(layout_file, frames, raw, ranges, out: Path, poses=(), undistort=True) -> dict:
    model, layout = load_model(layout_file)
    m = compare_model(model, layout, frames, raw, ranges, out, poses, undistort)
    m["layout"] = str(layout_file)
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    return m


# --- joint zero offsets: render-and-compare least squares --------------------------------------------------------
# q_true = q_range_fraction + delta. Observations at one real pose:
#   wrist   marker corners (20 detected; 21 from the rejected quads, its id is under a cable), and the jaw edges.
#           The jaws are rigid in the camera's link, so they fix the camera MOUNT pitch (a nuisance parameter, the
#           "+8 deg" of the twin report), not the joints; the markers fix the camera's world pose = joints + mount.
#   overview arm silhouette (the folded arm against the plants: weak, capped chamfer).
# Chamfer terms are DT(twin contour) sampled at fixed real edge pixels, so the residual length never changes.
FIT_SCALE, DT_CAP, JAW_CAP, JAC_STEP_DEG = 3, 8.0, 30.0, 0.5
PRIOR_DEG = [5, 5, 5, 5, 5, 5, 5, 10, 10]          # delta j1..j8, wrist mount pitch
PARAMS = [f"j{i}" for i in range(1, 9)] + ["wrist_mount_pitch"]


def _edges(img, roi):
    e = cv2.Canny(cv2.GaussianBlur(img, (3, 3), 0), 50, 130) > 0
    return np.argwhere(e & roi)[:, ::-1].astype(float)          # (x, y)


def _outline(mask, b):
    """Outline pixels of a mask; image-border segments (a mask cut by the frame) are not outline."""
    m8 = mask.astype(np.uint8)
    edge = (m8 - cv2.erode(m8, np.ones((3, 3), np.uint8))) > 0
    edge[:b], edge[-b:], edge[:, :b], edge[:, -b:] = False, False, False, False
    return edge


def _contour_dt(mask, cap=DT_CAP):
    """Distance (px at FIT_SCALE) to the outline of a mask, capped."""
    edge = _outline(mask, 2 * FIT_SCALE)
    if not edge.any():
        return np.full(mask.shape, cap * FIT_SCALE, np.float32)
    return np.minimum(cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 5), cap * FIT_SCALE)


def _sample(img, pts):
    p = (pts * FIT_SCALE).astype(np.float32).reshape(-1, 1, 2)
    return cv2.remap(img, p[..., 0], p[..., 1], cv2.INTER_LINEAR).ravel() / FIT_SCALE


class OffsetFit:
    def __init__(self, layout_file, frames, raw, ranges):
        self.model, self.layout = load_model(layout_file)
        m = self.model
        self.data = mujoco.MjData(m)
        self.W, self.H = self.layout["image_size"]
        self.q0 = self._q_unclipped(raw, ranges)
        self.wcam, self.ocam = m.camera("wrist_cam").id, m.camera("overview_cam").id
        self.quat0 = m.cam_quat[self.wcam].copy()
        root = m.body_rootid[m.site_bodyid[m.site("grasp_tcp").id]]
        self.arm_geoms = np.flatnonzero(m.body_rootid[m.geom_bodyid] == root)
        jaw_bodies = [m.body(n).id for n in ("Gripper_Jaw_01_v1_1", "Gripper_Jaw_02_v1_1", "Gripper_Gear_v1_1")]
        self.jaw_geoms = np.flatnonzero(np.isin(m.geom_bodyid, jaw_bodies))
        self.opt = mujoco.MjvOption(); self.opt.geomgroup[1] = 0
        self.r = mujoco.Renderer(m, self.H * FIT_SCALE, self.W * FIT_SCALE)
        self.real = {k: (cv2.resize(v, (self.W, self.H)) if v.shape[:2] != (self.H, self.W) else v) for k, v in frames.items()}
        mk = self.layout["markers"]
        self.dictionary = mk["dictionary"]
        self._setup_observations()

    def _q_unclipped(self, raw, ranges):
        """range_fraction without the URDF clip (j4 reads past its calibrated end at the folded pose)."""
        lo, hi = self.model.actuator_ctrlrange.T
        q = np.zeros(8)
        for i in range(8):
            a, b = ranges[i + 1]
            n = (raw[i + 1] - a) / (b - a)
            q[i] = hi[i] * (1 - n) if i == 7 else lo[i] + n * (hi[i] - lo[i])
        return q

    def pose(self, x):
        m = self.model
        a = math.radians(x[8]) / 2                                # mount pitch about the camera's own x axis
        mujoco.mju_mulQuat(m.cam_quat[self.wcam], self.quat0, np.array([math.cos(a), math.sin(a), 0, 0]))
        real_joints.set_arm_q(m, self.data, self.q0 + np.radians(x[:8]))

    def seg(self, cam):
        self.r.enable_segmentation_rendering()
        self.r.update_scene(self.data, camera=cam, scene_option=self.opt)
        s = self.r.render()
        self.r.disable_segmentation_rendering()
        return np.where(s[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), s[..., 0], -1)

    def model_corners(self, mid, with_depth=False):
        sites = [self.data.site_xpos[self.model.site(f"aruco_{mid}_c{k}").id] for k in range(4)]
        px, z = project(self.model, self.data, self.wcam, np.array(sites), self.W, self.H)
        return (px, z) if with_depth else px

    def _setup_observations(self):
        x0 = np.zeros(len(PARAMS))
        self.pose(x0)
        g = cv2.cvtColor(self.real["wrist"], cv2.COLOR_BGR2GRAY)
        det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary)),
                                      cv2.aruco.DetectorParameters())
        c, ids, rej = det.detectMarkers(g)
        self.marks = {int(i): (cc.reshape(4, 2).astype(float), 1.5) for i, cc in zip(([] if ids is None else ids.ravel()), c)}
        for it in self.layout["markers"]["items"]:                # an undecoded quad near a marker's model position
            mid = it["id"]
            if mid in self.marks:
                continue
            pm = self.model_corners(mid)
            best = None
            for qd in rej:
                qd = qd.reshape(4, 2).astype(float)
                for cand in (qd, qd[::-1]):
                    for s in range(4):
                        e = np.linalg.norm(np.roll(cand, s, 0) - pm, axis=1).mean()
                        if e < 20 and (best is None or e < best[0]):
                            best = (e, np.roll(cand, s, 0))
            if best is not None:
                self.marks[mid] = (best[1], 3.0)                  # estimated pose (21): looser
        H, W = self.H, self.W
        small = lambda m: cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        grow = lambda m, k: cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        # real jaws: the dark region (black plastic on bright wood) in the lower part of the wrist view, markers removed
        dark = cv2.morphologyEx((cv2.GaussianBlur(g, (5, 5), 0) < 70).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        dark[: H // 2] = 0                                        # jaw tips sit at ~2/3 height; above is the room
        for it in self.layout["markers"]["items"]:
            pm, z = self.model_corners(it["id"], with_depth=True)
            if (z > 0).all() and (pm > -W / 4).all() and (pm < [W * 1.25, H * 1.25]).all():   # in view, not grazing
                cv2.fillConvexPoly(dark, cv2.convexHull(((pm - pm.mean(0)) * 2.0 + pm.mean(0)).astype(np.int32)), 0)
        _, lab = cv2.connectedComponents(dark)
        self.real_jaw = np.isin(lab, np.setdiff1d(lab[-3:], [0]))  # the gripper enters from the bottom edge
        self.jaw_pts = np.argwhere(_outline(self.real_jaw, 2))[:, ::-1].astype(float)
        arm0 = small(np.isin(self.seg("overview_cam"), self.arm_geoms))
        self.arm_pts = _edges(self.real["overview"], grow(arm0, 15))
        self.mark_sigma = np.concatenate([[s] * 8 for _, s in self.marks.values()]) if self.marks else np.zeros(0)

    def terms(self, x):
        """Raw residuals in px: marker corner x/y errors, jaw chamfer, overview arm chamfer."""
        self.pose(x)
        mk = np.concatenate([(self.model_corners(i) - obs).ravel() for i, (obs, _) in self.marks.items()]) \
            if self.marks else np.zeros(0)
        jaw = _sample(_contour_dt(np.isin(self.seg("wrist_cam"), self.arm_geoms), JAW_CAP), self.jaw_pts)
        arm = _sample(_contour_dt(np.isin(self.seg("overview_cam"), self.arm_geoms)), self.arm_pts)
        return mk, jaw, arm

    def jac(self, x, *_):
        """Central differences with a fixed step: renders are pixel-quantised, scipy's relative steps are far too small."""
        cols = []
        for k in range(len(x)):
            d = np.zeros(len(x)); d[k] = JAC_STEP_DEG
            cols.append((self.residuals(x + d) - self.residuals(x - d)) / (2 * JAC_STEP_DEG))
        return np.array(cols).T

    def residuals(self, x, sigma_jaw=1.5, sigma_arm=2.5, n_jaw=25, n_arm=15):
        """Whitened residuals. Edge pixels are strongly correlated along a contour, so each chamfer group counts as
        n_* independent observations at most (weights / sqrt(len / n))."""
        mk, jaw, arm = self.terms(x)
        w = lambda r, s, n: r / s / math.sqrt(max(1.0, len(r) / n))
        return np.concatenate([mk / self.mark_sigma, w(jaw, sigma_jaw, n_jaw), w(arm, sigma_arm, n_arm),
                               np.asarray(x) / PRIOR_DEG])

    def summary(self, x):
        mk, jaw, arm = self.terms(x)
        per = {i: round(float(np.linalg.norm(self.model_corners(i) - obs, axis=1).mean()), 2) for i, (obs, _) in self.marks.items()}
        return {"wrist_marker_corner_px": per, "wrist_marker_corner_px_mean": round(float(np.mean(list(per.values()))), 2) if per else None,
                "wrist_jaw_chamfer_px": round(float(np.mean(jaw)), 2),
                "overview_arm_chamfer_px": round(float(np.mean(np.minimum(arm, DT_CAP))), 2)}


def fit_offsets(layout_file, frames, raw, ranges, out: Path | None = None) -> dict:
    from scipy.optimize import least_squares
    f = OffsetFit(layout_file, frames, raw, ranges)
    x0 = np.zeros(len(PARAMS))
    before = f.summary(x0)
    best = None
    for start in ([0] * 9, [0] * 8 + [-8], [0] * 8 + [8]):   # a few mount-pitch starts (the chamfer is local)
        res = least_squares(f.residuals, np.array(start, float), jac=f.jac, loss="soft_l1", f_scale=2.0,
                            x_scale=PRIOR_DEG, max_nfev=60)
        if best is None or res.cost < best.cost:
            best = res
    x = best.x
    J = best.jac
    dof = max(1, len(best.fun) - len(x))
    s2 = max(1.0, 2 * best.cost / dof)
    cov = np.linalg.pinv(J.T @ J) * s2
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    after = f.summary(x)
    combos = {"pitch_sum_j2+j3+j4+j6": [0, 1, 1, 1, 0, 1, 0, 0, 0], "pitch_sum+mount": [0, 1, 1, 1, 0, 1, 0, 0, -1],
              "j2+j3": [0, 1, 1, 0, 0, 0, 0, 0, 0], "j4+j6": [0, 0, 0, 1, 0, 1, 0, 0, 0], "roll_j5+j7": [0, 0, 0, 0, 1, 0, 1, 0, 0]}
    result_combos = {k: {"deg": round(float(np.dot(a, x)), 2), "std_deg": round(float(np.sqrt(np.dot(a, cov @ a))), 2)}
                     for k, a in ((k, np.array(v, float)) for k, v in combos.items())}
    result = {"mapping": "range_fraction", "q_range_fraction_rad": np.round(f.q0, 4).tolist(),
              "offsets_deg": {p: round(float(v), 2) for p, v in zip(PARAMS, x)},
              "std_deg": {p: round(float(v), 2) for p, v in zip(PARAMS, std)},
              "combinations": result_combos,
              "prior_std_deg": dict(zip(PARAMS, PRIOR_DEG)),
              "corr_strong": {f"{PARAMS[i]}~{PARAMS[j]}": round(float(corr[i, j]), 2) for i in range(9) for j in range(i + 1, 9)
                              if abs(corr[i, j]) > 0.7},
              "before": before, "after": after, "markers_used": {i: s for i, (_, s) in f.marks.items()},
              "points": {"wrist_jaw_edges": len(f.jaw_pts), "overview_arm_edges": len(f.arm_pts)},
              "cost": round(float(best.cost), 2), "nfev": int(best.nfev)}
    if out:
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for xx in (x0, x):
            f.pose(xx)
            tiles = []
            for name, cam in CAMS.items():
                f.r.update_scene(f.data, camera=cam, scene_option=f.opt)
                twin = cv2.resize(cv2.cvtColor(f.r.render(), cv2.COLOR_RGB2BGR), (f.W, f.H), interpolation=cv2.INTER_AREA)
                mc = {i: f.model_corners(i) for i in f.marks} if name == "wrist" else {}
                tiles.append(overlay(f.real[name], twin, {i: o for i, (o, _) in f.marks.items()} if name == "wrist" else {}, mc))
            rows.append(np.hstack(tiles))
        img = np.vstack(rows)
        for k, txt in enumerate(("before (range_fraction)", "after (fitted offsets)")):
            cv2.putText(img, txt, (6, 16 + k * rows[0].shape[0]), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / "offset_fit.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        (out / "offset_fit.json").write_text(json.dumps(result, indent=1))
    f.r.close()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", help="layout name/path (default: SIM_LAYOUT, else the selected SITE's)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--frames", type=Path, help="DIR with overview.jpg, wrist.jpg, joints.json instead of Station")
    src.add_argument("--capture", type=Path, help="a perception.calib_scene capture dir (frames/calib-*): median "
                                                  "frames at the capture pose plus every hand-placed wrist pose")
    ap.add_argument("--no-undistort", action="store_true", help="show/segment the raw real frames (corner errors "
                                                                 "always use undistorted real corners)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--fit-offsets", action="store_true", help="fit joint zero offsets (+ wrist mount pitch) instead")
    a = ap.parse_args()
    layout_file = build_scene.layout_path(a.layout)
    poses = []
    if a.capture:
        frames, raw, ranges, poses = load_capture(a.capture)
    elif a.frames:
        frames = {k: cv2.imread(str(a.frames / f"{k}.jpg")) for k in CAMS}
        j = json.loads((a.frames / "joints.json").read_text())
        raw, ranges = {int(k): v for k, v in j["raw"].items()}, {int(k): tuple(v) for k, v in j["ranges"].items()}
    else:
        frames, raw, ranges = asyncio.run(grab_live())
    out = a.out or HERE / "real_compare" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    if a.fit_offsets:
        print(json.dumps(fit_offsets(layout_file, frames, raw, ranges, out), indent=1))
        print("wrote", out)
        return
    m = compare(layout_file, frames, raw, ranges, out, poses, undistort=not a.no_undistort)
    for key, v in m.items():
        if not isinstance(v, dict) or "corner_px_error_mean" not in v:
            continue
        c = v["table_colour"]
        print(f"{key:14s} markers real {v['markers_real']} model {v['markers_model_in_view']}  corner px error mean "
              f"{v['corner_px_error_mean']} max {v['corner_px_error_max']} {v['corner_px_error_per_marker']}"
              + (f"  arm {v['arm_silhouette']}" if "arm_silhouette" in v else "")
              + f"  table colour error {c and c['mean_rgb_error']} rgb, dE {c and c['mean_delta_e']}")
    print("wrote", out)


if __name__ == "__main__":
    main()
