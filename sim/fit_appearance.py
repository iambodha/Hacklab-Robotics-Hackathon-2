"""Table texture and lighting of the home twin, from the real overview camera.

  python3 sim/fit_appearance.py [--layout home] [--frames sim/layouts/home_estimate] [--mm-per-px 2]

1. Texture: every point of the twin's table top is projected into the real overview frame with the calibrated camera
   (pose + intrinsics from the layout, i.e. the marker homography of the table plane) and sampled, so the texture is
   the rectified real tabletop. Marker papers (the twin draws its own), the arm and anything dark (cable), and table
   the overview does not see are filled by inpainting. Written to sim/assets/home_table_texture.png (image right =
   table +x, image up = table +y, as build_scene stretches it).
2. Marker print levels (ink / paper grey) from the real marker pixels.
3. Lighting: the texture already carries the real shading, so the light gain is fitted so that the rendered table
   matches the real one (mean colour over the table, compare_real's metric), keeping the ambient/key ratio.
Updates layout.table.texture, layout.markers.levels and layout.lighting (src "fit_appearance").
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_scene  # noqa: E402  (sets MUJOCO_GL)
import compare_real  # noqa: E402
import mujoco  # noqa: E402
import real_joints  # noqa: E402
import sync_from_calib  # noqa: E402

HERE = Path(__file__).resolve().parent


def real_camera(model, data, cam_layout, W, H):
    """rvec, tvec, K, dist of the real overview camera as the layout describes it."""
    cid = model.camera("overview_cam").id
    Rcv = data.cam_xmat[cid].reshape(3, 3) @ build_scene.CV_TO_MJ        # camera axes in the sim frame
    R, t = Rcv.T, -Rcv.T @ data.cam_xpos[cid]
    k = cam_layout.get("K_normalized")
    K = (np.array([[k["fx"] * W, 0, k["cx"] * W], [0, k["fy"] * H, k["cy"] * H], [0, 0, 1.0]]) if k
         else compare_real.camera_K(model, cid, W, H))
    return cv2.Rodrigues(R)[0], t, K, np.array(cam_layout.get("dist") or [0, 0, 0, 0, 0], float)


def make_texture(layout, model, data, real, mm_per_px):
    H, W = real.shape[:2]
    cam_layout = next(c for c in layout["cameras"] if c["name"] == "overview_cam")
    rvec, tvec, K, dist = real_camera(model, data, cam_layout, W, H)
    tw, td = layout["table"]["size"]
    nx, ny = int(round(tw * 1000 / mm_per_px)), int(round(td * 1000 / mm_per_px))
    tb = model.body("table").id
    Rt, pt = data.xmat[tb].reshape(3, 3), data.xpos[tb]
    xs = -tw / 2 + (np.arange(nx) + 0.5) * tw / nx
    ys = td / 2 - (np.arange(ny) + 0.5) * td / ny
    X, Y = np.meshgrid(xs, ys)
    local = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
    world = local @ Rt.T + pt
    cam = (world - (-cv2.Rodrigues(rvec)[0].T @ tvec)) @ cv2.Rodrigues(rvec)[0].T
    uv = cv2.projectPoints(world, rvec, tvec, K, dist)[0].reshape(ny, nx, 2).astype(np.float32)
    tex = cv2.remap(real, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    seen = ((uv[..., 0] >= 1) & (uv[..., 0] < W - 2) & (uv[..., 1] >= 1) & (uv[..., 1] < H - 2)
            & (cam[:, 2].reshape(ny, nx) > 0))
    hole = ~seen
    # not wood: arm, cable, marker paper, wall/floor seen past an edge (Lab distance to the median seen colour)
    lab = cv2.cvtColor(cv2.GaussianBlur(tex, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(float)
    ref = np.median(lab[seen], axis=0)
    off = np.sqrt((((lab - ref) * [0.6, 1.0, 1.0]) ** 2).sum(-1)) > 22
    hole |= cv2.dilate(off.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    mk = layout.get("markers") or {}
    half = mk.get("size_m", 0.1) / 2 + mk.get("paper_margin_m", 0.012) + 0.02
    to_px = lambda p: [(p[0] + tw / 2) / tw * nx, (td / 2 - p[1]) / td * ny]
    m8 = hole.astype(np.uint8)
    for it in mk.get("items", []):
        T = build_scene.marker_pose(it)
        corners = np.array([T[:3, :3] @ [sx * half, sy * half, 0] + T[:3, 3] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        loc = (corners - pt) @ Rt                                       # sim frame -> table body frame
        cv2.fillConvexPoly(m8, np.array([to_px(p) for p in loc], np.int32), 1)
    base = (np.array([0.0, 0.0, 0.0]) - pt) @ Rt                         # arm base footprint (+ margin)
    cv2.circle(m8, tuple(int(v) for v in to_px(base)), int(0.09 * 1000 / mm_per_px), 1, -1)
    filled = cv2.inpaint(tex, m8, 7, cv2.INPAINT_TELEA)
    return filled, m8 > 0, {"size_px": [nx, ny], "mm_per_px": mm_per_px, "seen_fraction": round(float(seen.mean()), 3),
                            "filled_fraction": round(float((m8 > 0).mean()), 3)}


def render_overview(model, data, W, H):
    opt = mujoco.MjvOption(); opt.geomgroup[1] = 0
    with mujoco.Renderer(model, H, W) as r:
        r.update_scene(data, camera="overview_cam", scene_option=opt)
        img = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
        mask = compare_real.table_mask(model, r, data, "overview_cam", None)
    return img, mask


def set_gain(model, base_amb, base_dif, g):
    model.vis.headlight.ambient[:] = np.clip(base_amb * g, 0, 1)
    lid = model.light("key").id
    model.light_diffuse[lid] = np.clip(base_dif * g, 0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", default="home")
    ap.add_argument("--frames", type=Path, default=HERE / "layouts" / "home_estimate")
    ap.add_argument("--mm-per-px", type=float, default=2.0)
    ap.add_argument("--ambient-share", type=float, default=0.75, help="share of the light that is ambient (rest: key light)")
    a = ap.parse_args()
    layout_file = build_scene.layout_path(a.layout)
    layout = json.loads(layout_file.read_text())
    real = cv2.imread(str(a.frames / "overview.jpg"))
    j = json.loads((a.frames / "joints.json").read_text())
    raw, ranges = {int(k): v for k, v in j["raw"].items()}, {int(k): tuple(v) for k, v in j["ranges"].items()}
    H, W = real.shape[:2]

    # 1. texture (geometry from the current layout; build without a texture first so the table frame is current)
    layout["table"].pop("texture", None)
    model, _ = compare_real.load_model_from(layout, layout_file)
    data = mujoco.MjData(model)
    real_joints.set_arm_q(model, data, real_joints.to_q(raw, ranges, *model.actuator_ctrlrange.T, real_joints.load_mapping(layout)))
    tex, holes, meta = make_texture(layout, model, data, real, a.mm_per_px)
    tex_path = HERE / "assets" / "home_table_texture.png"
    cv2.imwrite(str(tex_path), tex)
    print(f"texture {meta} -> {tex_path}")

    # 2. marker print levels: ink = dark cells inside the detected squares, paper = the ring just outside them
    gray = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY)
    ink, paper = [], []
    for pts in compare_real.detect(real, layout["markers"]["dictionary"]).values():
        c = pts.mean(0)
        inner, outer, ring_in = [np.zeros_like(gray) for _ in range(3)]
        cv2.fillConvexPoly(inner, ((pts - c) * 0.85 + c).astype(np.int32), 1)
        cv2.fillConvexPoly(outer, ((pts - c) * 1.3 + c).astype(np.int32), 1)
        cv2.fillConvexPoly(ring_in, ((pts - c) * 1.08 + c).astype(np.int32), 1)
        ink.append(np.percentile(gray[inner > 0], 15)); paper.append(np.percentile(gray[(outer > 0) & (ring_in == 0)], 85))
    levels = [round(float(np.median(ink)) / 255, 3), round(float(np.median(paper)) / 255, 3)] if ink else [0.0, 1.0]
    layout["markers"]["levels"] = levels
    print(f"marker print levels (ink, paper) = {levels}")

    # 3. lighting gain with the texture on
    layout["table"].update(texture=str(tex_path.relative_to(HERE)), specular=0.05)
    sh = a.ambient_share
    layout["lighting"] = {"headlight": {"ambient": [sh] * 3, "diffuse": [0, 0, 0], "specular": [0, 0, 0]},
                          "skybox": layout["lighting"].get("skybox", [[0.75, 0.75, 0.74], [0.45, 0.43, 0.40]]),
                          "lights": [{"name": "key", "directional": True, "pos": [0.0, 0.3, 2.0], "dir": [0.1, 0.2, -1.0],
                                      "diffuse": [1 - sh] * 3, "specular": [0.03] * 3, "castshadow": True}]}
    model, _ = compare_real.load_model_from(layout, layout_file)
    data = mujoco.MjData(model)
    real_joints.set_arm_q(model, data, real_joints.to_q(raw, ranges, *model.actuator_ctrlrange.T, real_joints.load_mapping(layout)))
    amb, dif = model.vis.headlight.ambient.copy(), model.light_diffuse[model.light("key").id].copy()
    real_c = compare_real.detect(real, layout["markers"]["dictionary"])

    def err(g):
        set_gain(model, amb, dif, g)
        img, mask = render_overview(model, data, W, H)
        m8 = mask.astype(np.uint8)
        for pts in real_c.values():
            cv2.fillConvexPoly(m8, cv2.convexHull(((pts - pts.mean(0)) * 1.6 + pts.mean(0)).astype(np.int32)), 0)
        c = compare_real.colour_error(real, img, m8 > 0)
        return c["mean_rgb_error"], c
    lo, hi = 0.6, 1.6
    for _ in range(14):                         # golden-section search on the gain
        m1, m2 = hi - 0.618 * (hi - lo), lo + 0.618 * (hi - lo)
        if err(m1)[0] < err(m2)[0]:
            hi = m2
        else:
            lo = m1
    g = (lo + hi) / 2
    e0, c0 = err(1.0)
    e, c = err(g)
    print(f"light gain {g:.3f}: table colour error {e0} -> {e} rgb (dE {c0['mean_delta_e']} -> {c['mean_delta_e']})")
    lt = layout["lighting"]
    lt["headlight"]["ambient"] = sync_from_calib.r(np.clip(amb * g, 0, 1), 3)
    lt["lights"][0]["diffuse"] = sync_from_calib.r(np.clip(dif * g, 0, 1), 3)
    lt["src"] = "fit_appearance"
    lt["_about"] = f"gain fitted to the real overview (table colour error {e0} -> {e} rgb)"
    layout["table"]["texture_meta"] = {**meta, "src": "fit_appearance", "from": str(a.frames / "overview.jpg")}
    out = json.loads(layout_file.read_text())                # write back only the appearance fields
    out["table"].update(texture=layout["table"]["texture"], specular=0.05, texture_meta=layout["table"]["texture_meta"])
    out["lighting"] = lt
    out["markers"]["levels"] = levels
    layout_file.write_text(sync_from_calib.dumps(out) + "\n")
    print("updated", layout_file)


if __name__ == "__main__":
    main()
