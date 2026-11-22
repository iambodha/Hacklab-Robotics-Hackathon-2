"""Table texture, table extent, lighting and background of a site's twin, from the real overview camera.

  python3 sim/fit_appearance.py [--layout NAME|FILE] [--frames DIR] [--mm-per-px 2]
                                [--measure-table [--near-edge Y]] [--background] [--rest-pose] [--compare FILE]
  python3 sim/fit_appearance.py --layout home --frames sim/layouts/home_estimate      # the home twin as before
  SITE=office python3 sim/fit_appearance.py --frames <capture dir> --measure-table --background --rest-pose \\
      --compare sites/office/appearance_compare.jpg

--layout defaults to the selected site's layout (SITE, sites/<site>/site.json). --frames is a directory with
overview.jpg (or overview_*.jpg, median-stacked per pixel so a passing hand drops out), optionally wrist.jpg /
wrist_*.jpg, and joints.json in either format: {"raw": {id: tick}, "ranges": {id: [lo, hi]}} (estimate_home) or
{"joints": {"j1": {"raw", "lo", "hi"}, ...}} (calibration capture; the arm pose during the overview frames).

0. --measure-table: left/right/far table edges and the table yaw from the rectified overview (table-coloured cells
   on the table plane, scanned outward from the markers); the near edge is not seen by the overview: --near-edge
   (sim y, m), else NEAR_MARGIN_M past the overview's bottom row. A lengthwise seam (two tables pushed together) is
   located too and used for an unseen side edge (tables assumed equal). Updates table.size/center/yaw/src.
1. Texture: every point of the twin's table top is projected into the real overview frame with the calibrated camera
   and sampled, so the texture is the rectified real tabletop. Marker papers (the twin draws its own), the arm (its
   projected silhouette + shadow margin), anything off-colour (cable), and table the overview does not see are filled
   smoothly from the surrounding table (push-pull; a measured seam is kept and continued). Written to the file the layout already names (home: sim/assets/home_table_texture.png), else to
   <layout folder>/table_texture.png (image right = table +x, image up = table +y, as build_scene stretches it).
2. Marker print levels (ink / paper grey) from the real marker pixels.
3. Lighting: the texture already carries the real shading, so the light gain is fitted so that the rendered table
   matches the real one (mean colour over the table, compare_real's metric), keeping the ambient/key ratio.
4. --background: floor and skybox colours scaled until the rendered floor / sky pixels have the real overview's mean
   colour there (whatever is behind the table: chair, carpet, walls).
5. --rest-pose: layout.rest_pose = the captured arm pose (the twin's previews render the rest keyframe).
Updates layout.table.texture, layout.markers.levels, layout.lighting (src "fit_appearance") and the fields above.
--compare FILE writes real | twin | 50 % blend of the overview (after the fit) and prints compare_real's metrics.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_scene  # noqa: E402  (sets MUJOCO_GL)
import compare_real  # noqa: E402
import mujoco  # noqa: E402
import real_joints  # noqa: E402
import sync_from_calib  # noqa: E402
from sites import DATA_ROOT  # noqa: E402  (build_scene puts the repo on sys.path)

HERE = Path(__file__).resolve().parent
SITE_TEXTURE = "table_texture.png"
# Table-colour test: Lab distance (L weighted 0.6) to the median table colour. Inside the marker box the office table
# spreads ~8 (median) to ~18 (80th percentile: shading, seams); wall, floor, chair and arm are > 40.
TABLE_LAB_TOL = 18.0
EDGE_CELL_M = 0.004       # rectified grid of the edge scans (about one overview pixel at the far edge)
EDGE_RUN = 3              # an edge = the first run of this many visible non-table cells (12 mm: ignores specks)
EDGE_CONFIRM = 25         # ... with mostly non-table in the next this many cells (10 cm: longer than a cup or bag,
                          # shorter than the gap to a light wall behind the table that can pass as table colour)
ARM_CLEAR_M = 0.15        # the far-edge scan skips |x| < this: the arm base and the chair right behind it
PAPER_CLEAR_M = 0.015     # marker paper + this counts as table in the scans (paper edges are not table edges)
SCAN_MARGIN_M = 0.08      # scans start this far inside the marker rows / columns (on the table for sure)
VIEW_PAD_M = 0.5          # rectified area = marker box + this on every side (the overview sees the table edges)
NEAR_MARGIN_M = 0.05      # default near edge: this far past the overview's bottom row (as sim/estimate_home.py)
SEAM_MIN_PEAK = 3.0       # a seam column is this much darker (Lab) than its 9-cell neighbourhood, on average
SEAM_HALF_M = 0.004       # texture band kept around a measured seam (the off-colour inpainting would erase it)
ARM_PAD_PX = 5            # arm silhouette growth (overview px): its soft shadow + the camera's few-px residual
FIT_ROUNDS = 3            # multiplicative colour fits (floor/sky): the render is ~linear in rgba, a few rounds settle
MIN_REGION_PX = 50        # a background region smaller than this is not fitted


def median_image(frames: Path, stem: str):
    """<stem>.jpg, else the per-pixel median of <stem>_*.jpg (None if neither exists)."""
    single = frames / f"{stem}.jpg"
    if single.exists():
        return cv2.imread(str(single))
    files = sorted(frames.glob(f"{stem}_*.jpg"))
    if not files:
        return None
    return np.median(np.stack([cv2.imread(str(f)) for f in files]), axis=0).astype(np.uint8)


def parse_joints(j: dict):
    """(raw {motor id: tick}, ranges {motor id: (lo, hi)}) from either joints.json format."""
    if "raw" in j:
        return {int(k): v for k, v in j["raw"].items()}, {int(k): tuple(v) for k, v in j["ranges"].items()}
    joints = j["joints"]
    return ({int(k[1:]): v["raw"] for k, v in joints.items()},
            {int(k[1:]): (v["lo"], v["hi"]) for k, v in joints.items()})


def shown(p: Path) -> str:
    """p relative to the hackathon data root (worktrees: <hackathon>/ws/<branch>) when inside it, for layout notes."""
    p = Path(p).resolve()
    return str(p.relative_to(DATA_ROOT)) if p.is_relative_to(DATA_ROOT) else str(p)


def load_frames(frames: Path):
    """(overview, wrist or None, raw, ranges, overview source description) of a frames directory."""
    overview = median_image(frames, "overview")
    if overview is None:
        raise FileNotFoundError(f"no overview.jpg or overview_*.jpg in {frames}")
    n = len(list(frames.glob("overview_*.jpg"))) if not (frames / "overview.jpg").exists() else 1
    src = shown(frames / "overview.jpg") if n == 1 else f"{shown(frames)}/overview_*.jpg (per-pixel median of {n})"
    raw, ranges = parse_joints(json.loads((frames / "joints.json").read_text()))
    return overview, median_image(frames, "wrist"), raw, ranges, src


def texture_target(layout: dict, layout_file: Path):
    """(file, layout string) of the table texture: the file the layout already names, else <layout folder>/
    table_texture.png. The string is relative to sim/, as compare_real.load_model resolves it there too."""
    cur = layout["table"].get("texture")
    f = build_scene.table_texture_file(cur, layout_file.parent) if cur else (layout_file.parent / SITE_TEXTURE).resolve()
    return f, os.path.relpath(f, HERE)


def real_camera(model, data, cam_layout, W, H):
    """rvec, tvec, K, dist of the real overview camera as the layout describes it."""
    cid = model.camera("overview_cam").id
    Rcv = data.cam_xmat[cid].reshape(3, 3) @ build_scene.CV_TO_MJ        # camera axes in the sim frame
    R, t = Rcv.T, -Rcv.T @ data.cam_xpos[cid]
    k = cam_layout.get("K_normalized")
    K = (np.array([[k["fx"] * W, 0, k["cx"] * W], [0, k["fy"] * H, k["cy"] * H], [0, 0, 1.0]]) if k
         else compare_real.camera_K(model, cid, W, H))
    return cv2.Rodrigues(R)[0], t, K, np.array(cam_layout.get("dist") or [0, 0, 0, 0, 0], float)


def project_seen(world, rvec, tvec, K, dist, W, H):
    """Pixels of world points (N, 3) and whether the camera really sees them: in front, inside the frame, and within
    the normalised radius of the frame's corners (beyond it the distortion polynomial folds back into the image)."""
    R = cv2.Rodrigues(rvec)[0]
    cam = world @ R.T + np.asarray(tvec).ravel()
    corners = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]], np.float32).reshape(-1, 1, 2)
    r_max = np.linalg.norm(cv2.undistortPoints(corners, K, dist).reshape(-1, 2), axis=1).max()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.linalg.norm(cam[:, :2] / cam[:, 2:3], axis=1)
    uv = cv2.projectPoints(world, rvec, tvec, K, dist)[0].reshape(-1, 2)
    seen = ((cam[:, 2] > 0) & (r < r_max) & (uv[:, 0] >= 1) & (uv[:, 0] < W - 2) & (uv[:, 1] >= 1) & (uv[:, 1] < H - 2))
    return uv.astype(np.float32), seen


def marker_paper_mask(layout, X, Y, clear=PAPER_CLEAR_M):
    """Cells (sim x, y grids) on a marker's paper (+ clear)."""
    mk = layout.get("markers") or {}
    half = mk.get("size_m", 0.1) / 2 + mk.get("paper_margin_m", 0.012) + clear
    out = np.zeros(X.shape, bool)
    for it in mk.get("items", []):
        T = build_scene.marker_pose(it)
        loc = np.stack([X - T[0, 3], Y - T[1, 3]], -1) @ T[:2, :2]
        out |= (np.abs(loc) < half).all(-1)
    return out


def first_off(is_table, visible, run=EDGE_RUN, confirm=EDGE_CONFIRM):
    """Index along a scan line (starting on the table) of the table edge: the first run of `run` visible non-table
    cells after which most of the next `confirm` visible cells are not table either (an object lying on the table is
    not an edge). None if the line does not start on visible table, or leaves the view first."""
    if not (len(is_table) and visible[0] and is_table[0]):
        return None
    for i in range(len(is_table) - run + 1):
        if not visible[i]:
            return None
        if visible[i:i + run].all() and not is_table[i:i + run].any():
            nxt = slice(i, i + confirm)
            if is_table[nxt][visible[nxt]].mean() < 0.5:
                return i
    return None


def robust_line(a, b, rounds=3):
    """b = slope * a + offset, refitted without outliers (> 2.5 median absolute residuals + 4 mm)."""
    keep = np.ones(len(a), bool)
    for _ in range(rounds):
        slope, off = np.polyfit(a[keep], b[keep], 1)
        r = np.abs(b - (slope * a + off))
        keep = r < 2.5 * np.median(r[keep]) + 0.004
    slope, off = np.polyfit(a[keep], b[keep], 1)
    return slope, off, keep


def seam_column(dist_lab, valid, xs, rows, cols):
    """x of the darkest thin line (Lab distance above its 9-cell median) over the given rows/columns, or None."""
    d = np.where(valid, dist_lab, 0).astype(np.float32)
    hp = d - cv2.medianBlur(np.clip(d, 0, 255).astype(np.uint8), 9).astype(np.float32)
    cs = np.flatnonzero(cols)
    prof = np.array([hp[rows, c][valid[rows, c]].mean() if valid[rows, c].any() else 0.0 for c in cs])
    if not len(prof) or prof.max() < SEAM_MIN_PEAK:
        return None
    return float(xs[cs[int(np.argmax(prof))]])


def measure_table(real, layout, model, data, near_edge=None):
    """Table edges in the sim frame from the rectified overview: {'yaw_deg', 'x_min', 'x_max', 'y_far', 'y_near',
    'seam_x', 'measured': [...], 'estimated': [...], ...}; x/y here are along the (yawed) table axes."""
    H, W = real.shape[:2]
    cam_layout = next(c for c in layout["cameras"] if c["name"] == "overview_cam")
    rvec, tvec, K, dist = real_camera(model, data, cam_layout, W, H)
    mk = np.array([build_scene.marker_pose(it)[:2, 3] for it in layout["markers"]["items"]])
    (mx0, my0), (mx1, my1) = mk.min(0), mk.max(0)
    xs = np.arange(mx1 + VIEW_PAD_M, mx0 - VIEW_PAD_M, -EDGE_CELL_M)     # columns: +x first (image left)
    ys = np.arange(my0 - VIEW_PAD_M, my1 + VIEW_PAD_M, EDGE_CELL_M)      # rows: far (-y) first
    X, Y = np.meshgrid(xs, ys)
    uv, seen = project_seen(np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1), rvec, tvec, K, dist, W, H)
    valid = seen.reshape(X.shape)
    uv = uv.reshape(X.shape + (2,))
    rect = cv2.remap(real, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR)
    lab = cv2.cvtColor(cv2.GaussianBlur(rect, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(float)
    paper = marker_paper_mask(layout, X, Y)
    inside = (X > mx0) & (X < mx1) & (Y > my0) & (Y < my1) & valid & ~paper
    d = np.sqrt((((lab - np.median(lab[inside], 0)) * [0.6, 1.0, 1.0]) ** 2).sum(-1))
    table = (d < TABLE_LAB_TOL) | paper

    far = []                                                             # scan from the base markers toward -y
    rows = np.arange(np.searchsorted(ys, my0 + SCAN_MARGIN_M), -1, -1)
    for c in np.flatnonzero(np.abs(xs) >= ARM_CLEAR_M):
        i = first_off(table[rows, c], valid[rows, c])
        if i is not None:
            far.append((xs[c], ys[rows[i]] + EDGE_CELL_M / 2))
    if len(far) < 10:
        raise RuntimeError(f"far table edge not found ({len(far)} columns)")
    far = np.array(far)
    slope, y_far0, keep = robust_line(far[:, 0], far[:, 1])
    yaw = math.atan(slope)
    rot = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])   # sim -> table axes
    to_table = lambda p: np.asarray(p) @ rot.T

    def side(sign):                                                      # scan outward from the marker columns
        pts = []
        start = np.argmin(np.abs(xs - ((mx1 if sign > 0 else mx0) + sign * SCAN_MARGIN_M)))
        cols = np.arange(start, -1, -1) if sign > 0 else np.arange(start, len(xs))
        for r, y in enumerate(ys):
            if y_far0 + EDGE_RUN * EDGE_CELL_M < y < my1 + VIEW_PAD_M:
                i = first_off(table[r, cols], valid[r, cols])
                if i is not None:
                    pts.append((xs[cols[i]] + sign * EDGE_CELL_M / 2, y))
        return np.array(pts).reshape(-1, 2)

    out = {"yaw_deg": round(math.degrees(yaw), 2), "y_far": float(np.median(to_table(far[keep])[:, 1])),
           "far_points": int(keep.sum()), "far_resid_mm": round(float(np.std(far[keep, 1] - slope * far[keep, 0] - y_far0)) * 1000, 1),
           "measured": ["far"], "estimated": []}
    for name, sign in (("x_max", 1), ("x_min", -1)):
        p = side(sign)
        out[f"{name}_points"] = len(p)
        if len(p) >= EDGE_RUN:
            out[name] = float(np.median(to_table(p)[:, 0]))
            out["measured"].append(f"{name} ({len(p)} rows)")
    near_rows = (ys > my1 - SCAN_MARGIN_M) & (ys < my1 + SCAN_MARGIN_M + 0.1)
    between = (xs > mx0 + SCAN_MARGIN_M) & (xs < mx1 - SCAN_MARGIN_M)
    seam = seam_column(d, valid & ~paper, xs, near_rows, between)
    out["seam_x"] = None if seam is None else float(to_table([seam, my1])[0])
    for name, other in (("x_max", "x_min"), ("x_min", "x_max")):
        if name not in out:
            if other in out and out["seam_x"] is not None:
                out[name] = 2 * out["seam_x"] - out[other]
                out["estimated"].append(f"{name} (mirror of {other} about the seam: two equal tables)")
            else:
                out[name] = float(mx1 + 0.1 if name == "x_max" else mx0 - 0.1)
                out["estimated"].append(f"{name} (marker column + 10 cm)")
    bottom = float(np.max(to_table(np.stack([X[valid], Y[valid]], 1))[:, 1]))
    out["overview_bottom_y"] = round(bottom, 3)
    if near_edge is None:
        out["y_near"] = bottom + NEAR_MARGIN_M
        out["estimated"].append(f"y_near (overview bottom row + {NEAR_MARGIN_M} m)")
    else:
        out["y_near"] = float(near_edge)
        out["estimated"].append("y_near (--near-edge)")
    for k in ("x_min", "x_max", "y_far", "y_near", "seam_x"):
        if out.get(k) is not None:
            out[k] = round(out[k], 4)
    return out


def table_from_edges(edges):
    """(size [w, d], center [x, y], yaw deg) of the layout table from measure_table's edges."""
    yaw = math.radians(edges["yaw_deg"])
    cu, cv = (edges["x_min"] + edges["x_max"]) / 2, (edges["y_far"] + edges["y_near"]) / 2
    c = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]) @ [cu, cv]
    return ([round(edges["x_max"] - edges["x_min"], 4), round(edges["y_near"] - edges["y_far"], 4)],
            [round(float(c[0]), 4), round(float(c[1]), 4)], edges["yaw_deg"])


def arm_silhouette(model, data, rvec, tvec, K, dist, W, H, pad_px=ARM_PAD_PX):
    """Pixels of the real overview covered by the twin arm at its current pose (per-geom convex hulls of the mesh
    vertices / box corners, projected with the real camera), grown by pad_px for its soft shadow and calibration."""
    root = model.body_rootid[model.site_bodyid[model.site("grasp_tcp").id]]
    mask = np.zeros((H, W), np.uint8)
    for g in np.flatnonzero(model.body_rootid[model.geom_bodyid] == root):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            m = model.geom_dataid[g]
            v = model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]]
        elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            v = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * model.geom_size[g]
        else:
            continue
        world = v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        R = cv2.Rodrigues(rvec)[0]
        if ((world @ R.T + np.asarray(tvec).ravel())[:, 2] <= 0).any():
            continue
        uv = cv2.projectPoints(world, rvec, tvec, K, dist)[0].reshape(-1, 2)
        cv2.fillConvexPoly(mask, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
    return cv2.dilate(mask, np.ones((2 * pad_px + 1,) * 2, np.uint8)) > 0


def push_pull_fill(img, hole):
    """img with the hole pixels replaced by a smooth multi-scale average of the known pixels around them (pyramid
    push-pull). Unlike marching inpainting (Telea), a thin dark ring at a hole's border (an arm shadow) does not get
    dragged across the whole hole."""
    w = (~hole).astype(np.float32)
    cur, cw, levels = img.astype(np.float32) * w[..., None], w, []
    while min(cw.shape) > 4:
        levels.append((cur, cw))
        cur, cw = cv2.pyrDown(cur), cv2.pyrDown(cw)
    est = cur / np.maximum(cw, 1e-6)[..., None]
    for cur, cw in reversed(levels):
        up = cv2.resize(est, (cw.shape[1], cw.shape[0]), interpolation=cv2.INTER_LINEAR)
        a = np.clip(cw, 0, 1)[..., None]
        est = cur / np.maximum(cw, 1e-6)[..., None] * a + up * (1 - a)
    out = img.copy()
    out[hole] = np.clip(est[hole], 0, 255).astype(np.uint8)
    return out


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
    uv, seen = project_seen(world, rvec, tvec, K, dist, W, H)
    uv, seen = uv.reshape(ny, nx, 2), seen.reshape(ny, nx)
    tex = cv2.remap(real, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    arm = arm_silhouette(model, data, rvec, tvec, K, dist, W, H).astype(np.uint8)
    hole = ~seen | (cv2.remap(arm, uv[..., 0], uv[..., 1], cv2.INTER_NEAREST) > 0)
    # not table: arm, cable, marker paper, wall/floor seen past an edge (Lab distance to the median seen colour)
    lab = cv2.cvtColor(cv2.GaussianBlur(tex, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(float)
    ref = np.median(lab[seen], axis=0)
    off = np.sqrt((((lab - ref) * [0.6, 1.0, 1.0]) ** 2).sum(-1)) > 22
    seam_x, band = layout["table"].get("seam_x"), None
    if seam_x is not None:                       # keep the seam line where no blob (arm, marker, cable) is near it
        band = np.abs(X - (seam_x - (pt @ Rt)[0])) < SEAM_HALF_M        # table axes -> table body frame
        blob_px = int(round(3 * SEAM_HALF_M * 1000 / mm_per_px)) | 1    # opening removes lines thinner than this
        blobs = cv2.morphologyEx(off.astype(np.uint8), cv2.MORPH_OPEN, np.ones((blob_px, blob_px), np.uint8))
        off &= ~(band & (cv2.dilate(blobs, np.ones((3 * blob_px,) * 2, np.uint8)) == 0))
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
    if band is None:
        filled = push_pull_fill(tex, m8 > 0)
    else:                                        # fill without the dark seam (it would bleed), then draw it back
        filled = push_pull_fill(tex, (m8 > 0) | band)
        kept = band & (m8 == 0)
        filled[kept] = tex[kept]
        if kept.any():                           # the seam continues behind the arm and over the unseen table
            filled[band & ~kept] = np.median(tex[kept], axis=0).astype(np.uint8)
    return filled, m8 > 0, {"size_px": [nx, ny], "mm_per_px": mm_per_px, "seen_fraction": round(float(seen.mean()), 3),
                            "filled_fraction": round(float((m8 > 0).mean()), 3)}


def render_overview(model, data, W, H, segment=False):
    opt = mujoco.MjvOption(); opt.geomgroup[1] = 0
    with mujoco.Renderer(model, H, W) as r:
        r.update_scene(data, camera="overview_cam", scene_option=opt)
        img = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
        mask = compare_real.table_mask(model, r, data, "overview_cam", None)
        if not segment:
            return img, mask
        r.enable_segmentation_rendering()
        r.update_scene(data, camera="overview_cam", scene_option=opt)
        seg = r.render()
        r.disable_segmentation_rendering()
    geom = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
    return img, mask, geom


def set_gain(model, base_amb, base_dif, g):
    model.vis.headlight.ambient[:] = np.clip(base_amb * g, 0, 1)
    lid = model.light("key").id
    model.light_diffuse[lid] = np.clip(base_dif * g, 0, 1)


def rgb_mean(img, mask):
    """Mean colour (RGB, 0..1) of the masked pixels of a BGR image (the background mixes bright wall and dark chair:
    the mean, not the median, is what compare_real's colour metric compares)."""
    return img[mask].mean(axis=0)[::-1] / 255.0


def fit_background(layout, layout_file, real, q, W, H):
    """table.floor_rgba and lighting.skybox (top and bottom colour) scaled until the rendered floor / sky pixels have
    the real overview's mean colour there (in the office: chair, carpet and walls behind the table). Edits layout;
    returns {region: pixels}."""
    lt, info = layout["lighting"], {}
    for _ in range(FIT_ROUNDS):
        model, _ = compare_real.load_model_from(layout, layout_file)
        data = mujoco.MjData(model)
        real_joints.set_arm_q(model, data, q)
        img, _, geom = render_overview(model, data, W, H, segment=True)
        floor = cv2.erode((geom == model.geom("floor").id).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        sky = geom < 0
        sky_rows = np.flatnonzero(sky.any(1))
        top = sky & (np.arange(H)[:, None] <= (np.median(sky_rows) if len(sky_rows) else -1))
        info = {"floor_px": int(floor.sum()), "sky_px": int(sky.sum())}
        scale = lambda c, m: np.round(np.clip(np.asarray(c, float) * rgb_mean(real, m) / np.maximum(rgb_mean(img, m), 1e-3), 0, 1), 3).tolist()
        if floor.sum() >= MIN_REGION_PX:
            layout["table"]["floor_rgba"] = [*scale(layout["table"].get("floor_rgba", [0.45, 0.45, 0.47, 1])[:3], floor), 1]
        if top.sum() >= MIN_REGION_PX and (sky & ~top).sum() >= MIN_REGION_PX:
            sb = lt.get("skybox", [[0.86, 0.86, 0.84], [0.55, 0.55, 0.55]])
            lt["skybox"] = [scale(sb[0], top), scale(sb[1], sky & ~top)]
    return info


def compare_image(layout_file, real, wrist, raw, ranges, out_file: Path):
    """compare_real's per-camera metrics (marker corners, table colour) + real | twin | blend of the overview."""
    with tempfile.TemporaryDirectory() as tmp:
        frames = {"overview": real, "wrist": wrist if wrist is not None else np.zeros_like(real)}
        m = compare_real.compare(layout_file, frames, raw, ranges, Path(tmp))
    model, layout = compare_real.load_model(layout_file)
    data = mujoco.MjData(model)
    real_joints.set_arm_q(model, data, real_joints.to_q(raw, ranges, *model.actuator_ctrlrange.T, real_joints.load_mapping(layout)))
    twin = render_overview(model, data, real.shape[1], real.shape[0])[0]
    tiles = [real, twin, cv2.addWeighted(real, 0.5, twin, 0.5, 0)]
    big = [cv2.resize(t, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC) for t in tiles]
    row = np.hstack(big)
    for k, txt in enumerate(("real (median)", "twin", "50% blend")):
        cv2.putText(row, txt, (8 + k * big[0].shape[1], 20), 0, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(row, txt, (8 + k * big[0].shape[1], 20), 0, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), row, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {k: v for k, v in m.items() if isinstance(v, dict) and "table_colour" in v
            and (wrist is not None or not k.startswith("wrist"))}


def dump_like(text: str, obj: dict) -> str:
    """obj as JSON in the style of an existing layout file: json indent 1 (the office layout as synced) or
    sync_from_calib.dumps (home), so a rewrite only shows the changed fields in a diff."""
    end = "\n" if text.endswith("\n") else ""
    if text.rstrip("\n") == json.dumps(json.loads(text), indent=1):
        return json.dumps(obj, indent=1) + end
    return sync_from_calib.dumps(obj) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", help="layout name or file (default: the selected site's layout)")
    ap.add_argument("--frames", type=Path, default=HERE / "layouts" / "home_estimate")
    ap.add_argument("--mm-per-px", type=float, default=2.0)
    ap.add_argument("--ambient-share", type=float, default=0.75, help="share of the light that is ambient (rest: key light)")
    ap.add_argument("--measure-table", action="store_true", help="table size/center/yaw from the rectified overview")
    ap.add_argument("--near-edge", type=float, help="near table edge (sim y, m; the overview does not see it)")
    ap.add_argument("--background", action="store_true", help="fit floor and skybox colours too")
    ap.add_argument("--rest-pose", action="store_true", help="set rest_pose to the captured arm pose")
    ap.add_argument("--compare", type=Path, help="write real | twin | blend of the overview here after the fit")
    ap.add_argument("--compare-only", action="store_true", help="only --compare the current layout (no fit)")
    a = ap.parse_args()
    layout_file = build_scene.layout_path(a.layout)
    layout = json.loads(layout_file.read_text())
    real, wrist, raw, ranges, src = load_frames(a.frames)
    H, W = real.shape[:2]
    if a.compare_only:
        print(json.dumps(compare_image(layout_file, real, wrist, raw, ranges, a.compare or Path(tempfile.gettempdir()) / "appearance_compare.jpg"), indent=1))
        return
    out = json.loads(layout_file.read_text())                # written back: only the fields this script owns
    tex_path, tex_name = texture_target(layout, layout_file)
    layout["table"].pop("texture", None)                     # build without a texture so the table frame is current
    model, _ = compare_real.load_model_from(layout, layout_file)
    data = mujoco.MjData(model)
    q = real_joints.to_q(raw, ranges, *model.actuator_ctrlrange.T, real_joints.load_mapping(layout))
    real_joints.set_arm_q(model, data, q)

    # 0. table extent
    if a.measure_table:
        edges = measure_table(real, layout, model, data, a.near_edge)
        size, center, yaw = table_from_edges(edges)
        note = (f"fit_appearance --measure-table from {src}: far/side edges = first non-table-coloured cells on the "
                f"rectified overview (table axes, yaw from the far edge, {edges['far_points']} columns, "
                f"{edges['far_resid_mm']} mm rms); measured {', '.join(edges['measured'])}; estimated "
                f"{'; '.join(edges['estimated'])}; overview sees to y = {edges['overview_bottom_y']} m")
        tbl = {"size": size, "center": center, "yaw": yaw, "seam_x": edges["seam_x"], "src": note,
               "edges": {k: edges[k] for k in ("x_min", "x_max", "y_far", "y_near", "seam_x")}}
        for k in ("center_x", "back_edge_y"):
            layout["table"].pop(k, None); out["table"].pop(k, None)
        layout["table"].update(tbl); out["table"].update(tbl)
        print(f"table edges {json.dumps(edges)}\n  -> size {size} center {center} yaw {yaw}")
        model, _ = compare_real.load_model_from(layout, layout_file)
        data = mujoco.MjData(model)
        real_joints.set_arm_q(model, data, q)

    # 1. texture
    tex, holes, meta = make_texture(layout, model, data, real, a.mm_per_px)
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
    layout["table"].update(texture=tex_name, specular=0.05)
    sh = a.ambient_share
    layout["lighting"] = {"headlight": {"ambient": [sh] * 3, "diffuse": [0, 0, 0], "specular": [0, 0, 0]},
                          "skybox": layout["lighting"].get("skybox", [[0.75, 0.75, 0.74], [0.45, 0.43, 0.40]]),
                          "lights": [{"name": "key", "directional": True, "pos": [0.0, 0.3, 2.0], "dir": [0.1, 0.2, -1.0],
                                      "diffuse": [1 - sh] * 3, "specular": [0.03] * 3, "castshadow": True}]}
    model, _ = compare_real.load_model_from(layout, layout_file)
    data = mujoco.MjData(model)
    real_joints.set_arm_q(model, data, q)
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
    layout["table"]["texture_meta"] = {**meta, "src": "fit_appearance", "from": src}
    out["table"].update(texture=layout["table"]["texture"], specular=0.05, texture_meta=layout["table"]["texture_meta"])

    # 4. background colours
    if a.background:
        info = fit_background(layout, layout_file, real, q, W, H)
        out["table"]["floor_rgba"] = layout["table"]["floor_rgba"]
        print(f"background: floor {layout['table']['floor_rgba']}, skybox {lt['skybox']} ({info})")
        lt["_about"] += f"; floor and skybox colours = mean real overview colour where the twin shows them ({info})"
    out["lighting"] = lt
    out["markers"]["levels"] = levels

    # 5. rest pose
    if a.rest_pose:
        out["rest_pose"] = [round(float(v), 3) for v in q]
        out["_rest_pose_about"] = f"the real follower's pose during the overview frames ({shown(a.frames / 'joints.json')})"
        print(f"rest_pose {out['rest_pose']}")
    layout_file.write_text(dump_like(layout_file.read_text(), out))
    print("updated", layout_file)
    if a.compare:
        print(json.dumps(compare_image(layout_file, real, wrist, raw, ranges, a.compare), indent=1))
        print("wrote", a.compare)


if __name__ == "__main__":
    main()
