"""Estimate the home table geometry from the real cameras (until perception's scene_calib.json exists).

  python3 sim/estimate_home.py [--frames DIR] [--n 5] [--wrist-fovy DEG] [--no-wrist-pitch]

1. Grabs overview + wrist frames and the follower's raw ticks through Station (or reads DIR/overview.jpg,
   DIR/wrist.jpg, DIR/joints.json), and saves them to sim/layouts/home_estimate/.
2. Overview camera: one shared focal length (+ k1) and one table plane, fitted jointly to every detected ArUco
   marker's corners (each marker alone is too small for a stable pose at 299x224). Gives the marker layout on the
   table plane (frame P: origin = first far marker, x along the table, z up).
3. Arm: the arm's foot in the overview gives the base position (prior, ~1 cm); the twin's wrist camera at the measured
   joints (FK, real_joints mapping, fovy = the overview's: same KD-USB camera model) must see the markers where the
   real wrist camera sees them, which fixes the base yaw and a small tilt of the wrist camera vs its CAD mount.
   (--jaw-pitch also reports the tilt implied by the jaw silhouette; on 22 Nov it disagreed with the markers.)
4. Table edges from the rectified overview (left/right/far) and the wrist view (near edge).
5. Writes sim/layouts/home_estimate/scene_calib_estimate.json in the scene-contract format and applies it to
   sim/layouts/home.json with src "estimate" (sync_from_calib never lets an estimate replace a scene_calib value).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_scene  # noqa: E402  (sets MUJOCO_GL)
import mujoco  # noqa: E402
import real_joints  # noqa: E402
import sync_from_calib  # noqa: E402

HERE = Path(__file__).resolve().parent
from sites import get_site
OUT = get_site().directory / "estimate"
DICT, SIZE = "DICT_4X4_50", 0.10
ROT = lambda a: np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])


def marker_obj(s=SIZE):
    return np.array([[-s / 2, s / 2, 0], [s / 2, s / 2, 0], [s / 2, -s / 2, 0], [-s / 2, -s / 2, 0]])


def corners_on_plane(x, y, yaw):
    return marker_obj() @ ROT(yaw).T + [x, y, 0]


def detect(img, dictionary=DICT):
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)),
                                  cv2.aruco.DetectorParameters())
    c, ids, rej = det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    found = {} if ids is None else {int(i): cc.reshape(4, 2).astype(float) for i, cc in zip(ids.ravel(), c)}
    return found, [r.reshape(4, 2).astype(float) for r in rej]


def match_rejected(rejected, predicted, tol=12.0):
    """A rejected candidate square (e.g. id bits hidden under a cable) at a predicted marker position, corners in
    the predicted order; None if nothing fits within tol px per corner."""
    best = None
    for cand in rejected:
        for k in range(4):
            c = np.roll(cand, -k, axis=0)
            e = np.linalg.norm(c - predicted, axis=1).max()
            if e < tol and (best is None or e < best[0]):
                best = (e, c)
    return None if best is None else best[1]


def lm(fun, p, iters=200):
    """Small Levenberg-Marquardt with a numeric Jacobian."""
    lam = 1e-3
    for _ in range(iters):
        r0 = fun(p)
        J = np.empty((len(r0), len(p)))
        for k in range(len(p)):
            dp = np.zeros(len(p)); dp[k] = 1e-6 * max(1.0, abs(p[k]))
            J[:, k] = (fun(p + dp) - r0) / dp[k]
        A, g = J.T @ J, J.T @ r0
        while True:
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -g)
            if (fun(p + step) ** 2).sum() < (r0 ** 2).sum():
                p, lam = p + step, lam * 0.3
                break
            lam *= 10
            if lam > 1e8:
                return p
        if np.abs(step).max() < 1e-10:
            break
    return p


# --- 2. overview camera + marker layout --------------------------------------------------------------------------

def fit_overview(C: dict, W: int, H: int) -> dict:
    ref = min(i for i in C if i in (22, 23)) if any(i in C for i in (22, 23)) else min(C)
    others = sorted(i for i in C if i != ref)

    def unpack(p):
        mk = {ref: (0.0, 0.0, 0.0), **{i: tuple(p[8 + 3 * j: 11 + 3 * j]) for j, i in enumerate(others)}}
        return p[0], p[1], p[2:5], p[5:8], mk

    def resid(p):
        f, k1, rv, tv, mk = unpack(p)
        K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
        out = [(cv2.projectPoints(corners_on_plane(*mk[i]), rv, tv, K, np.array([k1, 0, 0, 0, 0.0]))[0].reshape(-1, 2)
                - C[i]).ravel() for i in C]
        return np.concatenate(out + [[2.0 * k1]])          # weak prior: little distortion

    best = None
    for f0 in (130.0, 180.0, 250.0):
        K = np.array([[f0, 0, W / 2], [0, f0, H / 2], [0, 0, 1.0]])
        _, rv, tv = cv2.solvePnP(marker_obj(), C[ref], K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        R = cv2.Rodrigues(rv)[0]
        p0 = [f0, 0.0, *rv.ravel(), *tv.ravel()]
        for i in others:
            _, rvi, tvi = cv2.solvePnP(marker_obj(), C[i], K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            q, Ri = R.T @ (tvi.ravel() - tv.ravel()), R.T @ cv2.Rodrigues(rvi)[0]
            p0 += [q[0], q[1], math.atan2(Ri[1, 0], Ri[0, 0])]
        p = lm(resid, np.array(p0))
        err = float(np.linalg.norm(resid(p)[:-1].reshape(-1, 2), axis=1).mean())
        if best is None or err < best[0]:
            best = (err, p)
    err, p = best
    f, k1, rv, tv, mk = unpack(p)
    return {"f": float(f), "k1": float(k1), "rvec": rv, "tvec": tv, "markers_P": mk, "ref": ref, "err_px": err,
            "K": np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]]), "D": np.array([k1, 0, 0, 0, 0.0]), "W": W, "H": H}


def project_overview(fit, pts_P):
    return cv2.projectPoints(np.asarray(pts_P, float), fit["rvec"], fit["tvec"], fit["K"], fit["D"])[0].reshape(-1, 2)


# --- 3. arm base from the wrist view -----------------------------------------------------------------------------

class WristModel:
    """The twin arm with a wrist camera we can re-mount (pitch offset vs the CAD pose) and re-lens (fovy)."""

    def __init__(self, layout_file, layout):
        xml = build_scene.generated_dir(layout_file) / "workspace.xml"
        if not xml.exists():
            build_scene.build(layout_file, xml.parent, report=False)
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(self.m)
        self.lo, self.hi = self.m.actuator_ctrlrange.T
        self.cid = self.m.camera("wrist_cam").id
        w = layout["wrist_camera"]
        self.T_link = np.array(w["T_link_cam"] if "T_link_cam" in w else
                               json.loads((build_scene.REPO / w["pose_file"]).read_text())["T_link_camera"])
        gripper = self.m.body("Gripper_Base_v1_1").id
        self.gripper_geoms = np.array([self._under(self.m.geom_bodyid[g], gripper) for g in range(self.m.ngeom)])

    def _under(self, b, root):
        while b:
            if b == root:
                return True
            b = self.m.body_parentid[b]
        return False

    def set_camera(self, pitch_deg, fovy):
        a = math.radians(pitch_deg)
        R = self.T_link[:3, :3] @ np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (R @ build_scene.CV_TO_MJ).flatten())
        self.m.cam_pos[self.cid], self.m.cam_quat[self.cid], self.m.cam_fovy[self.cid] = self.T_link[:3, 3], quat, fovy

    def cam_pose(self, q):
        """(position, R with OpenCV axes as columns) of the wrist camera in the sim frame."""
        real_joints.set_arm_q(self.m, self.d, q)
        return self.d.cam_xpos[self.cid].copy(), self.d.cam_xmat[self.cid].reshape(3, 3) @ build_scene.CV_TO_MJ

    def jaw_mask(self, q, W, H):
        """Pixels of the wrist image covered by the gripper (jaws), lower half only."""
        real_joints.set_arm_q(self.m, self.d, q)
        opt = mujoco.MjvOption(); opt.geomgroup[1] = 0
        with mujoco.Renderer(self.m, H, W) as r:
            r.enable_segmentation_rendering()
            r.update_scene(self.d, camera="wrist_cam", scene_option=opt)
            seg = r.render()
        ok = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[..., 0] >= 0)
        out = np.zeros((H, W), bool)
        out[ok] = self.gripper_geoms[seg[..., 0][ok]]
        out[:H // 2] = False
        return out


def fit_wrist_pitch(wm, q, real, fovy, lo=-15, hi=25):
    """Tilt of the real wrist camera vs the CAD mount from the jaw silhouette (independent of the arm pose and the
    scene: the jaws are rigid to the camera except for the j8 opening, taken from the joint mapping)."""
    H, W = real.shape[:2]
    dark = cv2.cvtColor(real, cv2.COLOR_BGR2HSV)[..., 2] < 95
    dark[:H // 2] = False
    scores = []
    for p in np.arange(lo, hi + 0.1, 1.0):
        wm.set_camera(p, fovy)
        t = wm.jaw_mask(q, W, H)
        scores.append(((t & dark).sum() / max((t | dark).sum(), 1), float(p)))
    wm.set_camera(0.0, fovy)
    t0 = wm.jaw_mask(q, W, H)
    iou0 = (t0 & dark).sum() / max((t0 | dark).sum(), 1)
    best = max(scores)
    return best[1], float(best[0]), float(iou0)


def p_to_s(base, pts_P):
    """Table-plane frame P -> sim frame S, base = (x, y, yaw) of the arm base frame in P."""
    return (np.asarray(pts_P, float) - [base[0], base[1], 0]) @ ROT(base[2])


def project_wrist(base, pts_P, cpos, Rcv, f, W, H):
    pc = (p_to_s(base, pts_P) - cpos) @ Rcv
    return pc[:, :2] / pc[:, 2:3] * f + [W / 2, H / 2], pc[:, 2]


def rot_x(a):
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def fit_base(Cw, rejected, markers_P, cpos, Rcv, f, W, H, prior_xy=None, prior_w=300.0, pitch0=0.0):
    """x = (base x, base y, base yaw in P, wrist camera tilt about its x axis vs the given mount, rad).
    prior_xy: the arm foot seen by the overview (prior_w px per metre); the tilt has a weak prior (1 px per degree)."""
    def resid(x, Cw):
        R = Rcv @ rot_x(x[3])
        out = [(project_wrist(x, corners_on_plane(*markers_P[i]), cpos, R, f, W, H)[0] - px).ravel() for i, px in Cw.items()]
        if prior_xy is not None:
            out.append(prior_w * (np.asarray(x[:2]) - prior_xy))
        out.append([math.degrees(x[3] - pitch0)])
        return np.concatenate(out)

    best = None
    for th0 in np.radians(np.arange(0, 360, 30)):
        x0 = np.array([*(prior_xy if prior_xy is not None else [0.3, 0.05]), th0, pitch0])
        x = lm(lambda x: resid(x, Cw), x0, 80)
        z = np.concatenate([project_wrist(x, corners_on_plane(*markers_P[i]), cpos, Rcv @ rot_x(x[3]), f, W, H)[1] for i in Cw])
        e = float(np.abs(resid(x, Cw)[:8 * len(Cw)]).mean())
        if (z > 0).all() and (best is None or e < best[0]):
            best = (e, x)
    e, x = best
    # Markers whose id bits are hidden (cable): pick up the bare square where the fit predicts it, then refit.
    extra = {}
    for i, m in markers_P.items():
        if i in Cw:
            continue
        pr, z = project_wrist(x, corners_on_plane(*m), cpos, Rcv @ rot_x(x[3]), f, W, H)
        if (z > 0).all() and (pr > 0).all() and (pr < [W, H]).all():
            c = match_rejected(rejected, pr, 15.0)
            if c is not None:
                extra[i] = c
    if extra:
        x = lm(lambda x: resid(x, {**Cw, **extra}), x, 80)
        e = float(np.abs(resid(x, {**Cw, **extra})[:8 * (len(Cw) + len(extra))]).mean())
    return x, e, sorted(extra)


def px_to_plane(fit, u, v):
    """Overview pixel -> point on the table plane (frame P)."""
    n = cv2.undistortPoints(np.array([[[u, v]]], float), fit["K"], fit["D"]).reshape(2)
    R = cv2.Rodrigues(fit["rvec"])[0]
    c, ray = -R.T @ fit["tvec"].ravel(), R.T @ np.array([n[0], n[1], 1.0])
    return c + ray * (-c[2] / ray[2])


def arm_foot(img, fit, markers_P, front_m):
    """Arm base origin in P from the overview: the dark arm between the two far markers; its lowest image row is the
    front edge of the base on the table, front_m ahead of the base origin (towards the camera)."""
    far = sorted(markers_P, key=lambda i: -markers_P[i][1])[:2]
    c = project_overview(fit, [[*markers_P[i][:2], 0] for i in far])
    u0, u1 = sorted(c[:, 0]); span = u1 - u0
    v_lo, v_hi = int(c[:, 1].min()) - 40, int(c[:, 1].max()) + 25
    roi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[max(v_lo, 0):v_hi, int(u0 + 0.2 * span):int(u1 - 0.2 * span), 2] < 60
    rows = np.where(roi.sum(1) >= 3)[0]
    if not len(rows):
        return None
    vb = rows.max()
    us = np.where(roi[max(vb - 2, 0):vb + 1].any(0))[0]
    u, v = int(u0 + 0.2 * span) + float(np.median(us)), max(v_lo, 0) + vb + 0.5
    p = px_to_plane(fit, u, v)
    cam = -cv2.Rodrigues(fit["rvec"])[0].T @ fit["tvec"].ravel()
    fwd = (cam[:2] - p[:2]) / np.linalg.norm(cam[:2] - p[:2])
    return p[:2] - front_m * fwd, (u, v)


# --- 4. table edges ----------------------------------------------------------------------------------------------

def wood_mask(img, ref_px):
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(float)
    ref = np.median(np.array([lab[int(v), int(u)] for u, v in ref_px]), axis=0)
    d = np.sqrt((((lab - ref) * [0.5, 1.0, 1.0]) ** 2).sum(-1))
    return d < 16, ref


def overview_edges(img, fit, markers_P, res=0.004):
    """Left/right/far table edges in P from the rectified overview (wood-coloured fraction crossing 0.5)."""
    xs = np.array([v[0] for v in markers_P.values()]); ys = np.array([v[1] for v in markers_P.values()])
    ref_pts = [[x, y, 0] for x in np.linspace(xs.min() + 0.08, xs.max() - 0.08, 6) for y in np.linspace(ys.min() + 0.08, ys.max() - 0.08, 4)]
    mask, _ = wood_mask(img, project_overview(fit, ref_pts))
    X, Y = np.meshgrid(np.arange(xs.min() - 0.4, xs.max() + 0.4, res), np.arange(ys.max() + 0.3, ys.min() - 0.4, -res))
    uv = project_overview(fit, np.stack([X.ravel(), Y.ravel(), 0 * X.ravel()], 1)).reshape(X.shape + (2,))
    inside = (uv[..., 0] >= 0) & (uv[..., 0] < fit["W"] - 1) & (uv[..., 1] >= 0) & (uv[..., 1] < fit["H"] - 1)
    m = np.zeros(X.shape, bool)
    m[inside] = mask[uv[..., 1][inside].astype(int), uv[..., 0][inside].astype(int)]
    out = {}
    band_y = (Y[:, 0] < ys.max() - 0.06) & (Y[:, 0] > ys.min() + 0.06)        # rows between the marker rows
    for side, sl in (("x_min", slice(None, None, -1)), ("x_max", slice(None))):
        x0 = xs.min() if side == "x_min" else xs.max()
        cols = np.where((X[0] <= x0) if side == "x_min" else (X[0] >= x0))[0][sl]
        for c in cols:
            vis = inside[band_y, c]
            if vis.sum() > 5 and m[band_y, c][vis].mean() < 0.5:
                out[side] = float(X[0, c]); break
    span = xs.max() - xs.min()
    band_x = ((X[0] > xs.min() + 0.06) & (X[0] < xs.min() + 0.3 * span)) | ((X[0] < xs.max() - 0.06) & (X[0] > xs.max() - 0.3 * span))
    for r_ in np.where(Y[:, 0] >= ys.max())[0][::-1]:
        vis = inside[r_, band_x]
        if vis.sum() > 5 and m[r_, band_x][vis].mean() < 0.5:
            out["y_max"] = float(Y[r_, 0]); break
    return out


def wrist_near_edge(img, base, cpos, Rcv, f, markers_P, W, H):
    """Near table edge (min y in P) from the wrist view: per column, the strongest horizontal intensity edge in the
    band above the markers (colour alone fails: a white cloth behind the table looks like the wood)."""
    ref = []
    for i in markers_P:
        p, z = project_wrist(base, [[*markers_P[i][:2], 0]], cpos, Rcv, f, W, H)
        if z[0] > 0 and 0 < p[0][0] < W and 0 < p[0][1] < H:
            ref.append(p[0])
    if not ref:
        return None
    vmax = int(min(p[1] for p in ref)) - 10
    gy = np.abs(cv2.Sobel(cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(float), cv2.CV_64F, 0, 1, ksize=3))
    ys = []
    for u in list(range(10, int(W * 0.42))) + list(range(int(W * 0.58), W - 10)):
        lo_v = max(vmax - 45, 3)
        strong = np.where(gy[lo_v:vmax, u] > 30)[0]
        if not len(strong):
            continue
        v = lo_v + int(strong[-1])                       # the first strong edge going up from the markers
        ray = Rcv @ np.array([(u - W / 2) / f, (v - H / 2) / f, 1.0])
        if ray[2] < -1e-3:
            pS = cpos + ray * (-cpos[2] / ray[2])
            ys.append(pS @ ROT(base[2]).T + [base[0], base[1], 0])
    if len(ys) < 20:
        return None
    return float(np.median(np.array(ys)[:, 1]))


# --- 5. contract-format output -----------------------------------------------------------------------------------

def scene_calib(fit, base, markers_P, edges, wrist_fovy, raw, ranges) -> dict:
    T_S_P = np.eye(4); T_S_P[:3, :3] = ROT(base[2]).T; T_S_P[:3, 3] = -ROT(base[2]).T @ [base[0], base[1], 0]
    T_W_S = np.linalg.inv(sync_from_calib.RZ90)
    T_W_P = T_W_S @ T_S_P
    Rc = cv2.Rodrigues(fit["rvec"])[0]
    T_P_cam = np.eye(4); T_P_cam[:3, :3] = Rc.T; T_P_cam[:3, 3] = -Rc.T @ fit["tvec"]
    W, H, f = fit["W"], fit["H"], fit["f"]
    items = []
    for i, (x, y, yaw) in sorted(markers_P.items()):
        T = np.eye(4); T[:3, :3] = ROT(yaw); T[:3, 3] = [x, y, 0]
        items.append({"id": i, "role": "base" if y > -0.2 else "top", "T_world_marker": sync_from_calib.r(T_W_P @ T)})
    (x0, x1), (y0, y1) = (edges["x_min"], edges["x_max"]), (edges["y_min"], edges["y_max"])
    c_W = T_W_P @ [(x0 + x1) / 2, (y0 + y1) / 2, 0, 1]
    return {
        "version": 1, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "src": "estimate (sim/estimate_home.py)",
        "world_frame": "follower base_link origin on table plane, z up, x forward at j1=0",
        "table": {"plane_world": [0, 0, 1, 0], "height_m": 0.0,
                  "center_xy": sync_from_calib.r(c_W[:2], 4), "size_xy": sync_from_calib.r([x1 - x0, y1 - y0], 4),
                  "yaw_deg": round(math.degrees(math.atan2(T_W_P[1, 0], T_W_P[0, 0])), 3),
                  "edges_P": edges},
        "markers": {"dictionary": DICT, "size_m": SIZE, "items": items},
        "cameras": {
            "overview": {"queue": get_site().queue("overview", "TWIN"), "image_size": [W, H],
                         "K_normalized": {"fx": f / W, "fy": f / H, "cx": 0.5, "cy": 0.5}, "dist": [fit["k1"], 0, 0, 0, 0],
                         "T_world_cam": sync_from_calib.r(T_W_P @ T_P_cam), "reproj_px": round(fit["err_px"], 3),
                         "fovy_deg": round(math.degrees(2 * math.atan(H / 2 / f)), 3)},
            "wrist": {"queue": get_site().queue("wrist", "TWIN"), "image_size": [W, H],
                      "fovy_deg": round(wrist_fovy, 3), "link": "Gripper_Base_v1_1",
                      "source": "CAD (camera_pose/wrist_camera.json)"}},
        "arm": {"bus": "5B61033301", "urdf": "norma-core/hardware/elrobot/simulation/elrobot_follower.urdf",
                "joint_mapping": "range_fraction (sim/real_joints.py)", "T_world_base": np.eye(4).tolist(),
                "raw_ticks_at_estimate": raw, "ranges_at_estimate": ranges},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=Path, help="use saved overview.jpg/wrist.jpg/joints.json instead of Station")
    ap.add_argument("--n", type=int, default=5, help="overview frames averaged for the marker corners")
    ap.add_argument("--wrist-fovy", type=float, help="wrist camera vertical fov (default: the overview's fitted fovy)")
    ap.add_argument("--jaw-pitch", action="store_true", help="also report the wrist tilt implied by the jaw silhouette")
    ap.add_argument("--layout", type=Path, default=get_site().layout)
    ap.add_argument("--ids", type=int, nargs="+", help="table marker ids (default 20 21 22 23)")
    ap.add_argument("--near-margin", type=float, default=0.05,
                    help="table assumed to run this far past the overview's bottom row (m; the near edge is not seen)")
    ap.add_argument("--no-apply", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.frames:
        ovs = [cv2.imread(str(a.frames / "overview.jpg"))]; wr = cv2.imread(str(a.frames / "wrist.jpg"))
        j = json.loads((a.frames / "joints.json").read_text())
        raw, ranges = {int(k): v for k, v in j["raw"].items()}, {int(k): tuple(v) for k, v in j["ranges"].items()}
    else:
        import station_io

        async def grab():
            cli = await station_io.connect()
            jf = station_io.JointFollower(cli)
            ov, w = await asyncio.gather(station_io.grab_jpegs(cli, "overview", a.n), station_io.grab_jpegs(cli, "wrist", 1))
            return ov, w, await jf.read()
        ov_j, w_j, (raw, meta) = asyncio.run(grab())
        ranges = meta["ranges"]
        ovs = [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for b in ov_j]
        wr = cv2.imdecode(np.frombuffer(w_j[0], np.uint8), cv2.IMREAD_COLOR)
    H, W = ovs[0].shape[:2]
    layout = json.loads(a.layout.read_text())
    expected = set(a.ids) if a.ids else {20, 21, 22, 23}
    dets = [detect(o)[0] for o in ovs]
    seen = {i: [d[i] for d in dets if i in d] for i in expected}
    C = {i: np.mean(v, axis=0) for i, v in seen.items() if len(v) >= max(1, len(ovs) // 2)}
    best_frame = max(range(len(ovs)), key=lambda k: len(set(dets[k]) & expected))
    if len(C) < 3:
        raise SystemExit(f"overview sees only markers {sorted(C)} of {sorted(expected)} (occluded?); try again or use --frames")
    fit = fit_overview(C, W, H)
    mk = fit["markers_P"]
    missing = []
    if len(mk) == 3:     # 4th marker of the rectangle hidden (e.g. a cable over its id bits): parallelogram guess
        # the right-angle vertex is the one closest to the other two; the hidden marker is opposite it
        corner = min(mk, key=lambda i: sum(np.hypot(mk[i][0] - mk[k][0], mk[i][1] - mk[k][1]) for k in mk))
        o1, o2 = [k for k in mk if k != corner]
        guess = (mk[o1][0] + mk[o2][0] - mk[corner][0], mk[o1][1] + mk[o2][1] - mk[corner][1],
                 math.atan2(np.mean([math.sin(mk[k][2]) for k in mk]), np.mean([math.cos(mk[k][2]) for k in mk])))
        new_id = (expected - set(mk)).pop() if len(expected - set(mk)) == 1 else None
        if new_id is not None:
            c = match_rejected(detect(ovs[best_frame])[1], project_overview(fit, corners_on_plane(*guess)), 10.0)
            if c is not None:      # the bare square is visible: refit with it
                fit = fit_overview({**C, new_id: c}, W, H); mk = fit["markers_P"]
            else:
                mk[new_id] = guess; missing.append(new_id)
    print(f"overview: f={fit['f']:.1f}px fovy={math.degrees(2 * math.atan(H / 2 / fit['f'])):.1f} k1={fit['k1']:.3f} "
          f"reproj={fit['err_px']:.2f}px markers(P)={ {i: np.round(v, 3).tolist() for i, v in mk.items()} } guessed={missing}")

    wm = WristModel(a.layout, layout)
    q = real_joints.to_q(raw, ranges, wm.lo, wm.hi, real_joints.load_mapping(layout))
    wcal = layout["wrist_camera"].get("src") == "scene_calib"
    # Same KD-USB camera model as the overview: unless calibrated, the wrist lens gets the overview's fitted fovy.
    wrist_fovy = a.wrist_fovy or (layout["wrist_camera"]["fovy"] if wcal else math.degrees(2 * math.atan(H / 2 / fit["f"])))
    iou = iou0 = jaw_pitch = None
    if a.jaw_pitch and not wcal:      # diagnostic: tilt from the jaw silhouette (disagreed with the markers on 22 Nov)
        jaw_pitch, iou, iou0 = fit_wrist_pitch(wm, q, wr, wrist_fovy)
        print(f"jaw silhouette: tilt vs CAD mount {jaw_pitch:+.0f} deg (IoU {iou0:.2f} -> {iou:.2f})")
    wm.set_camera(0.0, wrist_fovy)
    cpos, Rcv = wm.cam_pose(q)
    f_w = H / 2 / math.tan(math.radians(wrist_fovy) / 2)
    Cw, rej_w = detect(wr)
    Cw = {i: c for i, c in Cw.items() if i in mk}
    base_front = float(layout.get("arm", {}).get("base_front_m", 0.057))   # base_link mesh: y from -0.011 to 0.057
    foot = arm_foot(ovs[best_frame], fit, mk, base_front)
    if foot is None:       # fall back: midway between the two far markers, weakly
        far = sorted(mk, key=lambda i: -mk[i][1])[:2]
        prior, prior_w = np.array([(mk[far[0]][0] + mk[far[1]][0]) / 2, (mk[far[0]][1] + mk[far[1]][1]) / 2]), 30.0
    else:
        (prior, foot_px), prior_w = foot, 300.0
        print(f"arm foot in the overview at px {np.round(foot_px, 1).tolist()} -> base origin prior {np.round(prior, 3).tolist()} (P)")
    if not Cw:
        raise SystemExit("wrist camera sees no marker: cannot place the arm (point the wrist at a marker)")
    base, err_w, extra = fit_base(Cw, rej_w, mk, cpos, Rcv, f_w, W, H, prior, prior_w)
    pitch = 0.0 if wcal else round(math.degrees(base[3]), 2)
    print(f"arm base in P: ({base[0]:.3f}, {base[1]:.3f}) yaw {math.degrees(base[2]) % 360:.1f} deg; wrist tilt vs mount "
          f"{math.degrees(base[3]):+.1f} deg; wrist reproj {err_w:.2f}px from markers {sorted(Cw)} + bare squares {extra}; "
          f"wrist fovy {wrist_fovy:.1f}; q={np.round(q, 3).tolist()}")
    if not wcal:
        wm.set_camera(pitch, wrist_fovy); cpos, Rcv = wm.cam_pose(q)

    edges = overview_edges(ovs[best_frame], fit, mk)
    # Near edge: the wood seen by the wrist camera, but at least past the overview's bottom row (the real table runs
    # out of that frame; a cloth lying on the table end limits what the wrist camera sees as wood).
    bottom = min(px_to_plane(fit, u, H - 1)[1] for u in (W * 0.3, W * 0.5, W * 0.7))
    wrist_edge = wrist_near_edge(wr, base, cpos, Rcv, f_w, mk, W, H)
    edges["y_min"] = min(wrist_edge if wrist_edge is not None else bottom, bottom - a.near_margin)
    defaults = {"x_min": min(v[0] for v in mk.values()) - 0.1, "x_max": max(v[0] for v in mk.values()) + 0.1,
                "y_max": max(v[1] for v in mk.values()) + 0.08, "y_min": min(v[1] for v in mk.values()) - 0.15}
    guessed_edges = [k for k in defaults if edges.get(k) is None]
    edges = {k: round(float(edges[k] if edges.get(k) is not None else defaults[k]), 4) for k in defaults}
    print(f"table edges in P: {edges} (fallback for {guessed_edges})")

    calib = scene_calib(fit, base, mk, edges, wrist_fovy, raw, {k: list(v) for k, v in ranges.items()})
    if not wcal:
        calib["cameras"]["wrist"].update(pitch_offset_deg=pitch, fovy_source="overview camera fit (same camera model)")
    calib["notes"] = {"jaw_silhouette_pitch_deg": jaw_pitch, "jaw_iou": [iou0, iou], "base_prior_P": sync_from_calib.r(prior, 4),
                      "markers_guessed": missing, "table_edges_guessed": guessed_edges,
                      "wrist_markers": sorted(Cw), "wrist_bare_squares": extra, "wrist_reproj_px": round(err_w, 2),
                      "base_P": sync_from_calib.r(base, 4)}
    (OUT / "scene_calib_estimate.json").write_text(sync_from_calib.dumps(calib) + "\n")
    if not a.frames:     # keep the inputs of a successful estimate (re-run offline with --frames)
        cv2.imwrite(str(OUT / "overview.jpg"), ovs[best_frame]); cv2.imwrite(str(OUT / "wrist.jpg"), wr)
        (OUT / "joints.json").write_text(json.dumps({"raw": raw, "ranges": ranges}))
    print("wrote", OUT / "scene_calib_estimate.json")
    if not a.no_apply:
        layout, changes = sync_from_calib.apply_calib(calib, layout, "estimate")
        for it in layout["markers"]["items"]:
            if it["id"] in missing and it.get("src") == "estimate":
                it["note"] = "id hidden in the overview (cable over the bits): parallelogram from the other three"
        a.layout.write_text(sync_from_calib.dumps(layout) + "\n")
        print("\n".join(changes)); print("updated", a.layout)


if __name__ == "__main__":
    main()
