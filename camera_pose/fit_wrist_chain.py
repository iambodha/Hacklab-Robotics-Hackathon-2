"""Joint fit of the wrist camera chain on hand-moved multi-pose data: arm base on the table + wrist camera mount
rotation + joint zero offsets, with weak priors, validated by leave-one-pose-out.

  python camera_pose/fit_wrist_chain.py --frames-dir frames/calib-20261122-022246 \
      --extra-wrist-dir calibration/wrist_pose_0122 --old-calib calibration/scene_calib.json \
      --out camera_pose/wrist_chain_fit_20261122.json

Observations
  overview   today's frames -> overview pose + marker poses on one table plane (perception.scene_fit); markers the
             overview did not see today are brought in from --old-calib by a rigid 2-D alignment on the shared markers
  foot       front edge of the arm foot in the median overview frame -> a point prior on the base (arm_base.py)
  wrist      marker corners seen by the wrist camera at every hand-moved pose (median over that pose's frames)
Unknowns: base (x, y, yaw) on the table; mount rotation about the camera's x (pitch), y, z axes on top of the CAD
mount (T_link_camera_cad @ Rx Ry Rz); joint offsets (q = range_fraction + offset). Priors: offsets 0 +- 5 deg,
mount 0 +- 20 deg. Pixel residuals at sigma 1 px; covariances are inflated by the reduced chi^2 (never shrunk).
j8 (gripper) does not move the camera, so it cannot be seen here.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from camera_pose.wrist_mount import CAMERA_JSON, rx  # noqa: E402

JOINTS = [f"j{i}" for i in range(1, 8)]
MOUNT = ["pitch", "roll_y", "yaw_z"]
SIGMA_PX = 1.0
PRIOR_OFFSET_DEG = 5.0
PRIOR_MOUNT_DEG = 20.0
PRIOR_Z_MM = 20.0


def ry(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1.0]])


def rz(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def mount_T(T_cad, m: dict) -> np.ndarray:
    return T_cad @ rx(m.get("pitch", 0.0)) @ ry(m.get("roll_y", 0.0)) @ rz(m.get("yaw_z", 0.0))


# ---- data -------------------------------------------------------------------------------------------------
def load(frames_dir: Path, extra_dirs: list[Path], old_calib: Path | None, cfg: dict) -> dict:
    from perception import arm_base, aruco, calib_scene, scene_fit
    from perception import geometry as g
    from perception.scene import load_intrinsics
    W, H = calib_scene.W, calib_scene.H
    size, name = float(cfg["marker_size_m"]), cfg["dictionary"]
    data = calib_scene.load_dir(frames_dir)
    ov = [f for f in data["overview"] if f is not None]
    med, std, cnt = aruco.average_corners(ov, name, min_frac=calib_scene.OVERVIEW_MIN_FRAC, destripe=aruco.DESTRIPE_OVERVIEW_PX)
    sig = {i: float(np.sqrt(np.mean(std[i] ** 2))) for i in med}
    base_ids = list(cfg["base_marker_ids"])
    ref = next((i for i in base_ids if i in med), min(med))
    fit = scene_fit.fit(med, size, W, H, intr=load_intrinsics("overview"), sigma_px=sig, ref_id=ref, free_f=False)
    info = {"overview_markers": {int(i): int(cnt[i]) for i in med}, "overview_frames": len(ov),
            "overview_rms_px": round(fit.rms_px(), 3)}
    if old_calib is not None:
        old = {m["id"]: np.array(m["T_world_marker"]) for m in json.loads(old_calib.read_text())["markers"]["items"]}
        common = [i for i in old if i in fit.markers]
        obj = aruco.marker_object_points(size)
        A = np.concatenate([g.apply(old[i], obj)[:, :2] for i in common])
        B = np.concatenate([fit.corners_table(i)[:, :2] for i in common])
        ca, cb = A.mean(0), B.mean(0)
        U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
        R = (U @ np.diag([1, np.sign(np.linalg.det(U @ Vt))]) @ Vt).T
        T_al = g.T_from(np.pad(R, ((0, 1), (0, 1))) + np.diag([0, 0, 1.0]), [*(cb - R @ ca), 0])
        res = (A - ca) @ R.T + cb - B
        info["old_alignment"] = {"common": common, "rms_mm": round(float(np.sqrt(np.mean(np.sum(res ** 2, 1)))) * 1000, 2)}
        for i, T in old.items():
            if i not in fit.markers:
                Tt = T_al @ T
                fit.markers[i] = (float(Tt[0, 3]), float(Tt[1, 3]), math.atan2(Tt[1, 0], Tt[0, 0]))
                info.setdefault("imported_markers", []).append(int(i))
    gray = np.median(np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in ov]), 0).astype(np.float32)
    foot = arm_base.detect_foot(gray, fit, [i for i in base_ids if i in fit.markers])
    foot_pt = arm_base.foot_front_point(foot, fit) if foot else None
    info["foot"] = None if foot is None else {"pixel": np.round(foot["uv_centre"], 2).tolist(),
                                              "width_mm": round(foot_pt[2] * 1000, 1)}
    T_tm = {i: fit.T_table_marker(i) for i in fit.markers}
    poses = []
    wr_extra = [f for f in data["wrist"] if f is not None]
    for k, p in enumerate(data["wrist_poses"]):
        frames = [f for f in p["frames"] if f is not None]
        if p["joints"] == data["joints"] or _same(p["joints"], data["joints"]):
            frames += wr_extra                       # the "current pose" frames were taken at this pose too
        poses.append({"name": f"pose{k + 1:02d}", "joints": p["joints"], "frames": frames})
    for d in extra_dirs:
        o = calib_scene.load_dir(d)
        poses.append({"name": d.name, "joints": o["joints"], "frames": [f for f in o["wrist"] if f is not None]})
    for p in poses:
        m, _, _ = aruco.average_corners(p["frames"], name, destripe=aruco.DESTRIPE_WRIST_PX) if p["frames"] else ({}, None, None)
        p["obs"] = {i: c for i, c in m.items() if i in T_tm}
        del p["frames"]
    return {"W": W, "H": H, "size": size, "T_tm": T_tm, "foot_pt": foot_pt, "poses": poses, "info": info,
            "intr_wr": load_intrinsics("wrist"), "fit": fit}


def _same(a, b, tol=8):
    return a is not None and b is not None and all(abs(a[j]["raw"] - b[j]["raw"]) <= tol for j in a if j in b)


# ---- model ------------------------------------------------------------------------------------------------
class Chain:
    """Parameter vector = base (x, y, yaw) + the free mount angles + the free joint offsets (degrees)."""

    def __init__(self, D: dict, mount_free: list[str], joints_free: list[str], fixed: dict | None = None):
        from perception import wrist
        from perception.aruco import marker_object_points
        self.D, self.mount_free, self.joints_free = D, mount_free, joints_free
        self.fixed = {**D.get("base_offsets", {}), **(fixed or {})}
        self.cam = wrist.wrist_camera()
        self.T_cad = np.array(json.loads(CAMERA_JSON.read_text())["T_link_camera_cad"], float)
        self.obj = marker_object_points(D["size"])
        self.names = ["x", "y", "yaw"] + mount_free + joints_free
        mapping = {"sign": D.get("sign", {}), "offsets_deg": {}, **({"mode": D["mode"]} if D.get("mode") else {})}
        self.q0 = [self.cam.joints_to_q(p["joints"], mapping) for p in D["poses"]]

    def split(self, x):
        v = dict(self.fixed)
        for n, d in zip(self.names[3:], x[3:]):
            v[n] = v.get(n, 0.0) + d
        return np.r_[x[:3], v.get("z_mm", 0.0) / 1000], {k: v.get(k, 0.0) for k in MOUNT}, {j: v.get(j, 0.0) for j in JOINTS}

    def T_table_cam(self, x, k):
        from camera_pose.elrobot_camera import JOINT_OF
        from perception import geometry as g
        b, m, off = self.split(x)
        q = dict(self.q0[k])
        for j, d in off.items():
            if JOINT_OF[j] in q:
                q[JOINT_OF[j]] += math.radians(d)
        return g.T_from(g.rot_z(b[2]), [b[0], b[1], b[3]]) @ self.cam.link_pose(q) @ mount_T(self.T_cad, m)

    def pose_residuals(self, x, k):
        from perception import geometry as g
        D, Tc = self.D, self.T_table_cam(x, k)
        out = [(g.project(Tc, D["intr_wr"], D["W"], D["H"], g.apply(D["T_tm"][i], self.obj))[0] - c).ravel()
               for i, c in sorted(D["poses"][k]["obs"].items())]
        return np.concatenate(out) if out else np.zeros(0)

    def residuals(self, x, poses):
        from perception import geometry as g
        r = [self.pose_residuals(x, k) / SIGMA_PX for k in poses]
        fp = self.D["foot_pt"]
        if fp is not None:
            from perception import arm_base
            Wh = np.linalg.inv(np.linalg.cholesky(fp[1]))
            p = g.apply(g.T_from(g.rot_z(x[2]), [x[0], x[1], 0]), [[*arm_base.FOOT_FRONT_CENTRE, 0]])[0, :2]
            r.append(Wh @ (p - fp[0]))
        r.append(np.array([x[3 + i] / (PRIOR_MOUNT_DEG if n in MOUNT else PRIOR_Z_MM if n == "z_mm" else PRIOR_OFFSET_DEG)
                           for i, n in enumerate(self.names[3:])]))
        return np.concatenate(r)

    def solve(self, poses, x0):
        from perception import geometry as g
        x, r, J = g.levmar(lambda x: self.residuals(x, poses), x0, iters=300)
        n_pix = sum(len(self.pose_residuals(x, k)) for k in poses)
        cov = np.linalg.pinv(J.T @ J) * max(1.0, float(r @ r) / max(1, n_pix - J.shape[1]))
        return x, cov

    def rms(self, x, poses):
        r = np.concatenate([self.pose_residuals(x, k) for k in poses])
        return float(np.sqrt(np.mean(np.sum(r.reshape(-1, 2) ** 2, 1))))


def base_init(D: dict, poses) -> np.ndarray:
    from perception import wrist
    ch = Chain(D, [], [])
    x0 = np.zeros(3)
    est = wrist.solve_base_from_wrist([D["poses"][k]["obs"] for k in poses], D["T_tm"],
                                      [ch.T_table_cam(x0, k) for k in poses], D["size"], D["W"], D["H"], intr=D["intr_wr"])
    return np.array(wrist.planar_of(est["T_table_base"]))


def fit_model(D, mount_free, joints_free, poses, fixed=None, inits=(0.0, 9.31)):
    ch = Chain(D, mount_free, joints_free, fixed)
    b0 = base_init(D, poses)
    best = None
    for p0 in inits if "pitch" in mount_free else (0.0,):
        x0 = np.r_[b0, [p0 if n == "pitch" else 0.0 for n in ch.names[3:]]]
        x, cov = ch.solve(poses, x0)
        c = float(ch.residuals(x, poses) @ ch.residuals(x, poses))
        if best is None or c < best[0]:
            best = (c, x, cov)
    return ch, best[1], best[2]


MODELS = {
    "A cad": ([], []),
    "B pitch": (["pitch"], []),
    "C mount3": (MOUNT, []),
    "D pitch+j2j3j4j6": (["pitch"], ["j2", "j3", "j4", "j6"]),
    "E pitch+j2j4": (["pitch"], ["j2", "j4"]),
    "F mount3+j2..j6": (MOUNT, ["j2", "j3", "j4", "j5", "j6"]),
    "G all (mount3+j1..j7)": (MOUNT, JOINTS),
    "H base z": ([], ["z_mm"]),
    "I j2": ([], ["j2"]),
    "J pitch+j2": (["pitch"], ["j2"]),
}
TWINFIX = {"pitch": 9.31, "j1": -2.0, "j2": 5.3, "j3": -1.5, "j4": -2.8, "j5": -2.7, "j6": 5.9, "j7": -1.1}


def main():
    from perception.scene import CONFIG_PATH
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--extra-wrist-dir", type=Path, action="append", default=[])
    ap.add_argument("--old-calib", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--sign", default="", help="joint sign overrides, e.g. j5=-1 (the j1/j5/j7 signs are unverified)")
    ap.add_argument("--base-offsets", default="", help="offsets (deg) held fixed under every model, e.g. j5=-18.1; "
                                                        "a model's free offset is fitted on top of it (prior centred there)")
    ap.add_argument("--mode", choices=["range_fraction", "ticks"], help="joint mapping (default: the robot profile's)")
    a = ap.parse_args()
    cfg = json.loads(CONFIG_PATH.read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        D = load(a.frames_dir, a.extra_wrist_dir, a.old_calib, cfg)
    D["sign"] = {k: int(v) for k, v in (kv.split("=") for kv in a.sign.split(",") if kv)}
    D["info"]["sign"] = D["sign"]
    D["mode"] = D["info"]["mapping_mode"] = a.mode
    D["base_offsets"] = {k: float(v) for k, v in (kv.split("=") for kv in a.base_offsets.split(",") if kv)}
    D["info"]["base_offsets_deg"] = D["base_offsets"]
    use = [k for k, p in enumerate(D["poses"]) if p["obs"]]
    print(json.dumps(D["info"]))
    for k, p in enumerate(D["poses"]):
        print(f"  {p['name']}: markers {sorted(p['obs'])}" + ("" if p["obs"] else "  (unused)"))
    report = {"info": D["info"], "poses": {p["name"]: sorted(int(i) for i in p["obs"]) for p in D["poses"]},
              "sigma_px": SIGMA_PX, "priors_deg": {"offsets": PRIOR_OFFSET_DEG, "mount": PRIOR_MOUNT_DEG}, "models": {}}
    for label, (mf, jf) in MODELS.items():
        ch, x, cov = fit_model(D, mf, jf, use)
        s = np.sqrt(np.maximum(np.diag(cov), 0))
        corr = cov / np.outer(s, s).clip(1e-30)
        loo = {}
        for k in use:
            tr = [i for i in use if i != k]
            chk, xk, _ = fit_model(D, mf, jf, tr)
            loo[D["poses"][k]["name"]] = round(chk.rms(xk, [k]), 2)
        pairs = sorted(((abs(corr[i, j]), ch.names[i], ch.names[j], corr[i, j]) for i in range(len(x))
                        for j in range(i + 1, len(x))), reverse=True)[:4]
        params = {n: [round(float(np.degrees(v) if n == "yaw" else v * 1000 if n in "xy" else v), 2),
                      round(float(np.degrees(e) if n == "yaw" else e * 1000 if n in "xy" else e), 2)]
                  for n, v, e in zip(ch.names, x, s)}
        loo_rms = float(np.sqrt(np.mean(np.square(list(loo.values())))))
        report["models"][label] = {"fit_rms_px": round(ch.rms(x, use), 2), "loo_px": loo, "loo_rms_px": round(loo_rms, 2),
                                   "params (x/y mm, yaw/angles deg) [value, sigma]": params,
                                   "per_pose_fit_px": {D["poses"][k]["name"]: round(ch.rms(x, [k]), 2) for k in use},
                                   "strongest_corr": [[p[1], p[2], round(float(p[3]), 3)] for p in pairs]}
        print(f"\n{label}: fit {ch.rms(x, use):.2f} px, leave-one-pose-out {loo_rms:.2f} px  {loo}")
        print("  " + ", ".join(f"{n} {v[0]:+.2f}+-{v[1]:.2f}" for n, v in params.items()))
        print("  corr " + ", ".join(f"{p[1]}/{p[2]} {p[3]:+.2f}" for p in pairs))
    ch, x, _ = fit_model(D, [], [], use, fixed=TWINFIX)
    loo = {}
    for k in use:
        chk, xk, _ = fit_model(D, [], [], [i for i in use if i != k], fixed=TWINFIX)
        loo[D["poses"][k]["name"]] = round(chk.rms(xk, [k]), 2)
    loo_rms = float(np.sqrt(np.mean(np.square(list(loo.values())))))
    report["models"]["T twinfix fixed (pitch 9.31 + its offsets, base refit)"] = {
        "fit_rms_px": round(ch.rms(x, use), 2), "loo_px": loo, "loo_rms_px": round(loo_rms, 2)}
    print(f"\nT twinfix fixed: fit {ch.rms(x, use):.2f} px, leave-one-pose-out {loo_rms:.2f} px {loo}")
    if a.out:
        a.out.write_text(json.dumps(report, indent=1) + "\n")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
