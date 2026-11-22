"""ElRobot wrist-camera geometry: joint readings -> camera pose -> pixel <-> table point.

Task 1 of camera_pose/HANDOFF.md. numpy only for the library; MuJoCo only for `validate`.

  python3 camera_pose/elrobot_camera.py validate          # MuJoCo render check (no robot needed)
  python3 camera_pose/elrobot_camera.py pose              # live: camera + TCP from the motor bus (read-only)
  python3 camera_pose/elrobot_camera.py locate U V --hfov 90     # live: wrist pixel -> table point
  python3 camera_pose/elrobot_camera.py project X Y Z --hfov 90  # live: base-frame point -> wrist pixel

Frames: everything is in the URDF base_link frame (metres). Camera axes follow OpenCV
(x right, y down, z forward). Pixels are image indices (u = column, v = row, (0,0) = top-left
pixel centre).

Joint mapping: the robot profile's `kinematics.arm_mapping` (sim/robot_profile.py, ROBOT_PROFILE,
default elrobot = "range_fraction"), the same as the executor, sim/real_arm_mcp.py and the twin:
q = lower + n*(upper - lower), n = (raw - range_min) / (range_max - range_min). A joint whose
calibrated range is missing or reset (< 200 ticks wide) falls back to "ticks" around the servo
centre 2048, like sim/real_joints.py. Mode "ticks" (HANDOFF finding 3) is still available:
q = mid(limits) + sign * (raw - mid(range)) * 2*pi/4096. The signs of j1, j5 and j7 are NOT verified
(task 3) -- override with `mapping={"sign": {"j7": -1}}`.

Intrinsics: unknown until task 2. Pass them explicitly ({"fx_n","fy_n","cx_n","cy_n","dist"},
normalised by image width/height so a resize doesn't break them) or use `assumed_intrinsics(hfov)`
for a rough guess, which is labelled as such everywhere it's printed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from check_joint_mapping import Urdf  # noqa: E402  (minimal serial-chain FK, reused as the handoff asks)

CAMERA_JSON = HERE / "wrist_camera.json"
LINK = "Gripper_Base_v1_1"
JOINT_OF = {f"j{i}": f"rev_motor_0{i}" for i in range(1, 8)}
RAD_PER_TICK = 2 * math.pi / 4096
RAW_CENTRE = 2048                  # servo centre; every calibrated follower range on this arm is centred on it
MIN_RANGE_TICKS = 200              # narrower calibrated range = missing/reset calibration
MODES = ("range_fraction", "ticks")


def profile_arm_mapping() -> str:
    """kinematics.arm_mapping of the robot profile, loaded like sim/robot_profile.load (ROBOT_PROFILE path or
    name in sim/robots/, else elrobot). Read from the JSON directly when robot_profile can't be imported
    (it needs mujoco)."""
    try:
        if str(REPO / "sim") not in sys.path:
            sys.path.append(str(REPO / "sim"))        # appended: sim/recorder.py must not shadow the recorder package
        import robot_profile
        prof = robot_profile.load().raw
    except ImportError:
        choice = os.environ.get("ROBOT_PROFILE") or "elrobot"
        p = Path(choice)
        if not p.exists():
            p = REPO / "sim/robots" / (choice if choice.endswith(".json") else choice + ".json")
        prof = json.loads(p.read_text())
    mode = prof["kinematics"]["arm_mapping"]
    if mode not in MODES:
        raise ValueError(f"robot profile arm_mapping {mode!r} is not supported for the ElRobot URDF FK ({MODES})")
    return mode


DEFAULT_MAPPING = {"mode": profile_arm_mapping(), "sign": {f"j{i}": 1 for i in range(1, 8)}}


def find_urdf() -> Path:
    for root in (os.environ.get("NORMA_CORE_REPO"), REPO / "norma-core", REPO.parent / "norma-core"):
        if root:
            p = Path(root) / "hardware/elrobot/simulation/elrobot_follower.urdf"
            if p.exists():
                return p
    raise SystemExit("elrobot_follower.urdf not found: set NORMA_CORE_REPO to a norma-core checkout "
                     "that includes hardware/elrobot/simulation")


def assumed_intrinsics(hfov_deg: float, W: int = 4, H: int = 3) -> dict:
    """Pinhole guess from a horizontal field of view, no distortion. NOT a calibration."""
    fx = (W / 2) / math.tan(math.radians(hfov_deg) / 2)
    return {"fx_n": fx / W, "fy_n": fx / H, "cx_n": (W - 1) / 2 / W, "cy_n": (H - 1) / 2 / H,
            "dist": [0, 0, 0, 0, 0], "assumed": True, "hfov_deg": hfov_deg}


def _K(intr: dict, W: int, H: int):
    return intr["fx_n"] * W, intr["fy_n"] * H, intr["cx_n"] * W, intr["cy_n"] * H


def _distort(x, y, d):
    k1, k2, p1, p2, k3 = (list(d) + [0] * 5)[:5]
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    return x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y


def _undistort(xd, yd, d, iters=20):
    x, y = xd, yd
    for _ in range(iters):
        dx, dy = _distort(x, y, d)
        x, y = x - (dx - xd), y - (dy - yd)
    return x, y


class WristCamera:
    def __init__(self, urdf: Path | None = None, camera_json: Path = CAMERA_JSON):
        self.urdf_path = Path(urdf) if urdf else find_urdf()
        self.urdf = Urdf(self.urdf_path, tip=LINK)
        cfg = json.loads(Path(camera_json).read_text())
        self.T_link_cam = np.array(cfg["T_link_camera"], dtype=float)
        self.tcp_link = np.r_[cfg["tcp_m"], 1.0]
        self.intrinsics = cfg.get("intrinsics")

    # ---- joints -------------------------------------------------------------------------
    def joints_to_q(self, state: dict, mapping: dict | None = None) -> dict:
        """state: {"j1": {"raw", "lo", "hi"} | norm float, ...} -> {rev_motor_0i: radians}.
        Norm floats (0..1 through the calibrated range) use the range_fraction mapping."""
        m = {**DEFAULT_MAPPING, **(mapping or {})}
        if m["mode"] not in MODES:
            raise ValueError(f"unknown joint mapping {m['mode']!r}; one of {MODES}")
        sign = {**DEFAULT_MAPPING["sign"], **m.get("sign", {})}
        q = {}
        for j, name in JOINT_OF.items():
            if j not in state:
                continue
            lo_q, hi_q = self.urdf.limits[name]
            s, v = sign[j], state[j]
            if isinstance(v, dict) and abs(v["hi"] - v["lo"]) < MIN_RANGE_TICKS:
                q[name] = (lo_q + hi_q) / 2 + s * (v["raw"] - RAW_CENTRE) * RAD_PER_TICK
            elif isinstance(v, dict) and m["mode"] == "ticks":
                q[name] = (lo_q + hi_q) / 2 + s * (v["raw"] - (v["lo"] + v["hi"]) / 2) * RAD_PER_TICK
            else:
                n = v if not isinstance(v, dict) else (v["raw"] - v["lo"]) / (v["hi"] - v["lo"])
                n = n if s > 0 else 1 - n
                q[name] = lo_q + n * (hi_q - lo_q)
        return q

    # ---- poses --------------------------------------------------------------------------
    def link_pose(self, q: dict) -> np.ndarray:
        return self.urdf.fk(q)

    def camera_pose(self, q: dict) -> np.ndarray:
        """4x4 base_link <- camera (OpenCV axes)."""
        return self.link_pose(q) @ self.T_link_cam

    def tcp(self, q: dict) -> np.ndarray:
        return (self.link_pose(q) @ self.tcp_link)[:3]

    # ---- projection ---------------------------------------------------------------------
    def project(self, points, q: dict, intr: dict, W: int, H: int):
        """Base-frame points (N,3) -> pixels (N,2) and depth (N,). depth <= 0 = behind the camera."""
        P = np.atleast_2d(np.asarray(points, float))
        Tci = np.linalg.inv(self.camera_pose(q))
        pc = (Tci[:3, :3] @ P.T).T + Tci[:3, 3]
        z = pc[:, 2]
        x, y = _distort(pc[:, 0] / z, pc[:, 1] / z, intr.get("dist") or [])
        fx, fy, cx, cy = _K(intr, W, H)
        return np.c_[fx * x + cx, fy * y + cy], z

    def in_view(self, uv, depth, W, H):
        return (depth > 0) & (uv[:, 0] >= -0.5) & (uv[:, 0] <= W - 0.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] <= H - 0.5)

    def pixel_ray(self, u: float, v: float, q: dict, intr: dict, W: int, H: int):
        """-> (origin, unit direction) in the base frame."""
        fx, fy, cx, cy = _K(intr, W, H)
        x, y = _undistort((u - cx) / fx, (v - cy) / fy, intr.get("dist") or [])
        T = self.camera_pose(q)
        d = T[:3, :3] @ np.array([x, y, 1.0])
        return T[:3, 3].copy(), d / np.linalg.norm(d)

    def pixel_to_table(self, u: float, v: float, q: dict, intr: dict, W: int, H: int, table_z: float = 0.0):
        """Intersect the pixel's ray with the plane z = table_z. None if it never hits the table."""
        o, d = self.pixel_ray(u, v, q, intr, W, H)
        if abs(d[2]) < 1e-9:
            return None
        t = (table_z - o[2]) / d[2]
        return None if t <= 0 else o + t * d


# ---------------------------------------------------------------------------------------
# MuJoCo validation (task 1 "done when")

def _mujoco_spec(cam: WristCamera, fovy: float, markers: list[np.ndarray]):
    import mujoco
    u = cam.urdf_path.resolve()
    xml = re.sub(r"(<robot[^>]*>)",
                 r'\1<mujoco><compiler meshdir="%s" strippath="false" fusestatic="false" '
                 r'discardvisual="false" boundmass="1e-4" boundinertia="1e-8"/></mujoco>' % u.parent,
                 u.read_text(), count=1)                     # in memory only; the shared URDF is untouched
    spec = mujoco.MjSpec.from_string(xml)
    R_mj = cam.T_link_cam[:3, :3] @ np.diag([1.0, -1.0, -1.0])   # OpenCV -> MuJoCo camera (looks along -z, y up)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R_mj.flatten())
    c = spec.body(LINK).add_camera(name="wrist")
    c.pos, c.quat, c.fovy = cam.T_link_cam[:3, 3], quat, fovy
    for i, p in enumerate(markers):
        spec.worldbody.add_geom(name=f"marker{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                size=[0.003, 0, 0], pos=p, contype=0, conaffinity=0)
    return spec


def validate(n_random: int = 6, W: int = 299, H: int = 224, fovy: float = 70.0, seed: int = 1) -> bool:
    import mujoco
    cam = WristCamera()
    fy = (H / 2) / math.tan(math.radians(fovy) / 2)
    intr = {"fx_n": fy / W, "fy_n": fy / H, "cx_n": (W - 1) / 2 / W, "cy_n": (H - 1) / 2 / H, "dist": []}
    rng = np.random.default_rng(seed)
    # Today's successful top-down grasp (21 Nov, follower ranges from the morning calibration).
    grasp = {"j1": (2172, 1066, 3030), "j2": (1919, 982, 3114), "j3": (3084, 885, 3211), "j4": (1240, 882, 3214),
             "j5": (2116, 128, 3968), "j6": (2551, 1006, 3090), "j7": (1680, 109, 3987)}
    poses = [("grasp 21 Nov", cam.joints_to_q({j: {"raw": r, "lo": lo, "hi": hi} for j, (r, lo, hi) in grasp.items()})),
             ("all mid", cam.joints_to_q({j: 0.5 for j in JOINT_OF}))]
    for k in range(n_random):
        poses.append((f"random {k + 1}", cam.joints_to_q({j: float(rng.uniform(0.15, 0.85)) for j in JOINT_OF})))
    worst_px, worst_fk, ok = 0.0, 0.0, True
    print(f"MuJoCo {mujoco.__version__}: render {W}x{H}, fovy {fovy} deg; 5 random markers 0.12-0.30 m in front "
          f"of the camera per pose (kept out of the bottom of the frame, where the jaws are)")
    for label, q in poses:
        z = rng.uniform(0.12, 0.30, 5)
        cam_pts = np.c_[rng.uniform(-0.5, 0.5, 5) * z, rng.uniform(-0.35, 0.15, 5) * z, z]
        Tc = cam.camera_pose(q)
        markers = [(Tc[:3, :3] @ p) + Tc[:3, 3] for p in cam_pts]
        m = _mujoco_spec(cam, fovy, markers).compile()
        d = mujoco.MjData(m)
        for name, val in q.items():
            d.qpos[m.joint(name).qposadr[0]] = val
        mujoco.mj_forward(m, d)
        T_sim = np.eye(4)
        T_sim[:3, :3], T_sim[:3, 3] = d.body(LINK).xmat.reshape(3, 3), d.body(LINK).xpos
        fk_err = float(np.abs(T_sim - cam.link_pose(q)).max())
        r = mujoco.Renderer(m, H, W)
        r.enable_segmentation_rendering()
        r.update_scene(d, camera="wrist")
        seg = r.render()
        r.close()
        uv, depth = cam.project(np.array(markers), q, intr, W, H)
        errs = []
        for i in range(len(markers)):
            gid = m.geom(f"marker{i}").id
            ys, xs = np.nonzero((seg[..., 0] == gid) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)))
            if len(xs) < 3:
                errs.append(None)          # occluded by the robot itself; nothing to compare
                continue
            errs.append(float(np.hypot(xs.mean() - uv[i, 0], ys.mean() - uv[i, 1])))
        seen = [e for e in errs if e is not None]
        worst_px, worst_fk = max(worst_px, max(seen, default=0)), max(worst_fk, fk_err)
        good = bool(seen) and max(seen) < 2.0 and fk_err < 1e-6
        ok &= good
        print(f"  {label:13s} FK vs sim {fk_err:.1e} | marker px error " +
              " ".join("occl" if e is None else f"{e:.2f}" for e in errs) + ("" if good else "  <-- FAIL"))
    # pixel -> table -> pixel round trip (pure maths)
    q = poses[0][1]
    rt = 0.0
    for u, v in [(30, 40), (150, 112), (270, 200), (80, 190)]:
        p = cam.pixel_to_table(u, v, q, intr, W, H, table_z=0.0)
        if p is not None:
            back, _ = cam.project(p, q, intr, W, H)
            rt = max(rt, float(np.hypot(*(back[0] - (u, v)))))
    print(f"worst marker error {worst_px:.2f} px (pass < 2), worst FK mismatch {worst_fk:.1e}, "
          f"pixel->table->pixel round trip {rt:.1e} px")
    print("PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------------------------------
# Live CLI (read-only: reads the follower's motor bus through robot_mcp.station_lib)

async def _live_state() -> dict:
    sys.path.insert(0, str(REPO))
    from robot_mcp import station_lib as sl
    cli = await sl.connect()
    bus = await sl.discover_bus_serial(cli)
    joints, ranges, raws = await sl.current_st3215(cli, bus)
    return {f"j{i + 1}": {"raw": raws[f"j{i + 1}"], "lo": ranges[i][0], "hi": ranges[i][1]} for i in range(8)}


def _intr(args, cam: WristCamera) -> dict:
    if cam.intrinsics:
        return cam.intrinsics
    if args.hfov is None:
        raise SystemExit("no calibrated intrinsics yet (HANDOFF task 2); pass --hfov DEG for a rough assumption")
    return assumed_intrinsics(args.hfov)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate"); v.add_argument("--random", type=int, default=6)
    sub.add_parser("pose")
    lo = sub.add_parser("locate"); lo.add_argument("u", type=float); lo.add_argument("v", type=float)
    pr = sub.add_parser("project"); pr.add_argument("xyz", type=float, nargs=3)
    for p in (lo, pr):
        p.add_argument("--hfov", type=float, help="assumed horizontal FOV (deg) until intrinsics are calibrated")
        p.add_argument("--width", type=int, default=299); p.add_argument("--height", type=int, default=224)
        p.add_argument("--table-z", type=float, default=0.0, help="table height in base frame (task 5 measures it)")
    a = ap.parse_args()
    if a.cmd == "validate":
        sys.exit(0 if validate(a.random) else 1)
    cam = WristCamera()
    st = asyncio.run(_live_state())
    q = cam.joints_to_q(st)
    T = cam.camera_pose(q)
    if a.cmd == "pose":
        o, z = T[:3, 3], T[:3, 2]
        print(f"camera position (m)  {o.round(4)}\nviewing direction    {z.round(3)}"
              f"  ({math.degrees(math.asin(-z[2])):.0f} deg below horizontal)\nTCP (m)              {cam.tcp(q).round(4)}")
        if z[2] < -1e-3:
            hit = o + (-o[2] / z[2]) * z
            print(f"optical axis hits z=0 at {hit[:2].round(4)}")
        n8 = (st['j8']['raw'] - st['j8']['lo']) / (st['j8']['hi'] - st['j8']['lo'])
        print(f"jaw opening norm {n8:.2f}; joint signs j1/j5/j7 unverified (HANDOFF task 3)")
        return
    intr = _intr(a, cam)
    tag = " [ASSUMED intrinsics]" if intr.get("assumed") else ""
    if a.cmd == "locate":
        p = cam.pixel_to_table(a.u, a.v, q, intr, a.width, a.height, a.table_z)
        if p is None:
            print("that pixel's ray does not hit the table plane" + tag)
        else:
            print(f"table point (m) {p.round(4)}; offset from TCP {(p - cam.tcp(q)).round(4)}{tag}")
    else:
        uv, depth = cam.project(a.xyz, q, intr, a.width, a.height)
        inside = cam.in_view(uv, depth, a.width, a.height)[0]
        print(f"pixel {uv[0].round(1)} depth {depth[0]:.3f} m{'' if inside else '  (outside view)'}{tag}")


if __name__ == "__main__":
    main()
