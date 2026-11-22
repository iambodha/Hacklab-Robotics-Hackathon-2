"""MCP server for the real ElRobot arm with freshness-checked cameras and jump-free motion.

State comes from the motor bus (st3215/inference), never from the inference/normvla
frame, which freezes whenever torque is off. Camera frames come from the cam_stream
relay and are rejected if stale or byte-identical to the previous capture.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image as PILImage  # noqa: E402
from mcp.server.fastmcp import FastMCP, Image  # noqa: E402

import station as stn  # noqa: E402

CAM_URL = os.environ.get("ROBOT_CAM_URL", "http://localhost:8890")
MAX_CAM_AGE = float(os.environ.get("ROBOT_MAX_CAM_AGE", "1.0"))
MAX_DELTA = float(os.environ.get("ROBOT_MAX_DELTA", "0.35"))
TRACK_TOL = int(os.environ.get("ROBOT_TRACK_TOL", "250"))
GRIPPER = os.environ.get("ROBOT_GRIPPER_JOINT", "j8")
NAMES, IDS = stn.NAMES, stn.IDS

server = FastMCP("elrobot-hardware")
_cli = None
_last_hash: dict[str, str] = {}


async def _client():
    global _cli
    if _cli is None or not getattr(_cli, "connected", True):
        _cli = await stn.connect()
    return _cli


async def _read() -> dict:
    return await stn.read_state(await _client())


async def _write(address: int, values: dict[int, int], width: int) -> None:
    await stn.sync_write(await _client(), address, values, width)


def _table(st: dict) -> str:
    rows = [f"{'joint':>5} {'norm':>7} {'raw':>5} {'goal':>5} {'range':>12} in_rng torque status load temp"]
    for i, mid in enumerate(IDS):
        m = st[mid]
        rows.append(f"{NAMES[i]:>5} {m['norm']:>7} {m['raw']:>5} {m['goal']:>5} [{m['lo']:>4},{m['hi']:>4}] "
                    f"{'y' if m['in_range'] else 'NO':>6} {m['torque']:>6} {hex(m['status']):>6} {m['load']:>4} {m['temp']:>4}")
    return "\n".join(rows)


def _cameras() -> list[dict]:
    with urllib.request.urlopen(f"{CAM_URL}/api/cameras", timeout=3) as r:
        return [c for c in json.load(r) if c.get("live")]


def _snapshot(h: str) -> bytes:
    with urllib.request.urlopen(f"{CAM_URL}/snapshot/{h}", timeout=3) as r:
        return r.read()


@server.tool(annotations={"readOnlyHint": True})
async def look() -> list:
    """FRESH images from every live camera (cam_stream relay) plus the live motor-bus joint table.
    Each image is checked: relay age <= ROBOT_MAX_CAM_AGE seconds and bytes differ from the previous
    capture of that camera; a stale camera is retried and then reported as STALE instead of shown."""
    out: list = []
    for cam in _cameras():
        h, note = cam["hash"], ""
        for _ in range(4):
            data = _snapshot(h)
            digest = hashlib.sha1(data).hexdigest()
            age = next((c["age"] for c in _cameras() if c["hash"] == h), None)
            if age is not None and age <= MAX_CAM_AGE and digest != _last_hash.get(h):
                break
            note = f"age={age}"
            await asyncio.sleep(0.3)
        else:
            out.append(f"camera {h[:8]} ({cam['unique_id']}): STALE ({note}) — not shown")
            continue
        _last_hash[h] = digest
        with PILImage.open(io.BytesIO(data)) as im:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="JPEG", quality=85)
        out.append(f"camera {h[:8]} ({cam['unique_id']}) fresh, age={age:.2f}s, captured {time.strftime('%H:%M:%S')}")
        out.append(Image(data=buf.getvalue(), format="jpeg"))
    out.append(_table(await _read()))
    return out


@server.tool(annotations={"readOnlyHint": True})
async def joint_state() -> str:
    """Live joint table from the motor bus: raw ticks, calibrated range, in-range flag, torque, status, load."""
    return _table(await _read())


@server.tool(annotations={"destructiveHint": True})
async def set_torque(enabled: bool) -> str:
    """Enable/disable torque on all motors. Enabling first writes goal = present position for every
    motor so nothing jumps. Disabling makes the arm limp: support it before calling with False."""
    st = await _read()
    if enabled:
        await _write(0x2A, {mid: st[mid]["raw"] for mid in IDS}, 2)
        await asyncio.sleep(0.1)
    await _write(0x28, {mid: int(enabled) for mid in IDS}, 1)
    await asyncio.sleep(0.4)
    return _table(await _read())


@server.tool(annotations={"destructiveHint": True})
async def move_to(targets: dict[str, float], step: float = 0.02, dt: float = 0.12) -> str:
    """Creep joints to ABSOLUTE normalized targets in [0,1] (calibrated range), e.g. {"j1":0.5}.
    Joints: j1 base yaw, j2 shoulder, j3 elbow, j4/j5/j6 wrist, j7 gripper rotation, j8 gripper jaws
    (higher = more open). Unlisted joints hold their present raw position. Moves `step` per `dt`.
    Refuses: torque off, a commanded joint currently outside its calibrated range (it would snap),
    or a request > ROBOT_MAX_DELTA from present. Aborts and holds if a joint stops tracking
    (|goal-present| > ROBOT_TRACK_TOL ticks) or a motor reports a fault status."""
    return await _move(targets, step, dt, TRACK_TOL)


async def _move(targets: dict[str, float], step: float, dt: float, track_tol: int) -> str:
    unknown = [k for k in targets if k not in NAMES]
    if unknown:
        raise ValueError(f"unknown joints {unknown}; valid {NAMES}")
    st = await _read()
    if any(st[mid]["torque"] == 0 for mid in IDS):
        raise RuntimeError("torque is off on some motors — call set_torque(true) first")
    plan = {}
    for name, want in targets.items():
        mid = IDS[NAMES.index(name)]
        m = st[mid]
        if not m["in_range"]:
            raise RuntimeError(f"{name} raw {m['raw']} is outside calibrated range [{m['lo']},{m['hi']}] — "
                               "commanding it would snap; recalibrate or move it into range by hand")
        want = min(1.0, max(0.0, float(want)))
        if abs(want - m["norm"]) > MAX_DELTA:
            raise ValueError(f"{name}: {m['norm']:.3f}->{want:.3f} exceeds max delta {MAX_DELTA}; use waypoints")
        plan[mid] = (m["norm"], want)
    hold = {mid: st[mid]["raw"] for mid in IDS}
    n = max([1] + [int(abs(b - a) / step + 0.999) for a, b in plan.values()])
    for k in range(1, n + 1):
        goals = dict(hold)
        for mid, (a, b) in plan.items():
            m = st[mid]
            goals[mid] = int(round(m["lo"] + (a + (b - a) * k / n) * (m["hi"] - m["lo"])))
        await _write(0x2A, goals, 2)
        await asyncio.sleep(dt)
        now = await _read()
        bad = [(NAMES[IDS.index(mid)], goals[mid], now[mid]["raw"], now[mid]["status"]) for mid in IDS
               if now[mid]["status"] or abs(goals[mid] - now[mid]["raw"]) > track_tol]
        if bad:
            await _write(0x2A, {mid: now[mid]["raw"] for mid in IDS}, 2)
            return f"ABORTED at step {k}/{n}: (joint, goal, present, status) {bad}; holding present pose.\n" + _table(now)
    await asyncio.sleep(0.3)
    return f"reached in {n} steps.\n" + _table(await _read())


@server.tool(annotations={"destructiveHint": True})
async def gripper(open_fraction: float) -> str:
    """Set the gripper jaws (j8) to a normalized target: higher = more open. Closing on an object
    stalls the jaw short of the target, which is a grip; it does not abort for the gripper."""
    return await _move({GRIPPER: open_fraction}, 0.05, 0.1, 4096)


# ---------------------------------------------------------------------------
# Wrist-camera geometry (camera_pose/elrobot_camera.py). Read-only: these tools never move the arm.

_WCAM = None
CAM_W, CAM_H = 299, 224        # wrist frames as stored by Station (resize_target 224)
TABLE_Z = float(os.environ.get("ROBOT_TABLE_Z", "0.0"))   # base-frame table height; HANDOFF task 5 measures it


def _wcam():
    global _WCAM
    if _WCAM is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from camera_pose.elrobot_camera import WristCamera
        _WCAM = WristCamera()
    return _WCAM


async def _q():
    """URDF radians from the motor bus by the robot profile's joint mapping (range_fraction, as the executor)."""
    st = await _read()
    return _wcam().joints_to_q({NAMES[i]: {"raw": st[mid]["raw"], "lo": st[mid]["lo"], "hi": st[mid]["hi"]}
                                for i, mid in enumerate(IDS)})


def _intrinsics(assumed_hfov_deg: float | None) -> dict:
    cam = _wcam()
    if cam.intrinsics:
        return cam.intrinsics
    if assumed_hfov_deg is None:
        raise ValueError("wrist camera intrinsics are not calibrated yet (camera_pose/HANDOFF.md task 2); "
                         "pass assumed_hfov_deg (e.g. 80) for a rough, clearly labelled estimate")
    from camera_pose.elrobot_camera import assumed_intrinsics
    return assumed_intrinsics(assumed_hfov_deg)


def _note(intr: dict) -> str:
    return (f" [ASSUMED intrinsics, hfov {intr['hfov_deg']} deg: rough]" if intr.get("assumed") else "") + \
           f" [table z={TABLE_Z} m{' (not measured yet)' if 'ROBOT_TABLE_Z' not in os.environ else ''}]"


@server.tool(annotations={"readOnlyHint": True})
async def wrist_camera_pose() -> str:
    """Where the wrist camera is and where it looks, plus the gripper tip (TCP), in the robot base
    frame (metres, z up), computed from the live motor-bus joints. Use it to reason about what the
    wrist image shows. Joint signs of j1/j5/j7 are not yet verified on hardware."""
    import math
    q = await _q()
    cam = _wcam()
    T = cam.camera_pose(q)
    o, z = T[:3, 3], T[:3, 2]
    out = [f"camera position {o.round(4).tolist()}",
           f"viewing direction {z.round(3).tolist()} ({math.degrees(math.asin(max(-1, min(1, -z[2])))):.0f} deg below horizontal)",
           f"TCP (between fingertip pads) {cam.tcp(q).round(4).tolist()}"]
    if z[2] < -1e-3:
        hit = o + ((TABLE_Z - o[2]) / z[2]) * z
        out.append(f"image centre looks at table point {hit.round(4).tolist()}")
    return "\n".join(out)


@server.tool(annotations={"readOnlyHint": True})
async def wrist_pixel_to_table(u: float, v: float, assumed_hfov_deg: float | None = None) -> str:
    """Table point (base frame, metres) seen at wrist-image pixel (u = column, v = row, 299x224
    frame as returned by look), and its offset from the current gripper tip. Use it to turn
    'the object is at pixel (u,v)' into how far to move."""
    intr = _intrinsics(assumed_hfov_deg)
    q = await _q()
    cam = _wcam()
    p = cam.pixel_to_table(u, v, q, intr, CAM_W, CAM_H, TABLE_Z)
    if p is None:
        return "that pixel's ray does not hit the table plane" + _note(intr)
    return f"table point {p.round(4).tolist()}; offset from TCP {(p - cam.tcp(q)).round(4).tolist()}" + _note(intr)


@server.tool(annotations={"readOnlyHint": True})
async def project_to_wrist(x: float, y: float, z: float, assumed_hfov_deg: float | None = None) -> str:
    """Pixel (u, v) in the wrist image (299x224) where base-frame point (x, y, z) appears, or
    'outside view'."""
    intr = _intrinsics(assumed_hfov_deg)
    q = await _q()
    cam = _wcam()
    uv, depth = cam.project([x, y, z], q, intr, CAM_W, CAM_H)
    inside = bool(cam.in_view(uv, depth, CAM_W, CAM_H)[0])
    return (f"pixel {uv[0].round(1).tolist()}, depth {depth[0]:.3f} m" + ("" if inside else " (outside view)")) + _note(intr)


# ---------------------------------------------------------------------------
# Scene perception (perception/, calibration/scene_calib.json). Read-only: frames come from Station's
# usbvideo queues, joints from the motor bus; nothing is ever sent to the arm. World frame: follower
# base_link origin on the table, z up, x = arm forward at j1=0, metres. Needs opencv-contrib-python.

_located: list[dict] = []          # objects located this session, shown by scene_map / annotated_view


def _perc():
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from perception import locate
    return locate


def _locator():
    loc = _perc()
    if not loc.scene_file_exists():
        raise RuntimeError("no calibration/scene_calib.json yet: run `python -m perception.calib_scene --live`")
    return loc.Locator(loc.reload())


async def _joints() -> dict:
    st = await _read()
    return {NAMES[i]: {"raw": st[mid]["raw"], "lo": st[mid]["lo"], "hi": st[mid]["hi"]} for i, mid in enumerate(IDS)}


async def _frame(cam: str):
    from perception import station_io as sio
    jpg = (await sio.grab_jpegs(await _client(), cam, 1))[0]
    return sio.decode_jpeg(jpg)


def _json(d) -> str:
    return json.dumps(d, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


@server.tool(annotations={"readOnlyHint": True})
async def calibration_status() -> str:
    """What the scene calibration knows and how good it is: markers, camera intrinsics source,
    overview camera pose, how the arm base / world frame was found, hold-one-out accuracy (mm),
    mm per pixel at each marker. World frame: arm base on the table, x forward, y left, z up, metres."""
    return _json(_locator().sc.status())


@server.tool(annotations={"readOnlyHint": True})
async def locate(camera: str = "overview", u: float | None = None, v: float | None = None,
                 bbox: list[float] | None = None, color: str | None = None, height_m: float = 0.0,
                 label: str | None = None) -> str:
    """MEASURE where something is on the table (world metres), from what you recognised in a camera image.
    camera: "overview" or "wrist" (299x224 pixels, u = column, v = row, as in annotated_view's ruler).
    Give a pixel (u, v) for a point, and/or bbox [u0, v0, u1, v1] around an object; optional color
    (red/orange/yellow/green/blue/purple/black/white) helps segmentation. height_m: the object's height
    if known (0 = flat); a tall object located with height 0 is biased away from the camera, the
    result tells you by how much per cm. Returns centre_xy, size_m [long, short], yaw_deg (long side vs +x),
    sigma_centre_mm, or for a bare pixel the table point xyz with sigma_mm."""
    loc = _locator()
    joints = await _joints() if camera == "wrist" else None
    if bbox is None and color is None:
        if u is None or v is None:
            raise ValueError("give a pixel (u, v), a bbox, or a color")
        r = loc.pixel_to_table(camera, u, v, height_m, joints)
    else:
        img = await _frame(camera)
        r = loc.locate_object(camera, img, bbox=bbox, pixel=None if u is None else (u, v), color=color,
                              height_m=height_m, joints=joints)
    if r.get("ok"):
        _located.append({"label": label or color or "object", "camera": camera, "time": time.strftime("%H:%M:%S"),
                         **{k: r[k] for k in ("xyz", "centre_xy", "size_m", "yaw_deg", "polygon_xy", "sigma_mm",
                                              "sigma_centre_mm") if k in r}})
        del _located[:-20]
    return _json(r)


@server.tool(annotations={"readOnlyHint": True})
async def triangulate(overview_u: float, overview_v: float, wrist_u: float, wrist_v: float) -> str:
    """3D point (world metres) of one physical point seen at a pixel in BOTH cameras (e.g. the top of
    an object), using the overview calibration and the live wrist pose from FK. Gives ray_gap_mm
    (how well the two rays meet) and sigma_mm."""
    return _json(_locator().triangulate((overview_u, overview_v), (wrist_u, wrist_v), await _joints()))


@server.tool(annotations={"readOnlyHint": True})
async def project_point(x: float, y: float, z: float = 0.0) -> str:
    """Where world point (x, y, z) in metres appears in each camera (pixel u, v in 299x224, depth, in_view).
    Use it to check a plan: e.g. project a target before moving to it."""
    loc = _locator()
    out = {"overview": loc.world_to_pixel("overview", [x, y, z])}
    if loc.sc.intr("wrist") is not None:
        out["wrist"] = loc.world_to_pixel("wrist", [x, y, z], await _joints())
    return _json(out)


@server.tool(annotations={"readOnlyHint": True})
async def scene_map() -> str:
    """The measured scene in world metres: table extent, markers, arm base, both cameras (position,
    view direction, footprint on the table), gripper TCP, and objects located so far this session."""
    loc = _locator()
    joints = await _joints()
    return _json(loc.scene_map(joints, _located))


@server.tool(annotations={"readOnlyHint": True})
async def annotated_view(camera: str = "overview") -> list:
    """Live camera frame with what the maths sees drawn in: a 5 cm table grid (orange lines every 10 cm,
    labelled with world x), world axes at the arm base (x red, y green, z blue), calibrated markers (red)
    vs detected (green), the wrist camera's view footprint and TCP (magenta, overview only), and objects
    located this session (yellow). Upscaled x3; the ruler on the border is in 299x224 pixels for locate()."""
    import cv2
    from perception import annotate, aruco
    loc = _locator()
    img = await _frame(camera)
    joints = await _joints()
    obs = aruco.detect(img, loc.sc.d["markers"]["dictionary"])
    extra = [{"polygon": [[*p, 0.0] for p in o["polygon_xy"]]} if "polygon_xy" in o else
             {"point": o["xyz"], "label": o["label"]} for o in _located]
    extra += [{"point": [*o["centre_xy"], 0.0], "label": o["label"]} for o in _located if "centre_xy" in o]
    if camera == "wrist":
        out = annotate.annotate_wrist(loc.sc, img, joints, obs, extra)
    else:
        out = annotate.annotate_overview(loc.sc, img, obs, joints, extra)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return [f"{camera} annotated, captured {time.strftime('%H:%M:%S')}, markers detected {sorted(obs)}",
            Image(data=buf.tobytes(), format="jpeg")]


if __name__ == "__main__":
    server.run()
