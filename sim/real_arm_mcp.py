"""elrobot-real: the REAL ElRobot behind the same MCP tools as sim_arm_mcp.py (elrobot-sim).

The operator agent that completes the task in the simulator sees exactly the same tool surface here:
look, joint_state, set_torque, move_to, gripper, tcp_pose, move_tcp, pixel_to_table. Only the backend differs:

  * joints, torque, bounded moves: arm_mcp (Station motor bus at STATION_HOST:8888), unchanged safety:
    refuses moves > ROBOT_MAX_DELTA, refuses joints outside their calibrated range, aborts and holds on a
    tracking error above ROBOT_TRACK_TOL or a motor fault
  * cameras: the cam_stream relay (scripts/cam_stream.py, :8890), freshness-checked like arm_mcp, then
    upscaled 2x with the same labelled pixel grid as the sim (coordinates as used by pixel_to_table)
  * kinematics: the sim's own MuJoCo model (sim/generated/workspace.xml, kinematics only, no physics) and its
    ArmIK, with the same joint mapping as the sim: q = lower + n*(upper - lower) through the URDF limits,
    n = the real calibrated-range norm
  * wrist camera: CAD mount pose from camera_pose/wrist_camera.json (in the model as wrist_cam)
  * overview camera: needs real extrinsics in sim/real_overview_camera.json (see OPERATOR_REAL.md); until
    then pixel_to_table("overview", ...) explains that and the agent uses the wrist camera

Real-world additions: move_tcp refuses targets with the TCP below ROBOT_MIN_TCP_Z (default 0.012 m: the
fingertips reach the table at TCP z ~ 0.01), stops waiting when joints stop moving (servos under load stop
short), and reports the remaining error. Every tool call is appended to sim/real_sessions/actions.jsonl.

Run (stdio; usually via sim/run_agent_real.sh):
  MUJOCO_GL=cgl uv run --no-project --with 'mcp>=1.12,<2' --with pillow --with numpy --with mujoco \\
      python sim/real_arm_mcp.py
Assumptions not yet verified on hardware: the signs of j1/j5/j7 (camera_pose/HANDOFF.md task 3; j1 matches
the direction observed on 21 Nov), the wrist intrinsics (fovy 72 as in the sim), table top at z = 0.
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import io
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")   # no rendering here; import only
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from mcp.server.fastmcp import FastMCP, Image  # noqa: E402
from PIL import Image as PILImage, ImageDraw  # noqa: E402

import build_scene  # noqa: E402  (ArmIK + the scene builder)
from arm_mcp import server as am  # noqa: E402  (real Station access; its own MCP server object is unused)

IMG_SCALE = int(os.environ.get("ROBOT_IMG_SCALE", "2"))
GRID = os.environ.get("ROBOT_GRID", "1") == "1"
MIN_Z = float(os.environ.get("ROBOT_MIN_TCP_Z", "0.012"))
WRIST_CAM = os.environ.get("ROBOT_WRIST_CAM", "e091eeda")       # cam_stream hash prefixes (per machine)
OVERVIEW_CAM = os.environ.get("ROBOT_OVERVIEW_CAM", "5cd55831")
OVERVIEW_CAL = Path(os.environ.get("ROBOT_OVERVIEW_CAL", HERE / "real_overview_camera.json"))
LOG_DIR = Path(os.environ.get("ROBOT_REAL_LOG_DIR", HERE / "real_sessions"))
W, H = 299, 224
NAMES = [f"j{i}" for i in range(1, 9)]
FRAME = ("Base frame, metres: origin = centre of the arm base on the table top, +z up, +y = straight ahead of "
         "the arm (towards the overview camera), +x = the arm's right-hand side (image-LEFT in the overview camera).")

_ws = HERE / "generated" / "workspace.xml"
if not _ws.exists():
    build_scene.build(report=False)
MODEL = mujoco.MjModel.from_xml_path(str(_ws))
IK = build_scene.ArmIK(MODEL)
LO, HI = MODEL.actuator_ctrlrange.T.copy()          # URDF limits (rad), exactly as the sim maps them
KD = mujoco.MjData(MODEL)                            # kinematics-only data for camera poses
WRIST_ID = MODEL.camera("wrist_cam").id

# Optional calibration override: Station's stored ranges can go stale (21 Nov: this machine's Station kept j6
# 1688-2408 / j8 1898-2198 while the follower was re-calibrated to 1006-3090 / 1228-2868). ROBOT_RANGES_FILE
# (JSON {"j1": [lo, hi], ...}, raw ticks) replaces lo/hi/norm/in_range for every read; Station is not changed.
RANGES_FILE = os.environ.get("ROBOT_RANGES_FILE")
if RANGES_FILE:
    _ranges = {NAMES.index(k) + 1: (int(v[0]), int(v[1]))
               for k, v in json.loads(Path(RANGES_FILE).read_text()).items() if k in NAMES}
    _station_read = am._read

    async def _read_with_ranges() -> dict:
        st = await _station_read()
        for mid, (lo, hi) in _ranges.items():
            m = st[mid]
            m.update(lo=lo, hi=hi, norm=round((m["raw"] - lo) / (hi - lo), 4), in_range=lo <= m["raw"] <= hi)
        return st

    am._read = _read_with_ranges   # arm_mcp's joint_state / set_torque / _table pick this up too

server = FastMCP("elrobot-real")
LOG_DIR.mkdir(exist_ok=True)
SESSION_LOG = LOG_DIR / "actions.jsonl"


def recorded(fn):
    """Append every tool call (args, result text or error, duration) to real_sessions/actions.jsonl."""
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        call = sig.bind(*args, **kwargs); call.apply_defaults()
        t0, err, text = time.time(), None, ""
        try:
            result = await fn(**call.arguments)
            text = result if isinstance(result, str) else "\n".join(r for r in result if isinstance(r, str))
            return result
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            raise
        finally:
            with SESSION_LOG.open("a") as f:
                f.write(json.dumps({"t": time.strftime("%H:%M:%S"), "tool": fn.__name__, "args": call.arguments,
                                    "s": round(time.time() - t0, 2), "error": err, "result": text[-1500:]}) + "\n")
    return wrapper


# --- joints: real calibrated norms <-> URDF radians (the sim's convention) ---------------------------

async def measured():
    st = await am._read()
    n = [st[mid]["norm"] for mid in am.IDS]
    if any(x is None for x in n):
        raise RuntimeError("a joint has no calibrated range; calibrate before using kinematics")
    q = np.array([LO[i] + n[i] * (HI[i] - LO[i]) for i in range(7)] + [HI[7] * (1 - n[7])])
    return q, st


def to_norm(q):
    return (q[:7] - LO[:7]) / (HI[:7] - LO[:7])


# --- inverse kinematics: as close to jaws-down as the target allows ------------------------------------
# build_scene.ArmIK.solve either gets jaws exactly down or, as a fallback, ignores orientation completely; from a
# folded seed that fallback left the gripper 76 deg from vertical on hardware. Here the lean is stepped up
# gradually, staying near the current joints first, so the result is the least-tilted reachable pose.
LEAN_SMALL, LEAN_LARGE = (0, 10, -10, 20, -20, 30, -30), (40, -40, 50, 60, 70)


def _target_rot(xyz, lean_deg: float, yaw_deg: float):
    """TCP rotation whose approach axis (+y) points down, leaned lean_deg away from the base (negative: toward it)."""
    r = np.asarray(xyz[:2], float)
    r = r / np.linalg.norm(r) if np.linalg.norm(r) > 1e-6 else np.array([0.0, 1.0])
    t, yw = np.radians(lean_deg), np.radians(yaw_deg)
    a = np.array([np.sin(t) * r[0], np.sin(t) * r[1], -np.cos(t)])
    j = np.array([np.cos(yw), np.sin(yw), 0.0])
    j = j - (j @ a) * a
    if np.linalg.norm(j) < 1e-6:
        j = np.array([-r[1], r[0], 0.0])
    j /= np.linalg.norm(j)
    return np.column_stack([j, a, np.cross(j, a)])


def _solve_rot(xyz, Rt, q0, restarts: int = 0, seed: int = 3):
    """Damped least squares to position + rotation Rt. First from q0; with restarts, the solution nearest q0."""
    m, d = MODEL, IK.data
    rng = np.random.default_rng(seed)
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    best = None
    for attempt in range(1 + restarts):
        q = np.array(q0, float)
        if attempt:
            q[:7] = rng.uniform(IK.low[:7], IK.high[:7])
        for _ in range(300):
            d.qpos[IK.qids] = q
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            err = np.asarray(xyz) - d.site_xpos[IK.site]
            mujoco.mj_jacSite(m, d, jp, jr, IK.site)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, (Rt @ d.site_xmat[IK.site].reshape(3, 3).T).reshape(-1))
            if quat[0] < 0: quat *= -1
            ln = np.linalg.norm(quat[1:])
            err_r = quat[1:] * (2 * np.arctan2(ln, quat[0]) / max(ln, 1e-9))
            if np.linalg.norm(err) < 0.002 and np.linalg.norm(err_r) < 0.02:
                if best is None or np.abs(q[:7] - q0[:7]).max() < np.abs(best[:7] - q0[:7]).max():
                    best = q.copy()
                break
            jac, e = np.vstack([jp[:, IK.dofs], .1 * jr[:, IK.dofs]]), np.r_[err, .1 * err_r]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), e)
            q[:7] = np.clip(q[:7] + np.clip(dq, -.08, .08), IK.low[:7] + .001, IK.high[:7] - .001)
        if attempt == 0 and best is not None:
            return best
    return best


def _solve_tcp(xyz, q0, yaw_deg: float):
    """(q, lean_deg) for the least-leaning reachable pose, preferring one near the current joints; (None, None)."""
    for leans, restarts in ((LEAN_SMALL, 0), (LEAN_SMALL, 12), (LEAN_LARGE, 0), (LEAN_LARGE, 12)):
        for lean in leans:
            q = _solve_rot(xyz, _target_rot(xyz, lean, yaw_deg), q0, restarts)
            if q is not None:
                return q, lean
    return None, None


# --- motion on real servos ----------------------------------------------------------------------------
# The ST3215s run P-only position control: under gravity a joint stops short of its goal, and a goal equal to
# the present position gives no force at all. So (a) every joint we are NOT moving keeps its current goal
# register (keeps the gripper's squeeze and the arm's gravity offsets; arm_mcp's moves reset them to the present
# position, which drops a held object and lets the arm sag), and (b) after creeping, planned joints are nudged
# closed-loop until the measured position arrives. Contact (motor load) or a fault holds the present pose.
LOAD_ABORT = int(os.environ.get("ROBOT_LOAD_ABORT", "550"))   # ST3215 present load, 0..1000
SETTLE_TOL = 6                                                # ticks
GRIP_SQUEEZE = int(os.environ.get("ROBOT_GRIP_SQUEEZE", "45"))   # ticks the jaw goal stays past a stalled grip


def _holds(st):
    """Current goal of every motor, falling back to the present position if the goal register is far off."""
    return {mid: (st[mid]["goal"] if abs(st[mid]["goal"] - st[mid]["raw"]) < 300 else st[mid]["raw"]) for mid in am.IDS}


async def _stop_here(st, mids):
    await am._write(0x2A, {**_holds(st), **{mid: st[mid]["raw"] for mid in mids}}, 2)


def _flagged(st, plan, grip=False):
    return {m for m in am.IDS if st[m]["status"] and (m != 8 or (8 in plan and not grip))}


async def _faults(now, plan, grip=False):
    """Motor status flags that are still set on a second read 0.15 s later (one-read glitches are ignored)."""
    first = _flagged(now, plan, grip)
    if not first:
        return []
    await asyncio.sleep(0.15)
    again = await am._read()
    return [(NAMES[m - 1], hex(again[m]["status"])) for m in sorted(first & _flagged(again, plan, grip))]


async def _drive(plan: dict, step: float = 0.02, dt: float = 0.12, track_tol: int | None = None,
                 settle: bool = True, contact: bool = True, grip: bool = False) -> str:
    """plan {motor_id: target norm}. Creep, then settle closed-loop. Returns 'reached ...' or 'ABORTED ...'.
    A status flag on the gripper motor alone (overload from squeezing an object) never aborts an arm move."""
    track_tol = am.TRACK_TOL if track_tol is None else track_tol
    st = await am._read()
    if any(st[mid]["torque"] == 0 for mid in am.IDS):
        raise RuntimeError("torque is off on some motors: call set_torque(true) first")
    for mid in plan:
        if not st[mid]["in_range"]:
            raise RuntimeError(f"{NAMES[mid - 1]} raw {st[mid]['raw']} is outside its calibrated range; commanding it would snap")
    over = [m for m in am.IDS if m != 8 and st[m]["status"]]
    if over:       # a motor overloaded while holding (seen on j2 after minutes stretched out): unload it, then go on
        await am._write(0x2A, {**_holds(st), **{m: st[m]["raw"] for m in over}}, 2)
        await asyncio.sleep(0.8)
        st = await am._read()
        still = [(NAMES[m - 1], hex(st[m]["status"])) for m in am.IDS if m != 8 and st[m]["status"]]
        if still:
            raise RuntimeError(f"motor fault {still} did not clear after unloading: nothing moved. Let the arm rest "
                               f"(support it, set_torque(false)) or power-cycle the 12 V before continuing")
    raw_goal = {mid: int(round(st[mid]["lo"] + min(1.0, max(0.0, n)) * (st[mid]["hi"] - st[mid]["lo"]))) for mid, n in plan.items()}
    start = {mid: st[mid]["raw"] for mid in plan}
    hold = _holds(st)
    span = {mid: st[mid]["hi"] - st[mid]["lo"] for mid in plan}
    n = max([1] + [int(abs(raw_goal[m] - start[m]) / (step * span[m]) + 0.999) for m in plan])
    for k in range(1, n + 1):
        goals = dict(hold)
        for m in plan:
            goals[m] = int(round(start[m] + (raw_goal[m] - start[m]) * k / n))
        await am._write(0x2A, goals, 2)
        await asyncio.sleep(dt)
        now = await am._read()
        bad = [(NAMES[m - 1], goals[m], now[m]["raw"]) for m in plan if abs(goals[m] - now[m]["raw"]) > track_tol]
        bad = bad or await _faults(now, plan, grip)
        hot = [(NAMES[m - 1], now[m]["load"]) for m in plan if contact and m != 8 and now[m]["load"] > LOAD_ABORT]
        if bad or hot:
            await _stop_here(now, [m for m in plan if m != 8])
            why = f"tracking error or motor fault {bad}" if bad else f"contact: load over {LOAD_ABORT} {hot}"
            return f"ABORTED at step {k}/{n}: {why}; holding present pose."
    if not settle:
        return f"reached in {n} steps."
    cmd = dict(raw_goal)
    for it in range(12):
        await asyncio.sleep(0.2)
        now = await am._read()
        err = {m: raw_goal[m] - now[m]["raw"] for m in plan}
        if all(abs(e) <= SETTLE_TOL for e in err.values()):
            return f"reached in {n} steps (+{it} corrections)."
        hot = [(NAMES[m - 1], now[m]["load"]) for m in plan if contact and m != 8 and now[m]["load"] > LOAD_ABORT]
        fault = [] if hot else await _faults(now, plan, grip)
        if fault or hot:
            await _stop_here(now, [m for m in plan if m != 8])
            return f"ABORTED while settling: {'contact ' + str(hot) if hot else 'motor fault ' + str(fault)}; holding present pose."
        for m, e in err.items():
            if abs(e) > SETTLE_TOL:
                c = cmd[m] + round(0.6 * e)
                c = max(raw_goal[m] - 150, min(raw_goal[m] + 150, c))
                cmd[m] = max(now[m]["lo"] + 20, min(now[m]["hi"] - 20, c))
        await am._write(0x2A, {**_holds(now), **cmd}, 2)
    left = {NAMES[m - 1]: e for m, e in err.items() if abs(e) > SETTLE_TOL}
    return f"reached in {n} steps; still short by {left} ticks (load or an obstacle)."


# --- cameras -------------------------------------------------------------------------------------------

def _grid(jpeg: bytes) -> bytes:
    im = PILImage.open(io.BytesIO(jpeg)).convert("RGB")
    if IMG_SCALE != 1:
        im = im.resize((im.width * IMG_SCALE, im.height * IMG_SCALE), PILImage.BICUBIC)
    if GRID:
        draw = ImageDraw.Draw(im, "RGBA")
        for x in range(0, im.width, 50):
            draw.line([(x, 0), (x, im.height)], fill=(255, 64, 64, 70 if x % 100 else 120))
            if x: draw.text((x + 2, 1), str(x), fill=(255, 40, 40, 255))
        for y in range(0, im.height, 50):
            draw.line([(0, y), (im.width, y)], fill=(255, 64, 64, 70 if y % 100 else 120))
            if y: draw.text((2, y + 1), str(y), fill=(255, 40, 40, 255))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


async def _fresh_frame(prefix: str):
    """(jpeg, note) from the relay; same freshness rule as arm_mcp.look (age and byte change)."""
    cams = {c["hash"][:8]: c for c in am._cameras()}
    cam = cams.get(prefix)
    if cam is None:
        return None, f"camera {prefix}: not live on the relay"
    note = ""
    for _ in range(4):
        data = am._snapshot(cam["hash"])
        digest = __import__("hashlib").sha1(data).hexdigest()
        age = next((c["age"] for c in am._cameras() if c["hash"] == cam["hash"]), None)
        if age is not None and age <= am.MAX_CAM_AGE and digest != am._last_hash.get(cam["hash"]):
            am._last_hash[cam["hash"]] = digest
            return data, f"fresh, age={age:.2f}s"
        note = f"age={age}"
        await asyncio.sleep(0.3)
    return None, f"STALE ({note})"


def _wrist_pose(q):
    KD.qpos[IK.qids] = q
    mujoco.mj_kinematics(MODEL, KD)
    mujoco.mj_camlight(MODEL, KD)
    return KD.cam_xpos[WRIST_ID].copy(), KD.cam_xmat[WRIST_ID].reshape(3, 3).copy(), float(MODEL.cam_fovy[WRIST_ID])


def _ray(pos, mat, fovy, u, v):
    """Same pinhole maths as sim_arm_mcp.pixel_ray (MuJoCo camera: looks along -z, y up)."""
    u, v = u / IMG_SCALE, v / IMG_SCALE
    f = (H / 2) / np.tan(np.radians(fovy) / 2)
    d = mat @ np.array([(u - W / 2) / f, -(v - H / 2) / f, -1.0])
    return pos, d / np.linalg.norm(d)


# --- tools (names, arguments and texts as in sim_arm_mcp.py) -------------------------------------------

@server.tool(annotations={"readOnlyHint": True})
@recorded
async def look() -> list:
    """FRESH images from every live camera plus the live motor-bus joint table. First image: wrist camera;
    second: overview camera. Both upscaled 2x (598x448) with a red pixel grid labelled every 50 px."""
    out: list = []
    for label, prefix in (("wrist camera", WRIST_CAM), ("overview camera", OVERVIEW_CAM)):
        data, note = await _fresh_frame(prefix)
        if data is None:
            out.append(f"{label} {prefix}: {note}, not shown")
            continue
        out.append(f"{label} {prefix} {note}, captured {time.strftime('%H:%M:%S')}")
        out.append(Image(data=_grid(data), format="jpeg"))
    out.append(am._table(await am._read()))
    return out


@server.tool(annotations={"readOnlyHint": True})
@recorded
async def joint_state() -> str:
    """Live joint table from the motor bus: raw ticks, calibrated range, in-range flag, torque, status, load."""
    return await am.joint_state()


@server.tool(annotations={"destructiveHint": True})
@recorded
async def set_torque(enabled: bool) -> str:
    """Enable/disable torque on all motors. Enabling first writes goal = present position so nothing jumps.
    Disabling makes the arm limp: support it before calling with False."""
    return await am.set_torque(enabled)


@server.tool(annotations={"destructiveHint": True})
@recorded
async def move_to(targets: dict[str, float], step: float = 0.02, dt: float = 0.12) -> str:
    """Creep joints to ABSOLUTE normalized targets in [0,1] (calibrated range), e.g. {"j1":0.5}.
    Refuses big jumps (> 0.35), joints outside their range, or torque off; aborts and holds on a tracking error."""
    unknown = [k for k in targets if k not in NAMES]
    if unknown:
        raise ValueError(f"unknown joints {unknown}; valid {NAMES}")
    st = await am._read()
    plan = {}
    for name, want in targets.items():
        mid = NAMES.index(name) + 1
        want = min(1.0, max(0.0, float(want)))
        if st[mid]["norm"] is None or abs(want - st[mid]["norm"]) > am.MAX_DELTA:
            raise ValueError(f"{name}: {st[mid]['norm']} -> {want:.3f} exceeds max delta {am.MAX_DELTA}; use waypoints")
        plan[mid] = want
    out = await _drive(plan, step, dt)
    return out + "\n" + am._table(await am._read())


@server.tool(annotations={"destructiveHint": True})
@recorded
async def gripper(open_fraction: float) -> str:
    """Set the gripper jaws (j8) to a normalized target in ONE call: 1.0 = fully open, 0.0 = closed. Closing stops
    by itself when the jaws stall on an object and then holds it with a limited squeeze: the result says
    "Jaws stalled on something" and j8 stays above its fully closed value (~0.03)."""
    st = await am._read()
    if any(st[mid]["torque"] == 0 for mid in am.IDS):
        raise RuntimeError("torque is off on some motors: call set_torque(true) first")
    if not st[8]["in_range"]:
        raise RuntimeError("j8 is outside its calibrated range; commanding it would snap")
    lo, hi = st[8]["lo"], st[8]["hi"]
    target = int(round(lo + min(OPEN_MAX, max(0.0, float(open_fraction))) * (hi - lo)))   # the jaws stop mechanically near 0.96
    start, hold = st[8]["raw"], _holds(st)
    n = max(1, int(abs(target - start) / (0.06 * (hi - lo)) + 0.999))
    prev, still = start, 0
    for k in range(1, n + 1):
        goal = int(round(start + (target - start) * k / n))
        await am._write(0x2A, {**hold, 8: goal}, 2)
        await asyncio.sleep(0.1)
        now = (await am._read())[8]["raw"]
        still = still + 1 if abs(now - prev) < 6 else 0
        if target < start and now - goal > 200 and still >= 3:    # closing, far behind and not moving for 0.3 s: stalled
            break
        prev = now
    await asyncio.sleep(0.35)
    st = await am._read()
    out = f"gripper at {st[8]['norm']:.3f}."
    if st[8]["raw"] - st[8]["goal"] > GRIP_SQUEEZE + 15:     # stalled on an object: hold it with a bounded squeeze,
        await am._write(0x2A, {**_holds(st), 8: st[8]["raw"] - GRIP_SQUEEZE}, 2)   # or the servo overloads (0x20)
        await asyncio.sleep(0.3)
        st = await am._read()
        out = f"Jaws stalled on something at {st[8]['norm']:.3f}: holding it with a limited squeeze."
    elif st[8]["goal"] - st[8]["raw"] > 25:                  # opened against the mechanical stop: stop pushing
        await am._write(0x2A, {**_holds(st), 8: st[8]["raw"]}, 2)
    return out + "\n" + am._table(st)


@server.tool(annotations={"readOnlyHint": True})
@recorded
async def tcp_pose() -> str:
    """Where the gripper is, from the measured joints and the arm's kinematic model: the TCP (point between
    the fingertip pads) in the base frame, whether the jaws point down, and the jaw axis direction."""
    q, _ = await measured()
    pos, rot = IK.fk(q)
    approach, jaw = rot[:, 1], rot[:, 0]
    tilt = np.degrees(np.arccos(np.clip(-approach[2], -1, 1)))
    return (f"tcp xyz = [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m; jaws point {tilt:.0f} deg from straight down; "
            f"jaw closing axis yaw = {np.degrees(np.arctan2(jaw[1], jaw[0])):.0f} deg.\n{FRAME}")


@server.tool(annotations={"destructiveHint": True})
@recorded
async def move_tcp(x: float, y: float, z: float, top_down: bool = True, yaw_deg: float = 0.0,
                   step: float = 0.02, dt: float = 0.12) -> str:
    """Move the gripper TCP (between the fingertip pads) to (x, y, z) in the base frame using the arm's inverse
    kinematics, executed with move_to in bounded steps (same refusals and tracking aborts). top_down=True keeps
    the jaws pointing straight down (where impossible it uses the nearest tilted pose and says so); yaw_deg turns
    the jaw closing axis about vertical. If the target is more than 3 cm away sideways, it travels at a safe
    height (max(target, 0.08 m)), lifting first, then descends. The gripper opening is not changed.
    Table top is z = 0; on the real arm the TCP may not go below z = 0.012 (fingertips at the table)."""
    target = np.array([x, y, z], float)
    if not (MIN_Z <= z <= 0.45 and np.hypot(x, y) <= 0.45):
        raise ValueError(f"target {target.round(3).tolist()} is outside the workspace "
                         f"(z {MIN_Z}..0.45 m, radius <= 0.45 m)")
    q, _ = await measured()
    cur, _ = IK.fk(q)
    waypoints = []
    if np.hypot(*(target[:2] - cur[:2])) > 0.03:
        safe = max(z, 0.08)
        if cur[2] < safe - 0.01: waypoints.append([cur[0], cur[1], safe])
        if z < safe: waypoints.append([x, y, safe])
    waypoints.append(target.tolist())
    seed, log, notes = q, [], []
    for wp in waypoints:
        if top_down:
            sol, lean = _solve_tcp(np.array(wp), seed, yaw_deg)
            if sol is not None and lean:
                notes.append(f"jaws-down unreachable at {np.round(wp, 3).tolist()}: gripper leans {abs(lean)} deg from "
                             f"vertical ({'away from' if lean > 0 else 'toward'} the base)")
            if sol is None:                      # last resort: position only, orientation free
                sol = IK.solve(np.array(wp), top_down=False, q0=seed)
                if sol is not None:
                    _, rot = IK.fk(sol)
                    notes.append(f"no near-vertical pose at {np.round(wp, 3).tolist()}: gripper is "
                                 f"{np.degrees(np.arccos(np.clip(-rot[2, 1], -1, 1))):.0f} deg from vertical; do not grasp like this")
        else:
            sol = IK.solve(np.array(wp), top_down=False, q0=seed)
        if sol is not None and float(np.abs(to_norm(sol) - to_norm(seed)).max()) > 0.3:
            notes.append("large change of arm configuration for this move")
        if sol is None:
            return (f"UNREACHABLE: no joint solution for tcp {np.round(wp, 3).tolist()} "
                    f"({'jaws down' if top_down else 'any orientation'}); nothing moved after: {log}\n" + await am.joint_state())
        seed = sol
        goal = to_norm(sol)
        for _ in range(6):                       # chunks of <= 0.3 norm per joint, the last one settles closed-loop
            qn, _ = await measured()
            now = to_norm(qn)
            if float(max(abs(goal - now))) < 0.004:
                break
            chunk = np.clip(goal - now, -0.3, 0.3)
            last = bool(np.all(np.abs(goal - now) <= 0.3))
            out = await _drive({i + 1: float(now[i] + chunk[i]) for i in range(7)}, step, dt, settle=last)
            if out.startswith("ABORTED"):
                pos, _ = IK.fk((await measured())[0])
                return f"ABORTED on the way to {np.round(wp, 3).tolist()}; tcp now {pos.round(3).tolist()}.\n" + out
            if "still short" in out:
                notes.append(out)
            if last:
                break
        log.append(np.round(wp, 3).tolist())
    pos, _ = IK.fk((await measured())[0])
    off = float(np.linalg.norm(pos - target))
    notes.append(f"tcp is {off * 1000:.0f} mm from the target") if off > 0.005 else None
    note = ("\nNOTE: " + "; ".join(notes)) if notes else ""
    return f"reached via {log}; tcp now [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}].{note}\n" + await am.joint_state()


@server.tool(annotations={"readOnlyHint": True})
@recorded
async def pixel_to_table(camera: str, u: float, v: float, height: float = 0.0) -> str:
    """Turn a pixel of a look() image into a base-frame point using the camera calibration (wrist: CAD mount pose
    + current joints; overview: fixed extrinsics, once measured). (u, v) are pixel coordinates in the returned
    598x448 image (red grid labels), u right, v down. Returns where the viewing ray hits the plane z = height
    (0 = table top; e.g. 0.01 for the centre of a 2 cm cube, ~0.09 for a box rim)."""
    if camera == "wrist":
        q, _ = await measured()
        pos, mat, fovy = _wrist_pose(q)
    elif camera == "overview":
        if not OVERVIEW_CAL.exists():
            return ("the overview camera is not calibrated on the real arm yet (no sim/real_overview_camera.json); "
                    "use the wrist camera: move_tcp to a high survey pose, look(), then pixel_to_table('wrist', ...)")
        cal = json.loads(OVERVIEW_CAL.read_text())
        pos, mat, fovy = np.array(cal["pos"]), np.array(cal["xmat"]).reshape(3, 3), float(cal["fovy"])
    else:
        raise ValueError("camera must be one of ['wrist', 'overview']")
    origin, d = _ray(pos, mat, fovy, float(u), float(v))
    if d[2] > -1e-3 and origin[2] > height:
        return f"the ray through ({u}, {v}) does not go down to z = {height} (it points at or above the horizon)"
    t = (height - origin[2]) / d[2]
    p = origin + t * d
    return (f"{camera} pixel ({u:.0f}, {v:.0f}) -> table point [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}] "
            f"({t:.2f} m from the camera, {np.hypot(p[0], p[1]):.3f} m from the arm base).\n{FRAME}")


async def _views(which=("wrist", "overview")) -> list:
    out = []
    for label, prefix in (("wrist", WRIST_CAM), ("overview", OVERVIEW_CAM)):
        if label in which:
            data, note = await _fresh_frame(prefix)
            out += [f"{label} camera {note}", Image(data=_grid(data), format="jpeg")] if data else [f"{label} camera: {note}"]
    return out


def _first(text: str) -> str:
    lines = text.splitlines()
    return lines[0] + "".join(" " + l for l in lines[1:3] if l.startswith("NOTE"))


def _stopped(text: str) -> bool:
    return text.startswith(("ABORTED", "UNREACHABLE"))


OPEN_MAX = float(os.environ.get("ROBOT_GRIPPER_OPEN_MAX", "0.96"))
CLOSED_NORM = float(os.environ.get("ROBOT_GRIPPER_CLOSED", "0.045"))   # j8 above this after closing = holding something
CARRY_Z = float(os.environ.get("ROBOT_CARRY_Z", "0.20"))


@server.tool(annotations={"destructiveHint": True})
@recorded
async def pick_at(x: float, y: float, object_height: float = 0.02, yaw_deg: float = 0.0) -> list:
    """Pick up a small object whose centre is at (x, y) on the table (base frame, from pixel_to_table on a CLOSE
    wrist view): open the jaws, hover above it, descend, close until the jaws stall, lift to z = 0.20. One call
    replaces move/close/lift. object_height is the object's height in metres; yaw_deg turns the jaws about
    vertical (the jaws close along x at 0). Returns what happened, whether something is held, and fresh images.
    Stops at the first refused or aborted move and says where the arm is."""
    log = []
    st = await am._read()
    if st[8]["norm"] < 0.88:
        log.append("open: " + _first(await gripper(open_fraction=1.0)))
    for name, z in (("hover", max(object_height + 0.05, 0.07)), ("descend", max(MIN_Z, object_height / 2 + 0.008))):
        out = await move_tcp(x=x, y=y, z=z, top_down=True, yaw_deg=yaw_deg)
        log.append(f"{name}: " + _first(out))
        if _stopped(out) or "do not grasp" in out:
            return ["PICK STOPPED before closing.\n" + "\n".join(log)] + await _views()
    log.append("close: " + _first(await gripper(open_fraction=0.0)))
    out = await move_tcp(x=x, y=y, z=CARRY_Z, top_down=True, yaw_deg=yaw_deg)
    log.append("lift: " + _first(out))
    st = await am._read()
    held = st[8]["norm"] > CLOSED_NORM
    verdict = (f"HOLDING something (j8 {st[8]['norm']:.3f}, load {st[8]['load']}): check the wrist image shows it between the jaws."
               if held else f"NOT holding (j8 closed fully to {st[8]['norm']:.3f}): the object is still on the table; re-localize and retry.")
    if _stopped(out):
        verdict = "LIFT STOPPED. " + verdict
    return [verdict + "\n" + "\n".join(log)] + await _views()


@server.tool(annotations={"destructiveHint": True})
@recorded
async def drop_at(x: float, y: float, z: float = 0.19) -> list:
    """Carry what the gripper holds to (x, y) at height z (it first rises to z, then travels level), open the jaws
    to release it, then retreat at the same height to above the parking spot (0.0, 0.19). Use z at least 6 cm above
    a box rim. Returns what happened and fresh images."""
    log = []
    out = await move_tcp(x=x, y=y, z=z, top_down=True)
    log.append("carry: " + _first(out))
    if _stopped(out):
        return ["DROP STOPPED before releasing (still holding).\n" + "\n".join(log)] + await _views()
    log.append("release: " + _first(await gripper(open_fraction=1.0)))
    out = await move_tcp(x=0.0, y=0.19, z=max(z, 0.12), top_down=True)
    log.append("retreat: " + _first(out))
    return ["RELEASED at " + str([round(x, 3), round(y, 3), round(z, 3)]) + ".\n" + "\n".join(log)] + await _views()


if __name__ == "__main__":
    server.run()
