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

import robot_profile  # noqa: E402
from robot_profile import env  # noqa: E402

# Everything robot- and setup-specific comes from the profile (sim/robots/<name>.json, chosen with ROBOT_PROFILE).
# The environment variables below override single values, as before.
P = robot_profile.load()
if not os.environ.get("ROBOT_BUS_SERIAL"):
    os.environ["ROBOT_BUS_SERIAL"] = P.bus_serial          # read by arm_mcp.station at import

from arm_mcp import server as am  # noqa: E402  (real Station access; its own MCP server object is unused)

ARM_IDS, GRIP_ID = list(P.motors.arm), int(P.motors.gripper)      # motor ids on the bus
ALL_IDS = ARM_IDS + [GRIP_ID]
NA = len(ARM_IDS)                                                 # number of arm joints (q index NA = the gripper)
NAME = {int(k): v for k, v in vars(P.motors.names).items()}       # motor id -> joint name
ID_OF = {v: k for k, v in NAME.items()}
NAMES = [NAME[m] for m in ALL_IDS]
GRIP_NAME = NAME[GRIP_ID]
GOAL_REG = int(P.motors.goal_register)
am.stn.NAMES[:] = NAMES                                           # arm_mcp's joint table and reads follow the profile
am.stn.IDS[:] = ALL_IDS

IMG_SCALE = env("ROBOT_IMG_SCALE", P.cameras.scale, int)
GRID = env("ROBOT_GRID", "1" if P.cameras.grid else "0", str) == "1"
DROP_LINE = os.environ.get("ROBOT_DROP_LINE", "1") == "1"         # 0: clean wrist images (system2 perception)
MIN_Z = env("ROBOT_MIN_TCP_Z", P.workspace.min_tcp_z)
WRIST_CAM = env("ROBOT_WRIST_CAM", P.cameras.wrist, str)          # cam_stream hash prefixes (per machine)
OVERVIEW_CAM = env("ROBOT_OVERVIEW_CAM", P.cameras.overview, str)
OVERVIEW_CAL = Path(os.environ.get("ROBOT_OVERVIEW_CAL", HERE / "real_overview_camera.json"))
LOG_DIR = Path(os.environ.get("ROBOT_REAL_LOG_DIR", HERE / "real_sessions"))
W, H = P.cameras.width, P.cameras.height
FRAME = P.frame_text
PARK, CARRY_Z = list(P.poses.park), env("ROBOT_CARRY_Z", P.poses.carry_z)
WS, MO, GR, PK, DR, ST, IKP, TS, GB, WP = (P.workspace, P.motion, P.gripper, P.pick, P.drop, P.stir, P.ik, P.touch, P.grab,
                                              P.wipe)

IK = robot_profile.Kin(P)                            # forward kinematics, wrist-camera pose, position-only IK
MODEL = IK.model
LO, HI = IK.low, IK.high                             # joint limits (rad) of [arm joints..., gripper]

# arm_mcp.station.bus_frame() calls cli.follow() on every read and never unsubscribes, so a long session piles
# hundreds of subscriptions onto one Station connection until reads time out (seen after a few minutes of an
# operator run). Here one subscription per connection is kept and each read takes the next fresh entry from it.
_state_queues: dict = {}


async def _bus_frame_once(cli):
    stn = am.stn
    q = _state_queues.get(id(cli))
    if q is None:
        _state_queues.clear()
        q = asyncio.Queue()
        cli.follow(stn.ST3215_QUEUE, q)
        _state_queues[id(cli)] = q
    while not q.empty():                 # drop entries that arrived while nobody was reading
        q.get_nowait()
    try:
        e = await asyncio.wait_for(q.get(), 5)
    except asyncio.TimeoutError:
        # The subscription went silent (22 Nov: after the event loop was blocked for ~15 s under CPU load, every read
        # timed out until restart). Subscribe again once; if that is silent too, the next call retries again.
        _state_queues.clear()
        q = asyncio.Queue()
        cli.follow(stn.ST3215_QUEUE, q)
        _state_queues[id(cli)] = q
        e = await asyncio.wait_for(q.get(), 5)
    st = stn.st3215.InferenceStateReader(memoryview(bytes(e.Data)))
    for bs in st.get_buses() or []:
        if stn.as_str(bs.get_bus().get_serial_number()) == stn.BUS:
            return bs.get_motors() or []
    return None


if os.environ.get("ROBOT_SINGLE_SUBSCRIPTION", "1") == "1":
    am.stn.bus_frame = _bus_frame_once

# Optional calibration override: Station's stored ranges can go stale (21 Nov: this machine's Station kept j6
# 1688-2408 / j8 1898-2198 while the follower was re-calibrated to 1006-3090 / 1228-2868). ROBOT_RANGES_FILE
# (JSON {"j1": [lo, hi], ...}, raw ticks) replaces lo/hi/norm/in_range for every read; Station is not changed.
RANGES_FILE = os.environ.get("ROBOT_RANGES_FILE")
if RANGES_FILE:
    _ranges = {ID_OF[k]: (int(v[0]), int(v[1]))
               for k, v in json.loads(Path(RANGES_FILE).read_text()).items() if k in NAMES}
    _station_read = am._read

    async def _read_with_ranges() -> dict:
        st = await _station_read()
        for mid, (lo, hi) in _ranges.items():
            m = st[mid]
            m.update(lo=lo, hi=hi, norm=round((m["raw"] - lo) / (hi - lo), 4), in_range=lo <= m["raw"] <= hi)
        return st

    am._read = _read_with_ranges   # arm_mcp's joint_state / set_torque / _table pick this up too

server = FastMCP(P.mcp_server_name)
LOG_DIR.mkdir(exist_ok=True)
SESSION_LOG = LOG_DIR / "actions.jsonl"


import operator_prompt  # noqa: E402

DOC_VALUES = {**operator_prompt.values(P.raw), "MAX_DELTA": am.MAX_DELTA}


def doc(fn):
    """Fill <<NAME>> placeholders in a tool's description from the robot profile (what the model reads)."""
    fn.__doc__ = operator_prompt.fill(fn.__doc__ or "", DOC_VALUES)
    return fn


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

ARM_MAPPING, GRIP_MAPPING = P.kinematics.arm_mapping, P.kinematics.gripper_mapping
TICK_RAD = 2 * np.pi / P.motors.ticks_per_turn
_last_ranges: dict = {}        # motor id -> (lo, hi) of the latest read, for the tick-based mapping


async def measured():
    """(q, state): q = [arm joints..., gripper] in radians from the motor bus, by the profile's joint mapping.
    range_fraction: q = lower + norm * (upper - lower), norm = position in the calibrated range (ElRobot).
    ticks_centered: q = sign * (raw - zero_tick) * 2 pi / ticks_per_turn (arms calibrated around a zero pose)."""
    st = await am._read()
    n = [st[mid]["norm"] for mid in ALL_IDS]
    if any(x is None for x in n):
        raise RuntimeError("a joint has no calibrated range; calibrate before using kinematics")
    for mid in ALL_IDS:
        _last_ranges[mid] = (st[mid]["lo"], st[mid]["hi"])
    if ARM_MAPPING == "range_fraction":
        arm = [LO[i] + n[i] * (HI[i] - LO[i]) for i in range(NA)]
    elif ARM_MAPPING == "ticks_centered":
        z, sg = P.kinematics.zero_ticks, P.kinematics.signs
        arm = [sg[i] * (st[ARM_IDS[i]]["raw"] - z[i]) * TICK_RAD for i in range(NA)]
    else:
        raise ValueError(f"unknown arm_mapping {ARM_MAPPING!r}")
    grip = HI[NA] * (1 - n[NA]) if GRIP_MAPPING == "upper_times_one_minus_norm" else LO[NA] + n[NA] * (HI[NA] - LO[NA])
    return np.array(arm + [grip]), st


def to_norm(q):
    """Arm joint angles -> normalized targets in each motor's calibrated range (what the motor layer takes)."""
    if ARM_MAPPING == "range_fraction":
        return (q[:NA] - LO[:NA]) / (HI[:NA] - LO[:NA])
    z, sg = P.kinematics.zero_ticks, P.kinematics.signs
    raw = [z[i] + sg[i] * q[i] / TICK_RAD for i in range(NA)]
    return np.array([(raw[i] - _last_ranges[ARM_IDS[i]][0]) / (_last_ranges[ARM_IDS[i]][1] - _last_ranges[ARM_IDS[i]][0]) for i in range(NA)])


# --- inverse kinematics: as close to jaws-down as the target allows ------------------------------------
# build_scene.ArmIK.solve either gets jaws exactly down or, as a fallback, ignores orientation completely; from a
# folded seed that fallback left the gripper 76 deg from vertical on hardware. Here the lean is stepped up
# gradually, so the result is the least-tilted reachable pose. The lean list, restarts and search order are the shared
# skill config (learning/skillcfg.py: ik.leans, ik.restarts, ik.search; defaults = the profile's ik section), read
# on every solve: a new approved version applies without a restart.
from learning import skillcfg as SC  # noqa: E402

SC.set_target("real")      # never the twin config (proposals made there are applied without a human)


def _yaw_now(q) -> float:
    """Current rotation of the jaw closing axis about vertical, in degrees (what yaw_deg sets)."""
    _, rot = IK.fk(q)
    return float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))


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


def _solve_rot(xyz, Rt, q0, restarts: int = 0, seed: int = IKP.seed):
    """Damped least squares to position + rotation Rt. First from q0; with restarts, the solution nearest q0."""
    m, d = MODEL, IK.data
    rng = np.random.default_rng(seed)
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    best = None
    for attempt in range(1 + restarts):
        q = np.array(q0, float)
        if attempt:
            q[:NA] = rng.uniform(IK.low[:NA], IK.high[:NA])
        for _ in range(IKP.iterations):
            d.qpos[IK.qids] = q
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            err = np.asarray(xyz) - d.site_xpos[IK.site]
            mujoco.mj_jacSite(m, d, jp, jr, IK.site)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, (Rt @ d.site_xmat[IK.site].reshape(3, 3).T).reshape(-1))
            if quat[0] < 0: quat *= -1
            ln = np.linalg.norm(quat[1:])
            err_r = quat[1:] * (2 * np.arctan2(ln, quat[0]) / max(ln, 1e-9))
            if np.linalg.norm(err) < IKP.pos_tol and np.linalg.norm(err_r) < IKP.rot_tol:
                if best is None or np.abs(q[:NA] - q0[:NA]).max() < np.abs(best[:NA] - q0[:NA]).max():
                    best = q.copy()
                break
            jac, e = np.vstack([jp[:, IK.dofs], IKP.rot_weight * jr[:, IK.dofs]]), np.r_[err, IKP.rot_weight * err_r]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + IKP.damping * np.eye(len(e)), e)
            q[:NA] = np.clip(q[:NA] + np.clip(dq, -IKP.max_step, IKP.max_step), IK.low[:NA] + IKP.limit_margin, IK.high[:NA] - IKP.limit_margin)
        if attempt == 0 and best is not None:
            return best
    return best


def _solve_tcp(xyz, q0, yaw_deg: float):
    """(q, lean_deg) for the least-leaning reachable pose, preferring one near the current joints; (None, None).
    skillcfg.lean_search, search "least_lean": pass 1 stays on the current arm configuration (no restarts); pass 2
    tries random restarts (solution nearest the current joints) only for leans smaller than pass 1 found, so a less
    tilted pose always wins. (22 Nov: the old order accepted 60 deg from the current joints before trying restarts at
    40, where the arm reaches the bottle cap at ~35 deg, which pushed twist_cap into the side approach.)"""
    return SC.lean_search(lambda lean, rs: _solve_rot(xyz, _target_rot(xyz, lean, yaw_deg), q0, rs))


# --- motion on real servos ----------------------------------------------------------------------------
# The ST3215s run P-only position control: under gravity a joint stops short of its goal, and a goal equal to
# the present position gives no force at all. So (a) every joint we are NOT moving keeps its current goal
# register (keeps the gripper's squeeze and the arm's gravity offsets; arm_mcp's moves reset them to the present
# position, which drops a held object and lets the arm sag), and (b) after creeping, planned joints are nudged
# closed-loop until the measured position arrives. Contact (motor load) or a fault holds the present pose.
LOAD_ABORT = env("ROBOT_LOAD_ABORT", MO.load_abort, int)      # servo present load, 0..1000
SETTLE_TOL = MO.settle_tol                                    # ticks
MAX_TICKS_PER_STEP = env("ROBOT_MAX_TICKS_PER_STEP", MO.max_ticks_per_step)   # joint speed limit per creep step
GRIP_SQUEEZE = env("ROBOT_GRIP_SQUEEZE", GR.squeeze_ticks, int)   # ticks the jaw goal stays past a stalled grip
OPEN_MAX = env("ROBOT_GRIPPER_OPEN_MAX", GR.open_max)         # the jaws stop mechanically before 1.0
STEP, DT = MO.step, MO.dt                                     # default creep: fraction of a joint's range per DT seconds


def _holds(st):
    """Current goal of every motor, falling back to the present position if the goal register is far off."""
    return {mid: (st[mid]["goal"] if abs(st[mid]["goal"] - st[mid]["raw"]) < MO.goal_reset_gap else st[mid]["raw"]) for mid in am.IDS}


async def _stop_here(st, mids):
    await am._write(GOAL_REG, {**_holds(st), **{mid: st[mid]["raw"] for mid in mids}}, 2)


def _flagged(st, plan, grip=False):
    return {m for m in am.IDS if st[m]["status"] and (m != GRIP_ID or (GRIP_ID in plan and not grip))}


async def _faults(now, plan, grip=False):
    """Motor status flags that are still set on a second read 0.15 s later (one-read glitches are ignored)."""
    first = _flagged(now, plan, grip)
    if not first:
        return []
    await asyncio.sleep(MO.fault_recheck_s)
    again = await am._read()
    return [(NAME[m], hex(again[m]["status"])) for m in sorted(first & _flagged(again, plan, grip))]


async def _drive(plan: dict, step: float = STEP, dt: float = DT, track_tol: int | None = None,
                 settle: bool = True, contact: bool = True, grip: bool = False,
                 window: int | None = None, iters: int | None = None) -> str:
    """plan {motor_id: target norm}. Creep, then settle closed-loop. Returns 'reached ...' or 'ABORTED ...'.
    A status flag on the gripper motor alone (overload from squeezing an object) never aborts an arm move."""
    track_tol = am.TRACK_TOL if track_tol is None else track_tol
    st = await am._read()
    if any(st[mid]["torque"] == 0 for mid in am.IDS):
        raise RuntimeError("torque is off on some motors: call set_torque(true) first")
    for mid in plan:
        if not st[mid]["in_range"]:
            raise RuntimeError(f"{NAME[mid]} raw {st[mid]['raw']} is outside its calibrated range; commanding it would snap")
    over = [m for m in am.IDS if m != GRIP_ID and st[m]["status"]]
    if over:       # a motor overloaded while holding (seen on j2 after minutes stretched out): unload it, then go on
        await am._write(GOAL_REG, {**_holds(st), **{m: st[m]["raw"] for m in over}}, 2)
        await asyncio.sleep(MO.unload_wait_s)
        st = await am._read()
        still = [(NAME[m], hex(st[m]["status"])) for m in am.IDS if m != GRIP_ID and st[m]["status"]]
        if still:
            raise RuntimeError(f"motor fault {still} did not clear after unloading: nothing moved. Let the arm rest "
                               f"(support it, set_torque(false)) or power-cycle the 12 V before continuing")
    raw_goal = {mid: int(round(st[mid]["lo"] + min(1.0, max(0.0, n)) * (st[mid]["hi"] - st[mid]["lo"]))) for mid, n in plan.items()}
    start = {mid: st[mid]["raw"] for mid in plan}
    hold = _holds(st)
    span = {mid: st[mid]["hi"] - st[mid]["lo"] for mid in plan}
    # Speed limit per joint in servo ticks, not only as a fraction of its range: j5 and j7 have almost twice the
    # range of the others, so a pure fraction drove them twice as fast and they fell behind (tracking abort on any
    # large gripper rotation).
    n = max([1] + [int(abs(raw_goal[m] - start[m]) / min(step * span[m], MAX_TICKS_PER_STEP) + 0.999) for m in plan])
    for k in range(1, n + 1):
        goals = dict(hold)
        for m in plan:
            goals[m] = int(round(start[m] + (raw_goal[m] - start[m]) * k / n))
        await am._write(GOAL_REG, goals, 2)
        await asyncio.sleep(dt)
        now = await am._read()
        bad = [(NAME[m], goals[m], now[m]["raw"]) for m in plan if abs(goals[m] - now[m]["raw"]) > track_tol]
        bad = bad or await _faults(now, plan, grip)
        hot = [(NAME[m], now[m]["load"]) for m in plan if contact and m != GRIP_ID and now[m]["load"] > LOAD_ABORT]
        if bad or hot:
            await _stop_here(now, [m for m in plan if m != GRIP_ID])
            why = f"tracking error or motor fault {bad}" if bad else f"contact: load over {LOAD_ABORT} {hot}"
            return f"ABORTED at step {k}/{n}: {why}; holding present pose."
    if not settle:
        return f"reached in {n} steps."
    cmd = dict(raw_goal)
    window = MO.settle_window if window is None else window      # how far past the target a goal may be pushed (= force)
    for it in range(MO.settle_iters if iters is None else iters):
        await asyncio.sleep(MO.settle_wait_s)
        now = await am._read()
        err = {m: raw_goal[m] - now[m]["raw"] for m in plan}
        if all(abs(e) <= SETTLE_TOL for e in err.values()):
            return f"reached in {n} steps (+{it} corrections)."
        hot = [(NAME[m], now[m]["load"]) for m in plan if contact and m != GRIP_ID and now[m]["load"] > LOAD_ABORT]
        fault = [] if hot else await _faults(now, plan, grip)
        if fault or hot:
            await _stop_here(now, [m for m in plan if m != GRIP_ID])
            return f"ABORTED while settling: {'contact ' + str(hot) if hot else 'motor fault ' + str(fault)}; holding present pose."
        for m, e in err.items():
            if abs(e) > SETTLE_TOL:
                c = cmd[m] + round(MO.settle_gain * e)
                c = max(raw_goal[m] - window, min(raw_goal[m] + window, c))
                cmd[m] = max(now[m]["lo"] + MO.range_margin, min(now[m]["hi"] - MO.range_margin, c))
        await am._write(GOAL_REG, {**_holds(now), **cmd}, 2)
    left = {NAME[m]: e for m, e in err.items() if abs(e) > SETTLE_TOL}
    return f"reached in {n} steps; still short by {left} ticks (load or an obstacle)."


# --- cameras -------------------------------------------------------------------------------------------

def _drop_line(q):
    """Pixels (in the returned 2x image) of the vertical line from the TCP down to the table: where a released
    object falls. None when the gripper is low (nothing to drop into from there)."""
    tcp, _ = IK.fk(q)
    if tcp[2] < DR.line_min_tcp_z:
        return None
    pos, mat, fovy = _wrist_pose(q)
    f = (H / 2) / np.tan(np.radians(fovy) / 2)
    pts = []
    for z in np.linspace(tcp[2] - DR.line_top_gap, 0.0, DR.line_points):
        pc = mat.T @ (np.array([tcp[0], tcp[1], z]) - pos)
        if pc[2] < -1e-6:
            pts.append(((W / 2 + f * pc[0] / -pc[2]) * IMG_SCALE, (H / 2 - f * pc[1] / -pc[2]) * IMG_SCALE))
    return pts if len(pts) >= 2 else None


def _grid(jpeg: bytes, drop=None) -> bytes:
    im = PILImage.open(io.BytesIO(jpeg)).convert("RGB")
    if IMG_SCALE != 1:
        im = im.resize((im.width * IMG_SCALE, im.height * IMG_SCALE), PILImage.BICUBIC)
    if GRID:
        draw = ImageDraw.Draw(im, "RGBA")
        gs = P.cameras.grid_step
        for x in range(0, im.width, gs):
            draw.line([(x, 0), (x, im.height)], fill=(255, 64, 64, 70 if x % (2 * gs) else 120))
            if x: draw.text((x + 2, 1), str(x), fill=(255, 40, 40, 255))
        for y in range(0, im.height, gs):
            draw.line([(0, y), (im.width, y)], fill=(255, 64, 64, 70 if y % (2 * gs) else 120))
            if y: draw.text((2, y + 1), str(y), fill=(255, 40, 40, 255))
    if drop:
        draw = ImageDraw.Draw(im, "RGBA")
        draw.line(drop, fill=(0, 230, 0, 255), width=3)
        for (x, y) in (drop[0], drop[-1]):
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(0, 230, 0, 255), width=2)
        draw.text((drop[-1][0] + 8, drop[-1][1] - 6), "DROP", fill=(0, 200, 0, 255))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=P.cameras.jpeg_quality)
    return buf.getvalue()


async def _wrist_image(data: bytes) -> bytes:
    """Wrist frame with the grid and, when the gripper is raised, the green DROP line."""
    q, _ = await measured()
    return _grid(data, _drop_line(q) if DROP_LINE else None)


async def _fresh_frame(prefix: str):
    """(jpeg, note) from the relay; same freshness rule as arm_mcp.look (age and byte change)."""
    cam = next((c for c in am._cameras() if c["hash"].startswith(prefix)), None)
    if cam is None:
        return None, f"camera {prefix}: not live on the relay"
    note = ""
    for _ in range(P.cameras.fresh_tries):
        data = am._snapshot(cam["hash"])
        digest = __import__("hashlib").sha1(data).hexdigest()
        age = next((c["age"] for c in am._cameras() if c["hash"] == cam["hash"]), None)
        if age is not None and age <= am.MAX_CAM_AGE and digest != am._last_hash.get(cam["hash"]):
            am._last_hash[cam["hash"]] = digest
            return data, f"fresh, age={age:.2f}s"
        note = f"age={age}"
        await asyncio.sleep(P.cameras.fresh_wait_s)
    return None, f"STALE ({note})"


def _wrist_pose(q):
    return IK.wrist_pose(q)


def _ray(pos, mat, fovy, u, v):
    """Same pinhole maths as sim_arm_mcp.pixel_ray (MuJoCo camera: looks along -z, y up)."""
    u, v = u / IMG_SCALE, v / IMG_SCALE
    f = (H / 2) / np.tan(np.radians(fovy) / 2)
    d = mat @ np.array([(u - W / 2) / f, -(v - H / 2) / f, -1.0])
    return pos, d / np.linalg.norm(d)


# --- tools (names, arguments and texts as in sim_arm_mcp.py) -------------------------------------------

@server.tool(annotations={"readOnlyHint": True})
@recorded
@doc
async def look() -> list:
    """FRESH images from every live camera plus the live motor-bus joint table. First image: wrist camera;
    second: overview camera. Both are <<IMG>> with a red pixel grid labelled every <<GRID_STEP>> px."""
    out: list = []
    for label, prefix in (("wrist camera", WRIST_CAM), ("overview camera", OVERVIEW_CAM)):
        data, note = await _fresh_frame(prefix)
        if data is None:
            out.append(f"{label} {prefix}: {note}, not shown")
            continue
        out.append(f"{label} {prefix} {note}, captured {time.strftime('%H:%M:%S')}")
        out.append(Image(data=await _wrist_image(data) if prefix == WRIST_CAM else _grid(data), format="jpeg"))
    out.append(am._table(await am._read()))
    return out


@server.tool(annotations={"readOnlyHint": True})
@recorded
@doc
async def joint_state() -> str:
    """Live joint table from the motor bus: raw ticks, calibrated range, in-range flag, torque, status, load."""
    return await am.joint_state()


@server.tool(annotations={"destructiveHint": True})
@recorded
@doc
async def set_torque(enabled: bool) -> str:
    """Enable/disable torque on all motors. Enabling first writes goal = present position so nothing jumps.
    Disabling makes the arm limp: support it before calling with False."""
    return await am.set_torque(enabled)


@server.tool(annotations={"destructiveHint": True})
@recorded
@doc
async def move_to(targets: dict[str, float], step: float = STEP, dt: float = DT) -> str:
    """Creep joints to ABSOLUTE normalized targets in [0,1] (calibrated range), e.g. {"<<FIRST_JOINT>>":0.5}.
    Refuses big jumps (> <<MAX_DELTA>>), joints outside their range, or torque off; aborts and holds on a tracking error."""
    unknown = [k for k in targets if k not in NAMES]
    if unknown:
        raise ValueError(f"unknown joints {unknown}; valid {NAMES}")
    st = await am._read()
    plan = {}
    for name, want in targets.items():
        mid = ID_OF[name]
        want = min(1.0, max(0.0, float(want)))
        if st[mid]["norm"] is None or abs(want - st[mid]["norm"]) > am.MAX_DELTA:
            raise ValueError(f"{name}: {st[mid]['norm']} -> {want:.3f} exceeds max delta {am.MAX_DELTA}; use waypoints")
        plan[mid] = want
    out = await _drive(plan, step, dt)
    return out + "\n" + am._table(await am._read())


def _jaw_width(norm: float) -> float:
    """Gap between the fingertip pads in metres for a normalized jaw position (from the profile's jaw travel)."""
    return max(0.0, 2 * GR.jaw_travel_each * float(norm))


def _touch(st, force: float | None = None, probed: bool | None = None) -> str:
    """One line of 'touch sense' from the gripper motor: opening, estimated object width, load, verdict."""
    g = st[GRIP_ID]
    is_open = g["norm"] >= GR.open_enough            # fully open jaws rest on their end stop: that load is not a grip
    holding = (not is_open) and ((g["norm"] > CLOSED_NORM and g["load"] >= GR.holding_load) if probed is None else probed)
    thin = holding and g["norm"] <= CLOSED_NORM
    f = f", force setting {force:.2f}" if force is not None else ""
    return (f"touch: jaws at {g['norm']:.3f} (gap about {_jaw_width(g['norm']) * 1000:.0f} mm), motor load {g['load']}/1000{f}, "
            f"{g['temp']} C -> " + ("HOLDING something thin (a sheet: napkin, paper, foil)" if thin else
                                    "HOLDING something about that wide" if holding else
                                    "closed on nothing" if g["norm"] < _thin_min() or (g["norm"] <= CLOSED_NORM and probed is not True) else "open" if is_open else
                                    "NOTHING between the jaws (they closed further when tested: the object slipped out)" if probed is False
                                    else "partly closed, not gripping"))


async def _probe_hold() -> tuple[bool, dict]:
    """Is something really between the jaws? Ask them to close a little further: free jaws follow, held jaws do not.
    (After a contact stop the jaws stay where they are even if the object has slipped out, so position and load
    alone cannot tell.) Returns (holding, state); the grip is restored to its limited squeeze afterwards."""
    st = await am._read()
    g = st[GRIP_ID]
    if g["norm"] >= GR.open_enough or g["norm"] < _thin_min():  # open, or closed as far as empty jaws go
        return False, st
    raw0 = g["raw"]
    await am._write(GOAL_REG, {**_holds(st), GRIP_ID: raw0 - GR.probe_ticks}, 2)
    await asyncio.sleep(GR.probe_wait_s)
    st = await am._read()
    moved = raw0 - st[GRIP_ID]["raw"]
    holding = moved < GR.probe_moved_fraction * GR.probe_ticks and st[GRIP_ID]["norm"] >= _thin_min()
    if holding and st[GRIP_ID]["norm"] <= CLOSED_NORM:
        # Thin regime: empty jaws at their end stop cannot close further either, so position alone is not proof.
        # A sheet between the pads also raises the motor load (measured: 124-128 held against 92-96 empty).
        empty_loads = [e[1] for e in _EMPTY if e[1] is not None]
        if empty_loads and g["load"] < float(np.median(empty_loads)) + GR.thin_load_margin:
            holding = False
    if holding:                                            # back to the gentle hold
        await am._write(GOAL_REG, {**_holds(st), GRIP_ID: st[GRIP_ID]["raw"] - GR.squeeze_min_ticks}, 2)
    return holding, st


@server.tool(annotations={"destructiveHint": True})
@recorded
@doc
async def gripper(open_fraction: float, force: float | None = None) -> str:
    """Set the gripper jaws (<<GRIPPER_JOINT>>) in ONE call: 1.0 = fully open, 0.0 = closed. `force` (0.1 .. 1.0, default
    <<GRIP_FORCE>>) limits how hard the jaws squeeze: it caps the motor's torque while closing and holding. Use 0.2-0.3
    for hard, smooth or light things that pop out when squeezed (bottle caps, marbles, plastic lids), about 0.5 for
    wrappers and soft items, up to 1.0 only for heavy objects. Closing is slow and stops at first contact; the reply
    has a "touch:" line with the jaw gap in mm (the object's width), the motor load and whether something is held."""
    st = await am._read()
    if any(st[mid]["torque"] == 0 for mid in am.IDS):
        raise RuntimeError("torque is off on some motors: call set_torque(true) first")
    G = GRIP_ID
    if not st[G]["in_range"]:
        raise RuntimeError(f"{GRIP_NAME} is outside its calibrated range; commanding it would snap")
    force = float(min(1.0, max(GR.force_min, GR.force_default if force is None else force)))
    lo, hi = st[G]["lo"], st[G]["hi"]
    target = int(round(lo + min(OPEN_MAX, max(0.0, float(open_fraction))) * (hi - lo)))   # the jaws stop mechanically before 1.0
    start, hold = st[G]["raw"], _holds(st)
    closing = target < start
    # torque limit: gentle while closing/holding, full for opening (friction) so the jaws never stick shut
    await am._write(GR.torque_limit_register, {G: max(GR.torque_min, int(round(force * GR.torque_full))) if closing else GR.torque_full}, 2)
    frac = GR.close_step_fraction if closing else GR.step_fraction
    n = max(1, int(abs(target - start) / (frac * (hi - lo)) + 0.999))
    prev, still, contact = start, 0, False
    for k in range(1, n + 1):
        goal = int(round(start + (target - start) * k / n))
        await am._write(GOAL_REG, {**hold, G: goal}, 2)
        await asyncio.sleep(GR.step_wait_s)
        now = (await am._read())[G]["raw"]
        still = still + 1 if abs(now - prev) < GR.stall_still else 0
        if closing and now - goal > GR.contact_lag and still >= GR.contact_reads:    # behind the goal and not moving: touching
            contact = True
            break
        prev = now
    await asyncio.sleep(GR.after_wait_s)
    st = await am._read()
    if closing and (contact or st[G]["raw"] - st[G]["goal"] > GR.contact_lag) and st[G]["norm"] > CLOSED_NORM:
        squeeze = max(GR.squeeze_min_ticks, int(round(GRIP_SQUEEZE * force)))        # hold with a small, force-scaled squeeze
        await am._write(GOAL_REG, {**_holds(st), G: st[G]["raw"] - squeeze}, 2)
        await asyncio.sleep(GR.squeeze_wait_s)
        st = await am._read()
        held, st = await _probe_hold()
        out = (f"Jaws stalled on something at {st[G]['norm']:.3f}: holding it with a limited squeeze." if held else
               f"Jaws touched something but it slipped out: nothing is held (jaws now at {st[G]['norm']:.3f}).")
        return out + "\n" + _touch(st, force, held) + "\n" + am._table(st)
    elif closing and st[G]["norm"] <= CLOSED_NORM:
        held, st = await _probe_hold()                                               # a thin sheet keeps the jaws a hair apart
        if held:
            return (f"Jaws closed on something thin at {st[G]['norm']:.3f} (empty jaws close to about {GR.empty_closed_norm}).\n"
                    + _touch(st, force, True) + "\n" + am._table(st))
        _note_empty(st[G]["norm"], st[G]["load"])                                    # this is what empty jaws read
        await am._write(GOAL_REG, {**_holds(st), G: st[G]["raw"]}, 2)                # closed on nothing: stop pushing the jaws together
        out = f"gripper closed fully at {st[G]['norm']:.3f}: nothing between the jaws."
    else:
        if st[G]["goal"] - st[G]["raw"] > GR.open_stop_slack:                        # opened against the mechanical stop: stop pushing
            await am._write(GOAL_REG, {**_holds(st), G: st[G]["raw"]}, 2)
        out = f"gripper at {st[G]['norm']:.3f}."
    return out + "\n" + _touch(st, force if closing else None) + "\n" + am._table(st)


@server.tool(annotations={"destructiveHint": True})
@recorded
@doc
async def grip_state() -> str:
    """Touch sense: what the gripper feels right now. Jaw opening and gap in mm, the motor's load (how hard it is
    squeezing, 0..1000), its temperature and torque limit, and whether something is really held: the jaws are
    asked to close a fraction more, and only a real object stops them. Call it after a lift or a carry."""
    held, st = await _probe_hold()
    motors = await am.stn.bus_frame(await am._client())
    limit = next((int.from_bytes(bytes(m.get_state())[GR.torque_limit_register:GR.torque_limit_register + 2], "little")
                  for m in motors or [] if int(m.get_id()) == GRIP_ID), None)
    g = st[GRIP_ID]
    probed = held if (_thin_min() <= g["norm"] < GR.open_enough) else None
    return _touch(st, None, probed) + f"\ntorque limit {limit}/{GR.torque_full}."


@server.tool(annotations={"readOnlyHint": True})
@recorded
@doc
async def tcp_pose() -> str:
    """Where the gripper is, from the measured joints and the arm's kinematic model: the TCP (point between
    the fingertip pads) in the base frame, whether the jaws point down, and the jaw axis direction."""
    q, _ = await measured()
    pos, rot = IK.fk(q)
    approach, jaw = rot[:, 1], rot[:, 0]
    tilt = np.degrees(np.arccos(np.clip(-approach[2], -1, 1)))
    level = np.degrees(np.arcsin(np.clip(abs(jaw[2]), 0, 1)))
    return (f"tcp xyz = [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m; jaws point {tilt:.0f} deg from straight down; "
            f"jaw closing axis yaw = {np.degrees(np.arctan2(jaw[1], jaw[0])):.0f} deg ({level:.0f} deg from level).\n{FRAME}")


@server.tool(annotations={"destructiveHint": True})
@recorded
@doc
async def move_tcp(x: float, y: float, z: float, top_down: bool = True, yaw_deg: float | None = None,
                   step: float = STEP, dt: float = DT) -> str:
    """Move the gripper TCP (between the fingertip pads) to (x, y, z) in the base frame using the arm's inverse
    kinematics, executed with move_to in bounded steps (same refusals and tracking aborts). top_down=True keeps
    the jaws pointing straight down (where impossible it uses the nearest tilted pose and says so); yaw_deg turns
    the jaw closing axis about vertical (0 = jaws close along x); leave it out to KEEP the gripper's current
    rotation, so something you picked up at an angle stays at that angle. If the target is more than <<DIRECT_CM>> cm away sideways, it travels at a safe
    height (max(target, <<TRAVEL_H>> m)), lifting first, then descends. The gripper opening is not changed.
    Table top is z = 0; on the real arm the TCP may not go below z = <<MIN_Z>> (fingertips at the table)."""
    target = np.array([x, y, z], float)
    if not (MIN_Z <= z <= WS.max_tcp_z and np.hypot(x, y) <= WS.max_radius):
        return (f"UNREACHABLE: {target.round(3).tolist()} is outside the arm's workspace (z {MIN_Z}..{WS.max_tcp_z} m, at most "
                f"{WS.max_radius} m from the base; this point is {np.hypot(x, y):.2f} m away). Nothing moved.")
    q, _ = await measured()
    cur, _ = IK.fk(q)
    if yaw_deg is None:
        yaw_deg = _yaw_now(q)
    waypoints = []
    if np.hypot(*(target[:2] - cur[:2])) > WS.direct_move_within:
        safe = max(z, WS.travel_height)
        if cur[2] < safe - WS.travel_slack: waypoints.append([cur[0], cur[1], safe])
        if z < safe: waypoints.append([x, y, safe])
    waypoints.append(target.tolist())
    seed, log, notes = q, [], []
    for wp in waypoints:
        if top_down:
            # worker thread: the least-lean search takes 1-2 s, and blocking the event loop stalls the Station stream
            sol, lean = await asyncio.to_thread(_solve_tcp, np.array(wp), seed, yaw_deg)
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
        if sol is not None and float(np.abs(to_norm(sol) - to_norm(seed)).max()) > MO.reconfig_note:
            notes.append("large change of arm configuration for this move")
        if sol is None:
            return (f"UNREACHABLE: no joint solution for tcp {np.round(wp, 3).tolist()} "
                    f"({'jaws down' if top_down else 'any orientation'}); nothing moved after: {log}\n" + await am.joint_state())
        seed = sol
        goal = to_norm(sol)
        for _ in range(MO.chunk_tries):          # chunks of <= MO.chunk norm per joint, the last one settles closed-loop
            qn, _ = await measured()
            now = to_norm(qn)
            if float(max(abs(goal - now))) < MO.arrive_tol:
                break
            chunk = np.clip(goal - now, -MO.chunk, MO.chunk)
            last = bool(np.all(np.abs(goal - now) <= MO.chunk))
            out = await _drive({ARM_IDS[i]: float(now[i] + chunk[i]) for i in range(NA)}, step, dt, settle=last)
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
    notes.append(f"tcp is {off * 1000:.0f} mm from the target") if off > MO.off_target_note else None
    note = ("\nNOTE: " + "; ".join(notes)) if notes else ""
    return f"reached via {log}; tcp now [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}].{note}\n" + await am.joint_state()


@server.tool(annotations={"readOnlyHint": True})
@recorded
@doc
async def pixel_to_table(camera: str, u: float, v: float, height: float = 0.0, view: str = "latest") -> str:
    """Turn a pixel of a look() image into a base-frame point using the camera calibration (wrist: CAD mount pose
    + current joints; overview: fixed extrinsics, once measured). (u, v) are pixel coordinates in the returned
    <<IMG>> image (red grid labels), u right, v down. Returns where the viewing ray hits the plane z = height
    (0 = table top; e.g. 0.01 for the centre of a 2 cm cube, ~<<BOX_H>> for a box rim). view="latest" means the wrist
    image taken where the arm is NOW; for a pixel of a survey() image pass view = one of <<SURVEY_NAMES>>."""
    if camera == "wrist" and view != "latest":
        if view not in SURVEY_VIEWS:
            raise ValueError(f"view must be 'latest' or one of {sorted(SURVEY_VIEWS)} (the images of the last survey())")
        pos, mat, fovy = SURVEY_VIEWS[view]
    elif camera == "wrist":
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


@server.tool(annotations={"readOnlyHint": True})
@recorded
async def camera_info() -> str:
    """Camera calibration as JSON, the model pixel_to_table uses (same contract as sim_arm_mcp.camera_info): per
    camera the look() image size, pinhole fx, fy, cx, cy in look() pixels and the pose in the base frame in the
    OpenCV convention (x right, y down, z forward): p_base = R_base_cam @ p_cam + t_base_cam. The wrist pose
    follows the current joints; the overview camera is listed only once sim/real_overview_camera.json exists."""
    q, _ = await measured()
    poses = {"wrist": _wrist_pose(q)}
    if OVERVIEW_CAL.exists():
        cal = json.loads(OVERVIEW_CAL.read_text())
        poses["overview"] = (np.array(cal["pos"]), np.array(cal["xmat"]).reshape(3, 3), float(cal["fovy"]))
    out = {}
    for name, (pos, mat, fovy) in poses.items():
        f = (H / 2) / np.tan(np.radians(fovy) / 2) * IMG_SCALE
        out[name] = {"width": W * IMG_SCALE, "height": H * IMG_SCALE, "fx": f, "fy": f,
                     "cx": W * IMG_SCALE / 2, "cy": H * IMG_SCALE / 2,
                     "R_base_cam": (mat @ np.diag([1.0, -1.0, -1.0])).round(6).tolist(),
                     "t_base_cam": pos.round(6).tolist()}
    return json.dumps(out)


# --- feeling the table --------------------------------------------------------------------------------
# The model puts the table at z = 0, but where the fingertips really meet it depends on the arm's sag at that reach
# and on calibration. A guarded descent finds it by touch and the height is remembered per spot.
TABLE_FILE = LOG_DIR / "table_touch.json"
try:
    TABLE_TOUCHES: list = json.loads(TABLE_FILE.read_text())
except (OSError, ValueError):
    TABLE_TOUCHES = []


def _table_tcp_z(x: float, y: float):
    """Measured gripper (TCP) height at which the fingertips touch the table near (x, y), or None if never felt there."""
    near = [(float(np.hypot(t[0] - x, t[1] - y)), t[2]) for t in TABLE_TOUCHES]
    near = [n for n in near if n[0] <= TS.reuse_radius and abs(n[1] - TS.nominal_table_tcp_z) <= TS.plausible]
    return min(near)[1] if near else None


def _remember_touch(x: float, y: float, z: float) -> None:
    if abs(z - TS.nominal_table_tcp_z) > TS.plausible:      # that was an object or a false stop, not the table
        return
    TABLE_TOUCHES.append([round(x, 4), round(y, 4), round(z, 4)])
    del TABLE_TOUCHES[:-TS.keep]
    try:
        TABLE_FILE.write_text(json.dumps(TABLE_TOUCHES))
    except OSError:
        pass


async def _descend_guarded(x: float, y: float, z_to: float, yaw: float, guard_top: float | None = None):
    """Go straight down to z_to, feeling for contact near the table. Returns (text, contact_z or None).
    Above the table zone it moves normally. In the zone it goes in small steps, each settled with LIMITED effort.
    Contact (measured on hardware): a joint stays short of its target while the two most loaded arm motors together
    carry more than TS.load_sum, on TS.reads steps in a row. Motor load alone is no sign: it swings by 100-200 with
    the arm's own motion. On contact the arm stops pushing and backs off a little.
    guard_top: start the guarded zone at this TCP height instead of just above the table (e.g. above a bottle cap)."""
    q, st = await measured()
    cur, seed = IK.fk(q)[0], q
    last = float(cur[2])
    start = max(z_to, TS.nominal_table_tcp_z + TS.guard_from if guard_top is None else guard_top)
    if cur[2] > start + TS.step:                              # the free part of the way down, at normal speed
        sol, _ = await asyncio.to_thread(_solve_tcp, np.array([x, y, start]), seed, yaw)
        if sol is None:
            return f"stopped at z = {last:.3f}: no joint solution lower down", None
        res = await _go(sol, settle=True)
        if res.startswith("ABORTED"):
            return res, None
        q, st = await measured()
        seed, last = sol, float(IK.fk(q)[0][2])
    hits = 0
    for z in list(np.arange(last - TS.step, z_to, -TS.step)) + [z_to]:
        sol, _ = await asyncio.to_thread(_solve_tcp, np.array([x, y, z]), seed, yaw)
        if sol is None:
            return f"stopped at z = {last:.3f}: no joint solution lower down", None
        seed = sol
        norm = to_norm(sol)
        res = await _drive({ARM_IDS[i]: float(v) for i, v in enumerate(norm)}, STEP, TS.dt, settle=True, contact=False,
                           window=TS.window, iters=TS.iters)
        if res.startswith("ABORTED"):
            return res, None
        qn, st = await measured()
        meas = float(IK.fk(qn)[0][2])
        short = max(abs(st[ARM_IDS[i]]["lo"] + float(norm[i]) * (st[ARM_IDS[i]]["hi"] - st[ARM_IDS[i]]["lo"]) - st[ARM_IDS[i]]["raw"])
                    for i in range(NA))
        loads = sorted((st[m]["load"] for m in ARM_IDS), reverse=True)
        hits = hits + 1 if (short >= TS.short_ticks and loads[0] + loads[1] >= TS.load_sum) else 0
        if hits >= TS.reads:
            await _stop_here(st, ARM_IDS)                     # stop pushing: goals = where the joints are
            up, _ = await asyncio.to_thread(_solve_tcp, np.array([x, y, meas + TS.backoff]), qn, yaw)
            if up is not None:
                await _go(up, settle=False)
            return (f"TOUCHED at gripper height z = {meas:.3f} (a joint stayed {short:.0f} ticks short with motor loads "
                    f"{loads[0]}+{loads[1]}); backed off {TS.backoff * 1000:.0f} mm"), meas
        last = meas
    return f"reached z = {last:.3f} without touching anything", None


@server.tool(annotations={"destructiveHint": True})
@recorded
@doc
async def grab_upright(x: float, y: float, z: float, force: float = GB.force) -> list:
    """Grab a thin object that STANDS UPRIGHT (a pen in a holder, a straw, a spoon in a cup) at about (x, y), with
    the fingertips at height z. The wrist camera cannot tell whether such an object is between the jaws or a little
    in front of or behind them, so this searches by touch: it closes gently, and if the jaws meet nothing it opens,
    shifts 1-2 cm forwards or backwards along the gripper's own direction and tries again. Choose z low on the
    object, so that little of it sticks out below the fingertips. Returns HOLDING / NOT holding, a touch line and
    fresh images."""
    log = []
    st = await am._read()
    if st[GRIP_ID]["norm"] < GR.open_enough:
        await gripper(open_fraction=1.0)
    out = await move_tcp(x=x, y=y, z=z, top_down=True)
    log.append("approach: " + _first(out))
    if _stopped(out):
        return ["GRAB not started.\n" + "\n".join(log)] + await _views()
    q, _ = await measured()
    yaw = _yaw_now(q)
    r = np.array([x, y]) / max(float(np.hypot(x, y)), 1e-6)        # outward from the base: the plane the gripper leans in
    for off in GB.offsets:
        if off:
            sol, _ = await asyncio.to_thread(_solve_tcp, np.array([x + off * r[0], y + off * r[1], z]), (await measured())[0], yaw)
            if sol is None:
                continue
            res = await _go(sol, settle=True)
            if res.startswith("ABORTED"):
                log.append(f"shift {off * 100:+.1f} cm: " + res)
                break
        closed = await gripper(open_fraction=0.0, force=force)
        touch = next((l for l in closed.splitlines() if l.startswith("touch:")), "")
        if "HOLDING" in touch:
            log.append(f"at {off * 100:+.1f} cm: " + touch)
            return [f"HOLDING it (found by touch {off * 100:+.1f} cm from the first position).\n" + "\n".join(log)] + await _views()
        log.append(f"at {off * 100:+.1f} cm: nothing between the jaws")
        await gripper(open_fraction=1.0)
    return ["NOT holding: nothing was felt at any of the positions tried. The object is probably off to the SIDE of the "
            "jaws: re-localize it and try again.\n" + "\n".join(log)] + await _views()


@server.tool(annotations={"destructiveHint": True})
@recorded
@doc
async def grab_tip(x: float, y: float, z_top: float, z_min: float, pull_to: float = 0.0, force: float = GB.tip_force) -> list:
    """Take something soft by its very TIP: a tissue or napkin sticking up out of a pack or box, a ribbon, a bag's
    corner. The gripper starts at height z_top above (x, y), closes gently, and if the jaws meet nothing it opens,
    goes a little lower and tries again, down to z_min. So the first thing the jaws close on is the top of the
    sheet, not the pack underneath. z_top: a little above where the tip is; z_min: no lower than about 2.5 cm above
    the top of the pack or box (the fingertips reach 1 cm below the gripper height). pull_to: if given, the
    sheet is then pulled straight up to that gripper height (e.g. 0.20); this pull is allowed to work against the
    sheet's resistance, which move_tcp would stop on as a collision. The grip is firm by default (a sheet being
    pulled out of a pack slips from a gentle grip). If the pack comes up with the sheet, something has to hold the
    pack down. Returns HOLDING / NOT holding, a touch line and fresh images."""
    log = []
    st = await am._read()
    if st[GRIP_ID]["norm"] < GR.open_enough:
        await gripper(open_fraction=1.0)
    out = await move_tcp(x=x, y=y, z=z_top, top_down=True)
    log.append("approach: " + _first(out))
    if _stopped(out):
        return ["GRAB not started.\n" + "\n".join(log)] + await _views()
    q, _ = await measured()
    yaw, z = _yaw_now(q), float(z_top)
    for _ in range(GB.tip_max_tries):
        closed = await gripper(open_fraction=0.0, force=force)
        touch = next((l for l in closed.splitlines() if l.startswith("touch:")), "")
        if "HOLDING" in touch:
            log.append(f"at height {z:.3f}: " + touch)
            if pull_to <= z:
                return [f"HOLDING it by the tip (first contact at gripper height {z:.3f}). Pull it out with grab_tip's pull_to, "
                        f"or lift with move_tcp.\n" + "\n".join(log)] + await _views()
            qn, _ = await measured()
            cur, seed, peak = IK.fk(qn)[0], qn, 0
            for zz in list(np.arange(cur[2] + GB.pull_step, pull_to, GB.pull_step)) + [pull_to]:
                sol, _ = await asyncio.to_thread(_solve_tcp, np.array([cur[0], cur[1], zz]), seed, yaw)
                if sol is None:
                    log.append(f"pull stopped at {zz:.3f}: no joint solution higher up")
                    break
                seed = sol
                res = await _drive({ARM_IDS[i]: float(v) for i, v in enumerate(to_norm(sol))}, STEP, DT, settle=True, contact=False,
                                   window=GB.pull_window, iters=GB.pull_iters)
                now = await am._read()
                loads = sorted((now[m]["load"] for m in ARM_IDS), reverse=True)
                peak = max(peak, loads[0] + loads[1])
                if res.startswith("ABORTED"):
                    log.append("pull: " + res)
                    break
            held, st = await _probe_hold()
            end = float(IK.fk((await measured())[0])[0][2])
            log.append(f"pulled up to gripper height {end:.3f} (peak arm motor load {peak}); " + _touch(st, None, held))
            return [("HOLDING it after the pull: check in the images that only the sheet came up, not the pack." if held else
                     "NOT holding after the pull: the sheet slipped out of the jaws. Take it again, lower on the sheet.")
                    + "\n" + "\n".join(log)] + await _views()
        log.append(f"at height {z:.3f}: nothing between the jaws")
        await gripper(open_fraction=1.0)
        z -= GB.tip_step
        if z < z_min - 1e-6:
            break
        sol, _ = await asyncio.to_thread(_solve_tcp, np.array([x, y, z]), (await measured())[0], yaw)
        if sol is None:
            break
        res = await _go(sol, settle=True)
        if res.startswith("ABORTED"):
            log.append(res)
            break
    return [f"NOT holding: nothing was felt between {z_top:.3f} and {max(z, z_min):.3f}. The tip is probably beside the jaws, "
            f"or lower than z_min: look again.\n" + "\n".join(log)] + await _views()


@server.tool(annotations={"destructiveHint": True})
@recorded
@doc
async def touch_table(x: float, y: float) -> str:
    """Feel where the table is at (x, y): the gripper goes down slowly until the fingertips touch, stops, and lifts
    again. Returns the gripper height at contact, which is remembered and used by later picks near that spot. Use it
    once on a FREE patch of table near the items when a pick hit the table, slid, or closed above a flat object."""
    q, _ = await measured()
    yaw = _yaw_now(q)
    out = await move_tcp(x=x, y=y, z=TS.probe_from, top_down=True)
    if _stopped(out):
        return "touch_table not done. " + _first(out)
    text, z = await _descend_guarded(x, y, TS.nominal_table_tcp_z - TS.max_below, yaw)
    if z is not None:
        _remember_touch(x, y, z)
    back = await move_tcp(x=x, y=y, z=TS.probe_from, top_down=True)
    if z is None:
        note = ""
    elif abs(z - TS.nominal_table_tcp_z) > TS.plausible:
        note = (f" That is {abs(z - TS.nominal_table_tcp_z) * 1000:.0f} mm from where the table should be, so it was an object or "
                f"something else, not the table: NOT remembered.")
    else:
        note = (f" The table is {abs(z - TS.nominal_table_tcp_z) * 1000:.0f} mm {'higher' if z > TS.nominal_table_tcp_z else 'lower'} "
                f"here than the model assumes; picks within {TS.reuse_radius * 100:.0f} cm of this spot now use it.")
    return text + "." + note + " " + _first(back)


async def _views(which=("wrist", "overview")) -> list:
    out = []
    for label, prefix in (("wrist", WRIST_CAM), ("overview", OVERVIEW_CAM)):
        if label in which:
            data, note = await _fresh_frame(prefix)
            if data:
                out += [f"{label} camera {note}", Image(data=await _wrist_image(data) if label == "wrist" else _grid(data), format="jpeg")]
            else:
                out.append(f"{label} camera: {note}")
    return out


def _first(text: str) -> str:
    lines = text.splitlines()
    return lines[0] + "".join(" " + l for l in lines[1:3] if l.startswith(("NOTE", "ABORTED")))


def _stopped(text: str) -> bool:
    return text.startswith(("ABORTED", "UNREACHABLE"))


CLOSED_NORM = env("ROBOT_GRIPPER_CLOSED", GR.closed_norm)   # jaws above this after closing = clearly holding something
# Thin things (one layer of tissue) keep the jaws only a hair further apart than empty jaws: measured 0.017-0.018
# against 0.012-0.013 empty, with the motor load 30 higher. So "empty" is learned from the closes that met nothing,
# and a close that ends a little wider than that is treated as a thin hold.
_EMPTY: list = [(GR.empty_closed_norm, None)]               # (jaw position, motor load) of recent empty closes


def _thin_min() -> float:
    return float(np.median([e[0] for e in _EMPTY])) + GR.thin_margin


def _note_empty(norm: float, load: int) -> None:
    _EMPTY.append((float(norm), int(load)))
    del _EMPTY[:-GR.empty_memory]


THIN_MIN = GR.empty_closed_norm + GR.thin_margin            # starting value; _thin_min() is the live one
FLAT_Z = env("ROBOT_FLAT_PICK_Z", PK.flat_z)               # TCP height for flat objects (fingertips at the table)


@server.tool(annotations={"destructiveHint": True})
@recorded
@doc
async def pick_at(x: float, y: float, object_height: float = PK.default_height, yaw_deg: float = 0.0,
                  grip_force: float | None = None) -> list:
    """Pick up a small object whose centre is at (x, y) on the table (base frame, from pixel_to_table on a CLOSE
    wrist view): open the jaws, hover above it, descend, close until the jaws stall, lift to z = <<CARRY_Z>>. One call
    replaces move/close/lift. object_height is the object's height in metres; yaw_deg turns the jaws about
    vertical (the jaws close along x at 0). grip_force (0.1 .. 1.0, default <<GRIP_FORCE>>) is how hard the jaws
    squeeze: 0.2-0.3 for hard smooth things that pop out (bottle caps), more for heavy ones. Returns what happened,
    a "touch:" line (jaw gap in mm = the object's width, motor load), whether something is held, and fresh images.
    Stops at the first refused or aborted move and says where the arm is."""
    log = []
    st = await am._read()
    if st[GRIP_ID]["norm"] < GR.open_enough:
        log.append("open: " + _first(await gripper(open_fraction=1.0)))
    flat = object_height < PK.flat_below          # flat things (wrappers): go down until the fingertips touch the table
    out = await move_tcp(x=x, y=y, z=max(object_height + PK.hover_above, PK.hover_min), top_down=True, yaw_deg=yaw_deg)
    log.append("hover: " + _first(out))
    if _stopped(out) or "do not grasp" in out:
        return ["PICK STOPPED before closing.\n" + "\n".join(log)] + await _views()
    felt = _table_tcp_z(x, y)                                   # table height felt near here earlier, if any
    table = TS.nominal_table_tcp_z if felt is None else felt     # gripper height at which the fingertips touch the table
    # flat: go down until touching. Otherwise: fingertips a little below the object's middle, never into the table.
    goal_z = table - TS.max_below if flat else max(table + PK.table_clearance, table - TS.nominal_table_tcp_z + object_height / 2 + PK.grasp_above_mid)
    text, hit = await _descend_guarded(x, y, goal_z, _yaw_now((await measured())[0]))
    log.append("descend: " + text + ("" if felt is None else f" (table felt at {felt:.3f} near here)"))
    if text.startswith("ABORTED"):
        return ["PICK STOPPED before closing.\n" + "\n".join(log)] + await _views()
    if hit is not None and flat:
        _remember_touch(x, y, hit)
    closed = await gripper(open_fraction=0.0, force=grip_force)
    log.append("close: " + _first(closed))
    log.append(next((l for l in closed.splitlines() if l.startswith("touch:")), ""))
    held0, st = await _probe_hold()
    if not held0:      # nothing held: do not carry air; back off to a close viewpoint
        await gripper(open_fraction=1.0)
        out = await move_tcp(x=x, y=y, z=PK.backoff_z, top_down=True, yaw_deg=yaw_deg)
        log.append("back off: " + _first(out))
        why = (f"the jaws closed fully to {st[GRIP_ID]['norm']:.3f}" if st[GRIP_ID]["norm"] < _thin_min() else
               "the jaws touched the object but it slipped out when squeezed: try a lower grip_force, or grip a different part of it")
        return [f"NOT holding ({why}). The jaws are open again and the gripper is {PK.backoff_z * 100:.0f} cm above "
                f"the pick point: find the object in the wrist image below, re-localize and retry.\n" + "\n".join(log)] + await _views()
    out = await move_tcp(x=x, y=y, z=CARRY_Z, top_down=True, yaw_deg=yaw_deg)
    log.append("lift: " + _first(out))
    held, st = await _probe_hold()
    g = st[GRIP_ID]
    log.append("after the lift, " + _touch(st, None, held))
    verdict = (f"HOLDING something ({GRIP_NAME} {g['norm']:.3f}, load {g['load']}): check the wrist image shows it between the jaws."
               if held else f"NOT holding: it was gripped but came out of the jaws during the lift ({GRIP_NAME} now {g['norm']:.3f}). "
                                f"Find it again in the images; try a lower grip_force or grip a different part of it.")
    if _stopped(out):
        verdict = "LIFT STOPPED. " + verdict
    return [verdict + "\n" + "\n".join(log)] + await _views()


@server.tool(annotations={"destructiveHint": True})
@recorded
@doc
async def drop_at(x: float, y: float, z: float = DR.default_z, yaw_deg: float | None = None) -> list:
    """Carry what the gripper holds to (x, y) at height z (it first rises to z, then travels level), open the jaws
    to release it, then retreat at the same height to above the parking spot (<<PARK_XY>>). Use z at least 6 cm above
    a box rim. yaw_deg turns the gripper for the release (leave it out to keep the current rotation): for a long
    object choose it so the object's length runs along the box's long side. Returns what happened and fresh images."""
    log = []
    out = await move_tcp(x=x, y=y, z=z, top_down=True, yaw_deg=yaw_deg)
    log.append("carry: " + _first(out))
    if _stopped(out):
        return ["DROP STOPPED before releasing (still holding).\n" + "\n".join(log)] + await _views()
    log.append("release: " + _first(await gripper(open_fraction=1.0)))
    out = await move_tcp(x=PARK[0], y=PARK[1], z=max(z, P.poses.retreat_min_z), top_down=True)
    log.append("retreat: " + _first(out))
    return ["RELEASED at " + str([round(x, 3), round(y, 3), round(z, 3)]) + ".\n" + "\n".join(log)] + await _views()


SURVEY_VIEWS: dict[str, tuple] = {}     # camera pose of each image of the last survey()
SURVEY_POSES = tuple(tuple(p) for p in P.poses.survey)


@server.tool(annotations={"destructiveHint": True})
@recorded
@doc
async def survey() -> list:
    """Look at the whole workspace: moves the gripper high to <<SURVEY_COUNT>> poses (<<SURVEY_NAMES>>) and
    returns one wrist image from each, plus the overview image. Use it at the start to find the items and the
    boxes: nothing about the table layout is known beforehand. To turn a pixel of one of these three images into a
    position, call pixel_to_table("wrist", u, v, height, view=<name of that image>)."""
    out: list = []
    st = await am._read()
    if st[GRIP_ID]["norm"] < GR.open_enough and st[GRIP_ID]["norm"] <= CLOSED_NORM + GR.survey_closed_margin:   # closed and empty jaws hide the middle of the view
        await gripper(open_fraction=1.0)
    for name, x, y, z in SURVEY_POSES:
        r = await move_tcp(x=x, y=y, z=z, top_down=True, yaw_deg=P.poses.survey_yaw)
        if _stopped(r):
            out.append(f"{name} pose not reached: {_first(r)}")
            continue
        await asyncio.sleep(P.poses.survey_settle_s)
        data, note = await _fresh_frame(WRIST_CAM)
        if data is None:
            out.append(f"{name}: wrist camera {note}")
            continue
        SURVEY_VIEWS[name] = _wrist_pose((await measured())[0])
        out += [f'wrist image "{name}" (gripper at x={x}, y={y}, z={z}); for its pixels use pixel_to_table(..., view="{name}")',
                Image(data=await _wrist_image(data), format="jpeg")]
    out += await _views(("overview",))
    return out


@server.tool(annotations={"destructiveHint": True})
@recorded
@doc
async def center_over(u: float, v: float, height: float = DR.default_box_height) -> list:
    """Move the gripper sideways, keeping its height, so that it ends up above the thing seen at pixel (u, v) of
    the LATEST wrist image (for a box: the centre of its opening). height is a rough guess of that thing's height
    above the table (box rim: about <<BOX_H>>); it does not need to be right, because you check the result: the returned
    wrist image has a green DROP line showing where a released object would fall. Repeat center_over on the new
    image until the DROP line ends inside the box opening, then call release_here()."""
    q, _ = await measured()
    tcp, _ = IK.fk(q)
    line = _drop_line(q)
    if line:
        a, b = np.array(line[0]), np.array(line[-1])
        ab, ap = b - a, np.array([float(u), float(v)]) - a
        t = float(np.clip(ap @ ab / max(ab @ ab, 1e-9), -DR.line_extend, 1 + DR.line_extend))
        if float(np.linalg.norm(ap - t * ab)) <= DR.deadband_px:
            return [f"already above it: pixel ({u:.0f}, {v:.0f}) lies on the DROP line, so a released object falls there. "
                    f"Did not move. Gripper is at [{tcp[0]:.3f}, {tcp[1]:.3f}, {tcp[2]:.3f}]."] + await _views(("wrist",))
    origin, d = _ray(*_wrist_pose(q), float(u), float(v))
    if d[2] > -1e-3:
        return [f"the ray through ({u}, {v}) does not go down: pick a pixel on the box opening"]
    p = origin + ((height - origin[2]) / d[2]) * d
    shift = float(np.hypot(p[0] - tcp[0], p[1] - tcp[1]))
    if shift > DR.center_max_shift:
        return [f"that pixel is {shift:.2f} m away sideways, which is too far for one correction: move_tcp closer first"]
    out = await move_tcp(x=float(p[0]), y=float(p[1]), z=float(max(tcp[2], DR.center_min_z)), top_down=True)
    return [f"moved {shift * 100:.1f} cm sideways. " + _first(out)] + await _views(("wrist",))


@server.tool(annotations={"destructiveHint": True})
@recorded
@doc
async def release_here() -> list:
    """Open the jaws where the gripper is now (dropping what it holds), then retreat at the same height to above
    the parking spot. Use it after center_over shows the DROP line inside the box opening."""
    q, _ = await measured()
    tcp, _ = IK.fk(q)
    log = ["release: " + _first(await gripper(open_fraction=1.0))]
    out = await move_tcp(x=PARK[0], y=PARK[1], z=float(max(tcp[2], P.poses.retreat_min_z)), top_down=True)
    log.append("retreat: " + _first(out))
    return [f"RELEASED at [{tcp[0]:.3f}, {tcp[1]:.3f}, {tcp[2]:.3f}].\n" + "\n".join(log)] + await _views()


def _solve_tip(tip, length: float, seed, yaw: float):
    """Joints that put the TIP of a tool held along the gripper axis (length below the TCP) at `tip`."""
    sol = seed
    for _ in range(ST.tip_iterations if length > 0 else 1):
        _, rot = IK.fk(sol)
        nxt, _ = _solve_tcp(np.asarray(tip, float) - length * rot[:, 1], sol, yaw)     # rot[:, 1] = approach axis
        if nxt is None:
            return None
        sol = nxt
    return sol


def _tip_of(q, length: float):
    pos, rot = IK.fk(q)
    return pos + length * rot[:, 1]


async def _go(sol, settle: bool, dt: float = DT) -> str:
    return await _drive({ARM_IDS[i]: float(v) for i, v in enumerate(to_norm(sol))}, STEP, dt, settle=settle)


@server.tool(annotations={"destructiveHint": True})
@recorded
@doc
async def stir(x: float, y: float, z: float, radius: float = ST.default_radius, turns: int = ST.default_turns,
               tool_length: float = 0.0, clearance: float = ST.default_clearance, rim_height: float = 0.0,
               cup_radius: float = 0.0) -> list:
    """Stir: move the TIP of a tool held pointing down in the jaws in smooth horizontal circles around (x, y) at
    height z, `turns` times, then lift it out. tool_length = how far the tool sticks out below the fingertips
    (metres; 0 = stir with the gripper tip itself). The tool is steered by its tip, so a leaning gripper is
    compensated. Sequence: tip to (x, y, z + clearance) above the cup, straight down to z, circles, straight back
    up. For a cup choose z = rim height - 0.03 (tip inside the cup), clearance so that z + clearance is above the
    rim, and a radius of at most a third of the cup's inner width (max <<STIR_MAX_R>>). Give rim_height and cup_radius
    (inner radius) when you know them: the tool then checks BEFORE moving that the leaning tool shaft will not rub
    the rim, and tells you what to change if it would. The reply says how far the tool leans. It stops on HARD
    resistance only; a light cup being nudged cannot be felt, so compare the cup in the images before and after."""
    radius, turns = float(min(max(radius, ST.min_radius), ST.max_radius)), int(min(max(turns, 1), ST.max_turns))
    L = float(min(max(tool_length, 0.0), ST.max_tool_length))
    q, _ = await measured()
    yaw, seed = _yaw_now(q), q
    # IK runs in a worker thread: solving a whole path inline froze the event loop and the Station reads timed out
    above = await asyncio.to_thread(_solve_tip, [x, y, z + clearance], L, seed, yaw)
    down = await asyncio.to_thread(_solve_tip, [x, y, z], L, above, yaw) if above is not None else None
    if above is None or down is None:
        return [f"STIR not started: the tool tip cannot reach {[round(x, 3), round(y, 3), round(z, 3)]} "
                f"(or the point {clearance} m above it) with tool_length {L}. Nothing moved."] + await _views()
    tcp_low, rot_low = IK.fk(down)
    tilt = float(np.degrees(np.arccos(np.clip(-rot_low[2, 1], -1, 1))))          # how far the tool leans from vertical
    if rim_height > 0 and cup_radius > 0:
        depth = max(0.0, rim_height - z)                                         # how deep the tip is below the rim
        reach = depth * np.tan(np.radians(tilt)) + radius                        # where the shaft crosses the rim, from the centre
        if reach > cup_radius - ST.rim_margin:
            ok_depth = max(0.0, (cup_radius - ST.rim_margin - radius) / max(np.tan(np.radians(tilt)), 1e-6))
            return [f"STIR not started: the tool would lean {tilt:.0f} deg here, so with the tip {depth * 100:.0f} cm below the rim its shaft "
                    f"crosses the rim {reach * 100:.1f} cm from the centre, but the cup's inner radius is {cup_radius * 100:.1f} cm: it would "
                    f"rub and push the cup. Options: stir shallower (tip at most {ok_depth * 100:.1f} cm below the rim, z >= {rim_height - ok_depth:.3f}), "
                    f"use a smaller radius, or hold the tool with less of it below the fingertips (now {L * 100:.0f} cm). "
                    f"Nothing moved."] + await _views()
    if rim_height > 0 and cup_radius > 0 and z < rim_height:
        # the way in: the gripper leans more the higher it is, so check the shaft against the rim at several depths
        def worst_on_the_way():
            w, seed2 = (0.0, 0.0, 0.0), above
            for d in np.linspace(0.0, rim_height - z, ST.rim_check_samples)[1:]:
                sol = _solve_tip([x, y, rim_height - d], L, seed2, yaw)
                if sol is None:
                    continue
                seed2 = sol
                t = float(np.degrees(np.arccos(np.clip(-IK.fk(sol)[1][2, 1], -1, 1))))
                off = d * np.tan(np.radians(t))
                if off > w[0]:
                    w = (float(off), float(d), t)
            return w
        off, d_at, t_at = await asyncio.to_thread(worst_on_the_way)
        if off > cup_radius - ST.rim_margin:
            return [f"STIR not started: on the way down, with the tip {d_at * 100:.0f} cm below the rim the tool leans {t_at:.0f} deg and its shaft "
                    f"would be {off * 100:.1f} cm off-centre at the rim, but the cup's inner radius is {cup_radius * 100:.1f} cm: it would push "
                    f"the cup while entering. Hold the tool with less of it below the fingertips (now {L * 100:.0f} cm) or stir shallower. "
                    f"Nothing moved."] + await _views()
    if tcp_low[2] < MIN_Z:
        return [f"STIR not started: the gripper itself would have to go to z = {tcp_low[2]:.3f}. Check z and tool_length."] + await _views()
    log = []
    for name, sol in (("above the cup", above), ("lowered", down)):
        for _ in range(MO.chunk_tries):          # big moves in chunks, like move_tcp
            now = to_norm((await measured())[0])
            goal = to_norm(sol)
            if float(np.abs(goal - now).max()) <= MO.chunk:
                break
            res = await _drive({ARM_IDS[i]: float(now[i] + np.clip(goal[i] - now[i], -MO.chunk, MO.chunk)) for i in range(NA)}, STEP, DT, settle=False)
            if res.startswith("ABORTED"):
                return [f"STIR stopped on the way ({name}). " + res] + await _views()
        res = await _go(sol, settle=True)
        if res.startswith("ABORTED"):
            return [f"STIR stopped on the way ({name}). " + res] + await _views()
        log.append(f"{name}: tip at {np.round(_tip_of((await measured())[0], L), 3).tolist()}")
    log.append(f"tool leans {tilt:.0f} deg from vertical" + (" (a lot: expect the shaft near the rim)" if tilt > ST.max_tilt_note else ""))
    seed, n_pts = down, ST.points_per_turn
    def plan():
        pts = [_solve_tip([x + radius, y, z], L, seed, yaw)]
        for k in range(1, n_pts + 1):                    # one turn; the same joint path is repeated
            if pts[-1] is None:
                break
            ang = 2 * np.pi * k / n_pts
            pts.append(_solve_tip([x + radius * np.cos(ang), y + radius * np.sin(ang), z], L, pts[-1], yaw))
        return pts
    ring = await asyncio.to_thread(plan)
    path = ring + ring[1:] * (turns - 1) if all(p is not None for p in ring) else [None]
    # Resistance: only HARD resistance can be felt. Measured in open air, the arm's own motion changes the motor loads
    # by 100-200 during a circle (also relative to a practice circle), so a light cup being nudged is below the noise.
    # Whether the cup moved has to be judged from the images.
    base = {m: v["load"] for m, v in (await am._read()).items() if m in ARM_IDS}
    over, worst = 0, 0
    if not path or any(p is None for p in path):
        log.append("circles not reachable at this radius: stirred nothing")
    else:
        for k, sol in enumerate(path):
            res = await _go(sol, settle=False, dt=ST.dt)
            if res.startswith("ABORTED"):
                log.append(f"stopped after {k / n_pts:.1f} turns: {res}")
                break
            now = await am._read()
            push = max(abs(now[m]["load"] - base[m]) for m in ARM_IDS)
            worst = max(worst, push)
            over = over + 1 if push > ST.contact_load_delta else 0
            if over >= ST.contact_points:                                        # hard resistance at consecutive points
                await _stop_here(now, ARM_IDS)
                log.append(f"STOPPED after {k / n_pts:.1f} turns: hard resistance (arm motor load changed by {push}): "
                           f"the tool is jammed against the cup or its bottom")
                break
        else:
            log.append(f"stirred {turns} turns of radius {radius * 100:.1f} cm with the tip at height {z:.3f}" +
                       f"; no hard resistance (largest motor load change {worst}, stop level {ST.contact_load_delta}; a light cup "
                       f"being nudged cannot be felt: compare the cup in the images before and after)")
    for sol in (down, above):
        res = await _go(sol, settle=True)
        if res.startswith("ABORTED"):
            log.append("lifting out: " + res)
            break
    else:
        log.append(f"lifted out: tip at {np.round(_tip_of((await measured())[0], L), 3).tolist()}")
    return ["STIR: " + "; ".join(log)] + await _views()


@server.tool(annotations={"destructiveHint": True})
@recorded
@doc
async def wipe(x: float, y: float, to_x: float, to_y: float, strokes: int = WP.default_strokes) -> list:
    """Wipe the table: with a napkin or cloth HELD in the jaws (pick it first), go down at (x, y) until the
    fingertips feel the table, sweep in a straight line to (to_x, to_y) and back, `strokes` times, pressing lightly,
    then lift. The line must be free of other objects (they would be swept along) and at most <<WIPE_MAX_CM>> cm long. The
    jaws stay closed on the napkin; afterwards drop it where it belongs with drop_at. Without anything held it wipes
    with the bare fingertips. Returns what it did, whether the napkin is still held, and fresh images."""
    a, b = np.array([x, y], float), np.array([to_x, to_y], float)
    length = float(np.linalg.norm(b - a))
    if not WP.min_length <= length <= WP.max_length:
        return [f"WIPE not started: the line is {length * 100:.0f} cm long; it must be between {WP.min_length * 100:.0f} and "
                f"{WP.max_length * 100:.0f} cm. Nothing moved."]
    for pt in (a, b):
        if float(np.hypot(*pt)) > WS.max_radius:
            return [f"WIPE not started: {np.round(pt, 3).tolist()} is beyond the arm's reach. Nothing moved."]
    strokes = int(min(max(strokes, 1), WP.max_strokes))
    held0, _ = await _probe_hold()
    log = [("holding something (the napkin)" if held0 else "nothing is held: wiping with the bare fingertips")]
    out = await move_tcp(x=float(a[0]), y=float(a[1]), z=WP.start_height, top_down=True)
    if _stopped(out):
        return ["WIPE not started. " + _first(out)] + await _views()
    q, _ = await measured()
    yaw = _yaw_now(q)
    text, hit = await _descend_guarded(float(a[0]), float(a[1]), TS.nominal_table_tcp_z - TS.max_below, yaw)
    log.append("down: " + text)
    if hit is None:
        log.append("the table was not felt: wiping at the lowest height reached")
    qn, st = await measured()
    z = float(IK.fk(qn)[0][2]) - TS.backoff if hit is not None else float(IK.fk(qn)[0][2])   # back on the surface, light pressure
    n = max(2, int(length / WP.waypoint_every) + 1)

    def plan():
        pts, seed = [], qn
        for k in range(n + 1):
            p = a + (b - a) * k / n
            sol, _ = _solve_tcp(np.array([p[0], p[1], z]), seed, yaw)
            if sol is None:
                return None
            pts.append(sol); seed = sol
        return pts
    line = await asyncio.to_thread(plan)
    if line is None:
        up = await move_tcp(x=float(a[0]), y=float(a[1]), z=WP.lift_after, top_down=True)
        return ["WIPE not done: part of the line cannot be reached at table height. " + _first(up) + "\n" + "\n".join(log)] + await _views()
    done, hard, stopped = 0, 0, ""
    for sidx in range(strokes):
        path = line[1:] if sidx % 2 == 0 else line[-2::-1]              # there, back, there ...
        for sol in path:
            res = await _go(sol, settle=False, dt=WP.dt)
            if res.startswith("ABORTED"):
                stopped = res
                break
            now = await am._read()
            loads = sorted((now[m]["load"] for m in ARM_IDS), reverse=True)
            hard = hard + 1 if loads[0] + loads[1] >= WP.hard_load_sum else 0
            if hard >= WP.hard_reads:                                   # pushing hard into something
                await _stop_here(now, ARM_IDS)
                stopped = f"hard resistance (motor loads {loads[0]}+{loads[1]}): something is in the way or the table is higher there"
                break
        if stopped:
            break
        done += 1
    qe, _ = await measured()
    end = IK.fk(qe)[0]
    up = await move_tcp(x=float(end[0]), y=float(end[1]), z=WP.lift_after, top_down=True)
    held1, st = await _probe_hold()
    log.append(f"wiped {done} of {strokes} strokes of {length * 100:.0f} cm at gripper height {z:.3f}" + (f"; STOPPED: {stopped}" if stopped else ""))
    log.append("lift: " + _first(up))
    log.append("after wiping, " + _touch(st, None, held1))
    if held0 and not held1:
        log.append("the napkin is no longer in the jaws: it stayed on the table")
    return [("WIPED. " if done == strokes else "WIPE stopped early. ") + "\n".join(log)] + await _views()


# --- bottle cap: attempt records + the twist skill (learning/, prompts/bottle-contract.md) --------------------------
from learning import tools as LT  # noqa: E402


class RealRig:
    """learning.twist's rig interface on the real servos (same motion layer as every tool above)."""
    # +1: a positive model rotation about +z turns the jaws counter-clockwise seen from above. The j7 sign is not yet
    # verified on hardware (see the module docstring): set BOTTLE_TWIST_SIGN=-1 if the cap turns the wrong way.
    twist_sign = env("BOTTLE_TWIST_SIGN", 1, int)

    async def halt(self):
        """Stop outstanding arm goals at measured position on deadline/cancellation."""
        st = await am._read()
        await _stop_here(st, ARM_IDS)

    async def state(self) -> dict:
        st = await am._read()
        return {NAME[m]: st[m] for m in ALL_IDS}

    async def q(self):
        return (await measured())[0]

    def fk(self, q):
        return IK.fk(q)

    def solve(self, xyz, R, seed):
        return _solve_rot(np.asarray(xyz, float), np.asarray(R, float), np.asarray(seed, float))

    async def move_down(self, x, y, z, yaw_deg):
        return await move_tcp(x=x, y=y, z=z, top_down=True, yaw_deg=yaw_deg)

    async def reachable(self, x, y, z, yaw_deg, seed):
        if not (MIN_Z <= z <= WS.max_tcp_z and np.hypot(x, y) <= WS.max_radius):
            return None
        _, lean = await asyncio.to_thread(_solve_tcp, np.array([x, y, z], float), np.asarray(seed, float), yaw_deg)
        return None if lean is None else float(lean)

    async def go(self, q, dt, contact):
        return await _drive({ARM_IDS[i]: float(v) for i, v in enumerate(to_norm(np.asarray(q)))}, STEP, dt,
                            settle=False, contact=contact)

    async def descend(self, x, y, z_to, yaw_deg, top):
        return await _descend_guarded(x, y, z_to, yaw_deg, guard_top=top)

    async def grip(self, open_fraction, force):
        text = await gripper(open_fraction=open_fraction, force=force)
        touch = next((l for l in text.splitlines() if l.startswith("touch:")), "")
        return text, ("holding it" in text or "HOLDING" in touch)

    async def holding(self):
        return (await _probe_hold())[0]

    async def frames(self) -> dict:
        out = {}
        for role, prefix in (("wrist", WRIST_CAM), ("overview", OVERVIEW_CAM)):
            data, _ = await _fresh_frame(prefix)
            if data:
                out[role] = data
        return out


RIG = RealRig()


async def _content(text: str, images: dict) -> list:
    out: list = [text]
    if images.get("wrist"):
        out += ["wrist camera", Image(data=await _wrist_image(images["wrist"]), format="jpeg")]
    if images.get("overview"):
        out += ["overview camera", Image(data=_grid(images["overview"]), format="jpeg")]
    return out


def _learn_doc(fn):
    fn.__doc__ = LT.TOOL_DOCS[fn.__name__]
    return fn


@server.tool()
@recorded
@_learn_doc
async def episode_begin(params: dict, hypothesis: str) -> list:
    return await _content(*await LT.episode_begin(LT.store(), RIG, params, hypothesis))


@server.tool(annotations={"readOnlyHint": True})
@recorded
@_learn_doc
async def episode_history(last_n: int = 10) -> str:
    return (await LT.episode_history(LT.store(), last_n))[0]


@server.tool(annotations={"destructiveHint": True})
@recorded
@_learn_doc
async def twist_cap(x: float, y: float, cap_top_z: float, cap_height: float | None = None) -> list:
    return await _content(*await LT.twist_cap(LT.store(), RIG, x, y, cap_top_z, cap_height))


@server.tool(annotations={"destructiveHint": True})
@recorded
@_learn_doc
async def check_cap() -> list:
    return await _content(*await LT.check_cap(LT.store(), RIG))


@server.tool(annotations={"destructiveHint": True})
@recorded
@_learn_doc
async def extract_cap() -> list:
    return await _content(*await LT.extract_cap(LT.store(), RIG))

@server.tool()
@recorded
@_learn_doc
async def episode_end(outcome: str, lesson: str, next_change: dict | None = None) -> list:
    return await _content(*await LT.episode_end(LT.store(), RIG, outcome, lesson, next_change))


# --- skill config: read it, propose changes (applying to the real arm is a human step: learning/SELF_TUNING.md) ------
def _skill_doc(fn):
    fn.__doc__ = LT.skill_doc(fn.__name__, "real")
    return fn


@server.tool(annotations={"readOnlyHint": True})
@recorded
@_skill_doc
async def skill_config() -> str:
    return (await LT.skill_config())[0]


@server.tool()
@recorded
@_skill_doc
async def propose_skill_change(changes: dict, reason: str, evidence: list[str], twin_check: bool = False) -> str:
    return (await LT.propose_skill_change(changes, reason, evidence, twin_check, author="operator (real_arm_mcp)",
                                          auto_apply=False))[0]


if __name__ == "__main__":
    server.run()
