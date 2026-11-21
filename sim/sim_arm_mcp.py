"""Simulated ElRobot behind the same MCP tools as arm_mcp/server.py (the real arm).

The agent gets exactly what it gets on hardware: two camera JPEGs (wrist + overview,
299x224 like Station) and the motor-bus joint table (norm, raw ticks, goal, calibrated
range, torque, status, load, temp). It never sees object poses or simulator state.
Physics runs continuously in real time on a background thread, so the world keeps
moving between calls, like a real arm.

Real-arm behaviour reproduced: normalized [0,1] joint targets (j8 jaws: higher = more
open), creeping moves with a max-delta refusal, tracking-error aborts, a gripper that
stalls on an object, ST3215 torque limit (2.94 N m, no gravity compensation) and an
overload trip: a servo held at its torque limit for ROBOT_SIM_OVERLOAD_S seconds
reports status 0x20 and goes limp until it is unloaded.

Env: ROBOT_SIM_GUI=1 (MuJoCo viewer), ROBOT_MAX_DELTA (0.35), ROBOT_TRACK_TOL (250),
ROBOT_GRIPPER_JOINT (j8), ROBOT_SIM_LAYOUT (sim/layout.json), ROBOT_SIM_SEED,
ROBOT_SIM_OVERLOAD_S (2.0; 0 disables), ROBOT_SIM_TRUTH_LOG (optional JSONL of ground
truth for a human judge; keep it out of the agent's reach).
Recording (human side, recorder.py): ROBOT_SIM_RECORD (1; 0 disables), ROBOT_SIM_RECORD_DIR
(sim/sessions), ROBOT_SIM_RECORD_FPS (10), ROBOT_SIM_DASHBOARD_PORT (8891; live page). On exit
the session gets side.mp4 + replay.html (make_replay.py).
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

GUI = os.environ.get('ROBOT_SIM_GUI') == '1'
os.environ.setdefault('MUJOCO_GL', 'glfw' if GUI else 'osmesa')
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image as PILImage, ImageDraw  # noqa: E402
from mcp.server.fastmcp import FastMCP, Image  # noqa: E402

import build_scene  # noqa: E402

HERE = Path(__file__).resolve().parent
LAYOUT = Path(os.environ.get('ROBOT_SIM_LAYOUT', HERE / 'layout.json'))
MAX_DELTA = float(os.environ.get('ROBOT_MAX_DELTA', '0.35'))
TRACK_TOL = int(os.environ.get('ROBOT_TRACK_TOL', '250'))
GRIPPER = os.environ.get('ROBOT_GRIPPER_JOINT', 'j8')
OVERLOAD_S = float(os.environ.get('ROBOT_SIM_OVERLOAD_S', '2.0'))
TRUTH_LOG = os.environ.get('ROBOT_SIM_TRUTH_LOG')
IMG_SCALE = int(os.environ.get('ROBOT_SIM_IMG_SCALE', '2'))  # upscale 299x224 frames for the model's eyes
GRID = os.environ.get('ROBOT_SIM_GRID', '1') == '1'          # labelled pixel grid on look() images
KINEMATICS = os.environ.get('ROBOT_SIM_KINEMATICS', '1') == '1'  # tcp_pose / move_tcp / pixel_to_table tools
NAMES = [f'j{i}' for i in range(1, 9)]
TICKS_PER_RAD = 4096 / (2 * np.pi)
# Hashes of the real Station camera queues (camera_pose/HANDOFF.md), so prompts carry over.
CAMERAS = [('wrist_cam', '322ab5514b8e14014c3743c1abfd4a07', 'sim-wrist-uvc'),
           ('overview_cam', '8c697f6e0c79a3d2fab2a4ea977ee738', 'sim-overview')]


class SimArm:
    def __init__(self):
        scene = HERE / 'generated/workspace.xml'
        sources = [LAYOUT, HERE / 'build_scene.py']
        if not scene.exists() or scene.stat().st_mtime < max(p.stat().st_mtime for p in sources):
            build_scene.build(LAYOUT, HERE / 'generated', report=False)
        self.layout = json.loads(LAYOUT.read_text())
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.qids = m.jnt_qposadr[m.actuator_trnid[:, 0]]
        self.dofs = m.jnt_dofadr[m.actuator_trnid[:, 0]]
        self.lo_rad, self.hi_rad = m.actuator_ctrlrange.T.copy()
        self.gain = m.actuator_gainprm[:, 0].copy()
        self.bias = m.actuator_biasprm[:, :3].copy()
        self.fmax = m.actuator_forcerange[:, 1].copy()
        self.torque = np.ones(m.nu, bool)
        self.status = np.zeros(m.nu, int)
        self.saturated_s = np.zeros(m.nu)
        self.lock = threading.RLock()
        self.rng = np.random.default_rng(int(os.environ['ROBOT_SIM_SEED']) if 'ROBOT_SIM_SEED' in os.environ else None)
        self.reset(randomize=False)
        self.viewer = None
        if GUI:
            from mujoco import viewer as mjviewer
            self.viewer = mjviewer.launch_passive(m, self.data, show_left_ui=False)
            v = self.viewer.cam
            v.lookat[:] = [0, 0.12, 0.05]; v.distance = 0.95; v.azimuth = -90; v.elevation = -38
        self.renderer = None
        self.ik = build_scene.ArmIK(m)  # separate MjData: kinematics only
        self.vis_option = mujoco.MjvOption()
        self.vis_option.geomgroup[1] = 0  # URDF collision duplicates of the visual meshes: same look, half the triangles
        self.running = True
        threading.Thread(target=self._physics, daemon=True).start()

    # --- real-time physics -------------------------------------------------
    def _physics(self):
        m, d = self.model, self.data
        wall0, sim0, last_sync = time.monotonic(), d.time, 0.0
        while self.running:
            with self.lock:
                if d.time < sim0:  # reset happened
                    wall0, sim0 = time.monotonic(), d.time
                target = sim0 + (time.monotonic() - wall0)
                steps = 0
                while d.time < target and steps < 50:
                    mujoco.mj_step(m, d); steps += 1
                    if steps % 10 == 0: self._servo_protection(10 * m.opt.timestep)
                if target - d.time > 0.5:  # fell behind: accept a slower-than-real-time clock
                    wall0, sim0 = time.monotonic(), d.time
            if self.viewer is not None and time.monotonic() - last_sync > 1 / 30:
                if not self.viewer.is_running(): self.viewer = None
                else:
                    with self.lock: self.viewer.sync()
                last_sync = time.monotonic()
            time.sleep(0.001)

    def _servo_protection(self, dt):
        if OVERLOAD_S <= 0: return
        force = np.abs(self.data.actuator_force)
        sat = self.torque & (force >= 0.97 * self.fmax) & (np.arange(self.model.nu) != 7)  # jaws may stall
        self.saturated_s = np.where(sat, self.saturated_s + dt, 0.0)
        for i in np.flatnonzero(self.saturated_s > OVERLOAD_S):
            self.status[i] = 0x20; self._set_torque(i, False); self.saturated_s[i] = 0
        # A tripped servo re-engages once it is not being driven against gravity any more.
        for i in np.flatnonzero(self.status == 0x20):
            if abs(self.data.qvel[self.dofs[i]]) < 0.05:
                self.data.ctrl[i] = self.data.qpos[self.qids[i]]
                self.status[i] = 0; self._set_torque(i, True)

    def _set_torque(self, i, on):
        self.torque[i] = on
        self.model.actuator_gainprm[i, 0] = self.gain[i] if on else 0
        self.model.actuator_biasprm[i, :3] = self.bias[i] if on else 0

    # --- joint conventions (match the real arm) ------------------------------
    def norm(self, q):
        n = (q - self.lo_rad) / (self.hi_rad - self.lo_rad)
        n[7] = 1 - q[7] / self.hi_rad[7]  # j8: 1 = open
        return n

    def rad(self, i, n):
        return self.hi_rad[i] * (1 - n) if i == 7 else self.lo_rad[i] + n * (self.hi_rad[i] - self.lo_rad[i])

    def ticks(self, i, q):
        return int(round(2048 + (-q if i == 7 else q) * TICKS_PER_RAD))

    def state(self):
        with self.lock:
            q, ctrl, force = self.data.qpos[self.qids].copy(), self.data.ctrl.copy(), self.data.actuator_force.copy()
        rows = {}
        for i, name in enumerate(NAMES):
            lo, hi = sorted((self.ticks(i, self.lo_rad[i]), self.ticks(i, self.hi_rad[i])))
            raw = self.ticks(i, q[i])
            rows[name] = {'norm': round(float(self.norm(q)[i]), 3), 'raw': raw, 'goal': self.ticks(i, ctrl[i]),
                          'lo': lo, 'hi': hi, 'in_range': lo - 20 <= raw <= hi + 20, 'torque': int(self.torque[i]),
                          'status': int(self.status[i]), 'load': int(min(1000, abs(force[i]) / self.fmax[i] * 1000)),
                          'temp': 31}
        return rows

    def table(self):
        st = self.state()
        out = [f"{'joint':>5} {'norm':>7} {'raw':>5} {'goal':>5} {'range':>12} in_rng torque status load temp"]
        for name in NAMES:
            m = st[name]
            out.append(f"{name:>5} {m['norm']:>7} {m['raw']:>5} {m['goal']:>5} [{m['lo']:>4},{m['hi']:>4}] "
                       f"{'y' if m['in_range'] else 'NO':>6} {m['torque']:>6} {hex(m['status']):>6} {m['load']:>4} {m['temp']:>4}")
        return '\n'.join(out)

    # --- cameras --------------------------------------------------------------
    def snapshot(self, cam):
        """Camera JPEG at the Station resolution, upscaled IMG_SCALE x with a labelled pixel grid
        (coordinates are in the returned image, as used by pixel_to_table)."""
        w, h = self.layout['image_size']
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, h, w)
        with self.lock:
            self.renderer.update_scene(self.data, camera=cam, scene_option=self.vis_option)
        im = PILImage.fromarray(self.renderer.render())
        if IMG_SCALE != 1:
            im = im.resize((w * IMG_SCALE, h * IMG_SCALE), PILImage.BICUBIC)
        if GRID:
            draw = ImageDraw.Draw(im, 'RGBA')
            for x in range(0, im.width, 50):
                draw.line([(x, 0), (x, im.height)], fill=(255, 64, 64, 70 if x % 100 else 120))
                if x: draw.text((x + 2, 1), str(x), fill=(255, 40, 40, 255))
            for y in range(0, im.height, 50):
                draw.line([(0, y), (im.width, y)], fill=(255, 64, 64, 70 if y % 100 else 120))
                if y: draw.text((2, y + 1), str(y), fill=(255, 40, 40, 255))
        buf = io.BytesIO()
        im.save(buf, format='JPEG', quality=85)
        return buf.getvalue()

    # --- the arm's own kinematics + camera calibration (no knowledge of objects) ---------
    def measured_q(self):
        with self.lock:
            return self.data.qpos[self.qids].copy()

    def pixel_ray(self, cam, u, v):
        """Ray (origin, direction) in the base frame through pixel (u, v) of the returned image."""
        w, h = self.layout['image_size']
        u, v = u / IMG_SCALE, v / IMG_SCALE
        cid = self.model.camera(cam).id
        with self.lock:
            pos, mat = self.data.cam_xpos[cid].copy(), self.data.cam_xmat[cid].reshape(3, 3).copy()
        f = (h / 2) / np.tan(np.radians(self.model.cam_fovy[cid]) / 2)
        d = mat @ np.array([(u - w / 2) / f, -(v - h / 2) / f, -1.0])  # MuJoCo camera looks along -z, y up
        return pos, d / np.linalg.norm(d)

    # --- operator actions ------------------------------------------------------
    def reset(self, randomize):
        with self.lock:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key('rest').id)
            for i in range(self.model.nu): self._set_torque(i, True)
            self.status[:] = 0; self.saturated_s[:] = 0
            if randomize:
                for spec in self.layout['objects']:
                    r = spec.get('randomize')
                    if not r: continue
                    adr = self.model.jnt_qposadr[self.model.joint(f"{spec['name']}_free").id]
                    self.data.qpos[adr:adr+2] = [self.rng.uniform(*r['x']), self.rng.uniform(*r['y'])]
                    self.data.qpos[adr+3:adr+7] = build_scene.yaw_quat(self.rng.uniform(*r.get('yaw', [0, 0])))
            mujoco.mj_forward(self.model, self.data)

    def truth(self):
        """Ground truth for the human judge only (never returned to the agent)."""
        with self.lock:
            out = {}
            for spec in self.layout['objects']:
                p = self.data.body(spec['name']).xpos.copy()
                rot = self.data.body(spec['name']).xmat.reshape(3, 3)
                yaw = float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))
                inside = None
                for b in self.layout['bins']:
                    body = self.data.body(b['name'])
                    local = body.xmat.reshape(3, 3).T @ (p - body.xpos)
                    if abs(local[0]) < b['size'][0]/2 and abs(local[1]) < b['size'][1]/2 and local[2] < b['size'][2]:
                        inside = b['name']
                out[spec['name']] = {'xyz': [round(float(v), 4) for v in p], 'yaw_deg': round(yaw, 1), 'in_bin': inside}
            return {'sim_time': round(float(self.data.time), 2), 'objects': out}


arm = SimArm()
server = FastMCP('elrobot-sim')
recorder = None
if os.environ.get('ROBOT_SIM_RECORD', '1') == '1':
    from recorder import Recorder
    recorder = Recorder(arm, Path(os.environ.get('ROBOT_SIM_RECORD_DIR', HERE / 'sessions')),
                        fps=float(os.environ.get('ROBOT_SIM_RECORD_FPS', '10')),
                        port=int(os.environ.get('ROBOT_SIM_DASHBOARD_PORT', '8891')))
    print(f'elrobot-sim: recording to {recorder.dir}; live dashboard {recorder.url}', file=sys.stderr, flush=True)


def recorded(fn):
    """Log every tool call (args, result or error, the images the agent got) for the human side."""
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        start = (recorder.now(), recorder.frame) if recorder else None
        call = sig.bind(*args, **kwargs); call.apply_defaults()
        args, kwargs = (), dict(call.arguments)
        try:
            result = await fn(*args, **kwargs)
        except Exception as e:
            _record(fn.__name__, kwargs, f'ERROR {type(e).__name__}: {e}', start=start)
            raise
        images = [(cam, item.data) for (cam, *_), item in zip(CAMERAS, [r for r in result if isinstance(r, Image)])] \
            if isinstance(result, list) else []
        text = result if isinstance(result, str) else '\n'.join(r for r in result if isinstance(r, str))
        _record(fn.__name__, kwargs, text, images, start)
        return result
    return wrapper


def _record(tool, args, text, images=(), start=None):
    truth = arm.truth()
    if TRUTH_LOG:
        with open(TRUTH_LOG, 'a') as f:
            f.write(json.dumps({'wall': time.strftime('%H:%M:%S'), 'tool': tool, 'args': args, **truth}) + '\n')
    if recorder:
        recorder.event(tool, args, text, images, truth, start)


@server.tool(annotations={'readOnlyHint': True})
@recorded
async def look() -> list:
    """FRESH images from every live camera plus the live motor-bus joint table."""
    out: list = []
    for cam, h, uid in CAMERAS:
        data = arm.snapshot(cam)
        out.append(f"camera {h[:8]} ({uid}) fresh, age=0.03s, captured {time.strftime('%H:%M:%S')}")
        out.append(Image(data=data, format='jpeg'))
    out.append(arm.table())
    return out


@server.tool(annotations={'readOnlyHint': True})
@recorded
async def joint_state() -> str:
    """Live joint table from the motor bus: raw ticks, calibrated range, in-range flag, torque, status, load."""
    return arm.table()


@server.tool(annotations={'destructiveHint': True})
@recorded
async def set_torque(enabled: bool) -> str:
    """Enable/disable torque on all motors. Enabling first writes goal = present position for every
    motor so nothing jumps. Disabling makes the arm limp: support it before calling with False."""
    with arm.lock:
        if enabled: arm.data.ctrl[:] = arm.data.qpos[arm.qids]
        for i in range(arm.model.nu): arm._set_torque(i, enabled)
        if enabled: arm.status[:] = 0
    await asyncio.sleep(0.4)
    return arm.table()


@server.tool(annotations={'destructiveHint': True})
@recorded
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
        raise ValueError(f'unknown joints {unknown}; valid {NAMES}')
    if not (0 < step <= 0.2 and 0.02 <= dt <= 2):
        raise ValueError('step must be in (0, 0.2] and dt in [0.02, 2] s')
    st = arm.state()
    if not arm.torque.all():
        raise RuntimeError('torque is off on some motors — call set_torque(true) first')
    plan = {}
    for name, want in targets.items():
        m = st[name]
        if not m['in_range']:
            raise RuntimeError(f"{name} raw {m['raw']} is outside calibrated range [{m['lo']},{m['hi']}] — "
                               'commanding it would snap; recalibrate or move it into range by hand')
        want = min(1.0, max(0.0, float(want)))
        if abs(want - m['norm']) > MAX_DELTA:
            raise ValueError(f"{name}: {m['norm']:.3f}->{want:.3f} exceeds max delta {MAX_DELTA}; use waypoints")
        plan[NAMES.index(name)] = (m['norm'], want)
    with arm.lock:
        hold = arm.data.qpos[arm.qids].copy()
    n = max([1] + [int(abs(b - a) / step + 0.999) for a, b in plan.values()])
    for k in range(1, n + 1):
        goals = hold.copy()
        for i, (a, b) in plan.items():
            goals[i] = arm.rad(i, a + (b - a) * k / n)
        with arm.lock:
            arm.data.ctrl[:] = goals
        await asyncio.sleep(dt)
        now = arm.state()
        bad = [(name, now[name]['goal'], now[name]['raw'], now[name]['status']) for name in NAMES
               if now[name]['status'] or abs(now[name]['goal'] - now[name]['raw']) > track_tol]
        if bad:
            with arm.lock:
                tripped = arm.status != 0
                arm.data.ctrl[~tripped] = arm.data.qpos[arm.qids][~tripped]
            return f'ABORTED at step {k}/{n}: (joint, goal, present, status) {bad}; holding present pose.\n' + arm.table()
    await asyncio.sleep(0.3)
    return f'reached in {n} steps.\n' + arm.table()


FRAME = ('Base frame, metres: origin = centre of the arm base on the table top, +z up, +y = straight ahead of '
         'the arm (towards the overview camera), +x = the arm\'s right-hand side (image-LEFT in the overview camera).')
CAMERA_NAMES = {'wrist': 'wrist_cam', 'overview': 'overview_cam'}


async def tcp_pose() -> str:
    """Where the gripper is, from the measured joints and the arm's kinematic model (like FK on
    the real arm): the TCP (point between the fingertip pads) in the base frame, whether the jaws
    point down, and the jaw axis direction (yaw, and its angle from level: a tilted closing axis
    puts one jaw under a held object, so it may not drop out when the jaws open)."""
    pos, rot = arm.ik.fk(arm.measured_q())
    approach, jaw = rot[:, 1], rot[:, 0]
    tilt = np.degrees(np.arccos(np.clip(-approach[2], -1, 1)))
    level = np.degrees(np.arcsin(np.clip(abs(jaw[2]), 0, 1)))
    return (f'tcp xyz = [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m; jaws point {tilt:.0f} deg from straight down; '
            f'jaw closing axis yaw = {np.degrees(np.arctan2(jaw[1], jaw[0])):.0f} deg ({level:.0f} deg from level).\n{FRAME}')


async def move_tcp(x: float, y: float, z: float, top_down: bool = True, yaw_deg: float = 0.0,
                   step: float = 0.02, dt: float = 0.12) -> str:
    """Move the gripper TCP (between the fingertip pads) to (x, y, z) in the base frame using the
    arm's inverse kinematics, then execute it with move_to in bounded steps (same refusals and
    tracking aborts). top_down=True keeps the jaws pointing straight down (where that is impossible,
    e.g. high up near the base or over the box, it uses the nearest tilted pose and says so); yaw_deg turns the jaw
    closing axis about vertical (0 = jaws close along x). If the target is more than 3 cm away
    sideways, it travels at a safe height (max(target, 0.08 m)), lifting first if it is lower, then descends.
    The gripper opening (j8) is not changed. Table top is z = 0; grasp small objects at z ~ 0.02."""
    target = np.array([x, y, z], float)
    if not (0.0 <= z <= 0.45 and np.hypot(x, y) <= 0.45):
        raise ValueError(f'target {target.round(3).tolist()} is outside the workspace (z 0..0.45 m, radius <= 0.45 m)')
    q = arm.measured_q()
    cur, _ = arm.ik.fk(q)
    waypoints = []
    if np.hypot(*(target[:2] - cur[:2])) > 0.03:
        safe = max(z, 0.08)
        if cur[2] < safe - 0.01: waypoints.append([cur[0], cur[1], safe])  # lift straight up first
        if z < safe: waypoints.append([x, y, safe])                         # travel high, then descend
    waypoints.append(target.tolist())
    seed, log, notes = q, [], []
    for wp in waypoints:
        if top_down:  # least-tilted reachable pose, as on the real arm (real_arm_mcp.py)
            sol, lean = arm.ik.solve_least_lean(np.array(wp), seed, yaw_deg)
            if sol is not None and lean:
                notes.append(f'jaws-down unreachable at {np.round(wp, 3).tolist()}: gripper leans {abs(lean)} deg from '
                             f'vertical ({"away from" if lean > 0 else "toward"} the base)')
            if sol is None:  # last resort: position only, orientation free
                sol = arm.ik.solve(np.array(wp), top_down=False, q0=seed)
                if sol is not None:
                    _, rot = arm.ik.fk(sol)
                    notes.append(f'no near-vertical pose at {np.round(wp, 3).tolist()}: gripper is '
                                 f'{np.degrees(np.arccos(np.clip(-rot[2, 1], -1, 1))):.0f} deg from vertical; do not grasp like this')
        else:
            sol = arm.ik.solve(np.array(wp), top_down=False, q0=seed)
        if sol is None:
            return (f'UNREACHABLE: no joint solution for tcp {np.round(wp, 3).tolist()} '
                    f'({"jaws down" if top_down else "any orientation"}); nothing moved after: {log}\n' + arm.table())
        seed = sol
        goal = arm.norm(sol)
        for _ in range(12):
            now = arm.norm(arm.measured_q())
            if max(abs(goal[:7] - now[:7])) < 0.004: break
            chunk = {NAMES[i]: float(now[i] + np.clip(goal[i] - now[i], -0.3, 0.3)) for i in range(7)}
            out = await _move(chunk, step, dt, TRACK_TOL)
            if out.startswith('ABORTED'):
                pos, _ = arm.ik.fk(arm.measured_q())
                return f'ABORTED on the way to {np.round(wp, 3).tolist()}; tcp now {pos.round(3).tolist()}.\n' + out
        log.append(np.round(wp, 3).tolist())
    pos, _ = arm.ik.fk(arm.measured_q())
    note = ('\nNOTE: ' + '; '.join(notes)) if notes else ''
    return f'reached via {log}; tcp now [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}].{note}\n' + arm.table()


async def pixel_to_table(camera: str, u: float, v: float, height: float = 0.0) -> str:
    """Turn a pixel of a look() image into a base-frame point using the camera calibration
    (overview: fixed extrinsics; wrist: CAD mount pose + current joints). (u, v) are pixel
    coordinates in the returned image (red grid labels), u to the right, v down. Returns where the
    viewing ray hits the horizontal plane z = height (0 = table top; use the object's top or centre
    height, e.g. 0.01 for a 2 cm cube, ~0.09 for the box rim)."""
    if camera not in CAMERA_NAMES:
        raise ValueError(f'camera must be one of {list(CAMERA_NAMES)}')
    origin, d = arm.pixel_ray(CAMERA_NAMES[camera], float(u), float(v))
    if d[2] > -1e-3 and origin[2] > height:
        return f'the ray through ({u}, {v}) does not go down to z = {height} (it points at or above the horizon)'
    t = (height - origin[2]) / d[2]
    p = origin + t * d
    return (f'{camera} pixel ({u:.0f}, {v:.0f}) -> table point [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}] '
            f'({t:.2f} m from the camera, {np.hypot(p[0], p[1]):.3f} m from the arm base).\n{FRAME}')


async def camera_info() -> str:
    """Camera calibration as JSON, the same model pixel_to_table uses: per camera ('wrist',
    'overview') the look() image size, pinhole intrinsics fx, fy, cx, cy in look() pixels, and the
    camera pose in the base frame in the OpenCV convention (x right, y down, z forward):
    p_base = R_base_cam @ p_cam + t_base_cam. The wrist pose follows the current joints."""
    w, h = arm.layout['image_size']
    out = {}
    for name, cam in CAMERA_NAMES.items():
        cid = arm.model.camera(cam).id
        with arm.lock:
            pos, mat = arm.data.cam_xpos[cid].copy(), arm.data.cam_xmat[cid].reshape(3, 3).copy()
        f = (h / 2) / np.tan(np.radians(arm.model.cam_fovy[cid]) / 2) * IMG_SCALE
        out[name] = {'width': w * IMG_SCALE, 'height': h * IMG_SCALE, 'fx': f, 'fy': f,
                     'cx': w * IMG_SCALE / 2, 'cy': h * IMG_SCALE / 2,
                     'R_base_cam': (mat @ np.diag([1.0, -1.0, -1.0])).round(6).tolist(),
                     't_base_cam': pos.round(6).tolist()}
    return json.dumps(out)


if KINEMATICS:
    server.tool(annotations={'readOnlyHint': True})(recorded(tcp_pose))
    server.tool(annotations={'destructiveHint': True})(recorded(move_tcp))
    server.tool(annotations={'readOnlyHint': True})(recorded(pixel_to_table))
    server.tool(annotations={'readOnlyHint': True})(recorded(camera_info))


@server.tool(annotations={'destructiveHint': True})
@recorded
async def gripper(open_fraction: float) -> str:
    """Set the gripper jaws (j8) to a normalized target: higher = more open. Closing on an object
    stalls the jaw short of the target, which is a grip; it does not abort for the gripper."""
    return await _move({GRIPPER: open_fraction}, 0.05, 0.1, 4096)


@server.tool(annotations={'destructiveHint': True})
@recorded
async def operator_reset_scene(randomize: bool = True) -> str:
    """SIMULATION ONLY — what a human operator does between trials: put the arm back in its rest
    pose and place the objects on the table again (randomly when `randomize`). Returns the
    joint table; look() to see where things are."""
    arm.reset(randomize)
    await asyncio.sleep(0.5)
    return arm.table()


if __name__ == '__main__':
    try:
        server.run()
    finally:
        arm.running = False
        if recorder:
            recorder.close()
            import make_replay
            make_replay.build(recorder.dir)
        if arm.viewer is not None:
            arm.viewer.close()
            time.sleep(1)
