"""A connector-driven twin: the ElRobot twin's behaviour (sim/sim_arm_mcp.py) for any arm a connector declares.

Joint count and names, which joint is the gripper and which way it opens, the TCP site, cameras, the safety envelope
and the scene all come from the connector, so a new robot gets a twin and the standard MCP tool surface (look,
joint_state, set_torque, move_to, gripper, tcp_pose, move_tcp, pixel_to_table, camera_info, operator_reset_scene,
skill_config) without editing this file.

Real-arm behaviour reproduced as in sim_arm_mcp.py: normalized [0,1] joint targets (gripper: 1 = open), creeping
moves with a max-delta refusal, tracking-error aborts, a gripper that stalls on an object, a torque-limited servo
that trips (status 0x20, limp) after OVERLOAD_S at its limit, physics in real time on a background thread.
Differences, on purpose: arm moves keep the gripper's goal (sim_arm_mcp resets every unlisted joint's goal to its
present position, which lets a stalled grip go slack while the arm carries the object), and move_tcp refuses
targets outside the connector's safety envelope (min_tcp_z, max_tcp_z, max_radius) as the real server does.
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

import mujoco
import numpy as np
from mcp.server.fastmcp import FastMCP, Image
from PIL import Image as PILImage, ImageDraw

TICKS_PER_RAD = 4096 / (2 * np.pi)       # ST3215 / STS3215: 4096 ticks per turn, 2048 = centre
OVERLOAD_S = float(os.environ.get('ROBOT_SIM_OVERLOAD_S', '2.0'))   # as sim_arm_mcp.py
SATURATED = 0.97                         # |force| >= this fraction of the limit counts as held at the limit
IMG_SCALE = int(os.environ.get('ROBOT_SIM_IMG_SCALE', '2'))
GRID = os.environ.get('ROBOT_SIM_GRID', '1') == '1'
GRID_STEP = 50                           # px between grid lines, labels every line (as sim_arm_mcp.py)
TRUTH_LOG = os.environ.get('ROBOT_SIM_TRUTH_LOG')
HOLD_LAG_RAD = 0.05                      # jaw goal this far past the jaw = stalled on something (sim_arm_mcp SimRig)
HOLD_MIN_NORM = 0.02                     # ... while the jaws are not fully shut
ARRIVE_TOL = 0.004                       # norm: move_tcp arrived (sim_arm_mcp.py)
CHUNK = 0.3                              # norm per move_to chunk inside move_tcp (< max delta)
DIRECT_WITHIN = 0.03                     # m: closer than this sideways, move_tcp goes straight; else travels high
HOLD_RESET_RAD = 0.35                    # an arm goal this far from the joint (after a trip) restarts from the joint


class GenericTwin:
    def __init__(self, connector, scene: Path, layout: dict):
        self.connector, self.layout = connector, layout
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.names = [j.name for j in connector.joints]
        if len(self.names) != m.nu or connector.joints[-1].role != 'gripper':
            raise ValueError(f'{connector.name}: {m.nu} actuators in the twin, joints declared {self.names} '
                             '(arm joints in actuator order, then the gripper)')
        self.g = m.nu - 1
        self.open_upper = connector.gripper.open_at_upper_limit
        self.qids = m.jnt_qposadr[m.actuator_trnid[:, 0]]
        self.dofs = m.jnt_dofadr[m.actuator_trnid[:, 0]]
        self.lo_rad, self.hi_rad = m.actuator_ctrlrange.T.copy()
        self.gain, self.bias = m.actuator_gainprm[:, 0].copy(), m.actuator_biasprm[:, :3].copy()
        self.fmax = m.actuator_forcerange[:, 1].copy()
        self.torque, self.status, self.saturated_s = np.ones(m.nu, bool), np.zeros(m.nu, int), np.zeros(m.nu)
        self.lock = threading.RLock()
        self.rng = np.random.default_rng(int(os.environ['ROBOT_SIM_SEED']) if 'ROBOT_SIM_SEED' in os.environ else None)
        self.ik = connector.make_ik(m)
        self.cameras = {c.role: c.twin_camera for c in connector.cameras if c.twin_camera}
        self.renderer = None
        self.vis_option = mujoco.MjvOption()
        self.reset(randomize=os.environ.get('ROBOT_SIM_RANDOMIZE') == '1')
        self.running = True
        threading.Thread(target=self._physics, daemon=True).start()

    # --- real-time physics -------------------------------------------------------------------------------
    def _physics(self):
        m, d = self.model, self.data
        wall0, sim0 = time.monotonic(), d.time
        while self.running:
            with self.lock:
                if d.time < sim0:
                    wall0, sim0 = time.monotonic(), d.time
                target, steps = sim0 + (time.monotonic() - wall0), 0
                while d.time < target and steps < 50:
                    mujoco.mj_step(m, d)
                    steps += 1
                    if steps % 10 == 0:
                        self._servo_protection(10 * m.opt.timestep)
                if target - d.time > 0.5:                   # fell behind: accept a slower-than-real-time clock
                    wall0, sim0 = time.monotonic(), d.time
            time.sleep(0.001)

    def _servo_protection(self, dt):
        if OVERLOAD_S <= 0:
            return
        force = np.abs(self.data.actuator_force)
        sat = self.torque & (force >= SATURATED * self.fmax) & (np.arange(self.model.nu) != self.g)   # jaws may stall
        self.saturated_s = np.where(sat, self.saturated_s + dt, 0.0)
        for i in np.flatnonzero(self.saturated_s > OVERLOAD_S):
            self.status[i] = 0x20
            self._set_torque(i, False)
            self.saturated_s[i] = 0
        for i in np.flatnonzero(self.status == 0x20):      # re-engages once it is not driven against gravity
            if abs(self.data.qvel[self.dofs[i]]) < 0.05:
                self.data.ctrl[i] = self.data.qpos[self.qids[i]]
                self.status[i] = 0
                self._set_torque(i, True)

    def _set_torque(self, i, on):
        self.torque[i] = on
        self.model.actuator_gainprm[i, 0] = self.gain[i] if on else 0
        self.model.actuator_biasprm[i, :3] = self.bias[i] if on else 0

    # --- joint conventions ------------------------------------------------------------------------------------
    def norm(self, q):
        n = (np.asarray(q) - self.lo_rad) / (self.hi_rad - self.lo_rad)
        if not self.open_upper:
            n[self.g] = 1 - n[self.g]
        return n

    def rad(self, i, n):
        if i == self.g and not self.open_upper:
            n = 1 - n
        return self.lo_rad[i] + n * (self.hi_rad[i] - self.lo_rad[i])

    def ticks(self, i, q):
        sign = -1 if (i == self.g and not self.open_upper) else 1
        return int(round(2048 + sign * q * TICKS_PER_RAD))

    def measured_q(self):
        with self.lock:
            return self.data.qpos[self.qids].copy()

    def state(self):
        with self.lock:
            q, ctrl, force = self.data.qpos[self.qids].copy(), self.data.ctrl.copy(), self.data.actuator_force.copy()
        n, rows = self.norm(q), {}
        for i, name in enumerate(self.names):
            lo, hi = sorted((self.ticks(i, self.lo_rad[i]), self.ticks(i, self.hi_rad[i])))
            raw = self.ticks(i, q[i])
            rows[name] = {'norm': round(float(n[i]), 3), 'raw': raw, 'goal': self.ticks(i, ctrl[i]), 'lo': lo, 'hi': hi,
                          'in_range': lo - 20 <= raw <= hi + 20, 'torque': int(self.torque[i]),
                          'status': int(self.status[i]), 'load': int(min(1000, abs(force[i]) / self.fmax[i] * 1000)),
                          'temp': 31}
        return rows

    def table(self):
        st = self.state()
        w = max(5, max(len(n) for n in self.names))
        out = [f"{'joint':>{w}} {'norm':>7} {'raw':>5} {'goal':>5} {'range':>12} in_rng torque status load temp"]
        for name in self.names:
            m = st[name]
            out.append(f"{name:>{w}} {m['norm']:>7} {m['raw']:>5} {m['goal']:>5} [{m['lo']:>4},{m['hi']:>4}] "
                       f"{'y' if m['in_range'] else 'NO':>6} {m['torque']:>6} {hex(m['status']):>6} {m['load']:>4} {m['temp']:>4}")
        return '\n'.join(out)

    def holding(self) -> bool:
        with self.lock:
            lag = self.data.ctrl[self.g] - self.data.qpos[self.qids[self.g]]
        closing_dir = -1 if self.open_upper else 1           # the direction the jaw angle moves when closing
        return bool(self.state()[self.names[self.g]]['norm'] > HOLD_MIN_NORM and lag * closing_dir > HOLD_LAG_RAD)

    # --- cameras --------------------------------------------------------------------------------------------
    def frame(self, cam):
        w, h = self.layout['image_size']
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, h, w)
        with self.lock:
            self.renderer.update_scene(self.data, camera=cam, scene_option=self.vis_option)
        return self.renderer.render().copy()

    def snapshot(self, cam):
        w, h = self.layout['image_size']
        im = PILImage.fromarray(self.frame(cam))
        if IMG_SCALE != 1:
            im = im.resize((w * IMG_SCALE, h * IMG_SCALE), PILImage.BICUBIC)
        if GRID:
            draw = ImageDraw.Draw(im, 'RGBA')
            for x in range(0, im.width, GRID_STEP):
                draw.line([(x, 0), (x, im.height)], fill=(255, 64, 64, 70 if x % 100 else 120))
                if x:
                    draw.text((x + 2, 1), str(x), fill=(255, 40, 40, 255))
            for y in range(0, im.height, GRID_STEP):
                draw.line([(0, y), (im.width, y)], fill=(255, 64, 64, 70 if y % 100 else 120))
                if y:
                    draw.text((2, y + 1), str(y), fill=(255, 40, 40, 255))
        buf = io.BytesIO()
        im.save(buf, format='JPEG', quality=85)
        return buf.getvalue()

    def cam_pose(self, cam):
        cid = self.model.camera(cam).id
        with self.lock:
            return self.data.cam_xpos[cid].copy(), self.data.cam_xmat[cid].reshape(3, 3).copy(), float(self.model.cam_fovy[cid])

    def pixel_ray(self, cam, u, v):
        w, h = self.layout['image_size']
        u, v = u / IMG_SCALE, v / IMG_SCALE
        pos, mat, fovy = self.cam_pose(cam)
        f = (h / 2) / np.tan(np.radians(fovy) / 2)
        d = mat @ np.array([(u - w / 2) / f, -(v - h / 2) / f, -1.0])
        return pos, d / np.linalg.norm(d)

    # --- operator ------------------------------------------------------------------------------------------
    def reset(self, randomize):
        with self.lock:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key('rest').id)
            for i in range(self.model.nu):
                self._set_torque(i, True)
            self.status[:] = 0
            self.saturated_s[:] = 0
            if randomize:
                for spec in self.layout['objects']:
                    r = spec.get('randomize')
                    if not r:
                        continue
                    adr = self.model.jnt_qposadr[self.model.joint(f"{spec['name']}_free").id]
                    self.data.qpos[adr:adr + 2] = [self.rng.uniform(*r['x']), self.rng.uniform(*r['y'])]
                    a = np.radians(self.rng.uniform(*r.get('yaw', [0, 0]))) / 2
                    self.data.qpos[adr + 3:adr + 7] = [np.cos(a), 0, 0, np.sin(a)]
            mujoco.mj_forward(self.model, self.data)

    def truth(self):
        """Ground truth for a human judge or a test (never returned to the agent)."""
        with self.lock:
            return {'sim_time': round(float(self.data.time), 2),
                    'objects': {s['name']: {'xyz': [round(float(v), 4) for v in self.data.body(s['name']).xpos]}
                                for s in self.layout['objects']}}


def make_server(twin: GenericTwin, connector, name: str | None = None) -> FastMCP:
    """The standard Generaloid twin tool surface for `twin`."""
    server = FastMCP(name or f'{connector.name}-sim')
    names, G, safety = twin.names, twin.names[twin.g], connector.safety
    arm = list(range(twin.g))
    frame = connector.frame_text
    joint_list = ', '.join(names[:-1]) + f'; {G} = gripper jaws (higher = more open)'

    def recorded(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            call = sig.bind(*args, **kwargs)
            call.apply_defaults()
            try:
                return await fn(**call.arguments)
            finally:
                if TRUTH_LOG:
                    with open(TRUTH_LOG, 'a') as f:
                        f.write(json.dumps({'tool': fn.__name__, 'args': call.arguments, **twin.truth()}, default=str) + '\n')
        return wrapper

    def tool(**ann):
        return lambda fn: server.tool(annotations=ann)(recorded(fn))

    async def _move(targets: dict, step: float, dt: float, track_tol: int | None, max_delta: float | None = None) -> str:
        max_delta = safety.max_joint_delta if max_delta is None else max_delta
        unknown = [k for k in targets if k not in names]
        if unknown:
            raise ValueError(f'unknown joints {unknown}; valid {names}')
        if not (0 < step <= 0.2 and 0.02 <= dt <= 2):
            raise ValueError('step must be in (0, 0.2] and dt in [0.02, 2] s')
        if not twin.torque.all():
            raise RuntimeError('torque is off on some motors: call set_torque(true) first')
        st, plan = twin.state(), {}
        for name, want in targets.items():
            m = st[name]
            if not m['in_range']:
                raise RuntimeError(f"{name} raw {m['raw']} is outside its calibrated range [{m['lo']},{m['hi']}]")
            want = min(1.0, max(0.0, float(want)))
            if abs(want - m['norm']) > max_delta:
                raise ValueError(f"{name}: {m['norm']:.3f}->{want:.3f} exceeds max delta {max_delta}; use waypoints")
            plan[names.index(name)] = (m['norm'], want)
        with twin.lock:
            hold = twin.data.ctrl.copy()                  # unlisted joints keep their goals (a grip stays squeezed)
            q = twin.data.qpos[twin.qids]
            for i in arm:
                if abs(hold[i] - q[i]) > HOLD_RESET_RAD:
                    hold[i] = q[i]
        n = max([1] + [int(abs(b - a) / step + 0.999) for a, b in plan.values()])
        for k in range(1, n + 1):
            goals = hold.copy()
            for i, (a, b) in plan.items():
                goals[i] = twin.rad(i, a + (b - a) * k / n)
            with twin.lock:
                twin.data.ctrl[:] = goals
            await asyncio.sleep(dt)
            now = twin.state()
            bad = [(nm, now[nm]['goal'], now[nm]['raw'], now[nm]['status']) for nm in names
                   if now[nm]['status'] or (track_tol is not None and nm != G and abs(now[nm]['goal'] - now[nm]['raw']) > track_tol)]
            if bad:
                with twin.lock:
                    keep = twin.status == 0
                    keep[twin.g] = False
                    twin.data.ctrl[keep] = twin.data.qpos[twin.qids][keep]
                return f'ABORTED at step {k}/{n}: (joint, goal, present, status) {bad}; holding present pose.\n' + twin.table()
        await asyncio.sleep(0.3)
        return f'reached in {n} steps.\n' + twin.table()

    @tool(readOnlyHint=True)
    async def look() -> list:
        """FRESH images from every camera plus the live motor-bus joint table."""
        out: list = []
        for role, cam in twin.cameras.items():
            out.append(f"{role} camera ({cam}) fresh, captured {time.strftime('%H:%M:%S')}")
            out.append(Image(data=twin.snapshot(cam), format='jpeg'))
        out.append(twin.table())
        return out

    @tool(readOnlyHint=True)
    async def joint_state() -> str:
        """Live joint table from the motor bus: raw ticks, calibrated range, in-range flag, torque, status, load."""
        return twin.table()

    @tool(destructiveHint=True)
    async def set_torque(enabled: bool) -> str:
        """Enable/disable torque on all motors. Enabling first writes goal = present position so nothing jumps.
        Disabling makes the arm limp: support it before calling with False."""
        with twin.lock:
            if enabled:
                twin.data.ctrl[:] = twin.data.qpos[twin.qids]
            for i in range(twin.model.nu):
                twin._set_torque(i, enabled)
            if enabled:
                twin.status[:] = 0
        await asyncio.sleep(0.4)
        return twin.table()

    async def move_to(targets: dict[str, float], step: float = 0.02, dt: float = 0.12) -> str:
        return await _move(targets, step, dt, safety.track_tol_ticks)
    move_to.__doc__ = (f'Creep joints to ABSOLUTE normalized targets in [0,1] (calibrated range), e.g. {{"{names[0]}": 0.5}}. '
                       f'Joints: {joint_list}. Unlisted joints keep their goal. Refuses: torque off, a joint outside '
                       f'its range, a step larger than {safety.max_joint_delta}; aborts and holds on a tracking error '
                       'or a motor fault.')
    tool(destructiveHint=True)(move_to)

    @tool(destructiveHint=True)
    async def gripper(open_fraction: float) -> str:
        """Set the gripper jaws to a normalized target: higher = more open. Closing on an object stalls the jaws
        short of the target, which is a grip (reported as HOLDING)."""
        closing = open_fraction < twin.state()[G]['norm']
        out = await _move({G: open_fraction}, 0.05, 0.1, None, max_delta=1.0)   # one call, as on the real arm
        held = closing and twin.holding()
        return ('Jaws stalled on something: HOLDING it.\n' if held else '') + out

    @tool(readOnlyHint=True)
    async def tcp_pose() -> str:
        """Where the gripper is, from the measured joints and the arm's kinematic model: the TCP (between the
        fingertips) in the base frame, how far the jaws point from straight down, and the jaw closing axis yaw."""
        pos, rot = twin.ik.fk(twin.measured_q())
        approach, jaw = rot[:, 1], rot[:, 0]
        tilt = np.degrees(np.arccos(np.clip(-approach[2], -1, 1)))
        return (f'tcp xyz = [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m; jaws point {tilt:.0f} deg from straight down; '
                f'jaw closing axis yaw = {np.degrees(np.arctan2(jaw[1], jaw[0])):.0f} deg.\n{frame}')

    @tool(destructiveHint=True)
    async def move_tcp(x: float, y: float, z: float, top_down: bool = True, yaw_deg: float = 0.0,
                       step: float = 0.02, dt: float = 0.12) -> str:
        """Move the gripper TCP to (x, y, z) in the base frame with the arm's inverse kinematics, executed with
        move_to in bounded steps (same refusals and aborts). top_down=True: jaws as close to straight down as
        reachable (the lean is reported). yaw_deg turns the jaw closing axis about vertical. More than 3 cm
        sideways, it travels at a safe height first. The gripper opening is not changed. Table top is z = 0."""
        target = np.array([x, y, z], float)
        if not (safety.min_tcp_z <= z <= safety.max_tcp_z and np.hypot(x, y) <= safety.max_radius):
            raise ValueError(f'target {target.round(3).tolist()} is outside the safety envelope (z {safety.min_tcp_z}..'
                             f'{safety.max_tcp_z} m, radius <= {safety.max_radius} m)')
        q = twin.measured_q()
        cur, _ = twin.ik.fk(q)
        waypoints = []
        if np.hypot(*(target[:2] - cur[:2])) > DIRECT_WITHIN:
            safe = max(z, safety.travel_height)
            if cur[2] < safe - 0.01:
                waypoints.append([cur[0], cur[1], safe])
            if z < safe:
                waypoints.append([x, y, safe])
        waypoints.append(target.tolist())
        seed, log, notes = q, [], []
        for wp in waypoints:
            sol = None
            if top_down:
                sol, lean = await asyncio.to_thread(twin.ik.solve_least_lean, np.array(wp), seed, yaw_deg)
                if sol is not None and lean:
                    notes.append(f'jaws-down unreachable at {np.round(wp, 3).tolist()}: gripper leans {abs(lean)} deg')
            if sol is None:
                sol = await asyncio.to_thread(twin.ik.solve, np.array(wp), False, seed)
                if sol is not None and top_down:
                    notes.append(f'no near-vertical pose at {np.round(wp, 3).tolist()}; do not grasp like this')
            if sol is None:
                return (f'UNREACHABLE: no joint solution for tcp {np.round(wp, 3).tolist()}; nothing moved after: {log}\n'
                        + twin.table())
            seed, goal = sol, twin.norm(sol)
            for _ in range(12):
                now = twin.norm(twin.measured_q())
                if max(abs(goal[arm] - now[arm])) < ARRIVE_TOL:
                    break
                chunk = {names[i]: float(now[i] + np.clip(goal[i] - now[i], -CHUNK, CHUNK)) for i in arm}
                out = await _move(chunk, step, dt, safety.track_tol_ticks)
                if out.startswith('ABORTED'):
                    pos, _ = twin.ik.fk(twin.measured_q())
                    return f'ABORTED on the way to {np.round(wp, 3).tolist()}; tcp now {pos.round(3).tolist()}.\n' + out
            log.append(np.round(wp, 3).tolist())
        pos, _ = twin.ik.fk(twin.measured_q())
        note = ('\nNOTE: ' + '; '.join(notes)) if notes else ''
        return f'reached via {log}; tcp now [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}].{note}\n' + twin.table()

    @tool(readOnlyHint=True)
    async def pixel_to_table(camera: str, u: float, v: float, height: float = 0.0) -> str:
        """Turn a pixel of a look() image into a base-frame point with the camera calibration: where the viewing
        ray through (u, v) (red grid labels, u right, v down) meets the horizontal plane z = height."""
        if camera not in twin.cameras:
            raise ValueError(f'camera must be one of {list(twin.cameras)}')
        origin, d = twin.pixel_ray(twin.cameras[camera], float(u), float(v))
        if d[2] > -1e-3 and origin[2] > height:
            return f'the ray through ({u}, {v}) does not go down to z = {height}'
        t = (height - origin[2]) / d[2]
        p = origin + t * d
        return (f'{camera} pixel ({u:.0f}, {v:.0f}) -> table point [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}] '
                f'({t:.2f} m from the camera, {np.hypot(p[0], p[1]):.3f} m from the arm base).\n{frame}')

    @tool(readOnlyHint=True)
    async def camera_info() -> str:
        """Camera calibration as JSON (the model pixel_to_table uses): per camera the look() image size, pinhole
        fx, fy, cx, cy in look() pixels, and the pose in the base frame, OpenCV axes: p_base = R @ p_cam + t."""
        w, h = twin.layout['image_size']
        out = {}
        for role, cam in twin.cameras.items():
            pos, mat, fovy = twin.cam_pose(cam)
            f = (h / 2) / np.tan(np.radians(fovy) / 2) * IMG_SCALE
            out[role] = {'width': w * IMG_SCALE, 'height': h * IMG_SCALE, 'fx': f, 'fy': f, 'cx': w * IMG_SCALE / 2,
                         'cy': h * IMG_SCALE / 2, 'R_base_cam': (mat @ np.diag([1.0, -1.0, -1.0])).round(6).tolist(),
                         't_base_cam': pos.round(6).tolist()}
        return json.dumps(out)

    @tool(destructiveHint=True)
    async def operator_reset_scene(randomize: bool = True) -> str:
        """SIMULATION ONLY: put the arm back in its rest pose and the objects on the table again (randomly when
        `randomize`). Returns the joint table; look() to see where things are."""
        twin.reset(randomize)
        await asyncio.sleep(0.5)
        return twin.table()

    if 'skill_config' in connector.skills_for('sim'):
        from learning import tools as LT

        async def skill_config() -> str:
            return (await LT.skill_config())[0]
        skill_config.__doc__ = LT.skill_doc('skill_config', 'twin')
        tool(readOnlyHint=True)(skill_config)

    return server


def mount_generic(connector, out_dir: Path):
    """Build the connector's twin scene, start the twin, return (twin, server)."""
    spec = connector.twin
    scene = spec.build_scene(out_dir)
    twin = GenericTwin(connector, scene, json.loads(Path(spec.layout).read_text()))
    print(f'generaloid: {connector.name} twin up ({scene})', file=sys.stderr, flush=True)
    return twin, make_server(twin, connector)
