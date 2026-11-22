"""Fast headless kitting twin: the kitting workspace.xml (sim/kitting_scene.py) stepped as fast as the CPU allows, the
scripted pick-and-place policy (plan.py) on top, domain randomization, and a judge that reads the simulator truth.

Same arm model, servos, pads and IK (sim/build_scene.ArmIK least-lean search, reading the shared skill config) as the
real-time twin server sim/sim_arm_mcp.py; what differs is only the clock (no real-time pacing) and that the motion
primitives are written here (servo goal ramps, guarded descent, torque-limited close) after sim_arm_mcp.SimRig and
executor/motion.py. One pick-and-place is ~10 s of simulated time and ~1 s of CPU.

What the policy may use: the spec, the sheet pose located in the rendered overview image (locate.py), joint readings,
jaw opening and servo loads. What only the judge uses: object poses from the simulator.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
import mujoco  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "sim")):
    if p not in sys.path:
        sys.path.append(p)
import build_scene  # noqa: E402
import kitting_scene  # noqa: E402

from . import locate as L  # noqa: E402
from .plan import PAD_REACH, ObjectPlan, order, plan_pick_place  # noqa: E402
from .spec import Pose2D, Spec, axis_wrap_deg, wrap_deg  # noqa: E402

# --- servo and sensing constants (sources in the comments) -------------------------------------------------------
CTRL_DT = 0.02            # s between servo goal updates (the Station streams goals at ~50 Hz)
JOINT_SPEED = 1.2         # rad/s max goal speed for travel moves (real: profile motion.step 0.02 of range per 0.12 s
                          # ~ 0.5-1 rad/s; faster here only shortens sim time, the arm still tracks within 2 ticks)
DESCENT_STEP = 0.003      # m per guarded-descent step (profile touch.step)
DESCENT_DT = 0.08         # s per descent step
GUARD_ABOVE = 0.015       # free descent to this far above the target, then guarded steps
BLOCKED_M = 0.004         # measured TCP this much above the commanded one = touching something (sim_arm_mcp.descend)
LOAD_ABORT = 550          # profile motion.load_abort (0..1000 of the servo torque)
TICK = 2 * math.pi / 4096
SETTLE_TOL = 6 * TICK     # profile motion.settle_tol (ticks): closed-loop settle, as executor/motion.Motion.drive
SETTLE_ITERS = 12         # profile motion.settle_iters
SETTLE_WAIT_S = 0.2       # profile motion.settle_wait_s
SETTLE_GAIN = 0.6         # profile motion.settle_gain: P-only servos stop short under gravity: push the goal past
SETTLE_WINDOW = 150 * TICK  # profile motion.settle_window (ticks a goal may be pushed past its target)
TOUCH_WINDOW, TOUCH_ITERS = 40 * TICK, 5   # profile touch.window / touch.iters: limited effort in guarded steps
GRIP_RAMP_S = 0.5
GRIP_MAX_S = 6.0          # a torque-limited close is slow in the twin (sim_arm_mcp.SimRig.grip waits up to 6 s)
HOLD_MARGIN = 0.02        # jaw norm above the empty-close reading = something between the pads
JAW_TORQUE = build_scene.GRIPPER_TORQUE   # 0.4 N m at grip force 1.0 (sim_arm_mcp.SimRig.grip)

# --- judge thresholds --------------------------------------------------------------------------------------------
REST_TILT_DEG = 20.0      # an object tilted more than this is not lying in the cell (e.g. leaning on a wall)
MOVED_MM = 3.0            # a neighbour displaced more than this was knocked
SETTLE_AFTER_S = 0.6      # s of physics after the release before the final pose is read


@dataclass
class DR:
    """Domain randomization ranges. Each has a reason; none() switches all off (tests, nominal runs)."""
    sheet: bool = True                       # sheet pose from spec placement.randomize
    human: bool = True                       # objects off their spots by the spec's human_placement noise
    light_scale: tuple = (0.6, 1.5)          # key light diffuse x (room light changes over a day)
    ambient: tuple = (0.5, 0.95)             # headlight ambient (home.json fitted 0.75)
    light_dir_jitter: float = 0.35           # added to the key light direction x/y
    cam_pos_m: float = 0.003                 # overview camera position error (scene_calib: marker rms 1-2 mm)
    cam_rot_deg: float = 0.4                 # overview camera rotation error (0.4 deg ~ 3 mm at 0.45 m)
    base_m: float = 0.002                    # arm base vs table calibration (kinematic bias, per episode)
    base_yaw_deg: float = 0.4
    mass_scale: tuple = (0.8, 1.25)          # object masses are estimates, not weighed
    friction_scale: tuple = (0.7, 1.3)       # paper / plastic / steel vs pad rubber, not measured

    @classmethod
    def none(cls) -> "DR":
        return cls(False, False, (1, 1), (0.75, 0.75), 0.0, 0.0, 0.0, 0.0, 0.0, (1, 1), (1, 1))


@dataclass
class ObjRun:
    """One object's pick and place: the phase records (episodes style) and the judge's truth."""
    name: str
    cls: str
    cell: str
    plan: dict
    sheet_true: Pose2D = field(default=None, repr=False)     # judge only
    phases: list = field(default_factory=list)
    outcome: str = "aborted"
    truth: dict = field(default_factory=dict)
    diagnosis: list = field(default_factory=list)

    def phase(self, phase, status="ok", note="", **m):
        self.phases.append({"phase": phase, "status": status, "note": note, "measurements": m})


class KitSim:
    def __init__(self, spec: Spec | None = None, scene: Path | None = None):
        self.spec = spec or Spec.load()
        sc, self.layout = kitting_scene.ensure_scene(self.spec)
        self.m = mujoco.MjModel.from_xml_path(str(scene or sc))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.ik = build_scene.ArmIK(m)
        self.qids, self.dofs = self.ik.qids, m.jnt_dofadr[m.actuator_trnid[:, 0]]
        self.fmax = m.actuator_forcerange[:, 1].copy()
        self.lo, self.hi = m.actuator_ctrlrange.T.copy()
        self.sheet_id = m.body("kit_sheet").id
        self.obj = {}
        for o in self.spec.objects():
            b = m.body(o["name"])
            j = m.joint(f"{o['name']}_free")
            geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b.id]
            self.obj[o["name"]] = {"body": b.id, "qadr": m.jnt_qposadr[j.id], "dadr": m.jnt_dofadr[j.id], "geoms": geoms,
                                   "mass0": m.body_mass[b.id], "inertia0": m.body_inertia[b.id].copy(), "class": o["class"]}
        self.nominal = {"light_diffuse": m.light_diffuse.copy(), "light_dir": m.light_dir.copy(),
                        "ambient": m.vis.headlight.ambient.copy(), "geom_friction": m.geom_friction.copy()}
        cid = m.camera("overview_cam").id
        self.cam_id, self.cam_pos0, self.cam_quat0 = cid, m.cam_pos[cid].copy(), m.cam_quat[cid].copy()
        self.cam = L.camera_model(self.layout, twin=True)       # what the policy believes (nominal calibration)
        self.renderer = None
        self.bias = Pose2D(0, 0, 0)
        self.empty_norm = None
        self.reset(np.random.default_rng(0), DR.none())

    # --- scene -------------------------------------------------------------------------------------------------
    def reset(self, rng: np.random.Generator, dr: DR) -> dict:
        m, d, S = self.m, self.d, self.spec
        mujoco.mj_resetDataKeyframe(m, d, m.key("rest").id)
        pl = S.raw["placement"]
        nom = S.nominal_sheet_pose()
        if dr.sheet:
            r = pl["randomize"]
            sheet = Pose2D(nom.x + rng.uniform(*r["x_m"]), nom.y + rng.uniform(*r["y_m"]), nom.yaw_deg + rng.uniform(*r["yaw_deg"]))
        else:
            sheet = nom
        m.body_pos[self.sheet_id][:2] = [sheet.x, sheet.y]
        m.body_quat[self.sheet_id] = build_scene.yaw_quat(sheet.yaw_deg)
        hp = pl["human_placement"]
        starts = {}
        for t in S.targets(sheet):
            o = self.obj[t["name"]]
            dx = np.clip(rng.normal(0, hp["sigma_mm"], 2), -hp["max_mm"], hp["max_mm"]) / 1000 if dr.human else np.zeros(2)
            dyaw = rng.normal(0, hp["yaw_sigma_deg"]) if dr.human else 0.0
            p = Pose2D(t["start"].x + dx[0], t["start"].y + dx[1], t["start"].yaw_deg + dyaw)
            a = o["qadr"]
            d.qpos[a:a + 2] = [p.x, p.y]
            d.qpos[a + 3:a + 7] = build_scene.yaw_quat(p.yaw_deg)
            starts[t["name"]] = p.as_list()
            k = rng.uniform(*dr.mass_scale)
            m.body_mass[o["body"]], m.body_inertia[o["body"]] = o["mass0"] * k, o["inertia0"] * k
            f = rng.uniform(*dr.friction_scale)
            for g in o["geoms"]:
                m.geom_friction[g, 0] = self.nominal["geom_friction"][g, 0] * f
        m.light_diffuse[:] = self.nominal["light_diffuse"] * rng.uniform(*dr.light_scale)
        m.light_dir[:] = self.nominal["light_dir"]
        m.light_dir[:, :2] += rng.uniform(-dr.light_dir_jitter, dr.light_dir_jitter, (m.nlight, 2))
        m.vis.headlight.ambient[:] = rng.uniform(*dr.ambient)
        m.cam_pos[self.cam_id] = self.cam_pos0 + rng.normal(0, dr.cam_pos_m, 3)
        rv = np.radians(rng.normal(0, dr.cam_rot_deg, 3))
        dq = np.zeros(4)
        mujoco.mju_axisAngle2Quat(dq, rv / (np.linalg.norm(rv) + 1e-12), float(np.linalg.norm(rv)))
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, self.cam_quat0, dq)
        m.cam_quat[self.cam_id] = q
        self.bias = Pose2D(*(rng.normal(0, dr.base_m, 2) if dr.base_m else (0.0, 0.0)), rng.normal(0, dr.base_yaw_deg) if dr.base_yaw_deg else 0.0)
        d.ctrl[:] = d.qpos[self.qids]
        mujoco.mj_forward(m, d)
        self.step_s(0.4)
        if self.empty_norm is None:
            self.empty_norm = self.jaw_check()
        return {"sheet": sheet.as_list(), "starts": starts, "bias": self.bias.as_list(),
                "light_scale": round(float(m.light_diffuse[0, 0] / max(1e-9, self.nominal["light_diffuse"][0, 0])), 3),
                "ambient": round(float(m.vis.headlight.ambient[0]), 3)}

    def render(self, cam: str = "overview_cam") -> np.ndarray:
        W, H = self.layout["image_size"]
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.m, H, W)
        opt = mujoco.MjvOption()
        opt.geomgroup[1] = 0
        self.renderer.update_scene(self.d, camera=cam, scene_option=opt)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    # --- truth (judge only) -------------------------------------------------------------------------------------
    def obj_pose(self, name: str) -> dict:
        b = self.d.body(name)
        R = b.xmat.reshape(3, 3)
        return {"xyz": b.xpos.copy(), "yaw_deg": math.degrees(math.atan2(R[1, 0], R[0, 0])),
                "tilt_deg": math.degrees(math.acos(np.clip(abs(R[2, 2]) if self.obj[name]["class"] != "screw" else
                                                           math.sqrt(max(0.0, 1 - R[2, 0] ** 2)), -1, 1)))}

    def all_xy(self) -> dict[str, np.ndarray]:
        return {n: self.d.body(n).xpos[:2].copy() for n in self.obj}

    # --- low level ----------------------------------------------------------------------------------------------
    def step_s(self, seconds: float):
        for _ in range(int(round(seconds / self.m.opt.timestep))):
            mujoco.mj_step(self.m, self.d)

    def q(self) -> np.ndarray:
        return self.d.qpos[self.qids].copy()

    def loads(self) -> np.ndarray:
        return np.abs(self.d.actuator_force) / self.fmax * 1000

    def jaw_norm(self) -> float:
        return float(1 - self.d.qpos[self.qids[7]] / self.hi[7])

    def tcp(self) -> np.ndarray:
        return self.ik.fk(self.q())[0]

    def _actual(self, xyz, yaw):
        """Policy frame -> where the arm really goes (the per-episode base calibration error)."""
        p = self.bias.apply(np.asarray(xyz[:2], float))[0]
        return np.array([p[0], p[1], xyz[2]]), yaw + self.bias.yaw_deg

    def go_q(self, q_goal, speed=JOINT_SPEED, contact_abort=True, window=SETTLE_WINDOW, iters=SETTLE_ITERS) -> dict:
        """Ramp the 7 arm goals to q_goal at <= speed rad/s, then settle closed-loop (goal pushed past the target by
        gain x error, at most window). -> {"ok", "peak_load", "why", "short_rad"}."""
        d, n = self.d, int(round(CTRL_DT / self.m.opt.timestep))
        start = d.ctrl[:7].copy()
        q_goal = np.asarray(q_goal, float)[:7]
        k = max(1, int(np.ceil(np.abs(q_goal - start).max() / (speed * CTRL_DT))))
        peak = 0.0
        for i in range(1, k + 1):
            d.ctrl[:7] = start + (q_goal - start) * i / k
            for _ in range(n):
                mujoco.mj_step(self.m, d)
            peak = max(peak, float(self.loads()[:7].max()))
            if contact_abort and peak > LOAD_ABORT:
                d.ctrl[:7] = d.qpos[self.qids[:7]]
                return {"ok": False, "peak_load": round(peak), "why": f"contact abort (load {peak:.0f})", "short_rad": None}
        cmd = q_goal.copy()
        err = q_goal - self.q()[:7]
        for _ in range(iters):
            self.step_s(SETTLE_WAIT_S)
            err = q_goal - self.q()[:7]
            peak = max(peak, float(self.loads()[:7].max()))
            if np.abs(err).max() <= SETTLE_TOL:
                break
            if contact_abort and peak > LOAD_ABORT:
                d.ctrl[:7] = d.qpos[self.qids[:7]]
                return {"ok": False, "peak_load": round(peak), "why": f"contact abort while settling (load {peak:.0f})",
                        "short_rad": float(np.abs(err).max())}
            cmd = np.clip(cmd + SETTLE_GAIN * np.where(np.abs(err) > SETTLE_TOL, err, 0.0), q_goal - window, q_goal + window)
            d.ctrl[:7] = np.clip(cmd, self.lo[:7], self.hi[:7])
        return {"ok": True, "peak_load": round(peak), "why": "", "short_rad": float(np.abs(err).max())}

    def solve(self, xyz, yaw, lean=None):
        """IK in the policy frame. lean None: least-lean search (both jaw directions); else that lean from the
        current joints (straight-line moves keep the orientation). -> (q, lean, yaw) or (None, None, yaw)."""
        p, yw = self._actual(xyz, yaw)
        q0 = self.q()
        if lean is not None:
            sol = self.ik.solve_rot(p, self.ik.target_rot(p, lean, yw), q0, 0)
            return (sol, lean, yaw) if sol is not None else (None, None, yaw)
        best = (None, None, yaw)
        for y in (yaw, axis_wrap_deg(yaw + 180) if abs(axis_wrap_deg(yaw + 180) - yaw) > 1 else None):
            if y is None:
                continue
            sol, ln = self.ik.solve_least_lean(p, q0, y + self.bias.yaw_deg)
            if sol is not None and (best[0] is None or abs(ln) < abs(best[1])):
                best = (sol, ln, y)
        return best

    def move_tcp(self, xyz, yaw) -> dict:
        """Point-to-point, via the travel height when moving sideways (as executor Skill.travel)."""
        from .plan import TRAVEL_Z
        cur = self.tcp()
        pts = []
        if np.hypot(*(np.asarray(xyz[:2]) - cur[:2])) > 0.03:
            if cur[2] < TRAVEL_Z - 0.01:
                pts.append([cur[0], cur[1], TRAVEL_Z])
            if xyz[2] < TRAVEL_Z:
                pts.append([xyz[0], xyz[1], TRAVEL_Z])
        pts.append(list(xyz))
        out = {"ok": True, "lean": None, "peak_load": 0}
        for p in pts:
            q, lean, _ = self.solve(p, yaw)
            if q is None:
                return {"ok": False, "why": f"unreachable {np.round(p, 3).tolist()} yaw {yaw:.0f}", "lean": None}
            r = self.go_q(q)
            out.update(lean=lean, peak_load=max(out["peak_load"], r["peak_load"]))
            if not r["ok"]:
                return {**out, **r}
        return out

    def descend(self, xy, z_to, yaw, lean) -> dict:
        """Straight down (constant orientation): free to GUARD_ABOVE over z_to, then DESCENT_STEP steps that stop
        when the measured TCP lags the commanded one (touching) or a servo load passes LOAD_ABORT."""
        z0 = float(self.tcp()[2])
        zs = []
        if z0 > z_to + GUARD_ABOVE + DESCENT_STEP:
            zs.append(("free", z_to + GUARD_ABOVE))
        z = min(z0, z_to + GUARD_ABOVE)
        while z > z_to + 1e-6:
            z = max(z_to, z - DESCENT_STEP)
            zs.append(("guarded", z))
        peak = 0.0
        for kind, z in zs:
            q, _, _ = self.solve([xy[0], xy[1], z], yaw, lean)
            if q is None:
                return {"ok": False, "contact_z": None, "why": f"no IK at z={z:.3f}", "z": float(self.tcp()[2])}
            r = self.go_q(q, contact_abort=True) if kind == "free" else \
                self.go_q(q, speed=0.6, contact_abort=True, window=TOUCH_WINDOW, iters=TOUCH_ITERS)
            peak = max(peak, r["peak_load"])
            meas = float(self.tcp()[2])
            if not r["ok"] or (kind == "guarded" and meas > z + BLOCKED_M):
                up, _, _ = self.solve([xy[0], xy[1], meas + 0.002], yaw, lean)
                if up is not None:
                    self.go_q(up, contact_abort=False)
                return {"ok": True, "contact_z": round(meas, 4), "why": r["why"] or f"touched at z={meas:.3f}",
                        "z": float(self.tcp()[2]), "peak_load": peak}
        return {"ok": True, "contact_z": None, "why": "", "z": float(self.tcp()[2]), "peak_load": peak}

    def grip(self, open_fraction: float, force: float | None = None) -> dict:
        """Jaw goal ramp with a torque limit (closing: force x JAW_TORQUE; opening: full), wait until the jaws stop."""
        m, d = self.m, self.d
        closing = open_fraction < self.jaw_norm()
        f = float(np.clip(0.35 if force is None else force, 0.1, 1.0)) if closing else 1.0
        m.actuator_forcerange[7] = [-JAW_TORQUE * f, JAW_TORQUE * f]
        start, goal = float(d.ctrl[7]), float(self.hi[7] * (1 - np.clip(open_fraction, 0, 1)))
        n = int(round(CTRL_DT / m.opt.timestep))
        k = int(GRIP_RAMP_S / CTRL_DT)
        for i in range(1, k + 1):
            d.ctrl[7] = start + (goal - start) * i / k
            for _ in range(n):
                mujoco.mj_step(m, d)
        t = 0.0
        while t < GRIP_MAX_S - GRIP_RAMP_S:
            for _ in range(n):
                mujoco.mj_step(m, d)
            t += CTRL_DT
            if abs(d.qvel[self.dofs[7]]) < 0.03:
                break
        norm = self.jaw_norm()
        held = closing and self.empty_norm is not None and norm > self.empty_norm + HOLD_MARGIN
        return {"norm": round(norm, 4), "holding": bool(held), "force": f, "load": round(float(self.loads()[7]))}

    def jaw_check(self) -> float:
        """Close empty in the air, read the jaw norm (the reference 'holding' is judged against), open again."""
        g = self.grip(0.0, 0.35)
        self.grip(1.0)
        return g["norm"]


# --- the policy and the judge --------------------------------------------------------------------------------------
def pick_place(sim: KitSim, pl: ObjectPlan, cell: str, sheet_true: Pose2D) -> ObjRun:
    """Run one object's plan on the twin, with phase records and the judge's truth/diagnosis."""
    S = sim.spec
    run = ObjRun(pl.name, pl.cls, cell, pl.describe(), sheet_true)
    dims = S.class_dims(pl.cls)
    others0 = sim.all_xy()
    before = sim.obj_pose(pl.name)
    run.phase("locate", measurements_from="sheet pose (overview) + spot", note=f"start {np.round(pl.pick_xy, 4).tolist()}",
              x=pl.pick_xy[0], y=pl.pick_xy[1], yaw=pl.yaw_pick)

    sim.grip(1.0)
    r = sim.move_tcp([*pl.pick_xy, pl.hover_z], pl.yaw_pick)
    if not r["ok"]:
        run.phase("approach", "abort", r.get("why", ""))
        run.outcome = "blocked"
        return _judge(sim, run, others0, before, dims)
    lean = r["lean"]
    run.phase("approach", "ok", lean_deg=lean, peak_load=r["peak_load"])
    pushed0 = sim.obj_pose(pl.name)["xyz"]
    dsc = sim.descend(pl.pick_xy, pl.grasp_z, pl.yaw_pick, lean)
    moved_on_descent = float(np.linalg.norm(sim.obj_pose(pl.name)["xyz"][:2] - pushed0[:2]) * 1000)
    if not dsc["ok"]:
        run.phase("grasp", "abort", dsc["why"])
        run.outcome = "blocked"
        return _judge(sim, run, others0, before, dims)
    g = sim.grip(0.0, pl.grip_force)
    yaw_before_close = before["yaw_deg"]
    run.phase("grasp", "ok" if g["holding"] else "fail", dsc["why"], tcp_z=round(dsc["z"], 4), contact_z=dsc["contact_z"],
              jaw_norm=g["norm"], jaw_load=g["load"], holding=g["holding"], force=g["force"],
              truth_moved_on_descent_mm=round(moved_on_descent, 1),
              truth_pad_bottom_mm=round((dsc["z"] - PAD_REACH) * 1000, 1), truth_object_top_mm=round(dims["height"] * 1000, 1))
    if not g["holding"]:
        sim.grip(1.0)
        sim.move_tcp([*pl.pick_xy, pl.hover_z], pl.yaw_pick)
        run.outcome = "missed"
        return _judge(sim, run, others0, before, dims, grasp_tcp_z=dsc["z"], moved_on_descent=moved_on_descent)
    held_yaw = sim.obj_pose(pl.name)["yaw_deg"]
    r = sim.move_tcp([*pl.pick_xy, pl.travel_z], pl.yaw_pick)
    lifted = sim.obj_pose(pl.name)
    still = sim.jaw_norm() > sim.empty_norm + HOLD_MARGIN
    run.phase("lift", "ok" if (r["ok"] and still) else "fail", "" if still else "jaws closed further: lost it",
              jaw_norm=round(sim.jaw_norm(), 4), truth_lift_mm=round(float(lifted["xyz"][2] - before["xyz"][2]) * 1000, 1),
              truth_yaw_change_deg=round(abs(axis_wrap_deg(held_yaw - yaw_before_close)), 1))
    if not still:
        sim.grip(1.0)
        run.outcome = "dropped"
        return _judge(sim, run, others0, before, dims, grasp_tcp_z=dsc["z"], moved_on_descent=moved_on_descent)
    hover_place = max(pl.place_z + 0.03, pl.travel_z)
    r = sim.move_tcp([*pl.place_xy, hover_place], pl.yaw_place)
    still = sim.jaw_norm() > sim.empty_norm + HOLD_MARGIN
    run.phase("carry", "ok" if (r["ok"] and still) else "fail", r.get("why", ""), jaw_norm=round(sim.jaw_norm(), 4))
    if not r["ok"] or not still:
        sim.grip(1.0)
        run.outcome = "dropped" if not still else "blocked"
        return _judge(sim, run, others0, before, dims, grasp_tcp_z=dsc["z"], moved_on_descent=moved_on_descent)
    dpl = sim.descend(pl.place_xy, pl.place_z, pl.yaw_place, r["lean"])
    at_release = sim.obj_pose(pl.name)
    tcp_err = sim.bias.apply(np.asarray(pl.place_xy))[0] - sim.tcp()[:2]      # vs where the arm meant to be
    run.phase("place", "ok" if dpl["ok"] else "abort", dpl["why"], tcp_z=round(dpl["z"], 4), contact_z=dpl["contact_z"],
              tcp_short_mm=round(float(np.linalg.norm(tcp_err)) * 1000, 1))
    sim.grip(pl.release_open)
    after_open = sim.obj_pose(pl.name)
    run.phase("release", "ok", open_fraction=pl.release_open, jaw_norm=round(sim.jaw_norm(), 4))
    sim.move_tcp([*pl.place_xy, pl.travel_z], pl.yaw_place)
    sim.step_s(SETTLE_AFTER_S)
    final = sim.obj_pose(pl.name)
    run.phase("retreat", "ok",
              truth_drop_mm=round(float(at_release["xyz"][2] - dims["height"] / 2 - kitting_scene.SHEET_Z) * 1000, 1),
              truth_moved_after_open_mm=round(float(np.linalg.norm(after_open["xyz"][:2] - at_release["xyz"][:2])) * 1000, 1),
              truth_moved_on_retreat_mm=round(float(np.linalg.norm(final["xyz"][:2] - after_open["xyz"][:2])) * 1000, 1),
              truth_release_offset_mm=round(float(np.linalg.norm(at_release["xyz"][:2] - np.asarray(pl.place_xy))) * 1000, 1))
    run.outcome = "released"
    return _judge(sim, run, others0, before, dims, grasp_tcp_z=dsc["z"], moved_on_descent=moved_on_descent)


def in_cell(spec: Spec, sheet_true: Pose2D, cell: str, pose: dict, cls: str) -> tuple[bool, float, dict]:
    """(footprint fully inside the cell and lying flat, centre error mm, details) against the TRUE sheet pose."""
    c = spec.cell_pose(sheet_true, cell)
    dims = spec.class_dims(cls)
    local = c.R().T @ (pose["xyz"][:2] - [c.x, c.y])
    err = float(np.linalg.norm(local) * 1000)
    yaw = math.radians(pose["yaw_deg"] - c.yaw_deg)
    a, b = dims["half_long"], dims["half_across"]
    if dims["round"]:
        ext = np.array([a, a])
    else:
        ext = np.abs(np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])) @ [a, b]
    half = spec.cells()[cell]["size"] / 2000
    inside = bool(np.all(np.abs(local) + ext <= half + 1e-4))
    centre_in = bool(np.all(np.abs(local) <= half))
    flat = pose["tilt_deg"] <= REST_TILT_DEG and pose["xyz"][2] < dims["height"] * 1.6 + 0.003
    return bool(inside and flat), err, {"centre_in_cell": centre_in, "footprint_in_cell": inside, "flat": bool(flat),
                                  "local_mm": np.round(local * 1000, 1).tolist(), "tilt_deg": round(pose["tilt_deg"], 1)}


def _judge(sim, run: ObjRun, others0, before, dims, grasp_tcp_z=None, moved_on_descent=0.0) -> ObjRun:
    S = sim.spec
    pose = sim.obj_pose(run.name)
    ok, err, det = in_cell(S, run.sheet_true, run.cell, pose, run.cls)
    knocked = {n: round(float(np.linalg.norm(xy - others0[n]) * 1000), 1) for n, xy in sim.all_xy().items()
               if n != run.name and np.linalg.norm(xy - others0[n]) * 1000 > MOVED_MM}
    if run.outcome == "released":
        run.outcome = "placed" if ok else "off_cell"
    run.truth = {"final_xyz": np.round(pose["xyz"], 4).tolist(), "final_yaw_deg": round(pose["yaw_deg"], 1),
                 "error_mm": round(err, 1), "in_cell": ok, **det, "knocked": knocked}
    run.diagnosis = diagnose(run, dims, grasp_tcp_z, moved_on_descent)
    return run


def diagnose(run: ObjRun, dims: dict, grasp_tcp_z, moved_on_descent) -> list[str]:
    """What the records say went wrong, as an operator reading them would put it (tags the proposer uses)."""
    ph = {p["phase"]: p for p in run.phases}
    tags = []
    g = ph.get("grasp", {}).get("measurements", {})
    if run.outcome == "missed":
        pad_bottom = g.get("truth_pad_bottom_mm")
        if g.get("contact_z") is not None and moved_on_descent < 2.0:
            tags.append("descent_blocked")              # touched the table/object above the grasp height
        elif moved_on_descent >= 2.0:
            tags.append("pushed_on_descent")            # a pad landed on the object and shoved it
        elif pad_bottom is not None and pad_bottom > 0.6 * dims["height"] * 1000:
            tags.append("grasp_too_high")               # the pads closed above most of the object
        else:
            tags.append("closed_empty")
    if run.outcome == "dropped":
        tags.append("slipped_in_carry")
    lift = ph.get("lift", {}).get("measurements", {})
    if (lift.get("truth_yaw_change_deg") or 0) > 15:
        tags.append("rotated_in_grasp")
    r = ph.get("retreat", {}).get("measurements", {})
    if r:
        if (r.get("truth_moved_after_open_mm") or 0) > 3 or (r.get("truth_drop_mm") or 0) > 6:
            tags.append("bounced_on_release")
        if (r.get("truth_moved_on_retreat_mm") or 0) > 3:
            tags.append("dragged_on_retreat")
    if run.outcome == "blocked":
        tags.append("unreachable")
    if run.truth.get("knocked"):
        tags.append("knocked_neighbour")
    if run.outcome == "off_cell" and not tags:
        tags.append("off_target")
    return tags


def run_episode(sim: KitSim, params: dict, rng: np.random.Generator, dr: DR, only: list[str] | None = None,
                use_perception: bool = True) -> dict:
    """One kit: reset with randomization, locate the sheet in the rendered overview (or take the truth), pick and
    place every object (or only those named). -> {"truth", "locate", "objects": [ObjRun], ...}."""
    S = sim.spec
    truth = sim.reset(rng, dr)
    sheet_true = Pose2D(*truth["sheet"])
    loc = {"ok": True, "pose": sheet_true, "why": "truth (perception off)", "markers": []}
    if use_perception:
        img = sim.render()
        loc = L.locate_sheet(img, S, sim.cam)
        if not loc["ok"]:
            return {"truth": truth, "locate": {k: v for k, v in loc.items() if k != "corners_px"}, "objects": [],
                    "sheet_error_mm": None, "aborted": f"sheet not located: {loc.get('why')}"}
    est = loc["pose"]
    sheet_err = float(np.hypot(est.x - sheet_true.x, est.y - sheet_true.y) * 1000)
    runs = []
    for t in order(S.targets(est)):
        if only and t["name"] not in only:
            continue
        pl = plan_pick_place(t, params[t["class"]], S.class_dims(t["class"])["round"])
        runs.append(pick_place(sim, pl, t["cell"], sheet_true))
    return {"truth": truth, "locate": {"pose": est.as_list(), "markers": loc.get("markers"), "residual_mm": loc.get("residual_mm"),
                                       "sigma_mm": loc.get("sigma_mm")},
            "sheet_error_mm": round(sheet_err, 2), "sheet_yaw_error_deg": round(wrap_deg(est.yaw_deg - sheet_true.yaw_deg), 3),
            "objects": runs, "aborted": None}
