"""Arm kinematics for the executor: the robot profile's joint mapping, FK, and the real arm's least-lean IK,
in the WORLD frame of prompts/scene-contract.md (the kinematic model's own base frame has +y forward; the world
frame has +x forward: p_world = T_world_base @ p_base, T_world_base from calibration/scene_calib.json).

The IK is the same algorithm as sim/real_arm_mcp.py (_target_rot/_solve_rot/_solve_tcp): jaws as close to
straight down as the target allows, stepping the lean up gradually and staying near the seed. It is copied rather
than imported because importing real_arm_mcp connects the arm_mcp Station client and an MCP server.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SIM = REPO / "sim"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SIM) not in sys.path:
    sys.path.append(str(SIM))      # appended: sim/recorder.py must not shadow the recorder package
os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")

DEFAULT_T_WORLD_BASE = np.array([[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


def scene_calib_path() -> Path | None:
    for p in (os.environ.get("EXECUTOR_SCENE_CALIB"), REPO / "calibration" / "scene_calib.json",
              REPO.parent / "perception" / "calibration" / "scene_calib.json"):
        if p and Path(p).exists():
            return Path(p)
    return None


def load_scene_calib() -> dict | None:
    p = scene_calib_path()
    return json.loads(p.read_text()) if p else None


class Kinematics:
    """q = [arm joints..., gripper] in radians. All public xyz are WORLD frame metres."""

    def __init__(self, profile=None, T_world_base=None):
        import mujoco  # noqa: F401  (fail early with a clear error if missing)
        import robot_profile
        self.P = profile or robot_profile.load()
        self.kin = robot_profile.Kin(self.P)
        self.lock = threading.RLock()
        P = self.P
        self.arm_ids = [int(m) for m in P.motors.arm]
        self.grip_id = int(P.motors.gripper)
        self.ids = self.arm_ids + [self.grip_id]
        self.name_of = {int(k): v for k, v in vars(P.motors.names).items()}
        self.names = [self.name_of[m] for m in self.ids]
        self.na = len(self.arm_ids)
        self.low, self.high = self.kin.low, self.kin.high
        if T_world_base is None:
            sc = load_scene_calib()
            T_world_base = (sc or {}).get("arm", {}).get("T_world_base", DEFAULT_T_WORLD_BASE)
        self.T = np.array(T_world_base, float)
        self.R, self.t = self.T[:3, :3], self.T[:3, 3]
        self.yaw_offset = float(np.degrees(np.arctan2(self.R[1, 0], self.R[0, 0])))   # world yaw = base yaw + offset

    # --- frames ----------------------------------------------------------------------
    def to_base(self, p_world):
        return self.R.T @ (np.asarray(p_world, float) - self.t)

    def to_world(self, p_base):
        return self.R @ np.asarray(p_base, float) + self.t

    # --- joint mapping (profile) ---------------------------------------------------------
    def q_from_norm(self, norm: dict[int, float], raw: dict[int, int] | None = None) -> np.ndarray:
        P, lo, hi = self.P, self.low, self.high
        n = [norm[m] for m in self.ids]
        if P.kinematics.arm_mapping == "range_fraction":
            arm = [lo[i] + n[i] * (hi[i] - lo[i]) for i in range(self.na)]
        else:   # ticks_centered
            z, sg, tick = P.kinematics.zero_ticks, P.kinematics.signs, 2 * np.pi / P.motors.ticks_per_turn
            arm = [sg[i] * (raw[self.arm_ids[i]] - z[i]) * tick for i in range(self.na)]
        g = n[self.na]
        grip = hi[self.na] * (1 - g) if P.kinematics.gripper_mapping == "upper_times_one_minus_norm" else lo[self.na] + g * (hi[self.na] - lo[self.na])
        return np.array(arm + [grip])

    def norm_from_q(self, q, ranges: dict[int, tuple[int, int]] | None = None) -> dict[int, float]:
        """Arm joint angles -> normalized targets per arm motor id."""
        P, lo, hi = self.P, self.low, self.high
        if P.kinematics.arm_mapping == "range_fraction":
            return {self.arm_ids[i]: float((q[i] - lo[i]) / (hi[i] - lo[i])) for i in range(self.na)}
        z, sg, tick = P.kinematics.zero_ticks, P.kinematics.signs, 2 * np.pi / P.motors.ticks_per_turn
        out = {}
        for i, m in enumerate(self.arm_ids):
            raw = z[i] + sg[i] * q[i] / tick
            a, b = ranges[m]
            out[m] = float((raw - a) / (b - a))
        return out

    # --- FK / IK ---------------------------------------------------------------------------
    def fk(self, q):
        """(tcp xyz world, rotation world <- tcp)."""
        with self.lock:
            p, r = self.kin.fk(np.asarray(q, float))
        return self.to_world(p), self.R @ r

    def tilt_deg(self, q) -> float:
        _, r = self.fk(q)
        return float(np.degrees(np.arccos(np.clip(-r[2, 1], -1, 1))))

    def yaw_now(self, q) -> float:
        """World yaw (deg) of the jaw closing axis about vertical."""
        _, r = self.fk(q)
        return float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))

    def tip_of(self, q, length: float):
        p, r = self.fk(q)
        return p + length * r[:, 1]

    def _target_rot(self, xyz_b, lean_deg, yaw_deg_b):
        r = np.asarray(xyz_b[:2], float)
        r = r / np.linalg.norm(r) if np.linalg.norm(r) > 1e-6 else np.array([0.0, 1.0])
        t, yw = np.radians(lean_deg), np.radians(yaw_deg_b)
        a = np.array([np.sin(t) * r[0], np.sin(t) * r[1], -np.cos(t)])
        j = np.array([np.cos(yw), np.sin(yw), 0.0])
        j = j - (j @ a) * a
        if np.linalg.norm(j) < 1e-6:
            j = np.array([-r[1], r[0], 0.0])
        j /= np.linalg.norm(j)
        return np.column_stack([j, a, np.cross(j, a)])

    def _solve_rot(self, xyz_b, Rt, q0, restarts=0):
        import mujoco
        K, IKP, NA = self.kin, self.P.ik, self.na
        m, d = K.model, K.data
        rng = np.random.default_rng(IKP.seed)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        best = None
        for attempt in range(1 + restarts):
            q = np.array(q0, float)
            if attempt:
                q[:NA] = rng.uniform(K.low[:NA], K.high[:NA])
            for _ in range(IKP.iterations):
                d.qpos[K.qids] = q
                mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
                err = np.asarray(xyz_b) - d.site_xpos[K.site]
                mujoco.mj_jacSite(m, d, jp, jr, K.site)
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, (Rt @ d.site_xmat[K.site].reshape(3, 3).T).reshape(-1))
                if quat[0] < 0:
                    quat *= -1
                ln = np.linalg.norm(quat[1:])
                err_r = quat[1:] * (2 * np.arctan2(ln, quat[0]) / max(ln, 1e-9))
                if np.linalg.norm(err) < IKP.pos_tol and np.linalg.norm(err_r) < IKP.rot_tol:
                    if best is None or np.abs(q[:NA] - q0[:NA]).max() < np.abs(best[:NA] - q0[:NA]).max():
                        best = q.copy()
                    break
                jac, e = np.vstack([jp[:, K.dofs], IKP.rot_weight * jr[:, K.dofs]]), np.r_[err, IKP.rot_weight * err_r]
                dq = jac.T @ np.linalg.solve(jac @ jac.T + IKP.damping * np.eye(len(e)), e)
                q[:NA] = np.clip(q[:NA] + np.clip(dq, -IKP.max_step, IKP.max_step), K.low[:NA] + IKP.limit_margin, K.high[:NA] - IKP.limit_margin)
            if attempt == 0 and best is not None:
                return best
        return best

    def solve(self, xyz_world, seed, yaw_deg: float | None = None, top_down: bool = True):
        """(q, lean_deg) for the least-leaning reachable pose near `seed`; (None, None) if unreachable.
        yaw_deg is the WORLD yaw of the jaw closing axis (None keeps the seed's). top_down=False: position only."""
        xyz_b = self.to_base(xyz_world)
        seed = np.asarray(seed, float)
        with self.lock:
            if not top_down:
                q = self.kin.solve(xyz_b, top_down=False, q0=seed)
                return (q, None) if q is not None else (None, None)
            yaw_w = self.yaw_now(seed) if yaw_deg is None else yaw_deg
            yaw_b = yaw_w - self.yaw_offset
            IKP = self.P.ik
            # Order differs from real_arm_mcp in one place: a larger lean NEAR the seed is tried before random
            # restarts (other arm configurations) with a small lean. Same result where a near solution exists;
            # it avoids big reconfigurations and cuts planning from ~6 s to ~0.1 s for high lifts close to the base.
            for leans, restarts in ((IKP.leans_small, 0), (IKP.leans_large, 0), (IKP.leans_small, IKP.restarts), (IKP.leans_large, IKP.restarts)):
                for lean in leans:
                    q = self._solve_rot(xyz_b, self._target_rot(xyz_b, lean, yaw_b), seed, restarts)
                    if q is not None:
                        return q, float(lean)
        return None, None

    def solve_tip(self, tip_world, length: float, seed, yaw_deg: float | None):
        """Joints that put the TIP of a tool held along the approach axis (length below the TCP) at tip_world."""
        sol = np.asarray(seed, float)
        for _ in range(self.P.stir.tip_iterations if length > 0 else 1):
            _, rot = self.fk(sol)
            nxt, _ = self.solve(np.asarray(tip_world, float) - length * rot[:, 1], sol, yaw_deg)
            if nxt is None:
                return None
            sol = nxt
        return sol

    def in_limits(self, q, margin: float = 0.0) -> bool:
        q = np.asarray(q, float)[:self.na]
        return bool(np.all(q >= self.low[:self.na] + margin) and np.all(q <= self.high[:self.na] - margin))
