"""Kinematics for any arm on a MuJoCo model: n arm joints + a gripper actuator, a `grasp_tcp` site whose +y axis is
the approach direction (out of the jaws) and +x the jaw closing axis.

Same algorithm as sim/build_scene.ArmIK (which is fixed at the ElRobot's 7 arm joints): damped least squares to
position + rotation, and the least-lean jaws-down search of the shared skill config (learning/skillcfg.lean_search),
so every connector's twin approaches like the real ElRobot does.
"""
from __future__ import annotations

import mujoco
import numpy as np

from generaloid.core.paths import ensure_sim_on_path

POS_TOL = 0.002          # m: solved when the TCP is this close (build_scene.ArmIK)
STEP_CLIP = 0.08         # rad per iteration (build_scene.ArmIK)
LIMIT_MARGIN = 0.001     # rad inside the joint limits (build_scene.ArmIK)
DAMPING = 1e-4           # damped least squares (build_scene.ArmIK)
ITERATIONS = 300


class GenericIK:
    def __init__(self, model, n_arm: int, site: str = "grasp_tcp", rot_tol: float = 0.02, rot_weight: float = 0.1):
        """rot_tol (rad) / rot_weight: orientation tolerance and weight vs position. The ElRobot values (0.02, 0.1) are
        build_scene.ArmIK's; a 5-DoF arm cannot hold an arbitrary orientation exactly and needs a looser tolerance."""
        self.model, self.data = model, mujoco.MjData(model)
        joints = model.actuator_trnid[:, 0]
        self.n = n_arm
        self.qids, self.dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints][:n_arm]
        self.site = model.site(site).id
        self.low, self.high = model.actuator_ctrlrange.T.copy()
        self.rot_tol, self.rot_weight = rot_tol, rot_weight

    def fk(self, q):
        self.data.qpos[self.qids] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy()

    @staticmethod
    def target_rot(xyz, lean_deg, yaw_deg):
        ensure_sim_on_path()
        import build_scene
        return build_scene.ArmIK.target_rot(xyz, lean_deg, yaw_deg)

    def _iterate(self, xyz, q, Rt=None):
        """Damped least squares from q; returns q when converged, else None."""
        m, d, n = self.model, self.data, self.n
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(ITERATIONS):
            d.qpos[self.qids] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err = np.asarray(xyz) - d.site_xpos[self.site]
            mujoco.mj_jacSite(m, d, jp, jr, self.site)
            if Rt is None:
                if np.linalg.norm(err) < 1.5 * POS_TOL:
                    return q
                jac, e = jp[:, self.dofs], err
            else:
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, (Rt @ d.site_xmat[self.site].reshape(3, 3).T).reshape(-1))
                if quat[0] < 0:
                    quat *= -1
                ln = np.linalg.norm(quat[1:])
                err_r = quat[1:] * (2 * np.arctan2(ln, quat[0]) / max(ln, 1e-9))
                if np.linalg.norm(err) < POS_TOL and np.linalg.norm(err_r) < self.rot_tol:
                    return q
                w = self.rot_weight
                jac, e = np.vstack([jp[:, self.dofs], w * jr[:, self.dofs]]), np.r_[err, w * err_r]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + DAMPING * np.eye(len(e)), e)
            q[:n] = np.clip(q[:n] + np.clip(dq, -STEP_CLIP, STEP_CLIP), self.low[:n] + LIMIT_MARGIN,
                            self.high[:n] - LIMIT_MARGIN)
        return None

    def solve(self, xyz, top_down: bool = False, q0=None, attempts: int = 16, seed: int = 3):
        """Position-only IK (orientation free), the last resort."""
        assert not top_down, "jaws-down IK: solve_least_lean"
        rng = np.random.default_rng(seed)
        for attempt in range(attempts):
            q = np.zeros(len(self.qids)) if q0 is None else np.array(q0, float)
            if attempt:
                q[:self.n] = rng.uniform(self.low[:self.n], self.high[:self.n])
            sol = self._iterate(xyz, q)
            if sol is not None:
                return sol
        return None

    def solve_rot(self, xyz, Rt, q0, restarts: int = 0, seed: int = 3):
        """Position + rotation Rt. First from q0; with restarts, the solution nearest q0."""
        rng = np.random.default_rng(seed)
        q0 = np.asarray(q0, float)
        best = None
        for attempt in range(1 + restarts):
            q = q0.copy()
            if attempt:
                q[:self.n] = rng.uniform(self.low[:self.n], self.high[:self.n])
            sol = self._iterate(xyz, q, Rt)
            if sol is not None and (best is None or np.abs(sol[:self.n] - q0[:self.n]).max()
                                    < np.abs(best[:self.n] - q0[:self.n]).max()):
                best = sol.copy()
            if attempt == 0 and best is not None:
                return best
        return best

    def solve_least_lean(self, xyz, q0, yaw_deg: float = 0.0, max_jump: float = 0.6):
        """(q, lean_deg) of the least-leaning reachable jaws-down pose (skillcfg.lean_search), (None, None) if none."""
        ensure_sim_on_path()
        from learning import skillcfg
        q0 = np.asarray(q0, float)
        return skillcfg.lean_search(lambda lean, rs: self.solve_rot(xyz, self.target_rot(xyz, lean, yaw_deg), q0, rs),
                                    near=lambda q: np.abs(q[:self.n] - q0[:self.n]).max() <= max_jump)
