"""Robot profile + kinematic model for the real-arm server. Nothing robot-specific is hard-coded in real_arm_mcp.py:
it reads a profile (sim/robots/<name>.json, chosen with ROBOT_PROFILE) and builds its kinematics from it.

Two kinematics backends:
  workspace_xml  an existing MuJoCo XML with one actuator per joint, a TCP site and a wrist camera (ElRobot: the
                 simulator's generated/workspace.xml, so the real arm and the sim share one model)
  urdf           any URDF, loaded for kinematics only (meshes are not needed): joints by name, a TCP point on a
                 link, and the wrist camera pose relative to a link (for a new arm such as the SO-101)
"""
from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent


def _ns(x):
    if isinstance(x, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in x.items()})
    return x


def load(path: str | None = None) -> SimpleNamespace:
    """Profile from `path`, else env ROBOT_PROFILE (a file path or a name in sim/robots/), else elrobot."""
    choice = path or os.environ.get("ROBOT_PROFILE") or "elrobot"
    p = Path(choice)
    if not p.exists():
        p = HERE / "robots" / (choice if choice.endswith(".json") else choice + ".json")
    raw = json.loads(p.read_text())
    prof = _ns(raw)
    prof.path, prof.raw = p, raw
    return prof


def env(name: str, default, cast=float):
    """An environment variable overrides one profile value (kept for the settings people already use)."""
    return cast(os.environ[name]) if os.environ.get(name) not in (None, "") else default


class Kin:
    """Forward/inverse kinematics of the arm on a MuJoCo model. q = [arm joints..., gripper joint] in radians."""

    def __init__(self, prof):
        k, self.ik = prof.kinematics, prof.ik
        self.n_arm = len(prof.motors.arm)
        if k.backend == "workspace_xml":
            import build_scene
            layout = build_scene.layout_path()
            xml = (build_scene.generated_dir(layout) / "workspace.xml"
                   if k.xml == "generated/workspace.xml" else HERE / k.xml)
            if not xml.exists():
                build_scene.build(layout, xml.parent, report=False)
            self.model = mujoco.MjModel.from_xml_path(str(xml))
            joints = self.model.actuator_trnid[:, 0]              # one actuator per joint: arm joints, then the gripper
            self.site, self.cam = self.model.site(k.tcp_site).id, self.model.camera(k.wrist_camera).id
            self.low, self.high = self.model.actuator_ctrlrange.T.copy()
        elif k.backend == "urdf":
            self.model = self._from_urdf(k, prof.path.parent)
            joints = np.array([self.model.joint(n).id for n in list(k.arm_joints) + [k.gripper_joint]])
            self.site, self.cam = self.model.site("tcp").id, self.model.camera("wrist").id
            self.low, self.high = self.model.jnt_range[joints].T.copy()
        else:
            raise ValueError(f"unknown kinematics backend {k.backend!r}")
        self.qids = self.model.jnt_qposadr[joints]
        self.dofs = self.model.jnt_dofadr[joints][:self.n_arm]
        self.data, self.cam_data = mujoco.MjData(self.model), mujoco.MjData(self.model)

    @staticmethod
    def _from_urdf(k, base: Path):
        path = Path(os.path.expandvars(os.path.expanduser(k.urdf)))
        if not path.is_absolute():
            path = base / path
        root = ET.parse(path).getroot()
        for link in root.iter("link"):                          # kinematics only: drop geometry so no meshes are needed
            for tag in ("visual", "collision"):
                for el in link.findall(tag):
                    link.remove(el)
        for old in root.findall("mujoco"):
            root.remove(old)
        ext = ET.SubElement(root, "mujoco")
        ET.SubElement(ext, "compiler", fusestatic="false", discardvisual="true", boundmass="1e-4", boundinertia="1e-8")
        spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
        # TCP frame convention used everywhere: +y = approach axis (out of the gripper), +x = jaw closing axis.
        site = spec.body(k.tcp_link).add_site(name="tcp", pos=list(k.tcp_xyz))
        if getattr(k, "tcp_R", None):                           # link frame <- TCP frame, when the link's axes differ
            q4 = np.zeros(4)
            mujoco.mju_mat2Quat(q4, np.array(k.tcp_R, float).flatten())
            site.quat = q4
        T = np.array(k.wrist_camera.T_link_camera, float)       # link <- camera, OpenCV axes (x right, y down, z forward)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (T[:3, :3] @ np.diag([1.0, -1.0, -1.0])).flatten())   # MuJoCo cameras look along -z, y up
        cam = spec.body(k.wrist_camera.link).add_camera(name="wrist")
        cam.pos, cam.quat, cam.fovy = T[:3, 3], quat, float(k.wrist_camera.fovy)
        return spec.compile()

    def fk(self, q):
        """(tcp xyz, tcp rotation matrix) in the base frame."""
        self.data.qpos[self.qids] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy()

    def wrist_pose(self, q):
        """(camera position, camera rotation, vertical field of view in degrees) of the wrist camera."""
        d = self.cam_data
        d.qpos[self.qids] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_camlight(self.model, d)
        return d.cam_xpos[self.cam].copy(), d.cam_xmat[self.cam].reshape(3, 3).copy(), float(self.model.cam_fovy[self.cam])

    def solve(self, xyz, top_down: bool = False, q0=None):
        """Position-only IK (orientation free): the last resort when no near-vertical pose exists."""
        assert not top_down, "orientation-constrained IK lives in real_arm_mcp._solve_tcp"
        ik, m, d, n = self.ik, self.model, self.data, self.n_arm
        rng = np.random.default_rng(ik.seed)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for attempt in range(ik.pos_only_attempts):
            q = (np.zeros(len(self.qids)) if q0 is None else np.array(q0, float)) if attempt == 0 else rng.uniform(self.low, self.high)
            for _ in range(ik.iterations):
                d.qpos[self.qids] = q
                mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
                err = np.asarray(xyz) - d.site_xpos[self.site]
                mujoco.mj_jacSite(m, d, jp, jr, self.site)
                if np.linalg.norm(err) < ik.pos_only_tol:
                    return q
                jac = jp[:, self.dofs]
                dq = jac.T @ np.linalg.solve(jac @ jac.T + ik.damping * np.eye(3), err)
                q[:n] = np.clip(q[:n] + np.clip(dq, -ik.max_step, ik.max_step), self.low[:n] + ik.limit_margin, self.high[:n] - ik.limit_margin)
        return None
