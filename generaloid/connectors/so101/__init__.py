"""SO-101 connector: the SO-101 follower (5-DoF arm + gripper, 6 STS3215 servos on one Station bus).

  sim   generaloid.twin.generic on the SO-101 scene of scene.py (norma-core's so101_robot_follower.urdf with meshes,
        the table of layout.json; both ported from origin/simulation_s101)
  real  sim/real_arm_mcp.py with ROBOT_PROFILE=so101 (sim/robots/so101.json, origin/robot-profiles): the same
        real-arm server as the ElRobot, every number from the profile. NEVER RUN ON HARDWARE: fill in the profile's
        `_to_verify` list first (bus serial, zero ticks and signs, wrist camera pose, camera ids).

This package is the worked example of docs/generaloid/WRITING_A_CONNECTOR.md.
"""
from __future__ import annotations

import json
import os
from functools import cached_property
from pathlib import Path

from generaloid.connectors.base import MOTION_SKILLS, Mount, TwinSpec
from generaloid.connectors.profile import ProfileConnector
from generaloid.core.paths import GENERATED, ensure_sim_on_path

HERE = Path(__file__).resolve().parent
LAYOUT = HERE / "layout.json"
# A 5-DoF arm cannot hold an arbitrary TCP orientation exactly (the jaw axis follows the arm's plane once the jaws
# lean): looser orientation tolerance and a higher orientation weight than the 7-DoF ElRobot (origin/simulation_s101).
IK_ROT_TOL = 0.06          # rad
IK_ROT_WEIGHT = 0.15


class So101Connector(ProfileConnector):
    name = "so101"
    title = "SO-101 follower (5-DoF arm + gripper, 6x STS3215)"
    profile_name = "so101"
    gripper_open_at_upper_limit = True           # URDF joint 6: -0.17 rad closed .. 1.75 rad open
    gripper_torque_nm = 0.5                      # scene.GRIPPER_TORQUE_NM
    hardware_verified = False

    SIM_SKILLS = MOTION_SKILLS | {"skill_config", "operator_reset_scene"}
    REAL_SKILLS = MOTION_SKILLS | {"pick", "drop", "survey"}

    @property
    def skills(self) -> frozenset[str]:
        return frozenset(self.SIM_SKILLS | self.REAL_SKILLS)

    def skills_for(self, mode: str) -> frozenset[str]:
        return frozenset(self.SIM_SKILLS if mode == "sim" else self.REAL_SKILLS)

    @property
    def frame_text(self) -> str:
        # The twin's frame (layout.json, as the ElRobot's). sim/robots/so101.json still describes the frame of the
        # TheRobotStudio URDF (+x ahead) that real mode uses: aligning the two is on the profile's _to_verify list.
        return ("Base frame, metres: origin = the joint-1 axis on the table top, +z up, +y = straight ahead of the arm "
                "(towards the overview camera), +x = the arm's right-hand side.")

    def frame_for(self, mode: str) -> str:
        return self.profile.frame_text if mode == "real" else self.frame_text

    @property
    def layout_path(self) -> Path:
        """layout.json, or ROBOT_SIM_LAYOUT if that file is an SO-101 layout ("robot_model": "so101")."""
        env = os.environ.get("ROBOT_SIM_LAYOUT")
        if env and Path(env).is_file() and json.loads(Path(env).read_text()).get("robot_model") == self.name:
            return Path(env)
        return LAYOUT

    @property
    def twin(self) -> TwinSpec:
        from . import scene
        layout = self.layout_path
        return TwinSpec(kind="generic", layout=layout, build_scene=lambda out: scene.build(out, layout),
                        rest_pose=tuple(json.loads(layout.read_text())["rest_pose"]))

    def make_ik(self, model):
        from generaloid.twin.ik import GenericIK
        return GenericIK(model, n_arm=len(self.arm_joints), rot_tol=IK_ROT_TOL, rot_weight=IK_ROT_WEIGHT)

    @cached_property
    def _ik(self):
        import mujoco
        return self.make_ik(mujoco.MjModel.from_xml_path(str(self.twin.build_scene(GENERATED / self.name))))

    def kinematics(self):
        return self._ik

    def reach_probe(self):
        return (0.0, 0.19, 0.06)                 # 5 cm above the cube of layout.json

    def mount(self, mode: str, station) -> Mount:
        os.environ["ROBOT_PROFILE"] = self.profile_name          # real_arm_mcp + learning.skillcfg read it
        ensure_sim_on_path()
        if mode == "sim":
            from generaloid.twin.generic import mount_generic
            twin, server = mount_generic(self, GENERATED / self.name)

            def close():
                twin.close()
            return Mount(self.name, mode, server, robot=twin, close=close)
        from generaloid.core.legacy import real_tools
        r = real_tools(self.profile_name)
        station.install()
        return Mount(self.name, mode, r.server, robot=r)


CONNECTOR = So101Connector()
