"""ElRobot connector: the follower arm of 21-22 Nov (7-DoF arm + jaws, 8 ST3215 servos on one Station bus).

Everything that works tonight, wrapped in place (nothing copied):
  sim   sim/sim_arm_mcp.py: the calibrated MuJoCo twin (venue / home layouts, screw-cap bottle, stir cup), with the
        twin's session recorder and the learning tools (episodes, twist_cap, skill config)
  real  sim/real_arm_mcp.py: the real arm through NormaCore Station (arm_mcp/station.py for the motor bus, the
        camera relay for frames), profile sim/robots/elrobot.json; pick/drop/survey/stir/wipe/grab and the bottle-cap
        twist that opened a fully closed cap on 22 Nov
Both servers register their tools at import; the Generaloid server adopts them unchanged.
"""
from __future__ import annotations

import importlib
import os
import time
from functools import cached_property

from generaloid.connectors.base import MOTION_SKILLS, Mount, TwinSpec
from generaloid.connectors.profile import ProfileConnector
from generaloid.core.paths import SIM, ensure_sim_on_path

GRIPPER_TORQUE_NM = 0.4          # sim/build_scene.py GRIPPER_TORQUE: jaw servo through the 11.5 mm/rad linkage


class ElRobotConnector(ProfileConnector):
    name = "elrobot"
    title = "ElRobot follower (7-DoF arm + jaws, 8x ST3215)"
    profile_name = "elrobot"
    gripper_open_at_upper_limit = False          # j8: 0 rad = jaws open (build_scene.py)
    gripper_torque_nm = GRIPPER_TORQUE_NM
    hardware_verified = True                     # 21-22 Nov: picks, stir, wipe, bottle cap (logs/learning/bottle/)

    SIM_SKILLS = MOTION_SKILLS | {"stir", "twist_cap", "episodes", "skill_config", "operator_reset_scene"}
    REAL_SKILLS = MOTION_SKILLS | {"pick", "drop", "survey", "stir", "wipe", "grab", "touch_table", "twist_cap",
                                   "episodes", "skill_config"}

    @property
    def skills(self) -> frozenset[str]:
        return frozenset(self.SIM_SKILLS | self.REAL_SKILLS)

    def skills_for(self, mode: str) -> frozenset[str]:
        return frozenset(self.SIM_SKILLS if mode == "sim" else self.REAL_SKILLS)

    @property
    def twin(self) -> TwinSpec:
        ensure_sim_on_path()
        import build_scene
        return TwinSpec(kind="elrobot-twin", layout=build_scene.layout_path())

    @cached_property
    def _ik(self):
        """build_scene.ArmIK on the twin's own model (sim/generated/workspace.xml): the least-lean jaws-down IK that the
        twin and the real arm share (skillcfg.lean_search)."""
        ensure_sim_on_path()
        import build_scene
        import mujoco
        xml = SIM / self.profile.kinematics.xml
        if not xml.exists():
            build_scene.build(report=False)
        return build_scene.ArmIK(mujoco.MjModel.from_xml_path(str(xml)))

    def kinematics(self):
        return self._ik

    def reach_probe(self):
        return (0.025, 0.16, 0.06)               # 5 cm above the venue cube (sim/layout.json black_cube)

    def mount(self, mode: str, station) -> Mount:
        os.environ["ROBOT_PROFILE"] = self.profile_name          # real_arm_mcp + learning.skillcfg read it
        ensure_sim_on_path()
        if mode == "sim":
            s = importlib.import_module("sim_arm_mcp")

            def close():                                         # what sim_arm_mcp's __main__ does on exit
                s.arm.running = False
                if s.recorder:
                    s.recorder.close()
                    import make_replay
                    make_replay.build(s.recorder.dir)
                if s.arm.viewer is not None:
                    s.arm.viewer.close()
                    time.sleep(1)
            return Mount(self.name, mode, s.server, robot=s, close=close)
        r = importlib.import_module("real_arm_mcp")              # no Station connection at import (arm_mcp is lazy)
        station.install()                                        # every bus/camera connect: this Station, guarded
        return Mount(self.name, mode, r.server, robot=r)


CONNECTOR = ElRobotConnector()
