"""Connector declarations read from a robot profile (sim/robots/<name>.json, sim/robot_profile.py).

The profiles already hold every robot-specific number of the real-arm server; a profile-backed connector turns them
into the Generaloid declarations so the numbers live in one place.
"""
from __future__ import annotations

import os
from functools import cached_property
from typing import ClassVar

from generaloid.core.paths import ensure_sim_on_path

from .base import Camera, Connector, Gripper, Joint, SafetyEnvelope

# Defaults of the existing servers (sim_arm_mcp.py / arm_mcp/server.py), overridable by the same env variables.
DEFAULT_MAX_DELTA = 0.35        # ROBOT_MAX_DELTA: largest normalized step per move_to call
DEFAULT_TRACK_TOL = 250         # ROBOT_TRACK_TOL: ticks of |goal - present| that abort a move


class ProfileConnector(Connector):
    profile_name: ClassVar[str]                  # name in sim/robots/ (or a path)
    gripper_open_at_upper_limit: ClassVar[bool]
    gripper_torque_nm: ClassVar[float]
    twin_cameras: ClassVar[dict[str, str]] = {"wrist": "wrist_cam", "overview": "overview_cam"}

    @cached_property
    def profile(self):
        ensure_sim_on_path()
        import robot_profile
        return robot_profile.load(self.profile_name)

    @cached_property
    def joints(self) -> list[Joint]:
        m = self.profile.motors
        names = {int(k): v for k, v in vars(m.names).items()}
        return [Joint(names[i], int(i), "arm") for i in m.arm] + [Joint(names[int(m.gripper)], int(m.gripper), "gripper")]

    @property
    def gripper(self) -> Gripper:
        return Gripper(joint=self.joints[-1].name, open_at_upper_limit=self.gripper_open_at_upper_limit,
                       max_width_m=float(self.profile.gripper.max_grip_width), torque_nm=self.gripper_torque_nm)

    @property
    def cameras(self) -> list[Camera]:
        c = self.profile.cameras
        return [Camera(role, self.twin_cameras.get(role), None if getattr(c, role) == "SET_ME" else getattr(c, role),
                       int(c.width), int(c.height)) for role in ("wrist", "overview")]

    @property
    def safety(self) -> SafetyEnvelope:
        w = self.profile.workspace
        return SafetyEnvelope(min_tcp_z=float(w.min_tcp_z), max_tcp_z=float(w.max_tcp_z), max_radius=float(w.max_radius),
                              max_joint_delta=float(os.environ.get("ROBOT_MAX_DELTA", DEFAULT_MAX_DELTA)),
                              track_tol_ticks=int(os.environ.get("ROBOT_TRACK_TOL", DEFAULT_TRACK_TOL)),
                              load_abort=int(self.profile.motion.load_abort), travel_height=float(w.travel_height))

    @property
    def frame_text(self) -> str:
        return self.profile.frame_text
