"""The connector interface: what a robot declares to plug into Generaloid.

A connector is one robot model. It declares
  joints        motor ids and names on the Station bus, arm joints first, then the gripper
  gripper       which joint, which direction is open, jaw width, torque limit
  cameras       roles (wrist, overview), their Station queue / relay id on hardware, their camera in the twin
  safety        the envelope every motion respects (TCP height and radius, max joint step, tracking, load)
  kinematics()  forward kinematics and least-lean jaws-down IK on the robot's MuJoCo model
  twin          the twin model (scene builder and layout)
  skills        which skills it supports (motion primitives, pick, stir, twist_cap, episodes, skill_config, ...)
and mounts itself in a mode ('sim' = twin, 'real' = hardware through the NormaCore Station) as an MCP tool surface.

The server (generaloid.core.server) knows only this interface. A new robot = a new package under
generaloid/connectors/<name>/ exposing `CONNECTOR`, nothing else changes (docs/generaloid/WRITING_A_CONNECTOR.md).

Add-on hook: after mounting, the server calls generaloid.comms.register(server, connector) if that module exists
(generaloid.core.hooks.CommsRegister). An add-on may rely on everything public here: name, title, modes, joints,
gripper, cameras, safety, skills, frame_text, describe().
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Protocol

from generaloid.core.hooks import CommsRegister  # noqa: F401  (re-exported: the add-on hook signature)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from generaloid.core.station import Station

# Skill vocabulary shared by all connectors (a connector lists the ones it supports).
MOTION_SKILLS = frozenset({"look", "joint_state", "set_torque", "move_to", "gripper", "tcp_pose", "move_tcp",
                           "pixel_to_table", "camera_info"})
KNOWN_SKILLS = MOTION_SKILLS | frozenset({"pick", "drop", "survey", "stir", "wipe", "grab", "touch_table", "twist_cap",
                                          "episodes", "skill_config", "operator_reset_scene"})


@dataclass(frozen=True)
class Joint:
    name: str                 # the name the MCP tools use (move_to targets, the joint table)
    motor_id: int             # servo id on the Station motor bus
    role: str                 # 'arm' | 'gripper'


@dataclass(frozen=True)
class Gripper:
    joint: str                # Joint.name of the jaws; tools always use normalized 1.0 = fully open
    open_at_upper_limit: bool  # model: the jaw joint's upper URDF limit is open (False for ElRobot j8: 0 rad = open)
    max_width_m: float        # widest object the jaws take
    torque_nm: float          # jaw servo torque in the twin (stalls on an object = a grip)


@dataclass(frozen=True)
class Camera:
    role: str                 # 'wrist' | 'overview' | ...
    twin_camera: str | None   # camera name in the twin's MuJoCo scene
    station_id: str | None    # Station usbvideo queue hash / camera relay prefix on hardware (None = set per machine)
    width: int                # Station frame size (resize_target 224)
    height: int


@dataclass(frozen=True)
class SafetyEnvelope:
    min_tcp_z: float          # m, lowest TCP above the table top
    max_tcp_z: float          # m
    max_radius: float         # m, from the base axis
    max_joint_delta: float    # largest normalized step per move_to call (ROBOT_MAX_DELTA)
    track_tol_ticks: int      # |goal - present| that aborts a move
    load_abort: int           # servo present load (0..1000) that counts as contact / aborts
    travel_height: float      # m, TCP height for sideways travel before descending


@dataclass(frozen=True)
class TwinSpec:
    kind: str                 # 'elrobot-twin' (sim/sim_arm_mcp.py) | 'generic' (generaloid.twin.generic)
    layout: Path              # table layout JSON (objects, bins, cameras)
    build_scene: Callable[[Path], Path] | None = None   # out_dir -> workspace.xml (generic twins)
    rest_pose: tuple[float, ...] = ()


class Kinematics(Protocol):
    """Kinematics on the robot's MuJoCo model. q = [arm joints..., gripper] in radians."""

    low: Any
    high: Any

    def fk(self, q) -> tuple[Any, Any]: ...                                     # (tcp xyz, tcp R); R[:,1] = approach

    def solve_least_lean(self, xyz, q0, yaw_deg: float = 0.0) -> tuple[Any, Any]: ...   # (q, lean_deg) or (None, None)


@dataclass
class Mount:
    """A mounted robot: its MCP tools plus how to shut it down."""
    connector: str
    mode: str
    tools: "FastMCP"                                   # tools registered here are adopted by the Generaloid server
    robot: Any = None                                  # the twin or the real-arm module (for in-process callers)
    close: Callable[[], None] = lambda: None
    addons: dict = field(default_factory=dict)         # add-on hook status, filled by the server


class Connector(ABC):
    name: ClassVar[str]
    title: ClassVar[str]
    modes: ClassVar[tuple[str, ...]] = ("sim", "real")
    hardware_verified: ClassVar[bool] = False          # True only after it ran on the real robot

    # --- declarations -------------------------------------------------------------------------------------
    @property
    @abstractmethod
    def joints(self) -> list[Joint]: ...

    @property
    @abstractmethod
    def gripper(self) -> Gripper: ...

    @property
    @abstractmethod
    def cameras(self) -> list[Camera]: ...

    @property
    @abstractmethod
    def safety(self) -> SafetyEnvelope: ...

    @property
    @abstractmethod
    def twin(self) -> TwinSpec: ...

    @property
    @abstractmethod
    def skills(self) -> frozenset[str]: ...

    @property
    @abstractmethod
    def frame_text(self) -> str: ...

    @abstractmethod
    def kinematics(self) -> Kinematics: ...

    def make_ik(self, model) -> Kinematics:
        """IK on a MuJoCo model of this arm (the generic twin builds one on its own model). Override for tuning."""
        from generaloid.twin.ik import GenericIK
        return GenericIK(model, n_arm=len(self.arm_joints))

    def reach_probe(self) -> tuple[float, float, float]:
        """A TCP point (base frame, m) the arm reaches jaws-down: the contract test checks fk(ik(p)) == p there."""
        return (0.0, 0.2, 0.08)

    # --- mounting -------------------------------------------------------------------------------------------
    @abstractmethod
    def mount(self, mode: str, station: "Station") -> Mount:
        """Bring the robot up in `mode` and return its MCP tools. 'real' must reach hardware only through `station`."""

    # --- derived ----------------------------------------------------------------------------------------------
    def skills_for(self, mode: str) -> frozenset[str]:
        """Skills available in one mode (default: all of them in every mode)."""
        return self.skills

    @property
    def arm_joints(self) -> list[Joint]:
        return [j for j in self.joints if j.role == "arm"]

    def describe(self) -> dict:
        return {"name": self.name, "title": self.title, "modes": list(self.modes),
                "hardware_verified": self.hardware_verified,
                "joints": [asdict(j) for j in self.joints], "gripper": asdict(self.gripper),
                "cameras": [asdict(c) for c in self.cameras], "safety": asdict(self.safety),
                "twin": {"kind": self.twin.kind, "layout": str(self.twin.layout)},
                "skills": sorted(self.skills), "frame": self.frame_text}
