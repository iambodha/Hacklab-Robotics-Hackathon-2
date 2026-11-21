"""Shared types: observations, the adapter interface, and the operation registry.

A RobotAdapter owns ONE robot. All motion is expressed in normalized joint
space [0,1] mapped through each joint's calibrated range — the adapter is
responsible for translating to native units (servo ticks, radians, ...).

A skill is a coroutine run inside an Operation: it gets a handle to report
progress and to check for cancellation, runs under the robot's ownership
lock (one motion source at a time), and returns a structured result.
"""
from __future__ import annotations

import asyncio
import itertools
import time
import traceback
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# Observation types

@dataclass
class JointState:
    name: str
    pos_norm: float                 # 0..1 through the calibrated range
    raw: int | None                 # native units (servo ticks), if known
    range_min: int
    range_max: int
    role: str = ""

    @property
    def calibrated(self) -> bool:
        return self.range_max > self.range_min


@dataclass
class Observation:
    joints: list[JointState]
    images: list[bytes] = field(default_factory=list)   # jpeg payloads
    image_source: str = "none"                          # "normvla" | "usbvideo" | "none"
    stamp: float = field(default_factory=time.time)

    def joint_map(self) -> dict[str, float]:
        return {j.name: j.pos_norm for j in self.joints}

    def ranges(self) -> list[tuple[int, int]]:
        return [(j.range_min, j.range_max) for j in self.joints]

    def uncalibrated(self) -> list[str]:
        return [j.name for j in self.joints if not j.calibrated]


class RobotAdapter:
    """One robot behind the MCP surface. Implementations: StationAdapter
    (NormaCore/ElRobot over normfs), MockAdapter (software). A future
    LeRobotAdapter implements the same interface for SO-100/101."""

    robot_id: str = "robot"
    backend: str = "abstract"
    joint_names: list[str] = []
    gripper_joint: str | None = None
    units: str = "normalized"       # joint space exposed to tools/skills
    motion_lock: asyncio.Lock = field(default_factory=asyncio.Lock)  # type: ignore

    def __init__(self):
        self.motion_lock = asyncio.Lock()

    def capabilities(self) -> dict:
        return {
            "robot_id": self.robot_id,
            "backend": self.backend,
            "joints": self.joint_names,
            "gripper_joint": self.gripper_joint,
            "units": self.units,
            "cameras": self.has_cameras(),
            "skills": [],            # filled in by the server from the registry
        }

    def has_cameras(self) -> bool:
        return True

    async def joint_state(self) -> Observation:
        """Camera-independent joint read. Must never block on a camera feed."""
        raise NotImplementedError

    async def observe(self, images: bool = True) -> Observation:
        """Joint state plus fresh camera frames when available."""
        raise NotImplementedError

    async def move_norm(self, targets: dict[str, float], max_step: float) -> Observation:
        """Move named joints to absolute normalized targets.

        Implementations MUST clamp targets to [0,1] AND to within `max_step`
        of the current position per joint, hold unspecified joints, and refuse
        to move uncalibrated joints. Returns the resulting joint state."""
        raise NotImplementedError

    async def hold(self) -> None:
        """Stop: command every joint to stay where it is (no torque-off —
        cutting torque on a loaded arm makes it collapse)."""
        raise NotImplementedError


# --------------------------------------------------------------------------
# Operations: a running (or finished) skill invocation

class CancelledError(Exception):
    """Raised inside a skill when its operation was cancelled."""


class Operation:
    _ids = itertools.count(1)

    def __init__(self, robot_id: str, skill: str, arguments: dict):
        self.id = f"op{next(Operation._ids)}"
        self.robot_id = robot_id
        self.skill = skill
        self.arguments = dict(arguments)
        self.status = "running"        # running | done | failed | cancelled
        self.progress = "starting"
        self.result: dict | None = None
        self.error: str | None = None
        self.cancel = asyncio.Event()
        self.task: asyncio.Task | None = None
        self.started_at = time.time()
        self.finished_at: float | None = None
        self.trace: list[str] = []

    def report(self, msg: str) -> None:
        self.progress = msg
        self.trace.append(msg)

    def check_cancel(self) -> None:
        if self.cancel.is_set():
            raise CancelledError()

    def as_dict(self) -> dict:
        return {
            "operation_id": self.id,
            "robot_id": self.robot_id,
            "skill": self.skill,
            "arguments": self.arguments,
            "status": self.status,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "elapsed_s": round((self.finished_at or time.time()) - self.started_at, 1),
            "trace": self.trace[-20:],
        }


_OPS: dict[str, Operation] = {}


def get_operation(op_id: str) -> Operation | None:
    return _OPS.get(op_id)


def running_operation(robot_id: str) -> Operation | None:
    for op in _OPS.values():
        if op.robot_id == robot_id and op.status == "running":
            return op
    return None


def start_operation(robot: RobotAdapter, skill_name: str, fn, arguments: dict) -> Operation:
    """Run skill coroutine `fn(adapter, op, **arguments)` under the robot's
    ownership lock. Exactly one operation (or direct motion tool) may command
    a robot at a time — enforced via robot.motion_lock."""
    op = Operation(robot.robot_id, skill_name, arguments)

    async def runner():
        async with robot.motion_lock:
            try:
                op.result = await fn(robot, op, **arguments)
                if op.status == "running":
                    op.status = "done"
            except CancelledError:
                op.status = "cancelled"
            except Exception as e:                    # noqa: BLE001 — report, don't crash the server
                op.status = "failed"
                op.error = f"{type(e).__name__}: {e}"
                op.trace.append(traceback.format_exc(limit=3))
            finally:
                op.finished_at = time.time()

    op.task = asyncio.create_task(runner())
    _OPS[op.id] = op
    return op
