"""Local mirror of the frozen records in prompts/interfaces.md (owner: executor/contracts.py).

MERGE NOTE for the integrator: this file exists only so `models/` runs and tests on its own. At merge time,
replace the body with `from executor.contracts import *` (field names, order, units and meanings here are
copied from executor/contracts.py at a35d4f2 and must stay identical). Nothing here adds fields.
"""
from __future__ import annotations

import dataclasses
import uuid
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class FrameRef:
    queue: str                     # Station queue (usbvideo/<hash>) or a sim/replay name
    seq: int                       # frame sequence number from the source (Station frame index)
    t_capture: float               # monotonic seconds (executor clock) when the frame was captured / arrived
    path_or_recording_ref: str     # "/api/frame/<obs_id>/<role>.jpg" or "recording:<id>@<t>"
    age_s: float = 0.0
    stale: bool = False            # True: consumers must treat its content as unknown
    joints_norm: list[float] | None = None


@dataclass
class JointState:
    names: list[str]
    raw: list[int]
    rad: list[float]
    norm: list[float]
    goal: list[int]
    load: list[int]
    current: list[int] | None
    temp: list[int]
    torque: list[int]
    status: list[int]
    t_mono: float
    tcp_world: list[float] | None = None
    gripper_open: float | None = None


@dataclass
class ObjectEstimate:
    name: str
    xyz_world: list[float]
    size: list[float] | None = None
    yaw: float | None = None       # radians about world z
    uncertainty_mm: float | None = None
    source: str = ""
    visible: bool = True
    t_seen: float | None = None


@dataclass
class Observation:
    obs_id: str
    t_mono: float
    t_wall: float
    setup_id: str
    frames: dict[str, FrameRef]
    joints: JointState
    objects: list[ObjectEstimate]
    freshness_s: float
    recording_id: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class ActionProposal:
    proposal_id: str
    source: str                    # "claude", "gemini-er", "policy:act", "operator", "script"
    based_on_obs: str
    setup_id: str
    skill: str
    skill_version: str
    params: dict                   # units/frame stated per param (e.g. xyz_world metres)
    preconditions: list[str]
    expires_at: float              # monotonic (executor clock); rejected after this
    rationale: str

    @staticmethod
    def new_id() -> str:
        return "prop-" + uuid.uuid4().hex[:10]


class Status(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    PARTIAL = "PARTIAL"
    BLOCKED = "BLOCKED"
    FAULT = "FAULT"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class ExecutionResult:
    execution_id: str
    proposal_id: str
    status: Status
    achieved: dict
    residuals: dict
    termination: str
    phases: list[dict]
    recording_id: str | None
    markers: list[str]


@dataclass
class Outcome:                     # independent evaluator, never the executor's own claim
    execution_id: str
    metric: str
    value: float
    unit: str
    success: bool | None           # None = unknown (occluded, stale, unmeasurable)
    evidence: list[str]


def to_dict(obj):
    """JSON-ready dict of any record above (enums as their value)."""
    if isinstance(obj, Enum):
        return obj.value
    if dataclasses.is_dataclass(obj):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    return obj
