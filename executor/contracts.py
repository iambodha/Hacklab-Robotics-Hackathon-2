"""Shared records (frozen by the integrator in prompts/interfaces.md). The executor workstream owns this file.

World frame, units and calibration: prompts/scene-contract.md (metres, radians, z up, origin at the follower
base on the table, +x forward at j1 = 0). Optional fields may be added; meanings, units and frames may not change.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum


class SetupId(str):
    """Hash of: scene_calib.json version, robot profile + calibrated ranges, tool attached, table layout name."""

    @classmethod
    def compute(cls, scene_version: str | None, profile: str, ranges: dict, tool: str = "none",
                layout: str = "default") -> "SetupId":
        key = json.dumps({"scene": scene_version, "profile": profile, "ranges": {str(k): list(v) for k, v in sorted(ranges.items())},
                          "tool": tool, "layout": layout}, sort_keys=True)
        return cls("setup-" + hashlib.sha1(key.encode()).hexdigest()[:12])


@dataclass
class FrameRef:
    queue: str                     # Station queue (usbvideo/<hash>) or a sim/replay name
    seq: int                       # frame sequence number from the source (Station frame index), not a byte hash
    t_capture: float               # monotonic seconds (executor clock) when the frame was captured / arrived
    path_or_recording_ref: str     # "/api/frame/<obs_id>/<role>.jpg" or "recording:<id>@<t>"
    age_s: float = 0.0             # at observation time
    stale: bool = False            # True: seq did not advance / too old; consumers must treat its content as unknown
    joints_norm: list[float] | None = None   # joint positions (calibrated norm) at capture time


@dataclass
class JointState:
    names: list[str]
    raw: list[int]
    rad: list[float]               # URDF radians by the profile mapping (gripper last)
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
    freshness_s: float             # age of the oldest data used at the moment of publishing
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


@dataclass
class ValidationResult:
    ok: bool
    reasons: list[str]
    checks: dict                   # check -> {"result": pass|fail|skipped, ...numbers}


class Status(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    PARTIAL = "PARTIAL"
    BLOCKED = "BLOCKED"
    FAULT = "FAULT"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

    @property
    def final(self) -> bool:
        return self not in (Status.QUEUED, Status.RUNNING)


@dataclass
class ExecutionResult:
    execution_id: str
    proposal_id: str
    status: Status
    achieved: dict                 # measured end state (joints, tcp, gripper), NOT the command
    residuals: dict                # target - achieved per quantity, with units
    termination: str               # reached, contact abort, stall, timeout, cancel, fault text
    phases: list[dict]             # name, start/end, status, notes
    recording_id: str | None
    markers: list[str]
    validation: ValidationResult | None = None
    skill: str = ""


@dataclass
class Outcome:                     # independent evaluator, never the executor's own claim
    execution_id: str
    metric: str
    value: float
    unit: str
    success: bool | None           # None = unknown (occluded, stale, unmeasurable)
    evidence: list[str]


def to_dict(obj) -> dict:
    """JSON-ready dict of any record above (enums as their value)."""
    def conv(x):
        if isinstance(x, Enum):
            return x.value
        if dataclasses.is_dataclass(x):
            return {f.name: conv(getattr(x, f.name)) for f in dataclasses.fields(x)}
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        if hasattr(x, "tolist"):
            return x.tolist()
        if isinstance(x, float):
            return round(x, 6)
        return x
    return conv(obj)


def proposal_from_dict(d: dict, now: float) -> ActionProposal:
    """Build a proposal from API JSON. `ttl_s` may replace `expires_at` (advisors cannot see the executor clock)."""
    exp = d.get("expires_at")
    if exp is None:
        exp = now + float(d.get("ttl_s", 10.0))
    return ActionProposal(proposal_id=d.get("proposal_id") or ActionProposal.new_id(), source=d.get("source", "operator"),
                          based_on_obs=d.get("based_on_obs", ""), setup_id=d.get("setup_id", ""), skill=d["skill"],
                          skill_version=str(d.get("skill_version", "")), params=dict(d.get("params") or {}),
                          preconditions=list(d.get("preconditions") or []), expires_at=float(exp),
                          rationale=d.get("rationale", ""))
