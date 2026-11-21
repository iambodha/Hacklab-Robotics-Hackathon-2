"""Minimal mirror of the shared records in prompts/interfaces.md (owned by executor/contracts.py).

Field names, meanings and units match the frozen interface exactly, so the integrator can replace this module
with `from executor.contracts import *`. World frame: metres, radians, z up, origin at the follower base on the
table (prompts/scene-contract.md).
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
    queue: str
    seq: int
    t_capture: float
    path_or_recording_ref: str
    age_s: float = 0.0
    stale: bool = False
    joints_norm: list[float] | None = None


@dataclass
class ObjectEstimate:
    name: str
    xyz_world: list[float]
    size: list[float] | None = None
    yaw: float | None = None
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
    joints: dict | None                # executor: JointState; skills only read tcp_world from it
    objects: list[ObjectEstimate]
    freshness_s: float
    recording_id: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class ActionProposal:
    proposal_id: str
    source: str
    based_on_obs: str
    setup_id: str
    skill: str
    skill_version: str
    params: dict
    preconditions: list[str]
    expires_at: float
    rationale: str

    @staticmethod
    def new_id() -> str:
        return "prop-" + uuid.uuid4().hex[:10]


@dataclass
class ValidationResult:
    ok: bool
    reasons: list[str]
    checks: dict


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


def to_dict(obj):
    """JSON-ready dict of any record above (enums as their value)."""
    if isinstance(obj, Enum):
        return obj.value
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, float):
        return round(obj, 6)
    return obj
