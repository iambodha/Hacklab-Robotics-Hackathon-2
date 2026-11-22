"""Skill model: phases + parameters + applicability + version + parent version.

A skill never talks to Station. Its phases are:
  plan      task + observation -> a concrete plan (pure, deterministic)
  validate  local checks on the plan before anything is proposed (the executor validates again, independently)
  execute   the plan as ActionProposals for the executor (one per motion unit)
  verify    an independent evaluator (skills/eval) turns before/after evidence into an Outcome
  recover   what to do after a PARTIAL/FAULT result or a failed Outcome (proposals or a note)

A SkillVersion is an immutable parameter set. `child()` makes a new version that names its parent, so every
change is traceable. Parameters are split into ADAPTABLE ones (bounded, may be changed by adaptation or the
improvement loop) and FIXED ones (safety/contact settings taken from the robot profile; never part of a version).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

from ._contracts import ActionProposal, Outcome, ValidationResult

PHASES = ("plan", "validate", "execute", "verify", "recover")


@dataclass(frozen=True)
class Param:
    name: str
    default: float
    lo: float
    hi: float
    unit: str
    about: str = ""
    integer: bool = False

    def clamp(self, v: float) -> float:
        v = min(max(float(v), self.lo), self.hi)
        return float(round(v)) if self.integer else v


@dataclass(frozen=True)
class SkillVersion:
    skill: str
    version: str
    parent: str | None
    params: dict
    created_by: str = "seed"            # seed | adaptation | loop:<agent> | human
    note: str = ""
    created: float = field(default_factory=time.time, compare=False)

    @staticmethod
    def make_id(skill: str, params: dict, parent: str | None) -> str:
        h = hashlib.sha1(json.dumps({"s": skill, "p": params, "parent": parent}, sort_keys=True).encode()).hexdigest()[:8]
        return f"{skill}-{h}"

    def child(self, changes: dict, created_by: str, note: str = "") -> "SkillVersion":
        params = {**self.params, **changes}
        return SkillVersion(self.skill, self.make_id(self.skill, params, self.version), self.version, params, created_by, note)

    def to_dict(self) -> dict:
        return {"skill": self.skill, "version": self.version, "parent": self.parent, "params": self.params,
                "created_by": self.created_by, "note": self.note, "created": self.created}


class Skill:
    """Base class. Subclasses define `name`, `spec` (adaptable params) and the phase methods."""
    name = ""
    spec: dict[str, Param] = {}

    def seed_version(self, **overrides) -> SkillVersion:
        params = {k: p.clamp(overrides.get(k, p.default)) for k, p in self.spec.items()}
        return SkillVersion(self.name, SkillVersion.make_id(self.name, params, None), None, params, "seed")

    def check_version(self, v: SkillVersion) -> list[str]:
        """Problems with a version: unknown params, missing params, values outside their bounds."""
        out = [f"unknown parameter {k}" for k in v.params if k not in self.spec]
        for k, p in self.spec.items():
            if k not in v.params:
                out.append(f"missing parameter {k}")
            elif not p.lo <= float(v.params[k]) <= p.hi:
                out.append(f"{k}={v.params[k]} outside [{p.lo}, {p.hi}] {p.unit}")
        return out

    # phases -------------------------------------------------------------------------------------
    def applicable(self, context: dict) -> list[str]:
        """Reasons the skill does NOT apply in this context (empty list = applicable)."""
        return []

    def plan(self, version: SkillVersion, task: dict):
        raise NotImplementedError

    def validate(self, plan) -> ValidationResult:
        raise NotImplementedError

    def execute(self, plan, *, based_on_obs: str, setup_id: str, now: float, source: str = "script",
                ttl_s: float = 30.0) -> list[ActionProposal]:
        raise NotImplementedError

    def verify(self, *args, **kw) -> Outcome:
        raise NotImplementedError

    def recover(self, plan, outcome: Outcome | None, results: list) -> dict:
        return {"action": "none", "why": "no recovery defined"}
