"""The wipe skill: cover a rectangular region with boustrophedon strokes of a held tissue.

Adaptable parameters live in the skill version; contact height and force limits are FIXED (limits.py) and are
attached to every proposal so the executor applies the hardware-validated settings.
"""
from __future__ import annotations

from ..config import get as strategy

import json
from pathlib import Path

from ..model import Param, Skill, SkillVersion
from .._contracts import ActionProposal, Outcome, ValidationResult
from . import limits as _limits
from .planner import Region, WipePlan, check_reach, plan_coverage

TOOLS = ("tissue", "napkin", "cloth")
HERE = Path(__file__).resolve().parent
_BOUNDS = json.loads((HERE / "bounds.json").read_text())        # protected
_DEFAULTS = json.loads((HERE / "defaults.json").read_text())    # editable seed values


class WipeSkill(Skill):
    name = "wipe"
    spec = {k: Param(k, _DEFAULTS[k], v["lo"], v["hi"], v["unit"], v.get("about", ""), v.get("integer", False))
            for k, v in _BOUNDS.items() if not k.startswith("_")}

    def __init__(self, limits: dict | None = None, T_world_base=None):
        self.limits = limits or _limits.load()
        self.T_world_base = T_world_base

    def applicable(self, context: dict) -> list[str]:
        out = []
        if context.get("tool") not in TOOLS:
            out.append(f"needs a held {'/'.join(TOOLS)} (tool attached: {context.get('tool')!r})")
        if context.get("surface", "flat") != "flat":
            out.append("needs a flat surface at table height")
        if context.get("objects_in_region"):
            out.append(f"objects in the region would be swept along: {context['objects_in_region']}")
        return out

    def plan(self, version: SkillVersion, task: dict) -> WipePlan:
        r = task["region"]
        region = r if isinstance(r, Region) else Region(*r)
        p = version.params
        return plan_coverage(region, p["w_hat"], p["beta"], int(task["budget"]), self.limits, task.get("axis", "auto"),
                             p["overshoot"], int(p["passes"]))

    def validate(self, plan: WipePlan) -> ValidationResult:
        L = self.limits
        reach = check_reach(plan, L, self.T_world_base)
        lengths = [s.length for s in plan.strokes]
        bad_len = [i for i, l in enumerate(lengths) if not L["min_stroke_m"] - 1e-9 <= l <= L["max_stroke_m"] + 1e-9]
        checks = {
            "reach": {"result": "fail" if reach else "pass", "problems": reach},
            "stroke_length": {"result": "fail" if bad_len else "pass", "min_m": min(lengths), "max_m": max(lengths),
                              "limits_m": [L["min_stroke_m"], L["max_stroke_m"]]},
            "budget": {"result": "pass" if len(plan.strokes) <= plan.budget else "fail", "strokes": len(plan.strokes),
                       "budget": plan.budget},
            "passes": {"result": "pass" if plan.passes <= L["max_passes"] else "fail", "passes": plan.passes},
            "coverage": {"result": "pass" if not plan.capped else "warn", "lanes": plan.lanes, "lanes_needed": plan.lanes_needed},
        }
        reasons = [f"{k}: {v.get('problems') or v}" for k, v in checks.items() if v["result"] == "fail"]
        return ValidationResult(not reasons, reasons, checks)

    def execute(self, plan: WipePlan, *, based_on_obs: str, setup_id: str, now: float, source: str = "script",
                ttl_s: float | None = None, version: str = "", plan_id: str = "") -> list[ActionProposal]:
        """One proposal per stroke, then a retreat (park) so the overview camera sees the board for inspection."""
        ttl_s = strategy("wipe.proposal.ttl_s") if ttl_s is None else ttl_s
        props = []
        for s in plan.strokes:
            props.append(ActionProposal(
                ActionProposal.new_id(), source, based_on_obs, setup_id, "wipe", version,
                {"from_xy_world": [round(v, 4) for v in s.start], "to_xy_world": [round(v, 4) for v in s.end],
                 "passes": plan.passes, "units": "m, world frame", "stroke_index": s.index, "lane": s.lane,
                 "plan_id": plan_id, "limits": dict(self.limits)},
                ["holding a tissue", "stroke line free of objects", "setup_id matches"],
                now + ttl_s, f"wipe lane {s.lane + 1}/{plan.lanes} (seg {s.segment + 1}/{plan.segments}), "
                             f"spacing {plan.spacing * 1000:.0f} mm for w_hat {plan.w_hat * 1000:.0f} mm"))
        props.append(ActionProposal(ActionProposal.new_id(), source, based_on_obs, setup_id, "park", version,
                                    {"purpose": "inspection: arm out of the overview camera's view of the board",
                                     "plan_id": plan_id}, [], now + ttl_s, "retreat before the inspection image"))
        return props

    def verify(self, *args, **kw) -> Outcome:
        from ..eval.wipe_eval import evaluate
        return evaluate(*args, **kw).outcome

    def recover(self, plan, outcome: Outcome | None, results: list) -> dict:
        faults = [r for r in results if getattr(r, "status", None) is not None and r.status.value in ("FAULT", "PARTIAL", "BLOCKED")]
        if faults:
            return {"action": "stop", "why": f"stroke {faults[0].proposal_id} ended {faults[0].status.value}: "
                                              f"{faults[0].termination}; inspect before continuing"}
        if outcome is None or outcome.success is None:
            return {"action": "reinspect", "why": "outcome unknown (arm in view / stale image): park and take a new image"}
        if not outcome.success:
            return {"action": "adapt_and_rewipe_residual", "why": f"{outcome.metric}={outcome.value:.2f} below target"}
        return {"action": "none", "why": "clean"}
