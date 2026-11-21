"""Validation before any motion. Every check is recorded with pass/fail/skipped and its numbers; any failure
means REJECTED and zero motion. The whole skill path is solved (skill.plan) here, from the measured joints."""
from __future__ import annotations

import time

from .contracts import ActionProposal, Observation, ValidationResult
from .skills import Plan, PlanError, Skill


def validate_in_sim(proposal: ActionProposal, plan: Plan | None) -> dict:
    """Twin preflight hook: run the plan in the MuJoCo twin and report collisions. No-op until the twin
    branch lands (then: set Executor.preflight to a function with this signature)."""
    return {"result": "skipped", "why": "twin preflight not wired yet"}


def validate(ex, p: ActionProposal, busy: bool) -> tuple[ValidationResult, Plan | None, float]:
    """(result, plan, seconds spent). `ex` is the Executor (clock, observations, backend, skills, setup id)."""
    t0 = time.monotonic()
    checks: dict[str, dict] = {}
    reasons: list[str] = []

    def fail(name, reason, **nums):
        checks[name] = {"result": "fail", **nums}
        reasons.append(reason)

    def ok(name, **nums):
        checks[name] = {"result": "pass", **nums}

    skill: Skill | None = ex.skills.get(p.skill)
    if skill is None:
        fail("skill", f"unknown skill {p.skill!r}", known=sorted(ex.skills))
        return ValidationResult(False, reasons, checks), None, time.monotonic() - t0
    if p.skill_version and p.skill_version != skill.version:
        fail("skill", f"skill {p.skill} version {p.skill_version} != executor's {skill.version}", version=skill.version)
    else:
        ok("skill", version=skill.version)

    now = ex.now()
    left = p.expires_at - now
    if left >= 0:
        ok("expiry", seconds_left=round(left, 3))
    else:
        fail("expiry", f"expired {-left:.2f} s ago", seconds_left=round(left, 3))

    obs: Observation | None = ex.observation(p.based_on_obs)
    if obs is None:
        fail("observation", f"unknown observation {p.based_on_obs!r} (observe() first; only recent ones are kept)")
    else:
        need = [r for r in skill.cameras if r in ex.backend.cameras]
        ages = {"joints": now - obs.joints.t_mono}
        stale = []
        for r in need:
            f = obs.frames.get(r)
            if f is None or f.stale:
                stale.append(r)
            if f is not None:
                ages[r] = now - f.t_capture
        age = max(ages.values())
        nums = {"age_s": round(age, 3), "limit_s": skill.freshness_s, "ages_s": {k: round(v, 3) for k, v in ages.items()},
                "cameras": need, "stale_cameras": stale}
        if stale:
            fail("freshness", f"camera(s) {stale} stale in observation {obs.obs_id} (frame sequence did not advance)", **nums)
        elif age > skill.freshness_s:
            fail("freshness", f"observation {obs.obs_id} is {age:.2f} s old; {p.skill} needs <= {skill.freshness_s} s", **nums)
        else:
            ok("freshness", **nums)

    current = ex.setup_id()
    sids = {"proposal": p.setup_id, "observation": obs.setup_id if obs else None, "current": current}
    if p.setup_id != current or (obs is not None and obs.setup_id != current):
        fail("setup", "setup_id mismatch (calibration, ranges, tool or layout changed since the observation)", **sids)
    else:
        ok("setup", **sids)

    (fail("busy", "another execution is running (one at a time)") if busy else ok("busy"))

    if skill.moves:
        (ok("motion_enabled", backend=ex.backend.name) if ex.backend.can_move else
         fail("motion_enabled", f"motion disabled on backend {ex.backend.name} (arm-enabled toggle off)", backend=ex.backend.name))

    plan = None
    try:
        st = ex.backend.read_state().motors
        bad = [ex.kin.name_of[m] for m in ex.kin.ids if not st[m]["in_range"] or st[m]["torque"] == 0 or st[m]["status"]]
        (fail("joints", f"joints not ready (out of range, torque off or fault): {bad}", joints=bad) if bad else ok("joints"))
    except Exception as e:  # noqa: BLE001
        st = None
        fail("joints", f"cannot read joints: {e}")

    if reasons:   # do not spend IK time on a proposal that is already rejected
        checks.setdefault("reachability", {"result": "skipped"})
        checks.setdefault("workspace", {"result": "skipped"})
        checks["twin_preflight"] = {"result": "skipped"}
        return ValidationResult(False, reasons, checks), None, time.monotonic() - t0

    try:
        plan = skill.plan(ex.ctx(q0=ex.motion.q_now(st)), p.params)
        checks.update({k: v for k, v in plan.checks.items()})
        checks.setdefault("reachability", {"result": "pass"})
        checks.setdefault("workspace", {"result": "pass"})
    except PlanError as e:
        checks.update(e.checks)
        checks.setdefault("reachability", {"result": "fail"})
        reasons.append(e.reason)
    except (KeyError, TypeError, ValueError) as e:
        fail("params", f"bad params for {p.skill}: {e}", params_doc=skill.params_doc)

    checks["twin_preflight"] = ex.preflight(p, plan) if plan is not None else {"result": "skipped"}
    if checks["twin_preflight"].get("result") == "fail":
        reasons.append("twin preflight: " + checks["twin_preflight"].get("why", "collision"))

    left = p.expires_at - ex.now()       # planning takes time: the proposal may have expired meanwhile
    if left < 0 and checks["expiry"]["result"] == "pass":
        fail("expiry", f"expired during validation ({-left:.2f} s ago)", seconds_left=round(left, 3))
    return ValidationResult(not reasons, reasons, checks), plan if not reasons else None, time.monotonic() - t0
