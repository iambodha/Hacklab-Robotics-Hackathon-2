"""Validation before any motion. Every check is recorded with pass/fail/skipped and its numbers; any failure
means REJECTED and zero motion. The whole skill path is solved (skill.plan) here, from the measured joints."""
from __future__ import annotations

import time

from .contracts import ActionProposal, Observation, ValidationResult
from .skills import Plan, PlanError, Skill


_DEFAULT_TWIN = None


def validate_in_sim(proposal: ActionProposal, plan: Plan | None = None, ex=None, obs: Observation | None = None,
                    q0=None) -> dict:
    """Twin preflight (executor/twin.py) of the proposal's WHOLE planned path: collisions with table, markers, bin(s),
    camera stand; twin reachability. -> {"result": pass|fail|skipped, "why", "seconds", ...numbers}, < 2 s.
    With an Executor `ex` and no plan, the skill is planned first from the measured joints (nothing moves)."""
    global _DEFAULT_TWIN
    try:
        if ex is not None:
            obs = obs or ex.observation(proposal.based_on_obs)
            if plan is None:
                skill = ex.skills.get(proposal.skill)
                if skill is None:
                    return {"result": "skipped", "why": f"unknown skill {proposal.skill!r}"}
                q0 = ex.motion.q_now() if q0 is None else q0
                plan = skill.plan(ex.ctx(q0=q0, obs=obs), proposal.params)
            pf = ex.twin
        else:
            if _DEFAULT_TWIN is None:
                from .kinematics import Kinematics
                from .twin import TwinPreflight
                _DEFAULT_TWIN = TwinPreflight(Kinematics(), warm=False)
            pf = _DEFAULT_TWIN
    except PlanError as e:
        return {"result": "fail", "why": f"no plan: {e.reason}", "checks": e.checks}
    except ImportError as e:
        return {"result": "skipped", "why": f"MuJoCo / twin not available: {e}"}
    if pf is None:
        return {"result": "skipped", "why": "twin preflight disabled (EXECUTOR_TWIN_PREFLIGHT=0)"}
    return pf(proposal, plan, obs, q0)


def executor_version(v: str) -> bool:
    """A numeric skill_version ("1", "2.1") names the executor's skill version and must match it. Anything else is the
    proposer's own version tag (skills: "wipe-3f2a9c1d", models: "gemini-advisor/0.1"): recorded, not enforced."""
    return bool(v) and v.replace(".", "").isdigit()


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
    if executor_version(p.skill_version) and p.skill_version != skill.version:
        fail("skill", f"skill {p.skill} version {p.skill_version} != executor's {skill.version}", version=skill.version)
    else:
        ok("skill", version=skill.version, **({"proposer_version": p.skill_version} if p.skill_version and
                                              p.skill_version != skill.version else {}))

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

    plan, q0 = None, None
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
        q0 = ex.motion.q_now(st)
        plan = skill.plan(ex.ctx(q0=q0, obs=obs), p.params)
        checks.update({k: v for k, v in plan.checks.items()})
        checks.setdefault("reachability", {"result": "pass"})
        checks.setdefault("workspace", {"result": "pass"})
    except PlanError as e:
        checks.update(e.checks)
        checks.setdefault("reachability", {"result": "fail"})
        reasons.append(e.reason)
    except (KeyError, TypeError, ValueError) as e:
        fail("params", f"bad params for {p.skill}: {e}", params_doc=skill.params_doc)

    checks["twin_preflight"] = ex.preflight(p, plan, ex, obs, q0) if plan is not None else {"result": "skipped"}
    if checks["twin_preflight"].get("result") == "fail":
        reasons.append("twin preflight: " + checks["twin_preflight"].get("why", "collision"))

    left = p.expires_at - ex.now()       # planning takes time: the proposal may have expired meanwhile
    if left < 0 and checks["expiry"]["result"] == "pass":
        fail("expiry", f"expired during validation ({-left:.2f} s ago)", seconds_left=round(left, 3))
    return ValidationResult(not reasons, reasons, checks), plan if not reasons else None, time.monotonic() - t0
