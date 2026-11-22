"""Evidence -> bounded proposals; never approves real hardware changes.

Rules operate on stored attempts so the exact same policy can be replayed offline.
Text evidence supplements motor telemetry because round one's short lift falsely
reported free; operator observations in the lesson are part of the record.
"""
from __future__ import annotations

from . import skillcfg as SC


def suggest(attempt: dict, cfg: dict | None = None) -> list[dict]:
    c = SC.active()["values"] if cfg is None else cfg
    p = attempt["begin"]["params"]
    end = attempt.get("end") or {}
    lesson = end.get("lesson", "").lower()
    phases = attempt.get("phases", [])
    outcome = end.get("outcome")
    evidence = [f"attempt {attempt['begin']['attempt']}: {outcome}; {end.get('lesson', '')}"]
    out = []

    def add(rule, changes, why):
        changes = {k: v for k, v in changes.items() if c.get(k) != v}
        if changes:
            _, errors = SC.validate(changes, c)
            if not errors:
                out.append(dict(rule=rule, changes=changes, reason=why, evidence=evidence))

    def param(name, value):
        lo, hi = c[f"params.bounds.{name}.lo"], c[f"params.bounds.{name}.hi"]
        return {f"params.default.{name}": max(lo, min(hi, value))}

    lifts = [x.get("measurements", {}) for x in phases if x["phase"] in ("check", "extract")]
    not_free = outcome in ("loosened", "no_progress") or any(x.get("resisted") for x in lifts)
    if not_free:
        add("more_strokes", param("strokes", int(p["strokes"]) + int(c["learn.strokes_step"])),
            "Cap remains attached after turning/checking: increase stroke count within bounds.")
        add("longer_lift", {"extract.lift_m": min(SC.registry()["extract.lift_m"]["hi"],
            max(c["learn.lift_clear_m"], c["extract.lift_m"] + c["learn.lift_step_m"]))},
            "Short lift did not prove extraction: increase the guarded lift, never the safety force limits.")
    if "bevel" in lesson or "top lip" in lesson:
        # Floor is immutable and object-dependent. Do not suggest the refused -11 mm of attempt 4.
        locate = next((x["measurements"] for x in phases if x["phase"] == "locate"), {})
        height = locate.get("cap_height")
        from .episodes import SAFETY
        from .twist import FINGERTIP_BELOW
        floor_offset = (c["twist.grasp_below_top_m"] - height - SAFETY["below_cap_edge_m"] + FINGERTIP_BELOW) * 1000 if height else p["approach_z_offset_mm"]
        deeper = max(floor_offset, p["approach_z_offset_mm"] - c["learn.z_step_mm"])
        add("deeper_grasp", param("approach_z_offset_mm", deeper), "Jaws rode onto the bevel: grip deeper, respecting the cap-edge floor.")
        add("top_approach", {"twist.approach": "top"}, "Bevel slip calls for a guarded top approach.")
        if p["lift_per_turn_mm"] > c["learn.lift_per_turn_mm"]:
            add("follow_thread_gently", param("lift_per_turn_mm", c["learn.lift_per_turn_mm"]), "Excess rise moved the jaws onto the bevel: follow the thread more gently.")
    if "overload" in lesson and "j8" in lesson:
        add("gripper_overload", param("grip_torque_limit", round(p["grip_torque_limit"] - c["learn.grip_step"], 3)), "Gripper overload: lower torque rather than raising a safety limit.")
    elif outcome == "slip" and "bevel" not in lesson:
        add("grip_slid", param("grip_torque_limit", round(p["grip_torque_limit"] + c["learn.grip_step"], 3)), "Measured slip: increase grip torque within the immutable safety envelope.")
    if outcome == "opened":
        from .capfree import free_signal
        records = [s for phase in phases for s in phase.get("measurements", {}).get("strokes", [])]
        signal = free_signal(records, c)
        if signal["free"]:
            ceiling = signal["confirmed_at"] + int(c["learn.strokes_margin"])
            if ceiling < p["strokes"]:
                add("trim_strokes", param("strokes", ceiling), "Opened with a confirmed free signal: trim excess strokes, preserving confirmation and configured margin.")
    if "timeout" in lesson:
        add("fit_the_call", {"twist.time_budget_s": SC.MCP_CLIENT_TIMEOUT_S - c["learn.reply_margin_s"]}, "A timed-out call must be split into resumable calls below the client timeout.")
    return out


def propose_after(store, n: int, author: str = "learning rule") -> list[dict]:
    from . import skillcheck
    out = []
    for idea in suggest(store.attempts()[n]):
        r = SC.propose(idea["changes"], idea["reason"], idea["evidence"], author, SC.target())
        if r["ok"]:
            pid = r["proposal"]["id"]
            proposal = skillcheck.run_checks(pid)
            if SC.target() == "twin" and SC.get("learn.auto_apply") == "on" and not SC.checks_failed(proposal):
                r["applied"] = SC.apply(pid, by="learning rule (twin)")
            r["proposal"] = SC.proposal(pid)
        out.append({"rule": idea["rule"], **r})
    return out


def proposals_text(rows: list[dict]) -> str:
    return "Learning rules: " + ("; ".join(
        f"{r['rule']}: proposal {r['proposal']['id']} ({r['proposal']['status']})" if r["ok"] else
        f"{r['rule']}: refused {r['errors']}" for r in rows) or "no supported change")
