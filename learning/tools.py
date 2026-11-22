"""The learning MCP tool bodies, shared by sim/real_arm_mcp.py and sim/sim_arm_mcp.py (same names, same texts).

Each returns (text, images) with images = {"wrist": jpeg, "overview": jpeg}; the servers turn that into MCP content.
"""
from __future__ import annotations

import asyncio
import json
import os

from . import capfree
from . import episodes as E
from . import skillcfg as SC
from .twist import Twister

TOOL_DOCS = {
    "episode_begin": """Start attempt N of the bottle-cap learning run. `params`: the learnable parameters you set for
this attempt (missing ones are inherited from the previous attempt; nothing is clamped: out-of-range values are
REFUSED with the reason). Change at most 3 against the previous attempt. `hypothesis`: what you expect the change to
do and why. Saves the before images, starts a recorder session. Returns the attempt number and the full parameters.
Bounds (the code defaults; skill_config() shows the bounds in force, params.bounds.*):
""" + E.bounds_text(),
    "episode_history": """All previous attempts as compact text: parameters, what changed, hypothesis, every phase with
its measurements (rotation achieved, j7 load, jaw slip, z change), judgements, outcome, lesson and the planned next
change, plus the trend. Read it before choosing the next attempt's parameters.""",
    "twist_cap": """The whole unscrewing skill for the OPEN attempt, with its parameters: jaws open, hover above
(x, y), turn to yaw_start_deg, guarded descent to the grasp height (cap_top_z - 10 mm + approach_z_offset_mm; never
below the cap's lower edge - 5 mm), torque-limited close (grip_close, grip_torque_limit), then strokes x [turn the
gripper counter-clockwise (seen from above) about the vertical cap axis by twist_deg_per_stroke at twist_speed while
rising lift_per_turn_mm per turn, watching the wrist-roll load (stops at j7_load_abort) and the jaws (slip); release;
turn back regrasp_backoff_deg; regrasp]. It ends early, KEEPING THE GRIP, when the j7 load says the thread let go
(skill config twist.stop_on_free) or before the call's time budget (twist.time_budget_s, under the client timeout):
then call twist_cap again with the same cap and it RESUMES (no re-approach) up to the attempt's `strokes`. (x, y): cap
centre from pixel_to_table on a CLOSE wrist view at height cap_top_z; cap_height: the cap's height in m. On any abort
it opens the jaws and lifts straight up. Returns per stroke: commanded vs achieved rotation, peak/mean j7 load, j8
change (slip), z change, why it stopped; then why the call ended, the j7 free signal and the strokes needed.""",
    "check_cap": """Short check (skill config check.lift_m, 15 mm): only with the cap gripped it lifts straight up: a
cap still on the thread holds the gripper back or slips out of the jaws; a free cap comes up in the jaws. CAUTION: on
the real arm 15 mm said "free" while the jaws had only slid up a cap still on the thread (attempts 1, 2, 7): use
extract_cap() to decide. Saves check images and logs a judgement. If the cap is free it stays in the jaws; otherwise
the jaws open and the gripper lifts away.""",
    "extract_cap": """Take the cap off: with the cap gripped (after twist_cap), lift straight up extract.lift_m (learned:
6 cm, as the operator did by hand in attempts 9-12), in small guarded steps. A free cap comes up in the jaws, clear of
the bottle and obvious on camera: it stays held up there. A cap still on the thread holds the gripper back: the lift
STOPS at once (it never yanks), the jaws open and the gripper lifts away; a cap the jaws only held by its top slides
out. The verdict combines the lift with the j7 free signal of the strokes. Saves extract images, logs a judgement.""",
    "episode_end": """Close the open attempt. outcome: one of opened|loosened|no_progress|slip|misaligned|aborted|fault.
lesson: one or two honest sentences on what this attempt showed (cite the measurements). next_change: the parameter
change you plan next, e.g. {"grip_torque_limit": 0.5}. Saves the after images, writes the attempt's report.md and
SUMMARY.md, stops the recording. Then the learning rules (learning/rules.py) read the attempt's evidence and record
their proposals for the next change (skill config learn.auto_propose); on the real arm they wait for a human.""",
    "skill_config": """The skill strategy settings in force (one level above the per-attempt params): the IK lean policy
(ik.leans, ik.search, ik.restarts), how twist_cap approaches the cap (twist.approach top|side|auto, twist.side_lean_deg,
twist.hovers_m, twist.max_lean_deg, twist.grasp_below_top_m, twist.release_open) and the bounds of the episode params
(params.bounds.<name>.lo/hi), each with its envelope; the safety limits (read-only); the config version (every
attempt records the version it ran with); pending proposals with their check results; recent versions.""",
    "propose_skill_change": """Request a change of skill strategy settings (see skill_config()) when the evidence
shows the strategy, not the attempt's params, is the problem. Examples: the gripper leans > 45 deg at the cap and
attempts slip on the cap's bevel -> {"twist.approach": "top"}; the operator needs more strokes than the bound allows ->
{"params.bounds.strokes.hi": 24}; tcp_pose/twist_cap notes say "jaws-down unreachable ... leans 60 deg" while smaller
leans are known to reach -> an ik.leans / ik.search change. `changes`: {key: new value}. `reason`: what the evidence
shows and what the change should do. `evidence`: references, e.g. ["attempt 7 approach lean_deg=60 mode=side",
"attempt 7 lesson: jaws slid onto the bevel"]. Validated against each key's envelope (safety keys never change), then
checked offline: IK keys and approach keys probe the last real cap poses (lean and approach mode before vs after);
twin_check=true also starts a short twin run before vs after (minutes, in the background; results appear in
skill_config()). <<APPLY>>""",
}

APPLY_TEXT = {
    "twin": "On the twin the change is APPLIED at once when no check fails: the next attempt uses the new version.",
    "real": "On the real arm nothing changes until a HUMAN approves it (python -m learning.skillcfg approve <id> --by "
            "<name>): tell the person at the table the proposal id; keep running attempts with the current version "
            "meanwhile. Never run the approve command yourself.",
}


def skill_doc(name: str, target: str) -> str:
    return TOOL_DOCS[name].replace("<<APPLY>>", APPLY_TEXT[target])


def store() -> E.EpisodeStore:
    return E.EpisodeStore()


def _open(st: E.EpisodeStore):
    n = st.open_attempt()
    if n is None:
        return None, None
    return n, st.attempts()[n]["begin"]["params"]


async def episode_begin(st: E.EpisodeStore, rig, params: dict, hypothesis: str, based_on=None):
    if isinstance(params, str):
        params = json.loads(params)
    images = await rig.frames()
    r = st.begin(params, hypothesis, based_on, images)
    if not r["ok"]:
        return "REFUSED, nothing started:\n- " + "\n- ".join(r["errors"]), {}
    rec = f"recording {r['recording_id']}" if r["recording_id"] else f"no recording ({r['note']})"
    return (f"attempt {r['attempt']} started ({rec}). Changed: {', '.join(r['changed']) or 'nothing'}.\n"
            f"params: {json.dumps(r['params'])}\nNext: locate the cap (look, pixel_to_table on a close wrist view), "
            f"then twist_cap(x, y, cap_top_z)."), images


async def episode_history(st: E.EpisodeStore, last_n: int = 10):
    return st.history_text(int(last_n)), {}


def attempt_strokes(st: E.EpisodeStore, n: int) -> list[dict]:
    """Every stroke the open attempt made so far (all its twist_cap calls)."""
    return [x for p in st.attempts()[n]["phases"] if p["phase"] == "twist"
            for x in (p.get("measurements") or {}).get("strokes", [])]


async def twist_cap(st: E.EpisodeStore, rig, x: float, y: float, cap_top_z: float, cap_height: float = 0.02):
    n, params = _open(st)
    if n is None:
        return "REFUSED: no attempt is open. Call episode_begin first.", {}

    def log(phase, status, meas, note):
        st.phase(n, phase, status, meas, note)
    prior = attempt_strokes(st, n)
    tw = Twister(rig, params, log, prior_strokes=prior)
    try:
        res = await tw.twist_cap(float(x), float(y), float(cap_top_z), float(cap_height))
    finally:
        path = st.attempt_dir(n) / "twist_trace.json"
        old = json.loads(path.read_text()) if prior and path.exists() else {"rows": [], "strokes": []}
        st.save_json(n, "twist_trace.json", {"attempt": n, "params": params, "x": x, "y": y, "cap_top_z": cap_top_z,
                                             "rows": old["rows"] + tw.trace, "strokes": old["strokes"] + tw.strokes})
        st.write_report(n)
    lines = [("TWIST DONE" if res["ok"] else "TWIST STOPPED") + f" (attempt {n})"
             + (f", resumed after {res['strokes_before']} strokes" if res.get("resumed") else "") + "."]
    for s in res["strokes"]:
        lines.append(f"stroke {s['stroke']}: commanded {s['commanded_deg']:.0f} deg, achieved {s['achieved_deg']:.0f} deg, "
                     f"j7 load peak {s['peak_j7_load']} mean {s['mean_j7_load']:.0f}, j8 change {s['j8_change_ticks']} ticks"
                     f"{' (SLIP)' if s['slip'] else ''}, z change {s['z_change_mm']:+.1f} mm"
                     + (f"; stopped: {s['abort_reason']}" if s["abort_reason"] else ""))
    if res["ok"]:
        lines.append(f"rotation this call {res['rotation_achieved_deg']:.0f} deg (attempt {res['rotation_attempt_deg']:.0f} deg "
                     f"in {res['strokes_total']} of {res['strokes_commanded']} strokes), gripper rose {res['z_change_mm']:+.1f} mm, "
                     f"{res['elapsed_s']:.0f} s; {res['why_stopped']}.")
        if res["strokes_needed"]:
            lines.append(f"STROKES NEEDED: {res['strokes_needed']} (the j7 load fell from there on).")
        nxt = ("twist_cap(...) again with the same cap to resume" if res["stopped_by"] == "time budget" else
               "extract_cap() to take the cap off")
        lines.append(f"{'still holding the cap: call ' + nxt if res['holding'] else 'NOT holding the cap at the end'}.")
    else:
        lines.append(f"ABORT REASON: {res['abort_reason']}" + (" (outcome: misaligned)" if res.get("misaligned") else ""))
    lines += [f"NOTE: {t}" for t in res["notes"]]
    return "\n".join(lines), await rig.frames()


async def check_cap(st: E.EpisodeStore, rig):
    return await _lift_tool(st, rig, "check")


async def extract_cap(st: E.EpisodeStore, rig):
    return await _lift_tool(st, rig, "extract")


def lift_verdict(r: dict, which: str, sig: dict | None = None) -> str:
    """The judgement text of a check / extract lift (sig: the attempt's j7 free signal, for context)."""
    lift = r.get("lift_m", 0) * 1000
    if r["cap_free"]:
        v = (f"CAP OFF: the gripper rose {r['cap_lift_mm']:.0f} of {lift:.0f} mm still holding it, unhindered" if which == "extract"
             else f"cap probably free: the gripper rose {r['cap_lift_mm']:.0f} mm still holding it, unhindered (the jaws "
                  f"could also have slid up the cap: confirm a gap between cap and bottle in the images, or extract_cap())")
    elif not r["holding_before"]:
        v = "nothing lifted: the jaws are not holding the cap (judge from the images only)"
    elif r["resisted"]:
        v = (f"cap still attached: the lift was held back (gripper rose {r['gripper_rose_mm']:.0f} of "
             f"{r.get('lift_planned_mm', 0):.0f} mm{'; ' + r['stopped'] if r.get('stopped') else ''})")
    elif not r["holding_after"]:
        v = ("cap still attached: it slipped out of the jaws during the lift" + (
            " (the jaws held only the top of a cap still on the thread)" if which == "extract" else ""))
    else:
        v = f"unclear: holding after lifting only {r['gripper_rose_mm']:.0f} mm"
    if sig and sig.get("ref_load") is not None:
        v += f"; {capfree.text(sig)}"
    return v


async def _lift_tool(st: E.EpisodeStore, rig, which: str):
    n, params = _open(st)
    if n is None:
        return "REFUSED: no attempt is open.", {}
    sig = capfree.free_signal(attempt_strokes(st, n), SC.active()["values"])
    tw = Twister(rig, params, lambda *a: None)
    r = await (tw.extract_cap() if which == "extract" else tw.check_cap())
    r["free_signal"] = sig
    images = await rig.frames()
    paths = st.save_images(n, which, images)
    verdict = lift_verdict(r, which, sig)
    st.judge(n, "camera", verdict, paths, signals=r)
    lines = [f"{which.upper()} (attempt {n}): {verdict}.", f"signals: {json.dumps(r, default=str)}"]
    if hasattr(rig, "truth"):                     # the twin knows how far the cap really turned (never on hardware)
        t = rig.truth()
        st.judge(n, "twin", f"cap turned {t['turned_deg']:.0f} deg in total, thread lift {t['lift_mm']:.1f} mm, "
                            f"{'FREE' if t['free'] else 'still on the thread'}", [], truth=t)
    if os.environ.get("BOTTLE_GEMINI_JUDGE", "1") == "1" and images:
        from .judge import gemini_cap_judge
        g = gemini_cap_judge(images)
        if g is not None:
            st.judge(n, "gemini-er", g["verdict"], paths, **{k: v for k, v in g.items() if k != "verdict"})
            lines.append(f"Gemini ER: {g['verdict']}")
    st.phase(n, which, "ok", {"cap_free": r["cap_free"], "cap_lift_mm": r["cap_lift_mm"], "resisted": r["resisted"],
                              "holding_after": r["holding_after"], "lift_mm": round(r["lift_m"] * 1000, 1),
                              "gripper_rose_mm": r["gripper_rose_mm"], "j7_free": sig["free"]}, verdict)
    if r["holding_before"] and not r["cap_free"]:
        lines.append("retreat: " + await tw._retreat(f"{which} done, cap not free"))
    st.write_report(n)
    return "\n".join(lines), images


async def episode_end(st: E.EpisodeStore, rig, outcome: str, lesson: str, next_change: dict | None = None,
                      author: str = "learning rule"):
    if isinstance(next_change, str):
        next_change = json.loads(next_change) if next_change.strip() else None
    images = await rig.frames()
    r = st.end(outcome, lesson, next_change, images)
    if not r["ok"]:
        return "REFUSED:\n- " + "\n- ".join(r["errors"]), {}
    e = r["record"]
    text = (f"attempt {r['attempt']} closed: {e['outcome']}, rotation {e['rotation_achieved_deg']:.0f} deg, cap lift "
            f"{e['cap_lift_mm'] if e['cap_lift_mm'] is not None else '-'} mm. Report {r['report']}; summary {r['summary']}."
            + (f" ({r['stop_note']})" if r["stop_note"] else ""))
    if SC.get("learn.auto_propose") == "on":
        from . import rules
        out = await asyncio.to_thread(rules.propose_after, st, r["attempt"], author)
        text += "\n" + rules.proposals_text(out)
    return text, images


async def skill_config():
    return SC.config_text(), {}


async def propose_skill_change(changes, reason: str, evidence=None, twin_check: bool = False, author: str = "operator",
                               auto_apply: bool = False):
    """auto_apply: the twin server applies a proposal whose checks pass; the real server never does."""
    if isinstance(changes, str):
        changes = json.loads(changes)
    if isinstance(evidence, str):
        evidence = [e.strip() for e in evidence.split("\n") if e.strip()]
    t = SC.target()
    r = SC.propose(changes or {}, reason, evidence or [], author, t)
    if not r["ok"]:
        return "REFUSED, nothing proposed:\n- " + "\n- ".join(r["errors"]), {}
    pid = r["proposal"]["id"]
    from . import skillcheck
    p = await asyncio.to_thread(skillcheck.run_checks, pid, "async" if twin_check else False)
    lines = [f"PROPOSAL {pid} recorded with its checks:", SC.proposal_text(p)]
    failed = SC.checks_failed(p)
    if t == "real" or not auto_apply:
        lines.append(f"PENDING: the {t} config is unchanged (version {SC.version(t)}) until a human approves: "
                     f"python -m learning.skillcfg approve {pid} --by <name>" + (f". Checks FAILED: {'; '.join(failed)}"
                                                                                  if failed else ""))
    elif failed:
        lines.append(f"NOT APPLIED: checks failed ({'; '.join(failed)}). Change the proposal or bring more evidence.")
    else:
        a = SC.apply(pid, by="auto (twin)")
        lines.append(f"APPLIED to the twin: version {a['version']}; the next attempt runs with it." if a["ok"] else
                     "NOT APPLIED: " + "; ".join(a["errors"]))
    return "\n".join(lines), {}
