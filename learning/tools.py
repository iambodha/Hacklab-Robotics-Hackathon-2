"""The learning MCP tool bodies, shared by sim/real_arm_mcp.py and sim/sim_arm_mcp.py (same names, same texts).

Each returns (text, images) with images = {"wrist": jpeg, "overview": jpeg}; the servers turn that into MCP content.
"""
from __future__ import annotations

import json
import os

from . import episodes as E
from .twist import Twister

TOOL_DOCS = {
    "episode_begin": """Start attempt N of the bottle-cap learning run. `params`: the learnable parameters you set for
this attempt (missing ones are inherited from the previous attempt; nothing is clamped: out-of-range values are
REFUSED with the reason). Change at most 3 against the previous attempt. `hypothesis`: what you expect the change to
do and why. Saves the before images, starts a recorder session. Returns the attempt number and the full parameters.
Bounds:
""" + E.bounds_text(),
    "episode_history": """All previous attempts as compact text: parameters, what changed, hypothesis, every phase with
its measurements (rotation achieved, j7 load, jaw slip, z change), judgements, outcome, lesson and the planned next
change, plus the trend. Read it before choosing the next attempt's parameters.""",
    "twist_cap": """The whole unscrewing skill for the OPEN attempt, with its parameters: jaws open, hover 5 cm above
(x, y), turn to yaw_start_deg, guarded descent to the grasp height (cap_top_z - 10 mm + approach_z_offset_mm; never
below the cap's lower edge - 5 mm), torque-limited close (grip_close, grip_torque_limit), then `strokes` x [turn the
gripper counter-clockwise (seen from above) about the vertical cap axis by twist_deg_per_stroke at twist_speed while
rising lift_per_turn_mm per turn, watching the wrist-roll load (stops at j7_load_abort) and the jaws (slip); release;
turn back; regrasp]. The last grip is kept for check_cap. (x, y): cap centre from pixel_to_table on a CLOSE wrist view
at height cap_top_z; cap_height: the cap's height in m. On any abort it opens the jaws and lifts straight up.
Returns per stroke: commanded vs achieved rotation, peak/mean j7 load, j8 change (slip), z change, why it stopped.""",
    "check_cap": """Did the cap come free? With the cap gripped, lifts 15 mm straight up: a cap still on the thread
holds the gripper back (arm loads rise) or slips out of the jaws; a free cap comes up in the jaws. Saves check
images and logs a judgement (signals + images, plus Gemini ER when a key is available). If the cap is free it stays
in the jaws; otherwise the jaws open and the gripper lifts away.""",
    "episode_end": """Close the open attempt. outcome: one of opened|loosened|no_progress|slip|misaligned|aborted|fault.
lesson: one or two honest sentences on what this attempt showed (cite the measurements). next_change: the parameter
change you plan next, e.g. {"grip_torque_limit": 0.5}. Saves the after images, writes the attempt's report.md and
SUMMARY.md, stops the recording.""",
}


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


async def twist_cap(st: E.EpisodeStore, rig, x: float, y: float, cap_top_z: float, cap_height: float = 0.02):
    n, params = _open(st)
    if n is None:
        return "REFUSED: no attempt is open. Call episode_begin first.", {}

    def log(phase, status, meas, note):
        st.phase(n, phase, status, meas, note)
    tw = Twister(rig, params, log)
    try:
        res = await tw.twist_cap(float(x), float(y), float(cap_top_z), float(cap_height))
    finally:
        st.save_json(n, "twist_trace.json", {"attempt": n, "params": params, "x": x, "y": y, "cap_top_z": cap_top_z,
                                             "rows": tw.trace, "strokes": tw.strokes})
        st.write_report(n)
    lines = [("TWIST DONE" if res["ok"] else "TWIST STOPPED") + f" (attempt {n})."]
    for s in res["strokes"]:
        lines.append(f"stroke {s['stroke']}: commanded {s['commanded_deg']:.0f} deg, achieved {s['achieved_deg']:.0f} deg, "
                     f"j7 load peak {s['peak_j7_load']} mean {s['mean_j7_load']:.0f}, j8 change {s['j8_change_ticks']} ticks"
                     f"{' (SLIP)' if s['slip'] else ''}, z change {s['z_change_mm']:+.1f} mm"
                     + (f"; stopped: {s['abort_reason']}" if s["abort_reason"] else ""))
    if res["ok"]:
        lines.append(f"total rotation achieved {res['rotation_achieved_deg']:.0f} deg, gripper rose {res['z_change_mm']:+.1f} mm, "
                     f"{'still holding the cap: call check_cap()' if res['holding'] else 'NOT holding the cap at the end'}.")
    else:
        lines.append(f"ABORT REASON: {res['abort_reason']}" + (" (outcome: misaligned)" if res.get("misaligned") else ""))
    lines += [f"NOTE: {t}" for t in res["notes"]]
    return "\n".join(lines), await rig.frames()


async def check_cap(st: E.EpisodeStore, rig):
    n, params = _open(st)
    if n is None:
        return "REFUSED: no attempt is open.", {}
    tw = Twister(rig, params, lambda *a: None)
    r = await tw.check_cap()
    images = await rig.frames()
    paths = st.save_images(n, "check", images)
    if r["cap_free"]:
        verdict = f"cap came free: lifted {r['cap_lift_mm']:.0f} mm in the jaws without resistance"
    elif not r["holding_before"]:
        verdict = "cannot tell from the signals: the jaws were not holding the cap"
    elif r["resisted"]:
        verdict = f"cap still attached: the lift was held back ({r.get('resist_note', '')})"
    else:
        verdict = "cap still attached: it slipped out of the jaws during the lift" if not r["holding_after"] else \
            f"unclear: holding after lifting only {r['cap_lift_mm']:.0f} mm"
    st.judge(n, "camera", verdict, paths, signals=r)
    lines = [f"CHECK (attempt {n}): {verdict}.", f"signals: {json.dumps(r)}"]
    if os.environ.get("BOTTLE_GEMINI_JUDGE", "1") == "1" and images:
        from .judge import gemini_cap_judge
        g = gemini_cap_judge(images)
        if g is not None:
            st.judge(n, "gemini-er", g["verdict"], paths, **{k: v for k, v in g.items() if k != "verdict"})
            lines.append(f"Gemini ER: {g['verdict']}")
    st.phase(n, "check", "ok", {"cap_free": r["cap_free"], "cap_lift_mm": r["cap_lift_mm"], "resisted": r["resisted"],
                                "holding_after": r["holding_after"]}, verdict)
    if not r["cap_free"]:
        lines.append("retreat: " + await tw._retreat("check done, cap not free"))
    st.write_report(n)
    return "\n".join(lines), images


async def episode_end(st: E.EpisodeStore, rig, outcome: str, lesson: str, next_change: dict | None = None):
    if isinstance(next_change, str):
        next_change = json.loads(next_change) if next_change.strip() else None
    images = await rig.frames()
    r = st.end(outcome, lesson, next_change, images)
    if not r["ok"]:
        return "REFUSED:\n- " + "\n- ".join(r["errors"]), {}
    e = r["record"]
    return (f"attempt {r['attempt']} closed: {e['outcome']}, rotation {e['rotation_achieved_deg']:.0f} deg, cap lift "
            f"{e['cap_lift_mm'] if e['cap_lift_mm'] is not None else '-'} mm. Report {r['report']}; summary {r['summary']}."
            + (f" ({r['stop_note']})" if r["stop_note"] else "")), images
