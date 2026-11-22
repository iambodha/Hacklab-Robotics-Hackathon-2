"""The whole learning loop on the twin with a scripted parameter policy (no LLM): proves the tools, the records and
the reports end to end. Each attempt: episode_history -> episode_begin -> twist_cap -> check_cap -> episode_end.

    MUJOCO_GL=cgl uv run --no-project --with numpy --with scipy --with pillow --with mujoco --with 'mcp>=1.12,<2' \\
        python -m learning.sim_loop [--attempts 3] [--learn-dir DIR] [--recorder-url URL]

Prints one JSON line per attempt and a final {"summary": ...} line. The cap position comes from the layout (a
scripted policy, not perception: the agent finds it with look / pixel_to_table on the real arm).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# (params change, hypothesis): a plausible operator's first three attempts
POLICY = [
    ({"grip_torque_limit": 0.12, "twist_deg_per_stroke": 45, "strokes": 2, "twist_speed": "medium"},
     "A gentle grip (0.12) is enough for a cap and keeps the pinch risk low."),
    ({"grip_torque_limit": 0.45},
     "Attempt 1 slipped: the jaws turned without the cap. A firmer grip (0.45) should transmit the torque."),
    ({"strokes": 6, "twist_deg_per_stroke": 90},
     "The cap turns with 0.45. 2 x 45 deg is not enough thread: 6 x 90 deg should free it."),
]


def classify(att: dict, cap_turned_deg: float) -> tuple[str, str]:
    """Outcome + lesson from the records, the way an honest operator would read them; cap_turned_deg = how far the cap
    itself turned in this attempt (the twin's judge record; on hardware: the wrist/overview images)."""
    strokes = [s for p in att["phases"] if p["phase"] == "twist" for s in p["measurements"].get("strokes", [])]
    rot = sum(max(0.0, s["achieved_deg"]) for s in strokes)
    slips = sum(1 for s in strokes if s["slip"] or (s["abort_reason"] or "").startswith("stall"))
    free = any(j["source"] == "twin" and j["truth"]["free"] for j in att["judges"])
    aborted = [p for p in att["phases"] if p["status"] == "abort"]
    if free:
        return "opened", f"{rot:.0f} deg in {len(strokes)} strokes freed the cap; check_cap lifted it without resistance."
    if aborted:
        return "aborted", f"aborted in {aborted[0]['phase']}: {aborted[0]['note'][:120]}"
    if rot >= 20 and cap_turned_deg < 0.25 * rot:
        return "slip", (f"the gripper turned {rot:.0f} deg but the cap only {cap_turned_deg:.0f} deg: the jaws slid around it "
                        f"(j8 did not move, so only the cap's own rotation shows it). The grip is too weak.")
    if slips and rot < 20:
        return "slip", f"{slips} of {len(strokes)} strokes slipped or stalled ({rot:.0f} deg achieved): the grip is too weak."
    if cap_turned_deg >= 20:
        return "loosened", (f"the cap turned {cap_turned_deg:.0f} deg with the gripper ({rot:.0f} deg) and is still on the "
                            f"thread: it needs more total rotation.")
    return "no_progress", f"only {cap_turned_deg:.0f} deg of cap rotation."


async def main(attempts: int) -> list[dict]:
    import sim_arm_mcp as S
    b = S.arm.layout["screw_cap"]
    x, y = b["pos"][0], b["pos"][1]
    top = b["pos"][2] + 2 * b["half_height"]
    out, turned = [], S.arm.screw_state()["turned_deg"]
    for i in range(attempts):
        change, hyp = POLICY[min(i, len(POLICY) - 1)]
        await S.episode_history(10)
        r = await S.episode_begin(change, hyp)
        if not r[0].startswith("attempt"):
            raise RuntimeError(r[0])
        await S.twist_cap(x, y, top, 2 * b["half_height"])
        await S.check_cap()
        st = S._store()
        n = st.open_attempt()
        now = S.arm.screw_state()["turned_deg"]
        outcome, lesson = classify(st.attempts()[n], now - turned)
        turned = now
        nxt = POLICY[i + 1][0] if i + 1 < len(POLICY) else {}
        await S.episode_end(outcome, lesson, nxt)
        e = S._store().attempts()[n]["end"]
        out.append({"attempt": n, "outcome": outcome, "rotation_achieved_deg": e["rotation_achieved_deg"],
                    "cap_lift_mm": e["cap_lift_mm"], "truth": S.arm.screw_state()})
        print(json.dumps(out[-1]), flush=True)
        if outcome == "opened":
            break
        await S.move_tcp(x, y, top + 0.06, True, 0.0)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--learn-dir", default=None)
    ap.add_argument("--recorder-url", default=None)
    ap.add_argument("--layout", default=str(REPO / "sim" / "generated" / "bottle_twin.json"))
    a = ap.parse_args()
    sys.path.insert(0, str(REPO))
    from learning import bottle_twin
    lay = bottle_twin.make_layout(Path(a.layout))
    os.environ["SIM_LAYOUT"] = os.environ["ROBOT_SIM_LAYOUT"] = str(lay)
    os.environ.setdefault("ROBOT_SIM_RECORD", "0")
    os.environ.setdefault("BOTTLE_GEMINI_JUDGE", "0")
    if a.learn_dir:
        os.environ["BOTTLE_SIM_LEARN_DIR"] = a.learn_dir
    if a.recorder_url:
        os.environ["BOTTLE_SIM_RECORDER_URL"] = a.recorder_url
    import recorder  # noqa: F401  (pin the recorder package before sim/ (with its own recorder.py) joins sys.path)
    sys.path.append(str(REPO / "sim"))
    res = asyncio.run(main(a.attempts))
    print(json.dumps({"summary": res}), flush=True)
    import sim_arm_mcp
    sim_arm_mcp.arm.running = False
