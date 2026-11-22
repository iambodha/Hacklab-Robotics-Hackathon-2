"""Synthetic bottle-learning records for testing the dashboard.

    python -m dashboard.fixtures OUT_DIR [--attempts 10] [--live SECONDS_PER_EVENT] [--partial]

Writes OUT_DIR/episodes.jsonl, attempt_<N>/{before,after}_{wrist,overview}.jpg (if Pillow is available),
twist_trace.json, report.md and SUMMARY.md, following prompts/bottle-contract.md. `--live` appends the
events one by one with a delay so the page can be watched updating; `--partial` leaves the last attempt
in progress (no episode_end) and ends the file with half a JSON line.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

OUTCOMES = ["misaligned", "slip", "slip", "no_progress", "loosened", "slip", "loosened", "loosened", "opened", "opened"]
LESSONS = {
    "misaligned": "Jaws closed on the bottle shoulder, not the cap: approach_z_offset too low.",
    "slip": "Jaws slipped on the cap during the stroke: grip too weak for the thread friction.",
    "no_progress": "Cap turned back with the jaws on release: back off before opening.",
    "loosened": "Cap turns and lifts; more strokes needed to clear the thread.",
    "opened": "Cap free after enough strokes with a firm grip and thread-following lift.",
}
BASE = {"grip_close": 0.6, "grip_torque_limit": 300, "approach_z_offset_mm": -12, "twist_deg_per_stroke": 30,
        "strokes": 2, "twist_speed": "slow", "lift_per_turn_mm": 0.0, "yaw_start_deg": 0, "regrasp_backoff_deg": 0,
        "j7_load_abort": 600}
PLAN = [{}, {"approach_z_offset_mm": -4}, {"grip_torque_limit": 420, "grip_close": 0.75}, {"grip_torque_limit": 520},
        {"regrasp_backoff_deg": 8}, {"twist_deg_per_stroke": 60, "strokes": 3}, {"grip_close": 0.85},
        {"lift_per_turn_mm": 2.5, "strokes": 5}, {"strokes": 7, "twist_speed": "medium"}, {"strokes": 8}]


def _image(path: Path, label: str, hue: int):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    img = Image.new("RGB", (320, 240), (40 + hue % 60, 50, 60 + hue % 90))
    d = ImageDraw.Draw(img)
    d.ellipse((120, 60, 200, 140), fill=(20, 20, 24))
    d.rectangle((135, 130, 185, 230), fill=(30, 30, 34))
    d.text((10, 10), label, fill=(240, 240, 240))
    img.save(path, "JPEG", quality=70)


def generate(out: Path, attempts: int = 10, live: float = 0.0, partial: bool = False, seed: int = 1):
    rnd = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    ep = out / "episodes.jsonl"
    ep.write_text("")
    params, t, rows = dict(BASE), time.time() - (0 if live else attempts * 120), []

    def emit(obj):
        with open(ep, "a") as f:
            f.write(json.dumps(obj) + "\n")
        if live:
            time.sleep(live)

    for n in range(1, attempts + 1):
        params = {**params, **PLAN[(n - 1) % len(PLAN)]}
        outcome = OUTCOMES[(n - 1) % len(OUTCOMES)]
        d = out / f"attempt_{n}"
        d.mkdir(exist_ok=True)
        for cam in ("wrist", "overview"):
            _image(d / f"before_{cam}.jpg", f"attempt {n} before {cam}", n * 17)
        emit({"type": "episode_begin", "attempt": n, "t": t, "params": params,
              "hypothesis": f"Attempt {n}: " + (", ".join(f"{k}={v}" for k, v in PLAN[(n - 1) % len(PLAN)].items()) or "baseline")
              + " should fix the last failure.", "based_on": list(range(1, n)), "recording_id": f"20261122-02{n:02d}00_bottle-attempt-{n}"})
        last = partial and n == attempts
        strokes = params["strokes"]
        seq = ["locate", "approach", "grasp"] + ["twist", "release", "regrasp"] * (strokes - 1) + ["twist", "release", "check", "retreat"]
        if outcome == "misaligned":
            seq = ["locate", "approach", "grasp"]
        for i, ph in enumerate(seq):
            t += 4
            if last and i == len(seq) // 2:
                break
            status = "fail" if (outcome == "misaligned" and ph == "grasp") or (outcome == "slip" and ph == "twist" and i > 4) else "ok"
            meas = {"j7_load": rnd.randint(150, 650)} if ph == "twist" else {}
            if outcome == "slip" and ph == "twist" and status == "fail":
                meas["slip"] = True
            emit({"type": "phase", "attempt": n, "t": t, "phase": ph, "status": status, "measurements": meas,
                  "note": "jaws slipped on the cap" if meas.get("slip") else ""})
        if last:
            with open(ep, "a") as f:
                f.write('{"type": "phase", "attempt": %d, "t": %f, "pha' % (n, t))
            break
        ts = [i * 0.05 for i in range(120)]
        peak = rnd.randint(250, 700)
        trace = [{"t": x, "j7_pos": 2000 + 300 * math.sin(x), "j7_goal": 2000 + 320 * math.sin(x),
                  "j7_load": int(peak * abs(math.sin(x * 1.3))), "j8_pos": 1500, "j8_current": 40 + rnd.randint(0, 20),
                  "tcp_z": 0.18 + 0.001 * x} for x in ts]
        (d / "twist_trace.json").write_text(json.dumps(trace))
        for cam in ("wrist", "overview"):
            _image(d / f"after_{cam}.jpg", f"attempt {n} after {cam}", n * 31)
        rot = {"misaligned": 0, "slip": rnd.randint(5, 30), "no_progress": 10}.get(outcome, strokes * params["twist_deg_per_stroke"] * 0.8)
        lift = {"opened": 6.0, "loosened": round(rot / 360 * 3, 1)}.get(outcome, 0.0)
        nxt = PLAN[n % len(PLAN)] if n < attempts else {}
        t += 5
        emit({"type": "episode_end", "attempt": n, "t": t, "outcome": outcome, "rotation_achieved_deg": rot,
              "cap_lift_mm": lift, "measurements": {"peak_j7_load": peak, "slip_events": 1 if outcome == "slip" else 0},
              "lesson": LESSONS[outcome], "next_change": nxt})
        emit({"type": "judge", "attempt": n, "t": t + 1, "source": "camera", "verdict": f"cap {outcome}", "evidence": [str(d / "after_wrist.jpg")]})
        rows.append((n, outcome, rot, lift, LESSONS[outcome]))
        (d / "report.md").write_text(f"# Attempt {n}\n\n**Outcome:** {outcome}  \n**Rotation:** {rot:.0f} deg\n\n"
                                     f"## Params\n\n| param | value |\n|---|---|\n" + "".join(f"| {k} | {v} |\n" for k, v in params.items())
                                     + f"\n## Lesson\n\n{LESSONS[outcome]}\n\n![after](after_wrist.jpg)\n")
        (out / "SUMMARY.md").write_text("# Bottle opening: summary\n\n| # | outcome | rotation (deg) | lift (mm) | lesson |\n|---|---|---|---|---|\n"
                                        + "".join(f"| {r[0]} | {r[1]} | {r[2]:.0f} | {r[3]} | {r[4]} |\n" for r in rows)
                                        + "\n## Trend\n\n- Grip and alignment fixed first, then more strokes and thread-following lift.\n")
        t += 60


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dashboard.fixtures")
    ap.add_argument("out", type=Path)
    ap.add_argument("--attempts", type=int, default=10)
    ap.add_argument("--live", type=float, default=0.0, help="seconds between events (0 = write everything at once)")
    ap.add_argument("--partial", action="store_true", help="leave the last attempt in progress, with a torn last line")
    a = ap.parse_args(argv)
    generate(a.out, a.attempts, a.live, a.partial)
    print(f"wrote {a.attempts} attempts to {a.out}")


if __name__ == "__main__":
    main()
