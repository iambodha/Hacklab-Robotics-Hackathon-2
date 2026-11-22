"""Gemini ER as an auto-labeller for recorder sessions: sub-task segments, success, language, keep/drop.

    uv run --no-project --with google-genai==3.3.0 python -m models.autolabel [ids or dirs] [--no-gemini]

Two passes per recording:
  1. sensors (no model): joint motion -> moving / idle spans, gripper close / open events, camera gaps,
     state stalls, recorder markers, whether an action stream exists. Cheap, always runs.
  2. Gemini Robotics ER (optional): keyframes from both cameras every `every_s` seconds (at most
     `max_frames` per camera), plus the sensor summary, -> a one-sentence task instruction, sub-task
     segments from SUBTASKS with success flags, issues (person in view, occlusion, dropped object), keep.
Output: models/labels/<recording id>.json, which lerobot_export --labels uses (episodes with task strings
and keep flags). Times are seconds since the recording started (recorder `t`).

This is how the larger model improves the small policy's data: it names what happened, cuts the
demo to the part that matters, and throws out the failures, before any training.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from recorder.fmt import FOLLOWER_BUS

from .gemini_er import GeminiER, GeminiError, Image, ParseError
from .lerobot_export import CAMERAS, JOINTS, RECORDING_DIRS, action_source, open_recording

SUBTASKS = ["idle", "approach", "grasp", "lift", "transport", "place", "release", "wipe_stroke", "inspect", "stir",
            "retreat", "recover", "other"]
LABEL_DIR = Path(__file__).resolve().parent / "labels"
MOVE_THRESH = 0.02        # norm/s summed over joints: below = idle
MIN_SPAN_S = 0.5
GAP_MS = 100.0


# ---- pass 1: sensors ------------------------------------------------------------------------------------
def sensor_pass(rec) -> dict:
    rows = rec.rows
    ts = [r["t"] for r in rows]
    vec = [[((r.get("buses") or {}).get(FOLLOWER_BUS, {}).get(j) or {}).get("norm") for j in JOINTS] for r in rows]
    speed = [0.0]
    for k in range(1, len(rows)):
        dt = max(ts[k] - ts[k - 1], 1e-3)
        a, b = vec[k - 1], vec[k]
        speed.append(sum(abs(y - x) for x, y in zip(a[:7], b[:7]) if x is not None and y is not None) / dt)
    # smooth over ~0.2 s and threshold
    w = 5
    sm = [sum(speed[max(0, k - w):k + w + 1]) / len(speed[max(0, k - w):k + w + 1]) for k in range(len(speed))]
    spans, cur = [], None
    for t, s in zip(ts, sm):
        state = "moving" if s > MOVE_THRESH else "idle"
        if cur and cur["kind"] == state:
            cur["t1"] = t
        else:
            cur = {"kind": state, "t0": t, "t1": t}
            spans.append(cur)
    merged = []
    for s in spans:                                   # absorb blips shorter than MIN_SPAN_S into the previous span
        if merged and s["t1"] - s["t0"] < MIN_SPAN_S:
            merged[-1]["t1"] = s["t1"]
        elif merged and merged[-1]["kind"] == s["kind"]:
            merged[-1]["t1"] = s["t1"]
        else:
            merged.append(dict(s))
    grip = [v[7] for v in vec]
    events = []
    for k in range(1, len(grip)):
        if grip[k] is None or grip[k - 1] is None:
            continue
        d = grip[k] - grip[max(0, k - 10)]
        if abs(d) > 0.15 and (not events or ts[k] - events[-1]["t"] > 1.0):
            events.append({"t": round(ts[k], 2), "type": "gripper_open" if d > 0 else "gripper_close"})
    gaps = {}
    for role in CAMERAS:
        h8 = rec.cam(role)
        crow = rec.cams[h8]["rows"] if h8 else []
        ct = [r["station_ns"] / 1e9 if r.get("station_ns") else r["t"] for r in crow]   # source clock, as the exporter
        d = sorted(b - a for a, b in zip(ct, ct[1:]))
        lim = max(GAP_MS, 2500 * d[len(d) // 2]) if d else GAP_MS          # > 100 ms and > 2.5 frame periods
        gaps[role] = [round(crow[k + 1]["t"], 2) for k, (a, b) in enumerate(zip(ct, ct[1:])) if (b - a) * 1000 > lim]
    moving = [s for s in merged if s["kind"] == "moving"]
    return {"duration_s": round(rec.duration, 2),
            "spans": [{**s, "t0": round(s["t0"], 2), "t1": round(s["t1"], 2)} for s in merged],
            "active": [round(moving[0]["t0"], 2), round(moving[-1]["t1"], 2)] if moving else None,
            "gripper_events": events, "camera_gaps": gaps,
            "stalls": [round(e["t"], 2) for e in rec.events if e.get("type") in ("state_stalled", "camera_lost")],
            "markers": [{"t": round(m["t"], 2), "label": m.get("label")} for m in rec.markers()],
            "action_source": action_source(rec)}


# ---- pass 2: Gemini ---------------------------------------------------------------------------------------
def keyframes(rec, every_s: float, max_frames: int) -> list[tuple[float, str, Image]]:
    dur = rec.duration
    n = max(1, min(max_frames, int(dur / every_s) + 1))
    times = [round(dur * (k + 0.5) / n, 2) for k in range(n)]
    out = []
    for t in times:
        for role in CAMERAS:
            hit = rec.frame_at(role, t, causal=True)
            if hit:
                row, data = hit
                out.append((t, role, Image(data, role=role, ref=f"recording:{rec.id}@{t}",
                                           w=int(row.get("w") or 0), h=int(row.get("h") or 0))))
    return out


def prompt_label(sensor: dict, task_hint: str | None) -> str:
    return ("These are keyframes from one recording of a single robot arm with a parallel gripper working on a "
            "table, seen from a fixed overview camera and a wrist camera, in time order. Joint sensors summary "
            f"(seconds since start): {json.dumps({k: sensor[k] for k in ('duration_s', 'spans', 'gripper_events', 'markers')})}.\n"
            + (f"The operator said the task was: {task_hint}.\n" if task_hint else "")
            + "Label the recording for robot-learning data. Use only these sub-task labels: " + ", ".join(SUBTASKS)
            + ". Segments must not overlap and must lie within the duration. Answer JSON: "
            "{\"task\": <one imperative sentence a person would say to request what the robot did, e.g. 'wipe the "
            "brown stain off the table'>, \"success\": true|false|null, \"segments\": [{\"t0\": s, \"t1\": s, \"label\": <sub-task>, "
            "\"success\": true|false|null, \"note\": <short>}], \"issues\": [<person in workspace | occluded | object "
            "dropped | camera problem | no robot motion | other, short>], \"keep\": true|false, \"why\": <one sentence>}. "
            "keep=false if the demo failed, a person interferes, or it shows nothing useful to imitate.")


def gemini_pass(rec, er: GeminiER, sensor: dict, task_hint: str | None, every_s: float, max_frames: int) -> dict:
    parts = []
    for t, role, im in keyframes(rec, every_s, max_frames):
        parts += [f"t={t:.2f} s, {role} camera:", im]
    obj = er.ask_json("autolabel", parts + [prompt_label(sensor, task_hint)], thinking_level="medium")
    return clean_gemini(obj, sensor["duration_s"])


def clean_gemini(obj, duration: float) -> dict:
    """Keep only well-formed, in-range, non-overlapping segments with known labels."""
    if not isinstance(obj, dict):
        raise ParseError("autolabel answer is not an object")
    segs, last = [], 0.0
    for s in sorted([s for s in obj.get("segments") or [] if isinstance(s, dict)], key=lambda s: float(s.get("t0", 0) or 0)):
        try:
            t0, t1 = max(0.0, float(s["t0"])), min(duration, float(s["t1"]))
        except (KeyError, TypeError, ValueError):
            continue
        t0 = max(t0, last)
        if s.get("label") not in SUBTASKS or t1 - t0 < 0.1:
            continue
        segs.append({"t0": round(t0, 2), "t1": round(t1, 2), "label": s["label"], "success": s.get("success"),
                     "note": str(s.get("note", ""))[:200]})
        last = t1
    return {"task": str(obj.get("task") or "").strip() or None, "success": obj.get("success"), "segments": segs,
            "issues": [str(i)[:100] for i in obj.get("issues") or []], "keep": obj.get("keep"),
            "why": str(obj.get("why", ""))[:300]}


# ---- combine --------------------------------------------------------------------------------------------
def decide(sensor: dict, gem: dict | None, task_hint: str | None) -> tuple[bool, list[str], list[dict]]:
    reasons = []
    if sensor["active"] is None:
        reasons.append("no robot motion")
    if sum(len(g) for g in sensor["camera_gaps"].values()) > 2 * len(CAMERAS):
        reasons.append(f"camera gaps > {GAP_MS:.0f} ms: {sensor['camera_gaps']}")
    if sensor["stalls"]:
        reasons.append(f"state stall / camera lost at {sensor['stalls']}")
    if sensor["action_source"] == "next_state":
        reasons.append("no action stream (no leader bus, no torque): action would be a next-state proxy")
    if gem:
        if gem.get("keep") is False:
            reasons.append("gemini: " + (gem.get("why") or "drop"))
        if gem.get("success") is False:
            reasons.append("gemini: task failed")
    keep = not reasons
    task = (gem or {}).get("task") or task_hint or "manipulate objects on the table"
    t0, t1 = sensor["active"] or (0.0, sensor["duration_s"])
    pad = 0.5
    episodes = [{"t0": round(max(0.0, t0 - pad), 2), "t1": round(min(sensor["duration_s"], t1 + pad), 2), "task": task,
                 "keep": keep}]
    return keep, reasons, episodes


def label(ref, er: GeminiER | None = None, task_hint: str | None = None, every_s: float = 1.0,
          max_frames: int = 12) -> dict:
    rec = open_recording(ref)
    sensor = sensor_pass(rec)
    gem, err = None, None
    if er is not None:
        try:
            gem = gemini_pass(rec, er, sensor, task_hint, every_s, max_frames)
        except (GeminiError, ParseError) as e:
            err = str(e)
    keep, reasons, episodes = decide(sensor, gem, task_hint)
    return {"recording_id": rec.id, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": er.model if er else None, "gemini_call": er.last.call_id if er and er.last else None,
            "gemini_error": err, "task": episodes[0]["task"], "keep": keep, "reasons": reasons,
            "episodes": episodes, "segments": (gem or {}).get("segments") or
            [{"t0": s["t0"], "t1": s["t1"], "label": "idle" if s["kind"] == "idle" else "other", "success": None,
              "note": "sensor span"} for s in sensor["spans"]],
            "gemini": gem, "sensor": sensor}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recordings", nargs="*")
    ap.add_argument("--no-gemini", action="store_true")
    ap.add_argument("--task", help="what the operator said the task was")
    ap.add_argument("--every", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=12)
    ap.add_argument("--out", default=str(LABEL_DIR))
    a = ap.parse_args()
    refs = a.recordings or sorted(p.name for d in RECORDING_DIRS for p in d.glob("2026*") if (p / "session.json").exists())
    er = None if a.no_gemini else GeminiER(source_tag="autolabel")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for r in refs:
        lab = label(r, er, a.task, a.every, a.max_frames)
        (out / f"{lab['recording_id']}.json").write_text(json.dumps(lab, indent=1))
        print(f"{lab['recording_id']}: keep={lab['keep']} task={lab['task']!r} segments={len(lab['segments'])} "
              f"reasons={lab['reasons']}" + (f" gemini_error={lab['gemini_error'][:100]}" if lab["gemini_error"] else ""))


if __name__ == "__main__":
    main()
