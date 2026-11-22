"""Are recorder sessions good enough to train ACT / SmolVLA? Rates, sync and coverage per recording.

    uv run --no-project python -m models.dataset_audit [recording ids or dirs] [--md models/AUDIT.md]

All timing uses Station's stamps (station_ns): state rows (bus clock), per-motor read stamps, camera
frame stamps. Arrival-time (`t`) numbers are shown only for comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recorder.fmt import FOLLOWER_BUS

from .lerobot_export import CAMERAS, JOINTS, LEADER_BUS, RECORDING_DIRS, _causal, action_source, open_recording

TARGET_FPS = 30                   # LeRobot SO-10x datasets, ACT and SmolVLA defaults


def _pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return None if not xs else round(xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))], 3)


def _diffs(ts):
    return [b - a for a, b in zip(ts, ts[1:])]


def audit(ref) -> dict:
    rec = open_recording(ref)
    rows = rec.rows
    st = [r["station_ns"] / 1e9 for r in rows if r.get("station_ns")]
    dur = (st[-1] - st[0]) if len(st) > 1 else 0.0
    d = _diffs(st)
    out = {"id": rec.id, "duration_s": round(dur, 2), "state_rows": len(rows),
           "state_hz": round((len(st) - 1) / dur, 1) if dur else None,
           "state_dt_ms": {"p50": _pct([x * 1e3 for x in d], .5), "p95": _pct([x * 1e3 for x in d], .95),
                           "max": _pct([x * 1e3 for x in d], 1.0)},
           "station_frames_per_s": round(rec.meta.get("counts", {}).get("state_frames_seen", 0) / dur, 1) if dur else None}
    # per-motor freshness: how old each motor's last read is at a state row
    ages, refresh = [], {}
    for j in JOINTS:
        stamps = [((r.get("buses") or {}).get(FOLLOWER_BUS, {}).get(j) or {}).get("stamp_ns") for r in rows]
        uniq = sorted({s for s in stamps if s})
        refresh[j] = round((len(uniq) - 1) / dur, 1) if dur and len(uniq) > 1 else 0.0
        ages += [(r["station_ns"] - s) / 1e6 for r, s in zip(rows, stamps) if s and r.get("station_ns")]
    out["motor_refresh_hz"] = {"min": min(refresh.values()), "max": max(refresh.values())}
    out["motor_age_ms"] = {"p50": _pct(ages, .5), "p95": _pct(ages, .95)}
    cams, cts = {}, {}
    for role in CAMERAS:
        h8 = rec.cam(role)
        if h8 is None:
            continue
        crow = rec.cams[h8]["rows"]
        ts = [r["station_ns"] / 1e9 for r in crow if r.get("station_ns")]
        cts[role] = ts
        g = _diffs(ts)
        lat = [r.get("latency") for r in crow]
        info = rec.cams[h8]["info"]
        cams[role] = {"frames": len(crow), "fps": round((len(ts) - 1) / (ts[-1] - ts[0]), 1) if len(ts) > 1 else None,
                      "gap_ms_p95": _pct([x * 1e3 for x in g], .95), "gap_ms_max": _pct([x * 1e3 for x in g], 1.0),
                      "dropped": info.get("dropped"), "late": info.get("late"),
                      "latency_ms_p50": _pct([x * 1e3 for x in lat if x is not None], .5),
                      "latency_ms_p95": _pct([x * 1e3 for x in lat if x is not None], .95),
                      "size": [info.get("w"), info.get("h")]}
    out["cameras"] = cams
    if len(cts) == 2:
        a, b = cts["wrist"], cts["overview"]
        skew = []
        for t in a:
            i = _causal(b, t)
            cands = [b[k] for k in (i, (i or 0) + 1) if k is not None and k < len(b)]
            if cands:
                skew.append(min(abs(t - c) for c in cands) * 1e3)
        out["camera_skew_ms"] = {"p50": _pct(skew, .5), "p95": _pct(skew, .95), "max": _pct(skew, 1.0)}
        fs = []
        for t in a:
            i = _causal(st, t)
            if i is not None:
                fs.append(abs(t - st[i]) * 1e3)
        out["frame_to_state_ms"] = {"p50": _pct(fs, .5), "p95": _pct(fs, .95)}
    # demonstration content
    raw = {j: [((r.get("buses") or {}).get(FOLLOWER_BUS, {}).get(j) or {}).get("pos") for r in rows] for j in JOINTS}
    out["motion_ticks"] = {j: (max(v) - min(v)) if all(x is not None for x in v) and v else None for j, v in raw.items()}
    out["torque_on_rows"] = sum(1 for r in rows if any(r.get("torque") or []))
    out["leader_recorded"] = any(LEADER_BUS in (r.get("buses") or {}) for r in rows[:50])
    out["commands"] = len(rec.commands)
    out["action_source"] = action_source(rec)
    out["markers"] = [m.get("label") for m in rec.markers()]
    out["verdict"] = verdict(out)
    return out


def verdict(a: dict) -> dict:
    v = {}
    v["state_rate_ok"] = bool(a["state_hz"] and a["state_hz"] >= TARGET_FPS * 1.4)
    v["motor_fresh_ok"] = bool(a["motor_age_ms"]["p95"] is not None and a["motor_age_ms"]["p95"] < 1000 / TARGET_FPS)
    v["camera_rate_ok"] = all(c["fps"] and c["fps"] >= TARGET_FPS * 0.9 and (c["gap_ms_max"] or 0) < 200
                              for c in a["cameras"].values()) and len(a["cameras"]) == 2
    v["camera_sync_ok"] = bool(a.get("camera_skew_ms") and a["camera_skew_ms"]["p95"] < 1000 / TARGET_FPS / 2)
    v["has_actions"] = a["action_source"] in ("leader", "goal")
    v["has_motion"] = any((t or 0) > 50 for t in a["motion_ticks"].values())
    v["usable_for_training"] = all(v.values())
    return v


def to_markdown(audits: list[dict]) -> str:
    L = ["# Recorder data audit for policy training (ACT / SmolVLA)", "",
         "Generated by `python -m models.dataset_audit`. Times on Station's clock (`station_ns`).",
         f"Target: {TARGET_FPS} fps rows (LeRobot SO-10x, ACT and SmolVLA defaults), two cameras.", "",
         "| recording | dur s | state Hz | motor refresh Hz | motor age p95 ms | overview fps / max gap ms | wrist fps / max gap ms "
         "| cam skew p95 ms | frame->state p95 ms | actions | motion (max ticks) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in audits:
        c = a["cameras"]
        cam = lambda r: f"{c[r]['fps']} / {c[r]['gap_ms_max']}" if r in c else "-"  # noqa: E731
        L.append(f"| {a['id']} | {a['duration_s']} | {a['state_hz']} | {a['motor_refresh_hz']['min']}-{a['motor_refresh_hz']['max']} "
                 f"| {a['motor_age_ms']['p95']} | {cam('overview')} | {cam('wrist')} | {(a.get('camera_skew_ms') or {}).get('p95')} "
                 f"| {(a.get('frame_to_state_ms') or {}).get('p95')} | {a['action_source']} | {max(t or 0 for t in a['motion_ticks'].values())} |")
    L += ["", "Checks per recording:", ""]
    for a in audits:
        L.append(f"- `{a['id']}`: " + ", ".join(f"{k} {'yes' if ok else '**no**'}" for k, ok in a["verdict"].items()))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recordings", nargs="*")
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args()
    refs = a.recordings or sorted(p.name for d in RECORDING_DIRS for p in d.glob("2026*") if (p / "session.json").exists())
    audits = [audit(r) for r in refs]
    if a.json:
        Path(a.json).write_text(json.dumps(audits, indent=1))
    md = to_markdown(audits)
    if a.md:
        Path(a.md).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
