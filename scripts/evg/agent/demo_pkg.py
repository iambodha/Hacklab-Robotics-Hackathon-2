"""Turn recorded demos into what the VLM sees: a text summary, waypoints, limits and a few keyframe images.

Per take (filtered samples): gripper opening is split at the midpoint of its range into open/closed.
  open -> closed  = grasp        closed -> open = release
Waypoints (joint positions in servo steps): start, pregrasp (last pose before the gripper closed),
grasp, lifted, release, end. The lowest j2 seen is the floor (see safety.py: j2 increase = gripper goes down).
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import demo_filter
from common import DEMOS

LABELS = "ABCDEFGH"


def _take(name, demos):
    folder = Path(demos) / name
    filt = folder / "filtered.jsonl"
    if not filt.exists():
        demo_filter.run(name, demos=demos)
    rows = demo_filter.load(filt)
    meta = {}
    try: meta = json.loads((folder / "meta.json").read_text())
    except (FileNotFoundError, ValueError): pass
    return folder, rows, meta


def _joints(r):
    return {j: r[str(j)][0] for j in range(1, 9)}


def _events(rows):
    g = [r["8"][0] for r in rows]
    lo, hi = min(g), max(g)
    if hi - lo < 200:
        return None, None
    mid = (lo + hi) / 2
    close = release = None
    seen_open = False
    for i, v in enumerate(g):
        if v > mid: seen_open = True
        if close is None and seen_open and v <= mid: close = i
        elif close is not None and v > mid: release = i; break
    return close, release


def build(names, demos=DEMOS, max_images=8):
    waypoints, floor, env, text, images = {}, 0, {}, [], []
    for n, name in enumerate(names):
        folder, rows, meta = _take(name, demos)
        if not rows:
            continue
        L = LABELS[n % len(LABELS)]
        close, release = _events(rows)
        idx = {"start": 0, "end": len(rows) - 1}
        if close is not None:
            idx["pregrasp"], idx["grasp"] = max(close - 1, 0), close
            j2g = rows[close]["2"][0]
            idx["lifted"] = next((i for i in range(close + 1, len(rows)) if rows[i]["2"][0] < j2g - 40), min(close + 1, len(rows) - 1))
        if release is not None:
            idx["release"] = release
        order = sorted(idx.items(), key=lambda kv: kv[1])
        for k, i in order:
            waypoints[f"{L}_{k}"] = dict(joints=_joints(rows[i]), t=rows[i]["t"], dwell_s=rows[i]["dwell_s"],
                                         grip_frame=rows[i].get("grip_frame"), over_frame=rows[i].get("over_frame"), folder=str(folder))
        floor = max([floor] + [r["2"][0] for r in rows])
        for j in range(1, 8):
            vals = [r[str(j)][0] for r in rows]
            e = env.setdefault(j, [min(vals), max(vals)]); e[0], e[1] = min(e[0], min(vals)), max(e[1], max(vals))
        desc = meta.get("description") or "(no description)"
        text.append(f"DEMO {L} ({name}): {desc}\n  duration {rows[-1]['t']:.0f}s, {len(rows)} kept poses after removing pauses. "
                    f"Gripper closed at t={rows[close]['t']:.1f}s" + (f", opened (release) at t={rows[release]['t']:.1f}s" if release else "")
                    if close is not None else f"DEMO {L} ({name}): {desc}\n  no grasp detected in this demo")
        for k, i in order:
            w = waypoints[f"{L}_{k}"]
            text.append(f"  waypoint {L}_{k}: " + " ".join(f"j{j}={v}" for j, v in w["joints"].items()) + f"  (held {w['dwell_s']}s)")
        for k in ("pregrasp", "grasp", "release"):
            w = waypoints.get(f"{L}_{k}")
            for cam, key in (("overview", "over_frame"), ("gripper", "grip_frame")):
                f = w and w.get(key)
                if f and (folder / f).exists():
                    images.append((f"Demo {L} at '{k}', {cam} camera:", (folder / f).read_bytes()))
    images = images[:max_images]
    text.append(f"Lowest point reached in the demos (floor): j2={floor}. Never command j2 above about {floor + 60}.")
    return dict(text="\n".join(text), images=images, waypoints=waypoints, floor_j2=floor, envelope=env)


def parts(pkg):
    out = ["=== DEMONSTRATION (recorded by a human teleoperating the arm) ===\n" + pkg["text"]]
    for label, jpeg in pkg["images"]:
        out += [label, jpeg]
    return out
