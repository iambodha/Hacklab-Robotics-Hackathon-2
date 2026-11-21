"""Filter a teleop demo: drop frames where the operator was just holding still to aim.

A sample is kept only if the arm moved far enough from the last kept sample (joint space),
or the gripper changed, or it is the first/last sample. How long the arm dwelled at a kept pose
is stored as `dwell_s` (aiming time can itself be informative; ignore it if not).

  python3 scripts/evg/demo_filter.py NAME [--eps 40] [--grip-eps 60] [--out filtered.jsonl]

Input : demos/NAME/states.jsonl   rows {"t": s, "1": [pos, goal, cur], ... "8": [...]}   (record_demo.py)
Output: demos/NAME/filtered.jsonl same rows + "dwell_s", "src_idx", and the nearest frames
        "grip_frame"/"over_frame" (file names, by timestamp).
Distances are in servo steps (4096 = 360 deg): 40 steps ~ 3.5 deg.
"""
import argparse, json, math, re, sys
from pathlib import Path

ARM_JOINTS = range(1, 8)
GRIPPER = "8"
DEMOS = Path(__file__).resolve().parents[2] / "demos"


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def dist(a, b):
    """Euclidean distance over arm joints 1-7, using measured position (index 0)."""
    return math.sqrt(sum((a[str(j)][0] - b[str(j)][0]) ** 2 for j in ARM_JOINTS))


def filter_rows(rows, eps=40, grip_eps=60):
    if not rows:
        return []
    kept = [0]
    for i in range(1, len(rows)):
        last = rows[kept[-1]]
        moved = dist(rows[i], last) >= eps
        gripped = abs(rows[i][GRIPPER][0] - last[GRIPPER][0]) >= grip_eps
        if moved or gripped:
            kept.append(i)
    if kept[-1] != len(rows) - 1:
        kept.append(len(rows) - 1)
    out = []
    for n, i in enumerate(kept):
        end = rows[kept[n + 1]]["t"] if n + 1 < len(kept) else rows[i]["t"]
        out.append({**rows[i], "src_idx": i, "dwell_s": round(end - rows[i]["t"], 2)})
    return out


def frame_index(folder, tag):
    """[(t, filename)] sorted by t for frames named <tag>_<t>.jpg."""
    pat = re.compile(rf"^{tag}_(\d+\.\d+)\.jpg$")
    found = [(float(m.group(1)), p.name) for p in folder.glob(f"{tag}_*.jpg") if (m := pat.match(p.name))]
    return sorted(found)


def nearest(index, t):
    return min(index, key=lambda x: abs(x[0] - t))[1] if index else None


def run(name, eps=40, grip_eps=60, out="filtered.jsonl", demos=DEMOS):
    folder = Path(demos) / name
    rows = load(folder / "states.jsonl")
    kept = filter_rows(rows, eps, grip_eps)
    grip, over = frame_index(folder, "grip"), frame_index(folder, "over")
    for r in kept:
        r["grip_frame"], r["over_frame"] = nearest(grip, r["t"]), nearest(over, r["t"])
    (folder / out).write_text("".join(json.dumps(r) + "\n" for r in kept))
    return len(rows), len(kept), folder / out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--eps", type=float, default=40, help="min joint-space move in steps to keep a sample")
    ap.add_argument("--grip-eps", type=float, default=60, help="min gripper change in steps to keep a sample")
    ap.add_argument("--out", default="filtered.jsonl")
    a = ap.parse_args()
    n, k, path = run(a.name, a.eps, a.grip_eps, a.out)
    print(f"{a.name}: {n} samples -> {k} kept ({100 * k / max(n, 1):.0f}%)  {path}")
