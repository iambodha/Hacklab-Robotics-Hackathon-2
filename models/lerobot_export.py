"""Recorder sessions -> a LeRobot dataset (format v3.0), without installing lerobot or torch.

    uv run --no-project --with pandas --with pyarrow python -m models.lerobot_export \
        --out models/exports/elrobot_demo --task "wipe the stain off the table" <recording id or dir> ...

One recording (or one labelled segment of it, see models/autolabel.py) = one episode. Rows are sampled on a
fixed grid at `fps` on Station's own clock (`station_ns`), causally: every row holds the newest state and
the newest frame of each camera captured at or before the row time, so nothing comes from the future.

Features:
  observation.state            float32[8]  follower j1..j7 + gripper, calibrated norm (0..1, Station's ranges)
  action                       float32[8]  leader norm if a leader bus was recorded (teleop), else the follower
                                           goal register converted to norm when torque is on, else the next
                                           row's follower state (a proxy: flagged in info.json "action_source")
  observation.images.overview  image       JPEG bytes as recorded (299x224), HF Image struct {bytes, path}
  observation.images.wrist     image
  timestamp, frame_index, episode_index, index, task_index

Files: meta/info.json, meta/stats.json, meta/tasks.parquet, meta/episodes/chunk-000/file-000.parquet,
data/chunk-000/file-000.parquet. Images live inside the data parquet (LeRobot's `image` dtype) rather than
as MP4 so no encoder is needed; `lerobot` can re-encode to video later. `read_dataset()` loads it back.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path

from recorder.fmt import FOLLOWER_BUS
from recorder.replay import Recording

LEADER_BUS = "5B61035034"
JOINTS = [f"j{i}" for i in range(1, 9)]
NAMES = ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "gripper"]
CAMERAS = ("overview", "wrist")
CODEBASE_VERSION = "v3.0"


RECORDING_DIRS = [Path(__file__).resolve().parents[2] / "recorder" / "recordings"]


def open_recording(ref) -> Recording:
    """A Recording from an object, a directory, or an id (searched in $RECORDER_DIR and ws/recorder/recordings)."""
    if isinstance(ref, Recording):
        return ref
    for d in RECORDING_DIRS:
        if (d / str(ref) / "session.json").exists():
            return Recording(d / str(ref))
    return Recording(ref)


def _times(rows: list[dict], key: str = "station_ns") -> list[float]:
    return [r[key] / 1e9 if r.get(key) else r["t"] for r in rows]


def _causal(ts: list[float], t: float) -> int | None:
    i = bisect.bisect_right(ts, t + 1e-9) - 1
    return i if i >= 0 else None


def _norm(bus: dict, j: str, field: str = "norm") -> float | None:
    m = bus.get(j) or {}
    if field == "norm":
        return m.get("norm")
    lo, hi, v = m.get("range_min"), m.get("range_max"), m.get(field)
    return None if not (lo is not None and hi is not None and hi > lo and v) else (v - lo) / (hi - lo)


def _vec(bus: dict | None, field: str = "norm") -> list[float] | None:
    if not bus:
        return None
    v = [_norm(bus, j, field) for j in JOINTS]
    return None if any(x is None for x in v) else [float(x) for x in v]


def action_source(rec: Recording) -> str:
    if any(LEADER_BUS in (r.get("buses") or {}) for r in rec.rows[:50]):
        return "leader"
    if any(((r.get("buses") or {}).get(FOLLOWER_BUS, {}).get("j1") or {}).get("torque") for r in rec.rows):
        return "goal"
    return "next_state"


def sample_episode(rec: Recording, fps: int, t_from: float | None = None, t_to: float | None = None) -> dict:
    """Rows on a fixed grid (seconds on Station's clock relative to the first state row)."""
    ts = _times(rec.rows)
    cam_h8 = {role: rec.cam(role) for role in CAMERAS}
    if not ts or any(h is None for h in cam_h8.values()):
        raise ValueError(f"{rec.id}: needs state rows and both cameras ({cam_h8})")
    cts = {role: _times(rec.cams[h8]["rows"]) for role, h8 in cam_h8.items()}
    base = ts[0]
    start = max([ts[0]] + [c[0] for c in cts.values()])
    end = min([ts[-1]] + [c[-1] for c in cts.values()])
    if t_from is not None:
        start = max(start, base + t_from)
    if t_to is not None:
        end = min(end, base + t_to)
    src = action_source(rec)
    n = int(math.floor((end - start) * fps)) + 1 if end > start else 0
    out = {"state": [], "action": [], "images": {r: [] for r in CAMERAS}, "timestamp": [], "t_rec": [],
           "skew_ms": [], "action_source": src}
    for k in range(n):
        t = start + k / fps
        i = _causal(ts, t)
        row = rec.rows[i]
        fol = (row.get("buses") or {}).get(FOLLOWER_BUS)
        state = _vec(fol)
        if state is None:
            continue
        if src == "leader":
            act = _vec(row["buses"].get(LEADER_BUS)) or state
        elif src == "goal":
            act = _vec(fol, "goal") or state
        else:
            nxt = rec.rows[_causal(ts, t + 1 / fps)]
            act = _vec((nxt.get("buses") or {}).get(FOLLOWER_BUS)) or state
        frames = {}
        for role, h8 in cam_h8.items():
            j = _causal(cts[role], t)
            fr = rec.cams[h8]["rows"][j]
            frames[role] = (fr, rec.frame_bytes(h8, fr["n"]))
        if any(b is None for _, b in frames.values()):
            continue
        out["state"].append(state)
        out["action"].append(act)
        for role, (fr, b) in frames.items():
            out["images"][role].append(b)
        out["timestamp"].append(round(k / fps, 6))
        out["t_rec"].append(round(row["t"], 4))
        out["skew_ms"].append(round(1000 * (max(cts[r][_causal(cts[r], t)] for r in CAMERAS) -
                                            min(cts[r][_causal(cts[r], t)] for r in CAMERAS)), 2))
    return out


def _stats(rows: list[list[float]]) -> dict:
    if not rows:
        return {}
    n, d = len(rows), len(rows[0])
    mean = [sum(r[i] for r in rows) / n for i in range(d)]
    std = [math.sqrt(sum((r[i] - mean[i]) ** 2 for r in rows) / n) for i in range(d)]
    return {"min": [min(r[i] for r in rows) for i in range(d)], "max": [max(r[i] for r in rows) for i in range(d)],
            "mean": mean, "std": std, "count": [n]}


def export(recordings: list, out: Path, fps: int = 30, task: str = "manipulate objects on the table",
           segments: dict | None = None, robot_type: str = "elrobot_follower") -> dict:
    """segments: {recording_id: [{"t0", "t1", "task", "keep"}...]} (from autolabel); default = whole recording."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(out)
    (out / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    tasks: dict[str, int] = {}
    data_cols = {k: [] for k in ("observation.state", "action", "observation.images.overview", "observation.images.wrist",
                                 "timestamp", "frame_index", "episode_index", "index", "task_index")}
    episodes, sources, skews, size = [], set(), [], None
    for rec in recordings:
        rec = open_recording(rec)
        segs = (segments or {}).get(rec.id) or [{"t0": None, "t1": None, "task": task, "keep": True}]
        for seg in segs:
            if not seg.get("keep", True):
                continue
            ep = sample_episode(rec, fps, seg.get("t0"), seg.get("t1"))
            n = len(ep["state"])
            if n < 2:
                continue
            ti = tasks.setdefault(seg.get("task") or task, len(tasks))
            e_idx, start = len(episodes), len(data_cols["index"])
            data_cols["observation.state"] += ep["state"]
            data_cols["action"] += ep["action"]
            for role in CAMERAS:
                data_cols[f"observation.images.{role}"] += [{"bytes": b, "path": None} for b in ep["images"][role]]
            data_cols["timestamp"] += ep["timestamp"]
            data_cols["frame_index"] += list(range(n))
            data_cols["episode_index"] += [e_idx] * n
            data_cols["index"] += list(range(start, start + n))
            data_cols["task_index"] += [ti] * n
            sources.add(ep["action_source"])
            skews += ep["skew_ms"]
            if size is None:
                h8 = rec.cam("overview")
                info = rec.cams[h8]["info"]
                size = (int(info.get("h") or 224), int(info.get("w") or 299))
            episodes.append({"episode_index": e_idx, "tasks": [seg.get("task") or task], "length": n,
                             "data/chunk_index": 0, "data/file_index": 0, "dataset_from_index": start,
                             "dataset_to_index": start + n, "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0,
                             "recording_id": rec.id, "t_from": seg.get("t0"), "t_to": seg.get("t1"),
                             "action_source": ep["action_source"]})
    if not episodes:
        raise ValueError("nothing to export (no episode with 2+ rows)")
    img = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    vec = pa.list_(pa.float32(), 8)
    schema = pa.schema([("observation.state", vec), ("action", vec), ("observation.images.overview", img),
                        ("observation.images.wrist", img), ("timestamp", pa.float32()), ("frame_index", pa.int64()),
                        ("episode_index", pa.int64()), ("index", pa.int64()), ("task_index", pa.int64())])
    pq.write_table(pa.table(data_cols, schema=schema), out / "data" / "chunk-000" / "file-000.parquet")
    pd.DataFrame({"task_index": list(tasks.values())}, index=pd.Index(list(tasks), name="task")) \
        .to_parquet(out / "meta" / "tasks.parquet")
    pq.write_table(pa.Table.from_pylist(episodes), out / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    H, W = size
    vec_f = lambda: {"dtype": "float32", "shape": [8], "names": NAMES}  # noqa: E731
    scalar = lambda dt: {"dtype": dt, "shape": [1], "names": None}  # noqa: E731
    features = {"observation.state": vec_f(), "action": vec_f(),
                **{f"observation.images.{r}": {"dtype": "image", "shape": [H, W, 3], "names": ["height", "width", "channels"]}
                   for r in CAMERAS},
                "timestamp": scalar("float32"), "frame_index": scalar("int64"), "episode_index": scalar("int64"),
                "index": scalar("int64"), "task_index": scalar("int64")}
    total = len(data_cols["index"])
    info = {"codebase_version": CODEBASE_VERSION, "robot_type": robot_type, "total_episodes": len(episodes),
            "total_frames": total, "total_tasks": len(tasks), "chunks_size": 1000, "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 200, "fps": int(fps), "splits": {"train": f"0:{len(episodes)}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet", "video_path": None,
            "features": features,
            "elrobot": {"action_source": sorted(sources), "state_units": "Station calibrated norm 0..1, gripper higher = open",
                        "clock": "station_ns, causal sampling", "camera_skew_ms_p95": _pct(skews, 0.95),
                        "source": "ws/models/models/lerobot_export.py"}}
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    stats = {"observation.state": _stats(data_cols["observation.state"]), "action": _stats(data_cols["action"])}
    for k in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        stats[k] = _stats([[float(x)] for x in data_cols[k]])
    (out / "meta" / "stats.json").write_text(json.dumps(stats))
    return info


def _pct(xs, q):
    xs = sorted(xs)
    return None if not xs else xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def read_dataset(root: Path) -> dict:
    """Load an export back: info, tasks {task: index}, episodes (list of dicts), data (column dict)."""
    import pandas as pd
    import pyarrow.parquet as pq
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    eps = pq.read_table(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pylist()
    data = pq.read_table(root / info["data_path"].format(chunk_index=0, file_index=0)).to_pydict()
    return {"info": info, "tasks": {t: int(r.task_index) for t, r in tasks.iterrows()}, "episodes": eps, "data": data}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recordings", nargs="+", help="recording ids or directories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--task", default="manipulate objects on the table")
    ap.add_argument("--labels", help="directory of autolabel JSON files (segments, tasks, keep flags)")
    a = ap.parse_args()
    segs = None
    if a.labels:
        segs = {}
        for f in Path(a.labels).glob("*.json"):
            lab = json.loads(f.read_text())
            segs[lab["recording_id"]] = lab.get("episodes") or []
    info = export(a.recordings, Path(a.out), a.fps, a.task, segs)
    print(json.dumps({k: info[k] for k in ("total_episodes", "total_frames", "fps", "elrobot")}, indent=1))


if __name__ == "__main__":
    main()
