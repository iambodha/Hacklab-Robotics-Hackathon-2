import json

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("pandas")

from models import lerobot_export as lx  # noqa: E402
from recorder.replay import Recording  # noqa: E402
from recorder.tests.conftest import build  # noqa: E402


def test_round_trip_on_recorder_fixture(tmp_path):
    rec = Recording(build(tmp_path / "recordings", seconds=2.0))
    out = tmp_path / "ds"
    info = lx.export([rec], out, fps=10, task="wipe the stain")
    ds = lx.read_dataset(out)
    d = ds["data"]
    n = info["total_frames"]
    assert info["codebase_version"] == "v3.0" and info["total_episodes"] == 1 and ds["tasks"] == {"wipe the stain": 0}
    assert 15 <= n <= 20 and len(d["index"]) == n and d["frame_index"] == list(range(n))
    assert ds["episodes"][0]["dataset_to_index"] == n and ds["episodes"][0]["recording_id"] == rec.id
    assert info["features"]["observation.images.wrist"]["shape"] == [224, 299, 3]
    assert info["elrobot"]["action_source"] == ["goal"]           # fixture motors have torque on, goal 2000
    ts = d["timestamp"]
    assert all(b > a for a, b in zip(ts, ts[1:])) and ts[1] - ts[0] == pytest.approx(0.1, abs=1e-6)
    # every row equals the newest recorded data at or before its time (causal, Station clock)
    st_t = [r["station_ns"] / 1e9 for r in rec.rows]
    base = max(st_t[0], *(c["rows"][0]["station_ns"] / 1e9 for c in rec.cams.values()))
    for k in (0, n // 2, n - 1):
        t = base + ts[k]
        row = rec.rows[lx._causal(st_t, t)]
        assert d["observation.state"][k] == pytest.approx(row["norm"], abs=1e-6)
        assert d["action"][k][0] == pytest.approx((2000 - 1000) / 2000, abs=1e-6)
        for role in ("wrist", "overview"):
            h8 = rec.cam(role)
            ct = [r["station_ns"] / 1e9 for r in rec.cams[h8]["rows"]]
            fr = rec.cams[h8]["rows"][lx._causal(ct, t)]
            assert d[f"observation.images.{role}"][k]["bytes"] == rec.frame_bytes(h8, fr["n"])
            assert fr["station_ns"] / 1e9 <= t + 1e-9                # never from the future
    stats = json.loads((out / "meta" / "stats.json").read_text())
    assert len(stats["observation.state"]["mean"]) == 8 and stats["observation.state"]["count"] == [n]


def test_segments_split_and_filter(tmp_path):
    rec = Recording(build(tmp_path / "recordings", seconds=2.0))
    segs = {rec.id: [{"t0": 0.0, "t1": 0.9, "task": "approach", "keep": True},
                     {"t0": 0.9, "t1": 1.9, "task": "grasp", "keep": True},
                     {"t0": 1.0, "t1": 1.5, "task": "bad", "keep": False}]}
    info = lx.export([rec], tmp_path / "ds", fps=10, segments=segs)
    ds = lx.read_dataset(tmp_path / "ds")
    assert info["total_episodes"] == 2 and ds["tasks"] == {"approach": 0, "grasp": 1}
    assert [e["tasks"] for e in ds["episodes"]] == [["approach"], ["grasp"]]
    assert sorted(set(ds["data"]["episode_index"])) == [0, 1]
