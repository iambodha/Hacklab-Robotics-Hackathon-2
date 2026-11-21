import asyncio
import json

import pytest

from recorder.fmt import NAMES, gripper_opening, read_state_entry, signed
from recorder.replay import Recording, ReplaySource

from .conftest import BUS, CAM, CAM2, HI, LO, build, pos_at, regs


def test_format_round_trip(rec_dir):
    meta = json.loads((rec_dir / "session.json").read_text())
    assert meta["source"] == "real" and meta["format_version"] == 1
    assert meta["name"] == "synthetic" and meta["note"] == "unit test"
    assert meta["stopped"] and meta["duration"] is not None
    assert meta["counts"]["state_rows"] == 100 and meta["counts"]["state_frames_seen"] == 800
    assert meta["counts"]["markers"] == 1
    assert meta["buses"][BUS]["motor_ids"] == list(range(1, 9))
    cams = meta["cameras"]
    assert cams[CAM[:8]]["role"] == "wrist" and cams[CAM2[:8]]["role"] == "overview"
    assert cams[CAM[:8]]["frames"] == 20 and cams[CAM[:8]]["dropped"] == 5  # index jumps 9 -> 15
    rows = [json.loads(l) for l in (rec_dir / "state.jsonl").read_text().splitlines()]
    assert [r["t"] for r in rows] == sorted(r["t"] for r in rows)
    r = rows[10]
    assert {"t", "wall", "station_ns", "buses", "norm", "goal"} <= set(r)
    b = r["buses"][BUS]
    assert b["role"] == "follower" and all(n in b for n in NAMES)
    j2 = b["j2"]
    assert {"pos", "goal", "speed", "load", "current", "volt", "temp", "torque", "moving", "range_min", "range_max",
            "frozen", "error", "regs"} <= set(j2)
    assert j2["load"] == -50 and j2["pos"] == pos_at(2, r["t"])
    assert b["gripper"] == gripper_opening(b["j8"]) is not None
    idx = [json.loads(l) for l in (rec_dir / "cams" / CAM[:8] / "index.jsonl").read_text().splitlines()]
    assert idx[0]["n"] == 0 and (rec_dir / "cams" / CAM[:8] / "0.jpg").exists()
    assert {"t", "wall", "station_ns", "index", "latency", "bytes"} <= set(idx[0])
    evs = [json.loads(l)["type"] for l in (rec_dir / "events.jsonl").read_text().splitlines()]
    assert evs[0] == "start" and evs[-1] == "stop" and "marker" in evs


def test_decode_helpers():
    assert signed(0x0400 | 7, 10) == -7 and signed(7, 10) == 7
    assert gripper_opening({"norm": 1.3}) == 1.0 and gripper_opening({"norm": None}) is None
    e = read_state_entry(regs(0x8000 | 5, goal=1234), LO, HI)
    assert e["raw"] == 4091 and e["goal"] == 1234 and e["in_range"] is False


def test_state_and_frame_lookup(rec_dir):
    rec = Recording(rec_dir)
    assert rec.state_at(0.509)["t"] == pytest.approx(0.5, abs=0.02)
    assert rec.state_at(0.519)["t"] == pytest.approx(0.52, abs=1e-3)       # nearest
    assert rec.state_at(0.519, causal=True)["t"] == pytest.approx(0.50, abs=1e-3)
    assert rec.state_at(-5)["t"] == rec.ts[0] and rec.state_at(99)["t"] == rec.ts[-1]
    row, data = rec.frame_at("wrist", 0.71)
    assert row["n"] == 7 and data == b"\xff\xd828ff0ca8-7\xff\xd9"
    row, _ = rec.frame_at(CAM2[:8], 0.74)                      # frames at .651, .751 -> nearest .751
    assert row["n"] == 7
    assert rec.frame_at(CAM2, 0.74, causal=True)[0]["n"] == 6
    assert rec.frame_at("nope", 1.0) is None
    assert [m["label"] for m in rec.markers()] == ["grasp attempt"]
    assert len(rec.rows_between(0.5, 1.0, every=0.1)) == 6


def test_iter_order(rec_dir):
    rec = Recording(rec_dir)
    items = list(rec.iter(realtime=False))
    ts = [t for t, _, _ in items]
    assert ts == sorted(ts)
    kinds = {k for _, k, _ in items}
    assert kinds == {"state", "frame", "event"}
    assert sum(1 for _, k, p in items if k == "frame" and p["cam"] == CAM[:8]) == 20
    fast = list(rec.iter(speed=100, realtime=True, start=1.0, end=1.5))
    assert fast and all(1.0 <= t <= 1.5 for t, _, _ in fast)


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_replay_source_clock_and_shape(rec_dir):
    clk = FakeClock()
    src = ReplaySource(rec_dir, clock=clk)
    st = asyncio.run(src.read_state())
    assert sorted(st) == list(range(1, 9))
    assert set(st[1]) == {"raw", "goal", "lo", "hi", "norm", "in_range", "torque", "status", "temp", "volt", "load"}
    assert st[3]["raw"] == pos_at(3, 0.0) and st[3]["lo"] == LO and st[2]["load"] == 50
    clk.t += 1.0
    assert src.read_state_sync()[3]["raw"] == pos_at(3, 1.0)
    assert src.frame("wrist") == b"\xff\xd828ff0ca8-9\xff\xd9"  # frame 10 is at t=1.003: causal never peeks ahead
    src.set_speed(2.0)
    clk.t += 0.25
    assert src.now() == pytest.approx(1.5)
    src.pause()
    clk.t += 10
    assert src.now() == pytest.approx(1.5) and not src.done
    src.resume()
    clk.t += 10
    assert src.done and src.now() == pytest.approx(src.rec.duration)
    cams = {c["label"]: c for c in src.cameras()}
    assert set(cams) == {"wrist", "overview"} and cams["wrist"]["hash"] == CAM


def test_replay_source_from_spec_loop(rec_dir):
    clk = FakeClock()
    src = ReplaySource.from_spec(f"replay:{rec_dir}?speed=1&start=0.5&loop=1", clock=clk)
    assert src.loop and src.now() == pytest.approx(0.5)
    clk.t += src.rec.duration
    assert src.now() == pytest.approx(0.5, abs=1e-6) and not src.done


def test_replay_source_raises_like_read_state(tmp_path):
    d = build(tmp_path / "r", seconds=0.2)
    rows = (d / "state.jsonl").read_text().splitlines()
    bad = [json.loads(l) for l in rows]
    for r in bad:
        r["buses"][BUS]["j4"]["error"] = "timeout"
    (d / "state.jsonl").write_text("\n".join(json.dumps(r) for r in bad) + "\n")
    with pytest.raises(RuntimeError, match="no complete error-free read"):
        ReplaySource(d).read_state_sync()
