"""End-to-end on a recording (replay mode) and the real-arm backend against fakes (nothing reaches a bus)."""
import time

import pytest

from executor.backends.base import MotionDisabled
from executor.backends.replay import ReplayBackend
from executor.backends.station import StationBackend
from executor.contracts import Status
from executor.core import Executor
from executor.perception import Locator
from executor.recording import RecorderLink
from executor.tests.conftest import proposal
from recorder.fmt import decode_motor
from recorder.replay import Recording
from recorder.tests.conftest import build, regs


def test_replay_end_to_end(kin, tmp_path):
    rec_dir = build(tmp_path / "fixture", seconds=2.0)
    backend = ReplayBackend(rec_dir, kin=kin)
    link = RecorderLink("local", root=tmp_path / "rec")
    ex = Executor(backend, kin, recorder=link, locator=Locator(), table_memory_path=tmp_path / "t.json")
    ex.start_task("replay-e2e", prompt="open the jaws halfway", advisor="script")
    backend.advance(0.5)
    obs = ex.observe()
    assert set(obs.frames) == {"wrist", "overview"}
    assert all(not f.stale for f in obs.frames.values())
    assert obs.frames["wrist"].seq == 4                   # Station frame index of the last frame at or before t=0.5
    assert obs.frames["wrist"].joints_norm is not None   # joints paired with the frame at capture
    eid = ex.submit(proposal(ex, obs, "gripper", {"open_fraction": 0.6}, source="script"))
    r = ex.wait(eid)
    assert r.status == Status.SUCCEEDED, r.termination
    assert abs(r.achieved["gripper_norm"] - 0.6) < 0.03
    rid = ex.rec.id
    ex.end_task("done")
    labels = [m["label"] for m in Recording(tmp_path / "rec" / rid).markers()]
    for want in ("task start", "proposal", "validation", "phase execute start", "phase verify SUCCEEDED", "result", "task end"):
        assert want in labels, (want, labels)
    assert r.recording_id == rid and r.markers


class FakeCapture:
    """The parts of recorder.live.LiveCapture StationBackend reads."""

    def __init__(self, bus="5B61033301"):
        import threading
        self.lock = threading.Lock()
        self.last_state_mono = time.monotonic()
        self.cams, self.loop = {}, None
        self.motors = {bus: {m: decode_motor(m, regs(2000 + m, goal=2000 + m), 1000, 3000, True, None, 0) for m in range(1, 9)}}

    def snapshot(self):
        return {"connected": True}


def test_station_backend_is_read_only_unless_gated(kin, tmp_path, monkeypatch):
    writes = []
    monkeypatch.delenv("EXECUTOR_REAL_MOTION", raising=False)
    b = StationBackend(capture=FakeCapture(), writer=lambda *a: writes.append(a))
    st = b.read_state().motors
    assert st[3]["raw"] == 2003 and st[3]["lo"] == 1000
    assert not b.can_move
    assert b.enable_motion(True).startswith("refused") and not b.can_move
    with pytest.raises(MotionDisabled):
        b.write_goals({1: 0.5})
    with pytest.raises(MotionDisabled):
        b.hold()
    ex = Executor(b, kin, recorder=RecorderLink("local", root=tmp_path / "rec"), locator=Locator())
    b.cap.last_state_mono = time.monotonic()
    obs = ex.observe()
    r = ex.status(ex.submit(proposal(ex, obs, "gripper", {"open_fraction": 0.5}, ttl=5)))
    assert r.status == Status.REJECTED and r.validation.checks["motion_enabled"]["result"] == "fail"
    assert writes == [], "nothing may reach the bus"


def test_station_backend_goal_conversion_when_enabled(monkeypatch):
    writes = []
    monkeypatch.setenv("EXECUTOR_REAL_MOTION", "1")
    b = StationBackend(capture=FakeCapture(), writer=lambda *a: writes.append(a))
    assert b.enable_motion(True) == "arm enabled" and b.can_move
    b.write_goals({2: 0.25, 8: 1.0})
    addr, values, width = writes[-1]
    assert addr == 42 and width == 2
    assert values[2] == 1500 and values[8] == 3000            # lo + n * (hi - lo)
    assert values[5] == 2005                                   # untouched motors keep their goal register
    b.enable_motion(False)
    with pytest.raises(MotionDisabled):
        b.write_goals({2: 0.3})
    assert len(writes) == 1
