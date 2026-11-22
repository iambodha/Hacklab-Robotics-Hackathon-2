"""Episode store: bounds, the contract's records, history text, reports, recorder sessions."""
import json
import threading

import pytest

from learning import episodes as E


class FakeRecorder(E.RecorderClient):
    def __init__(self):
        self.calls = []
        self.enabled = True

    def start(self, name, note=""):
        self.calls.append(("start", name))
        return f"rec-{name}", ""

    def marker(self, rec_id, label, data=None):
        self.calls.append(("marker", rec_id, label))
        return ""

    def stop(self, rec_id):
        self.calls.append(("stop", rec_id))
        return ""


@pytest.fixture
def store(tmp_path):
    return E.EpisodeStore(tmp_path / "learn", recorder=FakeRecorder())


def test_validate_rejects_out_of_range_without_clamping():
    full, changed, errors = E.validate({"twist_deg_per_stroke": 200, "strokes": 2.5, "twist_speed": "fast",
                                        "grip_torque_limit": 0.9, "bogus": 1})
    assert len(errors) == 5
    assert any("twist_deg_per_stroke=200 is outside its bounds 15.0..120.0" in e for e in errors)
    assert any("whole number" in e for e in errors)
    assert any("bogus" in e for e in errors)
    assert full["twist_deg_per_stroke"] == E.defaults()["twist_deg_per_stroke"]    # nothing clamped in
    full, changed, errors = E.validate({"twist_deg_per_stroke": 120, "strokes": 8, "approach_z_offset_mm": -15})
    assert not errors and set(changed) == {"twist_deg_per_stroke", "strokes", "approach_z_offset_mm"}
    assert isinstance(full["strokes"], int)


def test_safety_limits_cap_the_learnable_bounds():
    assert E.PARAMS["grip_torque_limit"][2] == E.SAFETY["max_grip_force"]
    assert E.PARAMS["j7_load_abort"][2] == E.SAFETY["j7_load_hard"]
    assert E.validate({"j7_load_abort": E.SAFETY["j7_load_hard"] + 1})[2]


def test_begin_end_records_follow_the_contract(store):
    r = store.begin({"twist_deg_per_stroke": 30}, "small strokes keep the grip", images={"wrist": b"\xff\xd8x"})
    assert r["ok"] and r["attempt"] == 1 and r["recording_id"] == "rec-bottle-attempt-1"
    assert (store.root / "attempt_1/before_wrist.jpg").read_bytes() == b"\xff\xd8x"
    assert store.begin({}, "again")["ok"] is False                     # attempt 1 still open
    stroke = {"stroke": 1, "commanded_deg": 30, "achieved_deg": 22.5, "peak_j7_load": 310, "mean_j7_load": 200,
              "j8_change_ticks": -4, "z_change_mm": 0.4, "slip": False, "abort_reason": None}
    store.phase(1, "twist", "ok", {**stroke, "strokes": [stroke]}, "stroke 1")
    store.phase(1, "check", "ok", {"cap_free": False, "cap_lift_mm": 0.0})
    with pytest.raises(ValueError):
        store.phase(1, "dance", "ok")
    assert store.end("great", "x")["ok"] is False                      # unknown outcome
    assert store.end("loosened", "")["ok"] is False                    # lesson required
    assert store.end("loosened", "lesson", {"strokes": 99})["ok"] is False
    e = store.end("loosened", "22 deg per stroke with no slip", {"strokes": 5})
    assert e["ok"]
    recs = [json.loads(l) for l in store.log.read_text().splitlines()]
    assert [r["type"] for r in recs] == ["episode_begin", "phase", "phase", "episode_end"]
    b, end = recs[0], recs[-1]
    for k in ("type", "attempt", "t", "params", "hypothesis", "based_on", "recording_id"):
        assert k in b
    for k in ("outcome", "rotation_achieved_deg", "cap_lift_mm", "measurements", "lesson", "next_change"):
        assert k in end
    assert end["rotation_achieved_deg"] == 22.5 and end["cap_lift_mm"] == 0.0
    assert ("stop", "rec-bottle-attempt-1") in store.recorder.calls
    assert ("marker", "rec-bottle-attempt-1", "twist:ok") in store.recorder.calls


def test_at_most_three_changes_and_inheritance(store):
    store.begin({"strokes": 2}, "h1")
    store.end("no_progress", "nothing turned")
    r = store.begin({"strokes": 3, "twist_deg_per_stroke": 60, "grip_torque_limit": 0.5, "lift_per_turn_mm": 2}, "h2")
    assert not r["ok"] and "change at most 3" in r["errors"][-1]
    r = store.begin({"twist_deg_per_stroke": 60}, "h2")
    assert r["ok"] and r["params"]["strokes"] == 2 and r["changed"] == ["twist_deg_per_stroke"]
    assert r["attempt"] == 2
    assert store.attempts()[2]["begin"]["based_on"] == [1]


def test_history_report_and_summary(store):
    assert "No attempts yet" in store.history_text()
    for n, (rot, outcome) in enumerate([(10, "slip"), (40, "loosened"), (95, "loosened"), (380, "opened")], 1):
        store.begin({"twist_deg_per_stroke": 15 + 20 * n}, f"hypothesis {n}", images={"wrist": b"jpg"})
        store.phase(n, "grasp", "ok", {"j8_norm": 0.6}, "touch: jaws at 0.6")
        store.end(outcome, f"lesson {n}", {"strokes": 4}, rotation_achieved_deg=rot, images={"wrist": b"jpg2"})
    h = store.history_text(last_n=2)
    assert "## attempt 3" in h and "## attempt 1" not in h and "lesson 4" in h
    assert "OUTCOME: opened" in h and "Trend:" in h and "improving" in h
    rep = (store.root / "attempt_2/report.md").read_text()
    assert "# Attempt 2: loosened" in rep and "hypothesis 2" in rep and "before_wrist.jpg" in rep
    assert "twist_deg_per_stroke *" in rep                         # changed parameter marked
    summ = (store.root / "SUMMARY.md").read_text()
    assert summ.count("[report](attempt_") == 4 and "10 -> 40 -> 95 -> 380" in summ
    assert "recordings/rec-bottle-attempt-4/viewer" in summ


def test_torn_last_line_is_skipped(store):
    store.begin({}, "h")
    with store.log.open("a") as f:
        f.write('{"type": "phase", "attem')
    assert store.open_attempt() == 1


def test_recorder_client_against_the_real_service(tmp_path):
    from recorder.server import Service, make_server
    svc = Service(root=tmp_path / "rec")
    httpd = make_server(svc, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        rc = E.RecorderClient(f"http://127.0.0.1:{httpd.server_address[1]}")
        st = E.EpisodeStore(tmp_path / "learn", recorder=rc)
        r = st.begin({}, "recorded attempt")
        assert r["recording_id"] and svc.active is not None
        st.phase(1, "approach", "ok", {}, "hover")
        st.end("no_progress", "it was recorded")
        assert svc.active is None
        rec = svc.recording(r["recording_id"])
        labels = [m.get("label") for m in rec.markers()]
        assert "approach:ok" in labels and "end:no_progress" in labels
        # a busy recorder does not stop the attempt
        svc.start("someone-else")
        r2 = st.begin({}, "recorder busy")
        assert r2["ok"] and r2["recording_id"] is None and "409" in r2["note"]
    finally:
        httpd.shutdown()
    dead = E.RecorderClient("http://127.0.0.1:9", timeout=0.5)
    assert dead.start("x")[0] is None and "unreachable" in dead.start("x")[1]
