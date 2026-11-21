import json

from models import autolabel as al
from models.gemini_er import FakeTransport, GeminiER, Image
from recorder.replay import Recording
from recorder.tests.conftest import build


def test_sensor_pass_on_fixture(tmp_path):
    rec = Recording(build(tmp_path / "recordings", seconds=2.0))
    s = al.sensor_pass(rec)
    assert s["active"] is not None and s["action_source"] == "goal"      # every joint ramps, torque on
    assert s["camera_gaps"] == {"overview": [], "wrist": []}
    assert s["markers"] == [{"t": 0.5, "label": "grasp attempt"}]


def test_gemini_labels_are_cleaned_and_used(tmp_path, log_dir):
    rec = Recording(build(tmp_path / "recordings", seconds=2.0))
    reply = json.dumps({"task": "pick up the cup", "success": True, "keep": True, "why": "clean demo", "issues": [],
                        "segments": [{"t0": -1, "t1": 0.8, "label": "approach", "success": True},
                                     {"t0": 0.5, "t1": 1.5, "label": "grasp", "success": True},   # overlaps: clipped
                                     {"t0": 1.5, "t1": 9.0, "label": "teleport"},                  # unknown label
                                     {"t0": 1.5, "t1": 9.0, "label": "lift"}]})                     # clamped to duration
    er = GeminiER(transport=FakeTransport([reply]), log_dir=log_dir)
    lab = al.label(rec, er, every_s=0.5, max_frames=4)
    assert lab["gemini_error"] is None and lab["task"] == "pick up the cup" and lab["keep"] is True
    segs = lab["segments"]
    assert [s["label"] for s in segs] == ["approach", "grasp", "lift"]
    assert segs[0]["t0"] == 0.0 and segs[1]["t0"] == 0.8 and segs[2]["t1"] <= rec.duration + 1e-6
    assert lab["episodes"][0]["task"] == "pick up the cup" and lab["episodes"][0]["keep"] is True
    parts = er.transport.calls[0]["parts"]
    imgs = [p for p in parts if isinstance(p, Image)]
    assert len(imgs) == 8 and {i.role for i in imgs} == {"wrist", "overview"}
    assert all(i.ref.startswith(f"recording:{rec.id}@") for i in imgs)
    assert er.transport.calls[0]["config"]["thinking_level"] == "medium"


def test_gemini_drop_and_failure_paths(tmp_path, log_dir):
    rec = Recording(build(tmp_path / "recordings", seconds=2.0))
    er = GeminiER(transport=FakeTransport(['{"task": "x", "success": false, "keep": false, "why": "dropped it"}',
                                           "garbage"]), log_dir=log_dir)
    lab = al.label(rec, er)
    assert lab["keep"] is False and any("dropped it" in r for r in lab["reasons"])
    lab2 = al.label(rec, er, task_hint="wipe")
    assert lab2["gemini_error"] and lab2["keep"] is True and lab2["task"] == "wipe"     # sensors alone pass
