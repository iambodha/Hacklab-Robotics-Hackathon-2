import json

import pytest

from models import advisor_gemini as ag
from models._contracts import FrameRef, JointState, Observation
from models.gemini_er import FakeTransport, GeminiER

from .conftest import tiny_jpeg

T_OBS = 1000.0


def obs(stale_wrist=False) -> Observation:
    js = JointState(names=[f"j{i}" for i in range(1, 9)], raw=[2000] * 8, rad=[0.0] * 8, norm=[0.5] * 8, goal=[0] * 8,
                    load=[0] * 8, current=None, temp=[30] * 8, torque=[0] * 8, status=[0] * 8, t_mono=T_OBS)
    fr = {"overview": FrameRef("usbvideo/9ea4cacd", 10, T_OBS, "mem:overview"),
          "wrist": FrameRef("usbvideo/28ff0ca8", 11, T_OBS, "mem:wrist", stale=stale_wrist)}
    return Observation("obs-1", T_OBS, 1.8e9, "setup-abc", fr, js, [], 0.1, "rec-1")


def advisor(replies, log_dir, **kw) -> ag.GeminiAdvisor:
    er = GeminiER(transport=FakeTransport(replies), log_dir=log_dir)
    return ag.GeminiAdvisor(er=er, frame_loader=lambda ref, role: tiny_jpeg(), **kw)


@pytest.fixture
def fake_world(monkeypatch):
    def p2w(cam, u, v, z=0.0, joints=None):
        return {"ok": True, "xyz": [round(u / 1000, 4), round(v / 1000, 4), z], "sigma_mm": 6.0}
    monkeypatch.setattr(ag.pb, "pixel_to_world", p2w)


def test_pick_proposal_fields_and_expiry(log_dir, fake_world):
    adv = advisor(['{"action": "pick", "args": {"object": "purple wrapper", "point": [500, 500], "camera": "overview"},'
                   ' "rationale": "nearest free object", "confidence": 0.9}'], log_dir)
    (p,) = adv.propose(obs(), "put the wrapper in the box")
    assert p.source == "gemini-er" and p.based_on_obs == "obs-1" and p.setup_id == "setup-abc"
    assert p.skill == "pick_at" and p.skill_version.startswith("gemini-advisor/")
    assert p.expires_at == T_OBS + ag.TTL_S["pick_at"]
    assert p.params["xyz_world"] == [0.1495, 0.112, 0.0] and p.params["pixel"] == [149.5, 112.0]
    assert p.params["frame_ref"] == "mem:overview" and "world" in p.params["xyz_frame"]
    assert "nearest free object" in p.rationale and "gemini-robotics-er-2-preview" in p.rationale
    assert any("setup_id" in c for c in p.preconditions)
    api = ag.to_api(p, now_mono=T_OBS + 2)
    assert api["ttl_s"] == pytest.approx(ag.TTL_S["pick_at"] - 2) and json.dumps(api)


def test_pixel_only_proposal_when_perception_unavailable(log_dir):
    adv = advisor(['{"action": "retry_grasp", "args": {"object": "cup", "point": [100, 900], "camera": "wrist"}}'],
                  log_dir, use_perception=False)
    (p,) = adv.propose(obs(), "pick the cup")
    assert p.skill == "pick_at" and p.params["retry"] is True and p.params["needs_localisation"] is True
    assert "xyz_world" not in p.params and p.params["camera"] == "wrist"
    assert any("localises pixel" in c for c in p.preconditions)


def test_wipe_region_proposal_has_world_region_and_strokes(log_dir, fake_world):
    adv = advisor(['{"action": "wipe_region", "args": {"box_2d": [400, 300, 600, 700], "camera": "overview"},'
                   ' "rationale": "stain left"}'], log_dir, stroke_spacing_m=0.01)
    (p,) = adv.propose(obs(), "wipe the stain")
    assert p.skill == "wipe_region" and p.expires_at == T_OBS + ag.TTL_S["wipe_region"]
    (x0, y0), (x1, y1) = p.params["region_world_xy"]
    assert x0 < x1 and y0 < y1 and p.params["area_cm2"] > 0
    lines = p.params["lines_world_xy"]
    assert len(lines) >= 2 and all(len(l) == 4 for l in lines)
    assert p.params["stroke_spacing_m"] == 0.01


def test_stale_frames_skipped_and_errors_give_no_proposal(log_dir):
    adv = advisor(["not json at all"], log_dir)
    assert adv.propose(obs(stale_wrist=True), "x") == []
    assert "no JSON" in adv.last_error
    assert len(adv.er.transport.calls[0]["parts"]) == 3          # one label + one image + prompt: wrist was stale
    adv2 = advisor(['{"action": "dance"}'], log_dir)
    assert adv2.propose(obs(), "x") == [] and "vocabulary" in adv2.last_error


def test_judge_outcome(log_dir):
    adv = advisor(['{"done": true, "confidence": 0.9, "evidence": "table clean"}',
                   '{"done": true, "confidence": 0.3, "evidence": "maybe"}'], log_dir)
    o = adv.judge(obs(), "exe-1", "wipe the stain")
    assert o.success is True and o.metric == "gemini_task_done" and o.value == 0.9
    assert any(e.startswith("gemini:gem-") for e in o.evidence)
    assert adv.judge(obs(), "exe-2", "wipe").success is None      # low confidence -> unknown


def test_residual_area_outcome(log_dir, fake_world):
    adv = advisor(['[{"box_2d": [0, 0, 1000, 1000], "label": "stain"}]', "[]"], log_dir)
    o = adv.residual(obs(), "exe-1", max_area_cm2=1.0)
    assert o.unit == "cm2" and o.value == pytest.approx(299 * 224 / 1e6 * 1e4, rel=1e-3) and o.success is False
    assert adv.residual(obs(), "exe-2").value == 0 and adv.residual is not None


def test_wipe_lines_and_spacing():
    lines = ag.wipe_lines([[0.2, -0.05], [0.3, 0.05]], 0.02)
    assert len(lines) == 6 and lines[0] == [0.2, -0.05, 0.3, -0.05]
    long = ag.wipe_lines([[0.0, 0.0], [0.6, 0.02]], 0.03)
    assert all(abs(l[2] - l[0]) <= ag.WIPE_MAX_LINE_M + 1e-9 for l in long)
    assert ag.adapt_spacing(0.03, 0.4) < 0.03 < ag.adapt_spacing(0.03, 0.0)
    assert ag.adapt_spacing(0.03, None) == 0.03 and ag.adapt_spacing(0.011, 0.9) == 0.01


def test_load_frame_from_recording(tmp_path, monkeypatch):
    from recorder.tests.conftest import build
    d = build(tmp_path / "recordings")
    monkeypatch.setattr(ag, "RECORDING_DIRS", [tmp_path / "recordings"])
    data = ag.load_frame(f"recording:{d.name}@1.0", "wrist")
    assert data.startswith(b"\xff\xd8") and b"28ff0ca8-" in data
