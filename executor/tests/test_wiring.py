"""The streams wired into the executor (fake backend, no hardware): one set of contracts, skills/wipe proposals and
the skills/eval outcome, perception.locate observations, Gemini proposals, twin preflight, pick_at / drop_at."""
import json
from pathlib import Path

import numpy as np
import pytest

from executor.backends import FakeBackend
from executor.contracts import Status
from executor.core import Executor
from executor.kinematics import load_scene_calib
from executor.perception import PerceptionLocator
from executor.recording import RecorderLink
from executor.skills import Plan
from executor.tests.conftest import proposal
from executor.tests.scenes import BLUE, render_overview
from executor.twin import TwinPreflight

ROOT = Path(__file__).resolve().parents[2]
CAP = (0.24, -0.06)                     # world xy of the blue cap in the synthetic overview image
BIN = {"xyz_world": [0.26, 0.12, 0.0], "size": [0.12, 0.10, 0.06], "yaw_deg": 0.0}


@pytest.fixture
def wex(kin, park, tmp_path):
    made = []

    def _make(images=None, **fake_kw):
        fake = FakeBackend(start_norm=dict(park), kin=kin, images=images, **fake_kw)
        ex = Executor(fake, kin, recorder=RecorderLink("local", root=tmp_path / "rec"), locator=PerceptionLocator(load_scene_calib()),
                      table_memory_path=tmp_path / "table.json")
        ex.twin._ready.wait(10)
        made.append(ex)
        return ex, fake
    yield _make
    for ex in made:
        ex.cancel()
        ex.end_task("test end")


# --- 1. one set of contracts ---------------------------------------------------------------------------------------
def test_one_set_of_contracts():
    import executor.contracts as ec
    import models._contracts as mc
    import skills._contracts as sc
    for name in ("ActionProposal", "Observation", "ObjectEstimate", "Outcome", "ExecutionResult", "Status", "ValidationResult",
                 "JointState", "FrameRef"):
        assert getattr(sc, name) is getattr(ec, name) and getattr(mc, name) is getattr(ec, name), name
    assert (ROOT / "skills/_contracts_vendored.py").read_bytes() == (ROOT / "executor/contracts.py").read_bytes(), \
        "skills/_contracts_vendored.py must be a byte-identical copy of executor/contracts.py (cp it)"


def test_skills_wipe_proposal_accepted_unchanged(wex):
    from skills.wipe import WipeSkill
    ex, fake = wex(table_z=0.008)
    sk = WipeSkill()
    v = sk.seed_version(w_hat=0.03)
    plan = sk.plan(v, {"region": (0.22, -0.07, 0.34, 0.03), "budget": 6})
    obs = ex.observe()
    props = sk.execute(plan, based_on_obs=obs.obs_id, setup_id=obs.setup_id, now=ex.now(), version=v.version, plan_id="p1")
    r = ex.wait(ex.submit(props[0]), 300)
    assert r.validation.ok, r.validation.reasons
    assert r.validation.checks["skill"]["proposer_version"] == v.version          # skills' version tag: recorded, not enforced
    assert r.status == Status.SUCCEEDED, (r.termination, r.phases)
    assert [p["name"] for p in r.phases] == ["plan", "validate", "execute", "verify", "recover"]
    assert r.achieved["contact_z"][0] == pytest.approx(0.008, abs=0.003)
    (a, b), = r.achieved["strokes_world"]
    assert np.allclose(a, props[0].params["from_xy_world"], atol=0.006) and np.allclose(b, props[0].params["to_xy_world"], atol=0.006)
    assert r.validation.checks["twin_preflight"]["result"] == "pass", r.validation.checks["twin_preflight"]
    late = ex.status(ex.submit(props[-1]))                     # the plan's park, on the 13 s old observation: too old
    assert late.status == Status.REJECTED and late.validation.checks["freshness"]["result"] == "fail"
    obs = ex.observe()
    props = sk.execute(plan, based_on_obs=obs.obs_id, setup_id=obs.setup_id, now=ex.now(), version=v.version, plan_id="p1")
    park = ex.wait(ex.submit(props[-1]), 300)                  # skill park with the wipe version tag
    assert park.validation.ok and park.status == Status.SUCCEEDED, (park.validation.reasons, park.termination)


def test_wipe_limits_must_match_the_executor(wex):
    from skills.wipe import WipeSkill
    ex, fake = wex(table_z=0.008)
    sk = WipeSkill(limits={**WipeSkill().limits, "hard_load_sum": 900})
    v = sk.seed_version(w_hat=0.03)
    plan = sk.plan(v, {"region": (0.22, -0.07, 0.34, 0.03), "budget": 6})
    obs = ex.observe()
    r = ex.status(ex.submit(sk.execute(plan, based_on_obs=obs.obs_id, setup_id=obs.setup_id, now=ex.now(), version=v.version)[0]))
    assert r.status == Status.REJECTED and r.validation.checks["limits"]["result"] == "fail"
    assert fake.motion_writes() == []


def _gemini(ex, replies, tmp_path):
    from models import advisor_gemini as ag
    from models.gemini_er import FakeTransport, GeminiER

    def frames(ref, role):
        obs_id = ref.split("/")[3]
        return ex.frame_jpeg(obs_id, role)
    return ag.GeminiAdvisor(er=GeminiER(transport=FakeTransport(replies), log_dir=tmp_path / "gem"), frame_loader=frames)


def _yx(xyz):
    from perception.locate import world_to_pixel
    u, v = world_to_pixel("overview", xyz)["pixel"]
    return [round(v / 224 * 1000), round(u / 299 * 1000)]


def test_models_gemini_proposal_accepted_unchanged(wex, tmp_path):
    ex, fake = wex(images={"overview": render_overview([(CAP, 0.016, 0.02, BLUE)])}, object_norm=0.35)
    adv = _gemini(ex, [json.dumps({"action": "pick", "args": {"object": "blue cap", "point": _yx([*CAP, 0.0]), "camera": "overview"},
                                   "rationale": "the blue cap", "confidence": 0.9})], tmp_path)
    obs = ex.observe()
    (p,) = adv.propose(obs, "take the blue object and put it in the bin")
    assert p.skill == "pick_at" and p.source == "gemini-er" and np.allclose(p.params["xyz_world"][:2], CAP, atol=0.01)
    r = ex.wait(ex.submit(p), 300)                       # the models record, unchanged
    assert r.validation.ok, r.validation.reasons
    assert r.status == Status.SUCCEEDED and r.achieved["holding"] is True, r.termination


@pytest.mark.parametrize("moved", [0.0, 0.06])
def test_slow_advisor_answer_is_relocalised(wex, tmp_path, moved):
    ex, fake = wex(images={"overview": render_overview([(CAP, 0.016, 0.02, BLUE)])}, object_norm=0.35)
    adv = _gemini(ex, [json.dumps({"action": "pick", "args": {"object": "blue cap", "point": _yx([*CAP, 0.0]), "camera": "overview"}})],
                  tmp_path)
    obs = ex.observe()
    (p,) = adv.propose(obs, "pick the blue cap")
    fake.advance(5.0)                                             # the model took 5 s; meanwhile the cap may have moved
    fake.set_image("overview", render_overview([((CAP[0], CAP[1] + moved), 0.016, 0.02, BLUE)]))
    r = ex.status(ex.submit(p))
    per = r.validation.checks["perception"]
    assert per["relocalised"]["from_obs"] == obs.obs_id and per["relocalised"]["moved_mm"] == pytest.approx(moved * 1000, abs=6)
    if moved:
        assert r.status == Status.REJECTED and "moved" in r.termination and fake.motion_writes() == []
    else:
        assert r.validation.ok and ex.wait(r.execution_id, 300).status == Status.SUCCEEDED


# --- 4. Gemini advisor -------------------------------------------------------------------------------------------------
def _gemini_advisor(ex, adv=None, **kw):
    from executor.advisor import make_advisor
    from executor.core import EventLog
    ev = EventLog()
    a = make_advisor("gemini", "http://127.0.0.1:0", ev, {k: s.describe() for k, s in ex.skills.items()}, {}, executor=ex,
                     advisor=adv, **kw)
    a.task = "take the blue object and put it in the bin"
    return a, ev


def test_gemini_advisor_proposes_through_the_executor_and_judges(wex, tmp_path):
    ex, fake = wex(images={"overview": render_overview([(CAP, 0.016, 0.02, BLUE)])}, object_norm=0.35)
    adv = _gemini(ex, [json.dumps({"action": "pick", "args": {"object": "blue cap", "point": _yx([*CAP, 0.0]), "camera": "overview"},
                                   "rationale": "grasp the cap first"}),
                       json.dumps({"done": True, "confidence": 0.9, "evidence": "cap between the jaws"})], tmp_path)
    a, ev = _gemini_advisor(ex, adv)
    why = a.run()
    assert why.startswith("done"), why
    kinds = [e["kind"] for e in ev.since(0)]
    assert "advisor_tool_result" in kinds and "advisor_outcome" in kinds
    (r,) = [ex.status(e) for e in ex.results]
    assert r.skill == "pick_at" and r.status == Status.SUCCEEDED and ex.proposals[r.execution_id].source == "gemini-er"


def test_gemini_advisor_without_key_reports_and_ends(wex, monkeypatch, tmp_path):
    ex, fake = wex()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("MODELS_SECRETS_FILE", str(tmp_path / "none.env"))
    a, ev = _gemini_advisor(ex)
    why = a.run()
    assert why == "disabled (no GEMINI_API_KEY)"
    msgs = [e["text"] for e in ev.since(0) if e["kind"] == "advisor_message"]
    assert msgs and "GEMINI_API_KEY" in msgs[0] and fake.motion_writes() == []


def test_gemini_advisor_with_rejected_key_reports_and_ends(wex, tmp_path):
    from models import advisor_gemini as ag
    from models.gemini_er import GeminiER, GeminiError
    ex, fake = wex()

    def reject(model, parts, config):
        raise GeminiError("ClientError: 400 INVALID_ARGUMENT. API key not valid. Please pass a valid API key.")
    adv = ag.GeminiAdvisor(er=GeminiER(transport=reject, log_dir=tmp_path / "gem"), frame_loader=lambda ref, role: ex.frame_jpeg(ref.split("/")[3], role))
    a, ev = _gemini_advisor(ex, adv)
    assert a.run() == "disabled (key rejected)"
    assert any("rejected the API key" in e.get("text", "") for e in ev.since(0)) and fake.motion_writes() == []
    assert "API key not valid" in a.last_error


# --- 3. perception -> observations ------------------------------------------------------------------------------------
def test_observation_has_blue_object_from_perception(wex):
    ex, fake = wex(images={"overview": render_overview([(CAP, 0.016, 0.02, BLUE)])})
    obs = ex.observe()
    (o,) = obs.objects
    assert o.name == "blue object" and o.visible and o.uncertainty_mm and 2 < o.uncertainty_mm < 25
    assert np.allclose(o.xyz_world[:2], CAP, atol=0.004) and "perception.locate" in o.source and "blue" in o.source
    assert 0.025 < o.size[0] < 0.045
    fake.frozen.add("overview")                                   # frozen camera: last estimate kept, but not visible
    ex.observe()
    fake.advance(2.0)
    stale = ex.observe()
    assert stale.objects and not stale.objects[0].visible and "memory" in stale.objects[0].source


# --- 6. pick_at / drop_at -----------------------------------------------------------------------------------------------
def test_pick_blue_then_drop_in_bin(wex):
    ex, fake = wex(images={"overview": render_overview([(CAP, 0.016, 0.02, BLUE)])}, object_norm=0.35)
    obs = ex.observe()
    r = ex.wait(ex.submit(proposal(ex, obs, "pick_at", {"object": "blue"})), 300)
    assert r.status == Status.SUCCEEDED, (r.termination, r.validation.reasons)
    per = r.validation.checks["perception"]
    assert per["result"] == "pass" and np.allclose(per["object"]["xyz_world"][:2], CAP, atol=0.004)
    assert r.achieved["holding"] is True and r.achieved["object_estimate"]["uncertainty_mm"] > 0
    assert r.validation.checks["twin_preflight"]["result"] == "pass"
    assert ex.evaluate(r.execution_id).success is True

    obs = ex.observe()
    d = ex.wait(ex.submit(proposal(ex, obs, "drop_at", {"bin": BIN, "object": "blue"})), 300)
    assert d.status == Status.SUCCEEDED, (d.termination, d.validation.reasons)
    t = d.validation.checks["target"]
    assert t["release_z"] == pytest.approx(0.06 + 0.06) and t["centre_xy"] == BIN["xyz_world"][:2]
    pf = d.validation.checks["twin_preflight"]
    assert pf["result"] == "pass" and pf["bin"]["size"] == BIN["size"] and pf["seconds"] < 2.0
    assert d.achieved["release_over_opening"] is True
    o = ex.evaluate(d.execution_id)                               # the image still shows the cap on the table: outside
    assert o.metric == "object_in_region" and o.success is False
    fake.set_image("overview", render_overview([(BIN["xyz_world"][:2], 0.016, 0.02, BLUE)]))
    assert ex.evaluate(d.execution_id).success is True           # seen inside the bin footprint


def test_pick_at_unknown_colour_is_rejected_without_motion(wex):
    ex, fake = wex(images={"overview": render_overview([])})
    obs = ex.observe()
    r = ex.status(ex.submit(proposal(ex, obs, "pick_at", {"object": "blue"})))
    assert r.status == Status.REJECTED and r.validation.checks["perception"]["result"] == "fail"
    assert "not found" in r.termination and fake.motion_writes() == []


def test_drop_at_from_observed_bin(wex, monkeypatch):
    from executor.perception import Hint
    ex, fake = wex(images={"overview": render_overview([((0.26, 0.12), 0.05, 0.06, (40, 40, 200))])})   # a red round bin
    ex.locator.hints = [Hint("red bin", "red", 0.06, 0.01, role="bin")]
    obs = ex.observe()
    (b,) = obs.objects
    r = ex.status(ex.submit(proposal(ex, obs, "drop_at", {})))
    assert r.validation.ok, r.validation.reasons
    assert np.allclose(r.validation.checks["target"]["centre_xy"], [0.26, 0.12], atol=0.006)
    assert r.validation.checks["twin_preflight"]["bin"]["from"].startswith("observation")
    ex.wait(r.execution_id, 300)


# --- 5. twin preflight ----------------------------------------------------------------------------------------------------
def test_preflight_passes_with_numbers(wex):
    ex, _ = wex()
    obs = ex.observe()
    p = proposal(ex, obs, "move_tcp", {"xyz_world": [0.25, 0.05, 0.10]})
    pf = ex.validate_in_sim(p)
    assert pf["result"] == "pass" and pf["layout"] == "home" and pf["samples"] > 0 and pf["seconds"] < 2.0
    assert pf["min_clearance_mm"] > 0 and pf["reach"]["max_tcp_mismatch_mm"] < 1.0


def test_preflight_rejects_path_through_bin_wall(wex):
    ex, fake = wex()
    obs = ex.observe()
    tall = {"xyz_world": [0.26, 0.0, 0.0], "size": [0.05, 0.24, 0.12], "yaw_deg": 0.0}   # 12 cm walls across the way
    p = proposal(ex, obs, "move_tcp", {"xyz_world": [0.33, 0.0, 0.06], "bin": tall})    # travel height is 8 cm
    r = ex.status(ex.submit(p))
    pf = r.validation.checks["twin_preflight"]
    assert r.status == Status.REJECTED and pf["result"] == "fail" and "preflight_bin" in pf["why"], pf
    assert pf["collisions"][0]["path"] == "move" and pf["collisions"][0]["depth_mm"] > 1
    assert fake.motion_writes() == []
    p2 = proposal(ex, ex.observe(), "move_tcp", {"xyz_world": [0.33, 0.0, 0.06]})      # without the bin: passes
    assert ex.validate_in_sim(p2)["result"] == "pass"


def test_preflight_names_the_marker_under_a_table_hit(wex, kin):
    import build_scene
    ex, _ = wex()
    item = next(i for i in json.loads(build_scene.layout_path("home").read_text())["markers"]["items"] if i["id"] == 22)
    xy = kin.to_world(build_scene.marker_pose(item)[:3, 3])[:2]
    q0 = ex.motion.q_now()
    q, _ = kin.solve([xy[0], xy[1], -0.015], q0)
    assert q is not None
    pf = ex.twin(None, Plan({"move": [q]}), None, q0)
    assert pf["result"] == "fail" and "marker 22" in pf["why"], pf


def test_preflight_skipped_when_twin_unavailable_or_out_of_time(kin):
    pf = TwinPreflight(kin, layout="no-such-layout", warm=False)
    plan = Plan({"move": [np.zeros(len(kin.ids))]})
    r = pf(None, plan)
    assert r["result"] == "skipped" and "not available" in r["why"]
    r = TwinPreflight(kin, budget_s=0.0, warm=False)(None, Plan({"move": [np.r_[np.full(kin.na, 0.3), 0.0]]}), None,
                                                     np.zeros(len(kin.ids)))
    assert r["result"] == "skipped" and "time budget" in r["why"]


# --- 2. wipe: executed through the phases, scored by skills/eval ----------------------------------------------------------
def test_executor_overview_camera_matches_calibration_with_distortion():
    from executor.evaluator import _camera
    from executor.perception import camera_info_from_scene
    from skills.eval import Camera
    sc = load_scene_calib()
    ref = Camera.from_scene_calib(sc)
    info = camera_info_from_scene(sc)["overview"]
    assert np.allclose(info["dist"][:5], sc["cameras"]["overview"]["dist"][:5])
    pts = np.array([[x, y, 0.0] for x in (0.15, 0.3, 0.45) for y in (-0.15, 0.0, 0.15)])
    uv = ref.project(pts)
    assert np.abs(_camera(info).project(pts) - uv).max() < 1e-6


def test_wipe_plan_scored_by_skills_eval(wex):
    from skills.eval import Camera, synth
    from skills.wipe import WipeSkill
    ex, fake = wex(table_z=0.008)
    cam = Camera.from_scene_calib(load_scene_calib())
    world = synth.SynthWorld().add_stain_rect(synth.TARGET, seed=1)
    fake.set_image("overview", synth.to_jpeg(world.render(cam, seed=1)))
    sk = WipeSkill()
    v = sk.seed_version(w_hat=0.03)
    plan = sk.plan(v, {"region": synth.TARGET, "budget": 6})
    results = []
    for i in range(len(plan.strokes) + 1):                         # every stroke proposed from a fresh observation
        obs = ex.observe()
        p = sk.execute(plan, based_on_obs=obs.obs_id, setup_id=obs.setup_id, now=ex.now(), version=v.version, plan_id="plan-1")[i]
        results.append(ex.wait(ex.submit(p), 300))
        assert results[-1].status == Status.SUCCEEDED, (results[-1].skill, results[-1].termination)
        if i == 0:
            o = ex.evaluate(results[0].execution_id)              # arm still above the board: unknown, not a guess
            assert o.metric == "contamination_reduction" and o.success is None
    done = [s for r in results[:-1] for s in r.achieved["strokes_world"]]
    world.wipe(done, w_true=0.03)
    fake.set_image("overview", synth.to_jpeg(world.render(cam, seed=2)))
    o = ex.evaluate(results[-1].execution_id)                     # after the plan's park: the board is in clear view
    assert o.metric == "contamination_reduction" and o.success is True and o.value >= 0.9, o
    assert any("skills/eval" in e for e in o.evidence)
