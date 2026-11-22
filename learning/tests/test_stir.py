"""The stir skill's twin: the scene (learning/stir/twin.py), the judge (learning/stir/judge.py), the records
(learning/stir/store.py), the practice policy (learning/stir/practice.Policy) and one short episode end to end.

The episode runs in a subprocess: sim_arm_mcp builds one twin per process at import (see test_sim_loop.py)."""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TASK = json.loads((ROOT / "learning/stir/task.json").read_text())
GEO = {"inner_radius": 0.036, "rim_height": 0.095, "wall": 0.004, "liquid_level": 0.6}


# --- scene ----------------------------------------------------------------------------------------------------------
def test_scene_composes_office_table_plus_mug_and_spoon():
    from learning.stir import twin
    lay = twin.compose(twin.load_task())
    office = json.loads((ROOT / "sites/office/layout.json").read_text())
    assert lay["table"] == office["table"] and lay["markers"] == office["markers"]
    cup, spoon = [o for o in lay["objects"] if o.get("shape") in ("cup", "spoon")]
    assert 0.075 <= 2 * (cup["radius"] + cup["wall"]) <= 0.085 and 0.09 <= cup["height"] <= 0.10   # a real mug
    assert 0.15 <= twin.spoon_length(TASK["spoon"]) <= 0.20                                         # a real spoon
    assert spoon["follow"] == "cup" and spoon["face_base"]
    assert lay["stir"]["tool_tip"] == "spoon_tip" and lay["stir"]["rim_height"] == cup["height"]
    assert lay["rest_pose"] == TASK["rest_pose"]


def test_site_layouts_untouched_and_committed_layout_in_sync():
    """The task objects live in task.json and the generated sim/layouts/stir_office.json, never in a site layout."""
    from learning.stir import twin
    from sites import get_site
    for site in ("home", "office"):
        lay = json.loads(get_site(site).layout.read_text())
        names = {o["name"] for o in lay.get("objects", [])}
        assert not names & {TASK["cup"]["name"], TASK["spoon"]["name"]} and "stir" not in lay
    committed = json.loads((ROOT / "sim/layouts/stir_office.json").read_text())
    fresh = twin.compose(twin.load_task())
    assert committed["stir"] == fresh["stir"]
    assert committed["objects"][0] == fresh["objects"][0]                 # the mug (the spoon pose is the settled one)
    assert committed["table"] == fresh["table"]


def test_spoon_rests_on_the_rim_and_the_mug_stands_still(tmp_path):
    """Built in MuJoCo: after 2 s the spoon still leans on the rim facing the arm base, bowl inside the mug."""
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("cv2")
    sys.path.insert(0, str(ROOT / "sim"))
    import build_scene
    p = tmp_path / "stir.json"
    p.write_text((ROOT / "sim/layouts/stir_office.json").read_text())
    build_scene.build(p, tmp_path / "gen", report=False)
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "gen/workspace.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("rest").id)
    d.ctrl[:] = d.qpos[m.jnt_qposadr[m.actuator_trnid[:, 0]]]
    mujoco.mj_forward(m, d)
    cup0, top0 = d.body("cup").xpos.copy(), d.site("spoon_top").xpos.copy()
    for _ in range(int(2.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    cup, tip, top = d.body("cup").xpos, d.site("spoon_tip").xpos, d.site("spoon_top").xpos
    assert math.dist(cup0, cup) < 0.001
    assert math.dist(top0, top) < 0.003                                   # settled pose: nothing slides at t = 0
    assert math.hypot(tip[0] - cup[0], tip[1] - cup[1]) < GEO["inner_radius"] and tip[2] < 0.04
    assert top[1] < cup[1] - 0.03 and abs(top[0] - cup[0]) < 0.01         # handle towards the arm base (-y)


# --- judge ----------------------------------------------------------------------------------------------------------
def circle(r=0.012, loops=3.5, z=-0.02, n=400, centre=(0.0, 0.0)):
    return [[centre[0] + r * math.cos(2 * math.pi * loops * k / n), centre[1] + r * math.sin(2 * math.pi * loops * k / n),
             z] for k in range(n)]


def window(pts, stir_s=10.0, wall=0.0, bottom=0.0, immersed=None):
    return {"pts": pts, "tip_immersed_s": stir_s if immersed is None else immersed, "tool_wall_s": wall,
            "tool_bottom_s": bottom}


GOOD = {"grasped": True, "aborted": None, "stir_s": 10.0, "cup_moved_m": 0.001, "cup_tilt_max_deg": 1.0,
        "slip_m": 0.001, "spoon_in_cup": True, "arm_cup_s": 0.0}


def test_judge_counts_loops_not_jitter():
    from learning.stir.judge import loops_about_centre
    assert loops_about_centre(circle(loops=3.5), 0.003)[0] == pytest.approx(3.5, abs=0.05)   # half a loop biases the path centre
    back_and_forth = circle(loops=1) + circle(loops=1)[::-1]
    assert loops_about_centre(back_and_forth, 0.003)[0] < 0.05            # net loops: going back cancels
    wiggle = [[0.001 * math.cos(k), 0.001 * math.sin(k), 0] for k in range(200)]
    assert loops_about_centre(wiggle, 0.003)[0] == 0.0                    # a 1 mm wiggle is not a loop


def test_judge_clean_stir_and_quality_terms():
    from learning.stir.judge import judge
    v = judge(window(circle()), GOOD, TASK["judge"], GEO)
    assert v["stirred"] and v["outcome"] == "stirred" and 0 < v["score"] == v["quality"] <= 1
    assert v["terms"]["coverage"] == pytest.approx(0.012 / (0.036 - 0.012), abs=0.01)
    wider = judge(window(circle(r=0.018)), GOOD, TASK["judge"], GEO)
    assert wider["score"] > v["score"]                                    # coverage pays
    lower = judge(window(circle(z=-0.035)), GOOD, TASK["judge"], GEO)
    assert lower["terms"]["depth"] > v["terms"]["depth"]                  # stirring lower pays


@pytest.mark.parametrize("change,outcome", [
    ({"extras": {"spoon_in_cup": False}}, "dropped_spoon"),
    ({"extras": {"cup_tilt_max_deg": 12.0}}, "tipped_cup"),
    ({"extras": {"cup_moved_m": 0.02}}, "pushed_cup"),
    ({"extras": {"arm_cup_s": 0.5}}, "arm_hit_cup"),
    ({"window": {"tool_wall_s": 3.0}}, "scraped_wall"),
    ({"extras": {"slip_m": 0.03}}, "slipped"),
    ({"window": {"tip_immersed_s": 2.0}}, "stirred_air"),
    ({"loops": 2.0}, "too_few_loops"),
    ({"extras": {"aborted": "tracking error"}}, "aborted"),
])
def test_judge_gates_name_the_outcome(change, outcome):
    from learning.stir.judge import judge
    w = {**window(circle(loops=change.get("loops", 3.5))), **change.get("window", {})}
    v = judge(w, {**GOOD, **change.get("extras", {})}, TASK["judge"], GEO)
    assert not v["stirred"] and v["outcome"] == outcome
    assert v["score"] < judge(window(circle()), GOOD, TASK["judge"], GEO)["score"]


def test_judge_no_grasp():
    from learning.stir.judge import judge
    v = judge(None, {**GOOD, "grasped": False}, TASK["judge"], GEO)
    assert v["outcome"] == "no_grasp" and v["score"] == 0


# --- records and policy ---------------------------------------------------------------------------------------------
def test_store_records_an_attempt(tmp_path):
    from learning.stir.judge import judge
    from learning.stir.store import StirStore
    st = StirStore(tmp_path)
    n = st.begin({"sim.stir.radius_m": 0.009}, "baseline")
    st.phase(n, "grasp", "ok", {"tries": 1})
    with pytest.raises(ValueError):
        st.phase(n, "twist")
    st.judge(n, judge(window(circle()), GOOD, TASK["judge"], GEO))
    st.end(n, "stirred", "fine", 0.5)
    n2 = st.begin({"sim.stir.radius_m": 0.012}, "wider")
    a = st.attempts()
    assert a[n2]["begin"]["changed"] == {"sim.stir.radius_m": {"before": 0.009, "after": 0.012}}
    assert a[n]["end"]["outcome"] == "stirred" and a[n]["judges"][0]["verdict"] == "stirred"
    assert "| 1 |" in (tmp_path / "SUMMARY.md").read_text() and (tmp_path / "attempt_1/report.md").exists()
    with pytest.raises(RuntimeError):
        st.begin({}, "attempt 2 still open")


def _policy():
    from learning.stir import practice as P
    cfg = {"sim.stir.radius_m": 0.009, "sim.stir.turns": 3, "sim.stir.height_m": 0.04, "sim.stir.step_dt_s": 0.05,
           "sim.stir.wall_margin_m": 0.006, "sim.stir.lean_extra_deg": 0.0, "sim.stir.grasp_from_top_m": 0.035,
           "sim.stir.ring_points": 16}
    env = {"sim.stir.radius_m": (0.003, 0.02), "sim.stir.turns": (1, 10), "sim.stir.height_m": (0.015, 0.08),
           "sim.stir.step_dt_s": (0.02, 0.2), "sim.stir.wall_margin_m": (0.002, 0.02),
           "sim.stir.lean_extra_deg": (0.0, 20.0), "sim.stir.grasp_from_top_m": (0.015, 0.06),
           "sim.stir.ring_points": (8, 32)}
    return P.Policy(cfg, env)


def V(outcome, score):
    return {"stirred": outcome == "stirred", "outcome": outcome, "score": score, "metrics": {"loops": 2.9},
            "evidence": ["x"]}


def test_policy_repairs_then_climbs_then_backs_off():
    pol = _policy()
    ph = {"stir": {"radius_m": 0.009, "radius_asked_m": 0.009}}
    assert pol.next(0, None, {})[0] == {}                                          # baseline first
    assert pol.next(1, V("too_few_loops", 0.07), ph)[0] == {"sim.stir.turns": 4}   # diagnosis: one more turn
    assert pol.next(2, V("stirred", 0.56), ph)[0] == {"sim.stir.radius_m": 0.012}  # clean: climb coverage
    ch, why = pol.next(3, V("scraped_wall", 0.0), ph)                             # did not pay: undo, next direction
    assert ch == {"sim.stir.radius_m": 0.009, "sim.stir.height_m": 0.034} and "did not pay" in why
    assert "sim.stir.radius_m" in pol.limits and pol.best[2] == 2
    assert pol.next(4, V("stirred", 0.60), ph)[0] == {"sim.stir.height_m": 0.028}  # better: keep going


def test_policy_capped_radius_moves_the_wall_margin():
    pol = _policy()
    pol.next(0, None, {})
    ch, why = pol.next(1, V("stirred", 0.5), {"stir": {"radius_m": 0.008, "radius_asked_m": 0.009}})
    assert ch == {"sim.stir.wall_margin_m": 0.005} and "capped" in why


# --- one short episode in the twin ---------------------------------------------------------------------------------
def test_one_short_stir_episode_in_the_twin(tmp_path):
    """One attempt, one turn, through the whole loop: config version, records, judge from ground truth."""
    pytest.importorskip("mujoco")
    pytest.importorskip("mcp")
    pytest.importorskip("cv2")
    env = {**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "cgl"), "ROBOT_SIM_RECORD": "0"}
    p = subprocess.run([sys.executable, "-m", "learning.stir.practice", "--attempts", "1", "--out", str(tmp_path),
                        "--start", "sim.stir.turns=1", "--start", "sim.stir.step_dt_s=0.03"],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-3000:]
    recs = [json.loads(l) for l in (tmp_path / "practice/episodes.jsonl").read_text().splitlines()]
    kinds = [r["type"] for r in recs]
    assert kinds[0] == "episode_begin" and kinds[-1] == "episode_end" and "judge" in kinds
    assert [r["phase"] for r in recs if r["type"] == "phase"] == ["locate", "grasp", "straighten", "lead_in", "stir",
                                                                   "release"]
    begin, j = recs[0], next(r for r in recs if r["type"] == "judge")
    assert begin["params"]["sim.stir.turns"] == 1 and begin["skillcfg_twin_version"] == 1
    m = j["metrics"]
    assert 0.8 <= m["loops"] <= 1.2 and m["cup_moved_m"] < 0.005 and m["spoon_in_cup"] and m["slip_m"] < 0.01
    assert j["verdict"] == "too_few_loops"                       # one turn is under the 3-loop gate, by design
    log = [json.loads(l) for l in (tmp_path / "skillcfg/changes.jsonl").read_text().splitlines()]
    assert log[0]["target"] == "twin" and log[0]["changes"]["sim.stir.turns"]["after"] == 1
    assert (tmp_path / "practice/SUMMARY.md").exists() and (tmp_path / "practice/attempt_1/path.json").exists()


def test_learning_curve_svg():
    from learning.stir.report import curve_svg
    rows = [{"attempt": 1, "outcome": "too_few_loops", "score": 0.07}, {"attempt": 2, "outcome": "stirred", "score": 0.57},
            {"attempt": 3, "outcome": "scraped_wall", "score": 0.0}, {"attempt": 4, "outcome": "stirred", "score": 0.6}]
    v = {"runs": 5, "stirred": 4, "mean_score": 0.5, "mean_radius_cm": 0.8, "mean_pace": 0.3}
    svg = curve_svg(rows, {"baseline": v, "learned": v})
    assert svg.startswith("<svg") and svg.count("<circle") == 4 and "scraped wall" in svg and "Validation" in svg
