"""The whole loop on the twin: 3 attempts with a scripted parameter policy (learning/sim_loop.py) through the
sim_arm_mcp tools, into episodes.jsonl, attempt_<N>/ files, reports, SUMMARY.md and recorder sessions.

Runs in a subprocess: sim_arm_mcp builds one twin per process at import, and other test modules load other
layouts into theirs."""
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def loop_run(tmp_path_factory):
    pytest.importorskip("mujoco")
    pytest.importorskip("mcp")
    pytest.importorskip("cv2")
    from recorder.server import Service, make_server
    tmp = tmp_path_factory.mktemp("bottle")
    svc = Service(root=tmp / "rec")
    httpd = make_server(svc, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    learn = tmp / "learn"
    env = {**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "cgl"), "BOTTLE_GEMINI_JUDGE": "0",
           "ROBOT_SIM_RECORD": "0"}
    try:
        p = subprocess.run([sys.executable, "-m", "learning.sim_loop", "--attempts", "3", "--learn-dir", str(learn),
                            "--recorder-url", f"http://127.0.0.1:{httpd.server_address[1]}",
                            "--layout", str(tmp / "bottle_twin.json")],
                           cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    finally:
        httpd.shutdown()
    assert p.returncode == 0, p.stderr[-3000:]
    summary = json.loads([l for l in p.stdout.splitlines() if l.startswith('{"summary"')][-1])["summary"]
    return learn, summary, svc


def _real_geometry_run(tmp_path, attempts, env_extra=None):
    env = {**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "cgl"), "BOTTLE_GEMINI_JUDGE": "0", "ROBOT_SIM_RECORD": "0",
           **(env_extra or {})}
    p = subprocess.run([sys.executable, "-m", "learning.sim_loop", "--attempts", str(attempts), "--real-geometry",
                        "--learn-dir", str(tmp_path / "learn"), "--layout", str(tmp_path / "bottle_twin_real.json")],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-3000:]
    summary = json.loads([l for l in p.stdout.splitlines() if l.startswith('{"summary"')][-1])["summary"]
    recs = [json.loads(l) for l in (tmp_path / "learn/episodes.jsonl").read_text().splitlines()]
    return summary, recs


def test_real_table_geometry_top_approach(tmp_path):
    """The 25 cm bottle ~0.28 m out (as on the real table). Since 22 Nov (f63acfd) twist_cap always comes from above,
    and the twin's IK uses the same least-lean policy as the real arm (skillcfg.lean_search): the gripper leans well
    under the 60-70 deg of the old twin order, hovers, descends vertically and turns the cap about its axis."""
    pytest.importorskip("mujoco")
    pytest.importorskip("cv2")
    summary, recs = _real_geometry_run(tmp_path, 2)
    assert [a["outcome"] for a in summary] == ["slip", "loosened"]
    assert summary[1]["truth"]["turned_deg"] >= 25
    app = [r for r in recs if r["type"] == "phase" and r["phase"] == "approach"]
    assert app and all(r["status"] == "ok" and r["measurements"]["mode"] == "top" and 30 < r["measurements"]["lean_deg"] < 60
                       for r in app), app
    assert all(r["skillcfg"] == {"target": "twin", "version": 0, "overrides": {}} for r in recs if r["type"] == "episode_begin")


def test_proposal_applies_in_the_twin_and_the_next_attempt_uses_it(tmp_path, monkeypatch):
    """propose twist.approach=top through the MCP tool body (as sim_arm_mcp's propose_skill_change) on a twin that
    approaches steep caps from the side -> IK probe of the real cap pose (side -> top) -> applied automatically on the
    twin -> the next sim attempt runs with that version and approaches from above."""
    pytest.importorskip("mujoco")
    pytest.importorskip("cv2")
    import asyncio
    from learning import skillcfg as SC, tools
    cfg = tmp_path / "cfg"
    monkeypatch.setenv("SKILLCFG_DIR", str(cfg))
    learn = tmp_path / "real_learn"                     # the real cap pose of 22 Nov as the probe's evidence
    learn.mkdir()
    (learn / "episodes.jsonl").write_text(json.dumps({"type": "phase", "attempt": 8, "phase": "locate", "measurements":
                                                      {"x": -0.006, "y": 0.269, "cap_top_z": 0.208}}) + "\n")
    monkeypatch.setenv("BOTTLE_LEARN_DIR", str(learn))
    SC.set_target("twin")
    r = SC.propose({"twist.side_lean_deg": 30.0}, "the pre-22 Nov side approach", ["f63acfd"], "test", "twin")
    assert SC.apply(r["proposal"]["id"], "test")["version"] == 1
    text, _ = asyncio.run(tools.propose_skill_change(
        {"twist.approach": "top"}, "leaning > 30 deg at the cap the side approach slides the jaws onto the bevel",
        ["attempt 8 approach lean_deg=40 mode=side", "attempt 8 lesson: jaws slid onto the cap's bevel"],
        author="operator (sim_arm_mcp)", auto_apply=True))
    assert "APPLIED to the twin: version 2" in text, text
    p = SC.proposals(t="twin")[-1]
    row = p["checks"]["ik_probe"]["rows"][0]
    assert p["checks"]["ik_probe"]["result"] == "pass" and (row["before"]["mode"], row["after"]["mode"]) == ("side", "top")
    summary, recs = _real_geometry_run(tmp_path, 1, {"SKILLCFG_DIR": str(cfg)})
    begin = next(r for r in recs if r["type"] == "episode_begin")
    assert begin["skillcfg"]["version"] == 2 and begin["skillcfg"]["overrides"]["twist.approach"] == "top"
    app = [r for r in recs if r["type"] == "phase" and r["phase"] == "approach"]
    assert app and all(r["measurements"]["mode"] == "top" for r in app), app


def test_three_attempts_learn_to_open_the_cap(loop_run):
    learn, summary, _ = loop_run
    assert [a["outcome"] for a in summary] == ["slip", "loosened", "opened"]
    assert summary[0]["truth"]["turned_deg"] < 10 <= summary[0]["rotation_achieved_deg"]      # jaws slid round the cap
    assert summary[-1]["truth"]["free"] and summary[-1]["cap_lift_mm"] >= 8


def test_records_follow_the_contract(loop_run):
    learn, _, _ = loop_run
    recs = [json.loads(l) for l in (learn / "episodes.jsonl").read_text().splitlines()]
    kinds = {r["type"] for r in recs}
    assert kinds == {"episode_begin", "phase", "judge", "episode_end"}
    phases = {r["phase"] for r in recs if r["type"] == "phase"}
    assert {"locate", "approach", "grasp", "twist", "release", "regrasp", "check"} <= phases
    begins = [r for r in recs if r["type"] == "episode_begin"]
    assert [b["attempt"] for b in begins] == [1, 2, 3] and begins[2]["based_on"] == [1, 2]
    assert all(b["recording_id"] for b in begins)
    assert {j["source"] for j in recs if j["type"] == "judge"} == {"camera", "twin"}
    for n in (1, 2, 3):
        d = learn / f"attempt_{n}"
        for f in ("before_wrist.jpg", "before_overview.jpg", "after_wrist.jpg", "check_wrist.jpg", "twist_trace.json",
                  "report.md"):
            assert (d / f).exists(), f"attempt_{n}/{f}"
        trace = json.loads((d / "twist_trace.json").read_text())
        row = trace["rows"][1]
        for k in ("j7_raw", "j7_goal", "j7_load", "j8_raw", "j8_load", "tcp_z", "cmd_deg", "yaw_deg"):
            assert k in row
        for s in trace["strokes"]:
            for k in ("commanded_deg", "achieved_deg", "peak_j7_load", "mean_j7_load", "j8_change_ticks", "z_change_mm",
                      "abort_reason"):
                assert k in s


def test_reports_and_recordings(loop_run):
    learn, _, svc = loop_run
    summary = (learn / "SUMMARY.md").read_text()
    assert "slip -> loosened -> opened" in summary and summary.count("[report](attempt_") == 3
    rep = (learn / "attempt_3/report.md").read_text()
    assert "# Attempt 3: opened" in rep and "## Strokes" in rep and "twist_trace.json" in rep
    recs = {m["name"]: m for m in svc.listing()}
    assert {"bottle-attempt-1", "bottle-attempt-2", "bottle-attempt-3"} <= set(recs)
    assert all(recs[f"bottle-attempt-{n}"].get("stopped") for n in (1, 2, 3))
    rec = svc.recording(recs["bottle-attempt-2"]["id"])
    labels = [m["label"] for m in rec.markers()]
    assert "grasp:ok" in labels and "twist:ok" in labels and "end:loosened" in labels
