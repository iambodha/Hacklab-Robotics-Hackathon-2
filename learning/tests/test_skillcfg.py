"""Skill config: registry and defaults, validation against envelopes and the safety class, versions with before/after
records, the live read path, approval rules, revert, the shared lean search, and the readers (twist, episodes, tools)."""
import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from learning import episodes as E
from learning import skillcfg as SC
from learning import twist as T

ROOT = Path(__file__).resolve().parents[2]


def _apply(changes, t="twin", **kw):
    r = SC.propose(changes, "test reason", ["test evidence"], "test", t)
    assert r["ok"], r
    a = SC.apply(r["proposal"]["id"], "tester", **kw)
    return r["proposal"]["id"], a


def test_no_file_means_the_code_defaults():
    a = SC.active("real")
    assert a["version"] == 0 and a["overrides"] == {}
    v = a["values"]
    prof = json.loads((ROOT / "sim/robots/elrobot.json").read_text())["ik"]
    assert v["ik.leans"] == prof["leans_small"] + prof["leans_large"] and v["ik.restarts"] == prof["restarts"]
    assert v["ik.search"] == "least_lean" and v["executor.ik.search"] == "near_first"
    assert (v["twist.side_lean_deg"], tuple(v["twist.hovers_m"]), v["twist.max_lean_deg"]) == (T.SIDE_LEAN, T.HOVERS, T.MAX_LEAN)
    assert (v["twist.grasp_below_top_m"], v["twist.release_open"], v["twist.approach"]) == (T.GRASP_BELOW_TOP, T.RELEASE_OPEN, "auto")
    for name, (kind, lo, hi, *_rest) in E.PARAMS.items():
        if kind != "enum":
            assert (v[f"params.bounds.{name}.lo"], v[f"params.bounds.{name}.hi"]) == (lo, hi)
    assert E.params_table() == E.PARAMS
    assert not (Path(os.environ["SKILLCFG_DIR"]) / "real.json").exists()


def test_safety_keys_are_listed_and_immutable():
    reg = SC.registry()
    for k in E.SAFETY:
        assert reg[f"safety.{k}"]["class"] == "safety"
    assert reg["workspace.min_tcp_z"]["class"] == "safety" and reg["motion.load_abort"]["class"] == "safety"
    r = SC.propose({"safety.max_grip_force": 0.9, "workspace.min_tcp_z": 0.0}, "more force", ["x"], t="twin")
    assert not r["ok"] and sum("safety limit" in e for e in r["errors"]) == 2
    # the bounds that touch a safety limit are capped by it
    assert reg["params.bounds.grip_torque_limit.hi"]["hi"] == E.SAFETY["max_grip_force"]
    assert reg["params.bounds.j7_load_abort.hi"]["hi"] == E.SAFETY["j7_load_hard"]
    assert not SC.propose({"params.bounds.grip_torque_limit.hi": 0.7}, "r", ["e"], t="twin")["ok"]


@pytest.mark.parametrize("changes, needle", [
    ({"twist.approach": "diagonal"}, "must be one of"),
    ({"twist.side_lean_deg": 120}, "outside the envelope"),
    ({"ik.leans": [0, 10, 10]}, "unique"),
    ({"ik.leans": [0, 95]}, "entry 95"),
    ({"ik.leans": []}, "needs 1..32"),
    ({"ik.restarts": 2.5}, "whole number"),
    ({"twist.hovers_m": "high"}, "must be a list"),
    ({"params.bounds.strokes.hi": 60}, "outside the envelope"),
    ({"params.bounds.strokes.lo": 10, "params.bounds.strokes.hi": 5}, "lo 10 > hi 5"),
    ({"params.bounds.strokes.lo": 5}, "must contain the default 3"),
    ({"bogus.key": 1}, "unknown key"),
])
def test_validation_refuses(changes, needle):
    r = SC.propose(changes, "reason", ["evidence"], t="twin")
    assert not r["ok"] and any(needle in e for e in r["errors"]), r
    assert SC.proposals() == []                                   # a refused request leaves no proposal


def test_reason_evidence_and_a_real_change_are_required():
    r = SC.propose({"twist.approach": "top"}, "", [], t="twin")
    assert not r["ok"] and len(r["errors"]) == 2
    r = SC.propose({"twist.approach": "auto"}, "r", ["e"], t="twin")
    assert not r["ok"] and "already the current" in r["errors"][0]


def test_apply_writes_a_version_and_the_running_reader_sees_it(tmp_path):
    assert SC.get("twist.approach", "twin") == "auto"
    pid, a = _apply({"twist.approach": "top", "params.bounds.strokes.hi": 30})
    assert a == {"ok": True, "target": "twin", "version": 1}
    assert SC.get("twist.approach", "twin") == "top" and SC.version("twin") == 1     # no restart, no reload call
    assert SC.get("twist.approach", "real") == "auto"                                # targets are separate
    cur = json.loads((SC.root() / "twin.json").read_text())
    assert cur["version"] == 1 and cur["values"] == {"twist.approach": "top", "params.bounds.strokes.hi": 30}
    rec = SC.changes_log("twin")[-1]
    assert rec["version"] == 1 and rec["proposal"] == pid and rec["author"] == "test" and rec["approved_by"] == "tester"
    assert rec["changes"]["twist.approach"] == {"before": "auto", "after": "top"}
    assert rec["reason"] == "test reason" and rec["evidence"] == ["test evidence"]
    assert SC.proposal(pid)["status"] == "applied" and SC.proposal(pid)["version"] == 1
    assert not SC.apply(pid, "again")["ok"]                                          # applied once only
    # a hand edit outside the envelope is ignored, not obeyed
    time.sleep(0.01)
    cur["values"]["twist.side_lean_deg"] = 500
    (SC.root() / "twin.json").write_text(json.dumps(cur))
    a = SC.active("twin")
    assert a["values"]["twist.side_lean_deg"] == 90.0 and a["ignored"] == ["twist.side_lean_deg"]


def test_revert_is_a_new_version():
    _apply({"twist.approach": "top"})
    _apply({"twist.release_open": 0.4})
    r = SC.revert("twin", 1, "tester", "release_open 0.4 made it worse")
    assert r == {"ok": True, "target": "twin", "version": 3}
    a = SC.active("twin")
    assert a["overrides"] == {"twist.approach": "top"}
    assert SC.changes_log("twin")[-1]["changes"] == {"twist.release_open": {"before": 0.4, "after": 0.25}}
    assert SC.revert("twin", 0, "tester", "all back")["ok"] and SC.active("twin")["overrides"] == {}
    assert not SC.revert("twin", 9, "tester", "")["ok"]
    assert not SC.revert("real", 0, "agent", "")["ok"]                      # the real config: human (CLI) only


def test_real_target_needs_a_human_and_passing_checks():
    r = SC.propose({"twist.approach": "top"}, "slips on the bevel", ["attempt 7"], "operator", "real")
    pid = r["proposal"]["id"]
    a = SC.apply(pid, "operator")
    assert not a["ok"] and "human approval" in a["errors"][0] and SC.version("real") == 0
    SC.update_proposal(pid, lambda p: p["checks"].update(ik_probe={"result": "fail", "summary": "loses reach"}))
    assert not SC.apply(pid, "bodha", human=True)["ok"]
    assert SC.apply(pid, "bodha", human=True, force=True)["ok"] and SC.get("twist.approach", "real") == "top"


def test_stale_proposal_is_refused():
    p1 = SC.propose({"twist.approach": "top"}, "r", ["e"], t="twin")["proposal"]["id"]
    p2 = SC.propose({"twist.approach": "side"}, "r", ["e"], t="twin")["proposal"]["id"]
    assert SC.apply(p1, "t")["ok"]
    a = SC.apply(p2, "t")
    assert not a["ok"] and "changed since this proposal" in a["errors"][0]
    assert SC.reject(p2, "t", "superseded")["ok"] and SC.proposal(p2)["status"] == "rejected"


def test_cli_approve_and_revert(capsys):
    pid = SC.propose({"twist.approach": "top"}, "r", ["e"], "operator", "real")["proposal"]["id"]
    assert SC.main(["approve", pid, "--by", "bodha"]) == 0 and "version 1" in capsys.readouterr().out
    assert SC.main(["revert", "real", "0", "--by", "bodha"]) == 0 and SC.active("real")["overrides"] == {}
    assert SC.main(["show", "--target", "real"]) == 0 and "version 2" in capsys.readouterr().out
    assert SC.main(["propose", "twist.release_open=0.3", "--target", "twin", "--reason", "r", "--evidence", "e",
                    "--no-check"]) == 0
    assert SC.proposals("pending", "twin")


def test_target_choice():
    assert SC.target() == "real"                               # unset: never the auto-applied twin config
    os.environ["SKILLCFG_TARGET"] = "twin"
    try:
        assert SC.target() == "twin"
        SC.set_target("real")
        assert SC.target() == "real"
    finally:
        del os.environ["SKILLCFG_TARGET"]
    with pytest.raises(ValueError):
        SC.set_target("lab")


# --- the shared lean search ---------------------------------------------------------------------------------------
def _fake_ik(near_from: int, restart_from: int):
    """try_lean: from the current joints only leans >= near_from solve; with restarts leans >= restart_from."""
    calls = []

    def try_lean(lean, rs):
        calls.append((lean, rs))
        ok = abs(lean) >= (restart_from if rs else near_from)
        return f"q{lean}{'r' if rs else ''}" if ok else None
    return try_lean, calls


def test_least_lean_beats_the_near_solution():
    leans = [0, 10, -10, 20, 35, 40, 60, 70]
    f, calls = _fake_ik(60, 35)
    assert SC.lean_search(f, leans, 12, "least_lean") == ("q35r", 35)          # the 22 Nov fix: 35, not 60
    f, _ = _fake_ik(60, 35)
    assert SC.lean_search(f, leans, 12, "near_first") == ("q60", 60)           # the old order
    f, calls = _fake_ik(0, 0)
    assert SC.lean_search(f, leans, 12, "least_lean") == ("q0", 0) and calls == [(0, 0)]
    f, _ = _fake_ik(99, 99)
    assert SC.lean_search(f, leans, 12, "least_lean") == (None, None)
    f, _ = _fake_ik(60, 35)
    assert SC.lean_search(f, leans, 0, "least_lean") == ("q60", 60)            # no restarts: near only
    f, _ = _fake_ik(10, 99)
    assert SC.lean_search(f, leans, 4, "least_lean", near=lambda q: q != "q10") == ("q-10", -10)


def test_lean_search_reads_the_config():
    f, _ = _fake_ik(60, 35)
    assert SC.lean_search(f, t="twin")[1] == 35
    _apply({"ik.search": "near_first"})
    f, _ = _fake_ik(60, 35)
    assert SC.lean_search(f, t="twin")[1] == 60
    _apply({"ik.leans": [0, 10, 20, 30, 70]})
    f, _ = _fake_ik(60, 35)
    assert SC.lean_search(f, t="twin")[1] == 70


# --- readers --------------------------------------------------------------------------------------------------------
def test_episode_bounds_follow_the_config(tmp_path):
    from learning.tests.test_episodes import FakeRecorder
    SC.set_target("twin")
    assert E.validate({"strokes": 30})[2]
    _apply({"params.bounds.strokes.hi": 30})
    full, changed, errors = E.validate({"strokes": 30})
    assert not errors and full["strokes"] == 30
    assert "strokes: 1..30" in E.bounds_text()
    st = E.EpisodeStore(tmp_path / "learn", recorder=FakeRecorder())
    r = st.begin({"strokes": 30}, "spin until it opens")
    assert r["ok"] and r["skillcfg"] == {"target": "twin", "version": 1, "overrides": {"params.bounds.strokes.hi": 30}}
    begin = st.attempts()[1]["begin"]
    assert begin["skillcfg"]["version"] == 1
    st.end("loosened", "30 strokes turned it")
    _apply({"twist.approach": "top"})
    st.begin({"strokes": 3}, "top approach")
    st.end("opened", "top approach")
    summary = (st.root / "SUMMARY.md").read_text()
    assert "| 1 | twin v1 |" in summary and "| 2 | twin v2 |" in summary
    assert "## Outcomes per skill-config version" in summary and "| twin v2 | 2 | opened |" in summary
    assert "Skill config: twin v1 (params.bounds.strokes.hi=30)" in (st.root / "attempt_1/report.md").read_text()
    assert "skill config: twin v2" in st.history_text()


class _Rig:
    twist_sign = 1

    def __init__(self, lean):
        self.lean = lean

    async def q(self):
        import numpy as np
        return np.zeros(8)

    def fk(self, q):
        import numpy as np
        return np.zeros(3), np.eye(3)

    async def reachable(self, x, y, z, yaw, seed):
        return self.lean

    async def state(self):
        return {"j8": {"norm": 1.0}}


def _mode(lean):
    seen = []
    tw = T.Twister(_Rig(lean), E.defaults(), lambda *a: None)

    async def side(*a):
        seen.append("side")
        return False

    async def down(*a):
        seen.append("top")
        return False
    tw._approach_side, tw._approach_down = side, down
    asyncio.run(tw.twist_cap(0.0, 0.27, 0.21))
    return seen[0], tw


def test_twister_approach_follows_the_config():
    SC.set_target("twin")
    assert _mode(60.0)[0] == "top"                                   # code default: always from above
    _apply({"twist.side_lean_deg": 45.0})
    assert _mode(60.0)[0] == "side" and _mode(40.0)[0] == "top"
    _apply({"twist.approach": "top"})
    mode, tw = _mode(60.0)
    assert mode == "top" and tw.cfg["version"] == 2
    _apply({"twist.max_lean_deg": 50.0})
    with pytest.raises(IndexError):                                 # refused before any approach: nothing seen
        _mode(60.0)


def test_tools_twin_applies_real_waits(monkeypatch):
    from learning import skillcheck, tools
    monkeypatch.setattr(skillcheck, "last_cap_poses", lambda *a, **k: [])
    SC.set_target("twin")
    text, _ = asyncio.run(tools.propose_skill_change({"twist.approach": "top"}, "slips on the bevel at 60 deg",
                                                     ["attempt 3 approach lean_deg=60 mode=side"], auto_apply=True))
    assert "APPLIED to the twin: version 1" in text and SC.get("twist.approach", "twin") == "top"
    assert "ik_probe: skipped" in text
    SC.set_target("real")
    text, _ = asyncio.run(tools.propose_skill_change('{"params.bounds.strokes.hi": 30}', "more strokes", "attempt 6",
                                                     auto_apply=True))
    assert "PENDING" in text and "approve" in text and SC.version("real") == 0
    text, _ = asyncio.run(tools.propose_skill_change({"safety.max_grip_force": 1.0}, "r", ["e"]))
    assert text.startswith("REFUSED")
    assert "pending proposals for real: 1" in asyncio.run(tools.skill_config())[0]
    assert "HUMAN approves" in tools.skill_doc("propose_skill_change", "real")


def test_real_server_has_no_apply_tool():
    src = (ROOT / "sim/real_arm_mcp.py").read_text()
    tools_ = [l.split("def ")[1].split("(")[0] for l in src.split("@server.tool")[1:] for l in [l[l.index("async def"):]]]
    assert {"skill_config", "propose_skill_change"} <= set(tools_)
    assert not any("approve" in t or "apply" in t or "revert" in t for t in tools_)
    assert 'SC.set_target("real")' in src and "auto_apply=False" in src


def test_last_cap_poses(tmp_path):
    recs = [{"type": "phase", "attempt": n, "phase": "locate", "measurements": {"x": x, "y": 0.268, "cap_top_z": 0.21}}
            for n, x in ((1, 0.0), (2, 0.001), (3, 0.02))]
    (tmp_path / "episodes.jsonl").write_text("\n".join(map(json.dumps, recs)) + "\n{torn")
    from learning.skillcheck import last_cap_poses
    poses = last_cap_poses(tmp_path)
    assert [p["attempt"] for p in poses] == [3, 2]                   # newest first, 5 mm apart


def test_ik_probe_reports_lean_and_mode():
    pytest.importorskip("mujoco")
    from learning import skillcheck
    d = SC.defaults()
    pose = {"attempt": 8, "x": -0.006, "y": 0.269, "cap_top_z": 0.208}       # the real cap, 22 Nov
    r = skillcheck.ik_probe({**d, "twist.side_lean_deg": 30.0}, d, [pose])
    row = r["rows"][0]
    assert r["result"] == "pass" and row["before"]["mode"] == "side" and row["after"]["mode"] == "top"
    assert 25 < row["after"]["grasp_lean_deg"] <= 45
    r = skillcheck.ik_probe(d, {**d, "ik.leans": [0, 10, 20]}, [pose])
    assert r["result"] == "fail" and "LOSES reach" in r["summary"]
