"""The executor reads and proposes through the shared skill config (learning/skillcfg.py): its IK search order
(executor.ik.search) and lean list (ik.leans) end to end from a proposal to a planned move, the config version in every
validation, and the real backend's proposals waiting for a human."""
import pytest

from executor.backends import FakeBackend
from executor.core import Executor
from executor.perception import Locator
from executor.recording import RecorderLink
from executor.tests.conftest import proposal
from learning import skillcfg as SC

FAR_HIGH = [0.28, 0.0, 0.20]          # world: the jaws cannot point straight down here (lean ~35-40 deg)


@pytest.fixture(autouse=True)
def _cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("SKILLCFG_DIR", str(tmp_path / "skillcfg"))
    monkeypatch.setattr(SC, "_target", None)
    monkeypatch.setattr("learning.skillcheck.last_cap_poses", lambda *a, **k: [])


def _move(ex, xyz):
    obs = ex.observe()
    r = ex.status(ex.submit(proposal(ex, obs, "move_tcp", {"xyz_world": xyz})))
    ex.wait(r.execution_id, 60)
    return r


def test_executor_reads_the_twin_config_and_stamps_its_version(make_ex):
    ex, _ = make_ex()
    assert ex.kin.cfg_target == "twin"
    c = ex.skill_config()
    assert c["version"] == 0 and c["values"]["executor.ik.search"] == "near_first" and c["pending"] == []
    assert c["registry"]["safety.max_grip_force"]["class"] == "safety"
    r = _move(ex, FAR_HIGH)
    assert r.validation.ok and r.validation.checks["skill_config"] == {"result": "pass", "target": "twin", "version": 0,
                                                                     "overrides": {}}
    assert ex.describe()["skill_config"]["version"] == 0


def test_proposal_changes_the_executor_ik_end_to_end(make_ex, monkeypatch):
    ex, _ = make_ex()
    seen = []
    real = SC.lean_search
    monkeypatch.setattr(SC, "lean_search", lambda *a, **k: seen.append(a[3]) or real(*a, **k))
    lean0 = _move(ex, FAR_HIGH).validation.checks["reachability"]["max_lean_deg"]
    assert set(seen) == {"near_first"} and lean0 > 20
    out = ex.propose_skill_change({"executor.ik.search": "least_lean"}, "plan the least tilt like the MCP servers",
                                  ["move_tcp to [0.28, 0, 0.2] leaned 40 deg"], "test")
    assert out["applied"] and out["version"] == 1
    seen.clear()
    r = _move(ex, FAR_HIGH)
    assert set(seen) == {"least_lean"} and r.validation.checks["skill_config"]["version"] == 1
    assert r.validation.checks["reachability"]["max_lean_deg"] <= lean0
    # the lean list too: jaws straight down only -> the same move is refused before any motion, revert restores it
    assert ex.propose_skill_change({"ik.leans": [0]}, "test", ["test"], "test")["version"] == 2
    r = _move(ex, FAR_HIGH)
    assert not r.validation.ok and any("unreachable" in x for x in r.validation.reasons)
    assert SC.revert("twin", 1, "test", "back")["ok"]
    assert _move(ex, FAR_HIGH).validation.ok
    kinds = [e["state"] for e in ex.events.since(0) if e["kind"] == "skill_config"]
    assert kinds == ["applied", "applied"]


def test_refused_and_real_backend_proposals(kin, park, tmp_path):
    fake = FakeBackend(start_norm=dict(park), kin=kin)
    fake.name = "real"                      # what the Station backend is called
    ex = Executor(fake, kin, recorder=RecorderLink("local", root=tmp_path / "rec"), locator=Locator(),
                  table_memory_path=tmp_path / "table.json", twin=False)
    try:
        assert ex.kin.cfg_target == "real"
        bad = ex.propose_skill_change({"safety.arm_load_hard": 900}, "r", ["e"])
        assert not bad["ok"] and "safety limit" in bad["errors"][0]
        out = ex.propose_skill_change({"executor.ik.search": "least_lean"}, "r", ["e"], "claude")
        assert out["ok"] and not out["applied"] and SC.version("real") == 0
        assert [p["id"] for p in ex.skill_config()["pending"]] == [out["proposal"]["id"]]
        assert SC.apply(out["proposal"]["id"], "bodha", human=True)["ok"]
        assert ex.skill_config()["values"]["executor.ik.search"] == "least_lean"
    finally:
        ex.end_task("test end")
        kin.cfg_target = "twin"
