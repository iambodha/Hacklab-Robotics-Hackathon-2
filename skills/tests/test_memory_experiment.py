"""Skill memory (versions, setup invalidation, journal) and the matched experiment runner."""
import json

import pytest

from skills._contracts import SetupId
from skills.eval import synth
from skills.experiment import PromotionRule, SimWipeBackend, Trial, decide, positions, run_matched, train_adaptation
from skills.memory import Memory
from skills.wipe import WipeSkill

SK = WipeSkill()


def setup(tool="tissue", scene="1"):
    comp = {"scene": scene, "profile": "elrobot", "ranges": {"j1": [1056, 3040]}, "tool": tool, "layout": "home"}
    return SetupId.compute(scene, "elrobot", {"j1": [1056, 3040]}, tool, "home"), comp


def test_versions_and_promotion(tmp_path):
    m = Memory(tmp_path)
    sid, comp = setup()
    m.register_setup(sid, comp)
    v0 = SK.seed_version(w_hat=0.05)
    m.add_version(v0, sid, "incumbent")
    v1 = v0.child({"w_hat": 0.03}, "adaptation", "from trial 0")
    m.add_version(v1, sid)
    assert m.incumbent("wipe", sid).version == v0.version and m.status(v1.version, sid) == "candidate"
    with pytest.raises(PermissionError):
        m.promote(v1.version, sid, approved_by="")
    m.promote(v1.version, sid, approved_by="operator")
    assert m.incumbent("wipe", sid).version == v1.version and m.status(v0.version, sid) == "retired"
    assert m.lineage(v1.version) == [v1.version, v0.version]


def test_setup_change_invalidates(tmp_path):
    m = Memory(tmp_path)
    sid, comp = setup()
    m.register_setup(sid, comp)
    m.add_version(SK.seed_version(), sid, "incumbent")
    sid2, comp2 = setup(tool="cloth")
    r = m.register_setup(sid2, comp2)
    assert r["invalidated"] == [sid] and not m.setup_valid(sid)
    assert "tool" in m.setup(sid)["reason"]
    assert m.incumbent("wipe", sid) is None and m.incumbent("wipe", sid2) is None
    with pytest.raises(ValueError):
        m.promote(SK.seed_version().version, sid, approved_by="x")
    m.register_setup(sid, comp)                     # the tissue is back: its knowledge is valid again
    assert m.setup_valid(sid) and not m.setup_valid(sid2) and m.incumbent("wipe", sid) is not None


def test_journal_is_append_only_and_rebuilds(tmp_path):
    m = Memory(tmp_path / "a")
    sid, comp = setup()
    m.register_setup(sid, comp)
    v = SK.seed_version()
    m.add_version(v, sid, "incumbent")
    m.record_experiment(setup_id=sid, skill="wipe", version=v.version, kind="run", initial={"region": [0, 0, 1, 1]},
                        versions={"code": "abc"}, change="none", outcome={"value": 0.5}, timing={"s": 1}, interventions=[])
    m.record_adaptation("wipe", sid, {"accepted": False, "from_version": v.version})
    lines = (tmp_path / "a" / "memory.jsonl").read_text().splitlines()
    assert len(lines) == 4 and all(json.loads(l)["table"] for l in lines)
    r = Memory.rebuild(tmp_path / "a" / "memory.jsonl", tmp_path / "b")
    assert r.incumbent("wipe", sid).params == v.params and len(r.experiments("wipe")) == 1


def test_positions_are_held_out():
    held = positions(synth.BOARD, (0.12, 0.10), 5, avoid=[synth.TARGET], min_dist=0.04, seed=2)
    c0 = ((synth.TARGET[0] + synth.TARGET[2]) / 2, (synth.TARGET[1] + synth.TARGET[3]) / 2)
    for r in held:
        c = ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)
        assert ((c[0] - c0[0]) ** 2 + (c[1] - c0[1]) ** 2) ** 0.5 >= 0.04
        assert synth.BOARD.x0 <= r[0] and r[2] <= synth.BOARD.x1 and synth.BOARD.y0 <= r[1] and r[3] <= synth.BOARD.y1


def test_rule_decision_logic():
    rule = PromotionRule(min_trials=2)
    good = {"value": 0.95, "success": True, "strokes": 4, "faults": []}
    bad = {"value": 0.8, "success": False, "strokes": 3, "faults": []}
    rows = [{"baseline": bad, "candidate": good, "budget": 4}] * 2
    assert decide(rule, rows)[0]
    rows2 = [{"baseline": bad, "candidate": {**good, "faults": ["stall"]}, "budget": 4}] * 2
    assert not decide(rule, rows2)[0]
    rows3 = [{"baseline": good, "candidate": good, "budget": 4}] * 2          # not better than the incumbent
    assert not decide(rule, rows3)[0]
    rows4 = [{"baseline": bad, "candidate": {**good, "value": None, "success": None}, "budget": 4}] * 2
    assert not decide(rule, rows4)[0]


def test_adapted_skill_beats_original_on_held_out_positions(tmp_path):
    """The headline demo in simulation: overestimated footprint -> measure -> adapt -> wins on fresh stains."""
    m = Memory(tmp_path)
    sid, comp = setup()
    m.register_setup(sid, comp)
    be = SimWipeBackend(w_true=0.03)
    base = SK.seed_version(w_hat=0.06)                     # footprint overestimated: 6 cm assumed, 3 cm true
    m.add_version(base, sid, "incumbent")
    cand, recs = train_adaptation(SK, base, [Trial(0, synth.TARGET, 11, 4)], be, memory=m, setup_id=sid)
    assert recs[0]["accepted"] and abs(recs[0]["observed"] - 0.03) <= 0.005
    assert cand.parent == base.version and cand.params["w_hat"] < base.params["w_hat"]
    held = positions(synth.BOARD, (0.12, 0.10), 4, avoid=[synth.TARGET], min_dist=0.04, seed=1)
    rep = run_matched(base, cand, [Trial(i, r, 50 + i, 4) for i, r in enumerate(held)], be, PromotionRule(), m, sid,
                      change="w_hat from adaptation")
    assert [r["order"] for r in rep.rows] == ["BC", "CB", "CB", "BC"]
    assert all(r["candidate"]["strokes"] <= 4 and r["baseline"]["strokes"] <= 4 for r in rep.rows)
    assert rep.promote, rep.table()
    assert rep.summary["candidate_mean"] >= 0.9 > rep.summary["baseline_mean"]
    kinds = [e["kind"] for e in m.experiments("wipe", sid)]
    assert kinds[0] == "preregistration" and kinds.count("matched:candidate") == 4
    assert "PROMOTE" in rep.table()
