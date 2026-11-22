"""Level-2 loop end to end on the synthetic failure, in a throwaway git repo; guardrails against forbidden edits."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from skills.loop import guard, runner
from skills.loop.agent import HeuristicAgent, _write
from skills.loop.bundle import load, synthetic_failure
from skills.loop.sandbox import Sandbox
from skills.memory import Memory

pytestmark = pytest.mark.skipif(bool(os.environ.get("SKILLS_LOOP_INNER")), reason="no loop inside the loop's own sandbox")
ROOT = Path(__file__).resolve().parents[2]


def _git(*a, cwd):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=cwd, check=True, capture_output=True)


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    """A minimal copy of this repo (skills + recorder + robot profiles) as its own git repo."""
    r = tmp_path_factory.mktemp("repo")
    ign = shutil.ignore_patterns("__pycache__", ".loop", "data", "*.pyc", ".pytest_cache")
    shutil.copytree(ROOT / "skills", r / "skills", ignore=ign)
    shutil.copytree(ROOT / "recorder", r / "recorder", ignore=ign)
    shutil.copytree(ROOT / "sites", r / "sites", ignore=ign)  # registry dependency of recorder / bundles
    shutil.copytree(ROOT / "sim" / "robots", r / "sim" / "robots", ignore=ign)
    shutil.copy(ROOT / ".gitignore", r / ".gitignore")
    _git("init", "-q", cwd=r)
    _git("add", "-A", cwd=r)
    _git("commit", "-qm", "base", cwd=r)
    return r


@pytest.fixture(scope="module")
def failure(tmp_path_factory):
    return synthetic_failure(tmp_path_factory.mktemp("synthetic"))


def test_synthetic_failure_bundle(failure):
    tr = load(failure)
    assert tr["outcome"]["success"] is False and tr["outcome"]["value"] < 0.9
    assert tr["incumbent"]["params"]["w_hat"] == 0.06
    assert tr["evaluation"]["width"]["valid"] and abs(tr["evaluation"]["width"]["w_observed_mm"] - 30) <= 5
    labels = [m["label"] for m in tr["recording"]["markers"]]
    assert {"inspect:before", "inspect:after", "outcome", "skill:plan"} <= set(labels)
    assert tr["evidence"] and (failure / tr["evidence"][0]).exists()


def test_loop_end_to_end_and_human_approval(repo, failure, tmp_path):
    mem = Memory(tmp_path / "mem")
    rep = runner.run(failure, agent=HeuristicAgent(), repo=repo, loop_dir=tmp_path / "loop", memory=mem)
    s = rep["steps"]
    assert s["guard"]["ok"] and s["guard"]["changed"] == [["M", "skills/wipe/defaults.json"]]
    assert s["regression"]["ok"], s["regression"]
    assert s["replay"]["incumbent"]["reproduced"] and s["replay"]["candidate"]["reproduced"]
    cf_i = s["replay"]["incumbent"]["counterfactual"][0]["reduction"]
    cf_c = s["replay"]["candidate"]["counterfactual"][0]["reduction"]
    assert cf_c > cf_i + 0.1                                         # better on the very stain that failed
    assert abs(rep["candidate_params"]["w_hat"] - 0.03) <= 0.005    # smaller spacing proposed
    b = s["bench"]["summary"]
    assert b["candidate_mean"] >= 0.9 > b["baseline_mean"] and b["over_budget"] == 0
    assert rep["decision"] == "awaiting_human_approval"
    rd = Path(rep["run_dir"])
    assert (rd / "CANDIDATE.md").exists() and "awaiting_human_approval" in (rd / "CANDIDATE.md").read_text()
    assert not (rd / "candidate").exists() and not (rd / "incumbent").exists()      # sandboxes cleaned up
    # nothing is promoted until a human approves
    assert mem.incumbent("wipe", rep["setup_id"]).version == rep["incumbent"]["version"]
    with pytest.raises(PermissionError):
        runner.approve(mem, rep["promotion_id"], "")
    out = runner.approve(mem, rep["promotion_id"], "operator")
    assert mem.incumbent("wipe", rep["setup_id"]).version == rep["candidate_version"] == out["incumbent"]
    assert mem.lineage(rep["candidate_version"])[1] == rep["incumbent"]["version"]
    with pytest.raises(PermissionError):
        runner.approve(mem, rep["promotion_id"], "operator")


class CheatingAgent:
    """Tries to pass by loosening the evaluator's success threshold (and the bounds)."""
    name = "cheater"

    def run(self, bundle, wt, run_dir):
        p = Path(wt) / "skills/eval/wipe_eval.py"
        p.write_text(p.read_text().replace("success_reduction: float = 0.90", "success_reduction: float = 0.10"))
        b = Path(wt) / "skills/wipe/bounds.json"
        b.write_text(b.read_text().replace('"lo": 0.01', '"lo": 0.001'))
        return _write(run_dir, {"agent": self.name, "cause": "x", "proposal": {"kind": "code_patch"}})


def test_loop_rejects_forbidden_edits_before_running_anything(repo, failure, tmp_path):
    rep = runner.run(failure, agent=CheatingAgent(), repo=repo, loop_dir=tmp_path / "loop", memory=Memory(tmp_path / "m"))
    assert rep["decision"] == "rejected" and "guard" in rep["why"]
    v = " ".join(rep["steps"]["guard"]["violations"])
    assert "skills/eval/wipe_eval.py" in v and "skills/wipe/bounds.json" in v
    assert "regression" not in rep["steps"] and "bench" not in rep["steps"]


def test_unknown_cause_gives_a_discriminating_check(repo, failure, tmp_path):
    b = tmp_path / "bundle"
    shutil.copytree(failure, b)
    tr = json.loads((b / "trace.json").read_text())
    tr["evaluation"]["width"] = {"valid": False, "why": "not identifiable", "w_observed_mm": 60.0}
    (b / "trace.json").write_text(json.dumps(tr))
    rep = runner.run(b, agent=HeuristicAgent(), repo=repo, loop_dir=tmp_path / "loop", memory=Memory(tmp_path / "m"))
    assert rep["decision"] == "needs_check" and rep["diagnosis"]["cause"] == "unknown"
    chk = rep["diagnosis"]["proposal"]["check"]
    assert [p["skill"] for p in chk["proposals"]] == ["wipe", "park"]


@pytest.mark.parametrize("path,content,ok", [
    ("skills/wipe/defaults.json", None, True),
    ("skills/wipe/planner.py", "# tweak\n", True),
    ("skills/tests/candidate/test_new.py", "def test_x():\n    pass\n", True),
    ("skills/eval/dirt.py", "# x\n", False),
    ("skills/eval/new_helper.py", "x = 1\n", False),
    ("skills/tests/test_eval.py", "# loosen\n", False),
    ("skills/wipe/limits.py", "# x\n", False),
    ("skills/experiment.py", "# x\n", False),
    ("skills/loop/guard.py", "ALLOW = ['*']\n", False),
    ("sim/robots/elrobot.json", "{}", False),
    ("executor/validate.py", "x = 1\n", False),
    ("README.md", "x\n", False),
])
def test_guard_paths(repo, tmp_path, path, content, ok):
    sb = Sandbox.create(repo, tmp_path / "sb")
    try:
        p = sb.path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        if content is None:
            d = json.loads(p.read_text())
            d["w_hat"] = 0.03
            p.write_text(json.dumps(d))
        else:
            p.write_text((p.read_text() if p.exists() else "") + content)
        g = guard.check(sb.path, sb.base, repo / "skills/wipe/bounds.json")
        assert g.ok is ok, g.violations
    finally:
        sb.remove()


def test_guard_param_bounds_and_hidden_files(repo, tmp_path):
    sb = Sandbox.create(repo, tmp_path / "sb")
    try:
        p = sb.path / "skills/wipe/defaults.json"
        d = json.loads(p.read_text())
        d["w_hat"] = 0.5                                         # outside the protected bounds
        p.write_text(json.dumps(d))
        g = guard.check(sb.path, sb.base, repo / "skills/wipe/bounds.json")
        assert not g.ok and any("outside protected bounds" in v for v in g.violations)
        p.write_text(json.dumps({**d, "w_hat": 0.03}))
        (sb.path / "skills/.loop").mkdir(parents=True, exist_ok=True)    # gitignored location
        (sb.path / "skills/.loop/override.py").write_text("x = 1\n")
        g = guard.check(sb.path, sb.base, repo / "skills/wipe/bounds.json")
        assert not g.ok and any("override.py" in v for v in g.violations)
        _git("add", "-A", cwd=sb.path)                            # committing does not hide a change either
        (sb.path / "skills/.loop/override.py").unlink()
        e = sb.path / "skills/eval/wipe_eval.py"
        e.write_text(e.read_text() + "\n")
        _git("commit", "-qam", "sneak", cwd=sb.path)
        g = guard.check(sb.path, sb.base, repo / "skills/wipe/bounds.json")
        assert not g.ok and any("wipe_eval.py" in v for v in g.violations)
    finally:
        sb.remove()
