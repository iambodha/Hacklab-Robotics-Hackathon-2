"""PROTECTED. Level-2 improvement loop orchestration (offline, between runs):

    failing trace bundle
      -> sandboxes: candidate worktree (agent edits) + incumbent worktree (baseline arm), both at the same commit
      -> agent: diagnosis (may be "cause unknown") + one patch or one discriminating check
      -> guard: path allowlist + protected bounds, checked from the incumbent copy BEFORE anything runs
      -> regression suite (skills + recorder tests) in the candidate sandbox
      -> replay test (recorder ReplaySource) in both sandboxes: reproduce the failure, counterfactual on the same stain
      -> sim benchmark: matched held-out trials, each arm in its own sandbox, decided by the pre-registered rule
      -> candidate report (CANDIDATE.md, candidate.json, patch.diff) + promotion record
         "awaiting_human_approval" | "rejected" | "needs_check"; hardware promotion only via approve()
"""
from __future__ import annotations

import dataclasses
import json
import shutil
import sys
import time
import uuid
from pathlib import Path

from ..eval import synth
from ..experiment import PromotionRule, Trial, decide, positions
from ..memory import Memory, code_version
from . import guard
from .bundle import load
from .sandbox import Sandbox

REPO = Path(__file__).resolve().parents[2]


def _defaults(root: Path) -> dict:
    return {k: v for k, v in json.loads((root / "skills/wipe/defaults.json").read_text()).items() if not k.startswith("_")}


def bench_trials(trace: dict, n: int = 4, seed: int = 0) -> list[Trial]:
    """Held-out positions: same region size as the failing task, away from where it failed."""
    r = trace["task"]["region"]
    size = (r[2] - r[0], r[3] - r[1])
    from ..eval.geometry import Board
    held = positions(Board(*trace["board"]), size, n, avoid=[r], min_dist=0.04, seed=seed)
    return [Trial(i, p, 1000 + seed * 10 + i, int(trace["task"]["budget"])) for i, p in enumerate(held)]


def _jrun(sb: Sandbox, module: str, args: list[str], out: Path, timeout=600) -> tuple[dict | None, dict]:
    r = sb.python("-m", module, *args, "--out", str(out), timeout=timeout)
    try:
        return json.loads(out.read_text()), r
    except (OSError, json.JSONDecodeError):
        return None, r


def run(bundle: str | Path, *, agent, repo: str | Path = REPO, loop_dir: str | Path | None = None,
        memory: Memory | None = None, rule: PromotionRule = PromotionRule(), keep: bool = False, n_trials: int = 4) -> dict:
    repo = Path(repo).resolve()
    loop_dir = Path(loop_dir or repo / "skills" / ".loop")
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    rd = (loop_dir / "runs" / run_id).resolve()
    rd.mkdir(parents=True)
    shutil.copytree(bundle, rd / "bundle")
    trace = load(rd / "bundle")
    memory = memory or Memory(loop_dir / "memory")
    t0 = time.time()
    report = {"run_id": run_id, "run_dir": str(rd), "bundle": str(rd / "bundle"), "skill": trace["skill"],
              "setup_id": trace["setup_id"], "incumbent": trace["incumbent"], "rule": rule.to_dict(), "rule_hash": rule.hash(),
              "code_version": code_version(repo), "agent": getattr(agent, "name", str(agent)), "steps": {}}
    # pre-registration: the decision rule is on record before the agent or any test runs
    memory.register_setup(trace["setup_id"], {"source": "loop", "setup_id": trace["setup_id"]}, scope=f"loop:{trace['setup_id']}")
    report["prereg_exp"] = memory.record_experiment(
        setup_id=trace["setup_id"], skill=trace["skill"], version=trace["incumbent"]["version"], kind="loop:preregistration",
        initial={"bundle": str(bundle), "run_id": run_id}, versions={"code": report["code_version"]}, change="",
        outcome={"rule": rule.to_dict(), "rule_hash": rule.hash()}, timing={"t": t0}, interventions=[])
    cand = Sandbox.create(repo, rd / "candidate")
    inc = Sandbox.create(repo, rd / "incumbent", cand.base)
    report["base"] = cand.base
    try:
        diag = agent.run(rd / "bundle", cand.path, rd)
        report["diagnosis"] = diag
        g = guard.check(cand.path, cand.base, inc.path / "skills/wipe/bounds.json")
        report["steps"]["guard"] = g.to_dict()
        diff = cand.diff()
        (rd / "patch.diff").write_text(diff)
        kind = (diag.get("proposal") or {}).get("kind", "none")
        if not g.ok:
            return _finish(report, rd, memory, "rejected", f"guard: {g.violations}")
        if kind in ("check", "none") or not g.changed:
            return _finish(report, rd, memory, "needs_check" if kind == "check" else "no_candidate",
                           (diag.get("proposal") or {}).get("description", "no change proposed"))
        inc_params = dict(trace["incumbent"]["params"])
        base_d, cand_d = _defaults(inc.path), _defaults(cand.path)
        changes = {k: v for k, v in cand_d.items() if base_d.get(k) != v}
        cand_params = {**inc_params, **changes}
        report["changes"] = changes
        report["candidate_params"] = cand_params
        # regression suite
        r = cand.python("-m", "pytest", "skills/tests", "recorder/tests", "-q", "-x", "-p", "no:cacheprovider", timeout=900)
        report["steps"]["regression"] = {"ok": r["returncode"] == 0, "tail": r["stdout"][-1500:], "stderr": r["stderr"][-800:]}
        if r["returncode"] != 0:
            return _finish(report, rd, memory, "rejected", "regression suite failed")
        # replay test
        rp = {}
        for name, sb, params in (("incumbent", inc, inc_params), ("candidate", cand, cand_params)):
            res, raw = _jrun(sb, "skills.loop.replay_check", ["--bundle", str(rd / "bundle"), "--params", json.dumps(params)],
                             rd / f"replay_{name}.json")
            rp[name] = res or {"error": raw["stderr"][-1500:]}
        report["steps"]["replay"] = rp
        if not (rp["incumbent"].get("reproduced") and rp["candidate"].get("reproduced")):
            return _finish(report, rd, memory, "rejected", "replay did not reproduce the recorded outcome")
        # sim benchmark on held-out positions, both arms on identical trials
        trials = bench_trials(trace, n_trials)
        (rd / "trials.json").write_text(json.dumps([dataclasses.asdict(t) for t in trials]))
        w = (trace.get("evaluation") or {}).get("width") or {}
        wts = [w["w_observed_mm"] / 1000] if w.get("valid") else [0.02, 0.03, 0.04, 0.05]
        arms = {}
        for name, sb, params in (("incumbent", inc, inc_params), ("candidate", cand, cand_params)):
            args = ["--params", json.dumps(params), "--trials", str(rd / "trials.json")]
            for wt in wts:
                args += ["--w-true", str(wt)]
            res, raw = _jrun(sb, "skills.loop.bench", args, rd / f"bench_{name}.json")
            arms[name] = res or {"rows": [], "error": raw["stderr"][-1500:]}
        rows = [{"trial": b["trial"]["index"], "w_true": b["w_true"], "budget": b["trial"]["budget"], "baseline": b, "candidate": c}
                for b, c in zip(arms["incumbent"]["rows"], arms["candidate"]["rows"])]
        ok, reasons, summary = decide(rule, rows)
        report["steps"]["bench"] = {"w_true": wts, "trials": [dataclasses.asdict(t) for t in trials], "summary": summary,
                                    "promote": ok, "reasons": reasons, "twin": arms["candidate"].get("twin"),
                                    "table": [{"trial": r["trial"], "w_true": r["w_true"], "incumbent": r["baseline"]["value"],
                                               "candidate": r["candidate"]["value"], "strokes": [r["baseline"]["strokes"],
                                                                                                 r["candidate"]["strokes"]]} for r in rows]}
        for name, row_key in (("incumbent", "baseline"), ("candidate", "candidate")):
            memory.record_experiment(setup_id=trace["setup_id"], skill=trace["skill"],
                                     version=trace["incumbent"]["version"] if name == "incumbent" else "candidate:" + run_id,
                                     kind=f"loop:bench:{name}", initial={"trials": report["steps"]["bench"]["trials"], "w_true": wts},
                                     versions={"code": report["code_version"], "base": cand.base}, change=json.dumps(changes),
                                     outcome={"rows": [r[row_key] for r in rows]}, timing={"t": time.time()}, interventions=[])
        return _finish(report, rd, memory, "awaiting_human_approval" if ok else "rejected",
                       "sim benchmark passed the pre-registered rule" if ok else f"benchmark: {reasons}")
    finally:
        report["duration_s"] = round(time.time() - t0, 2)
        if not keep:
            cand.remove()
            inc.remove()


def _finish(report: dict, rd: Path, memory: Memory, decision: str, why: str) -> dict:
    from ..model import SkillVersion
    report["decision"], report["why"] = decision, why
    inc = report["incumbent"]
    cand_version = None
    if "candidate_params" in report:
        parent = SkillVersion(inc["skill"], inc["version"], inc.get("parent"), inc["params"])
        cv = parent.child(report["changes"], f"loop:{report['agent']}", report["diagnosis"].get("cause", ""))
        cand_version = cv.version
        if memory.version(inc["version"], report["setup_id"]) is None:
            memory.add_version(parent, report["setup_id"], "incumbent")
        memory.add_version(cv, report["setup_id"], "candidate" if decision == "awaiting_human_approval" else "rejected")
    report["candidate_version"] = cand_version
    report["promotion_id"] = memory.record_promotion(
        skill=report["skill"], setup_id=report["setup_id"], candidate=cand_version or "-", incumbent=inc["version"],
        rule={"rule": report["rule"], "hash": report["rule_hash"]},
        evidence={"run_dir": str(rd), "diagnosis": report.get("diagnosis"), "bench": report["steps"].get("bench", {}).get("summary"),
                  "guard": report["steps"].get("guard"), "patch": str(rd / "patch.diff")}, decision=decision)
    (rd / "candidate.json").write_text(json.dumps(report, indent=1, default=str))
    (rd / "CANDIDATE.md").write_text(render(report))
    return report


def render(r: dict) -> str:
    d = r.get("diagnosis") or {}
    p = d.get("proposal") or {}
    s = r["steps"]
    lines = [f"# Candidate report {r['run_id']}", "",
             f"- skill: {r['skill']}  setup: {r['setup_id']}  base commit: {r.get('base')}  agent: {r['agent']}",
             f"- incumbent: {r['incumbent']['version']} {r['incumbent']['params']}",
             f"- **decision: {r['decision']}** ({r['why']})",
             f"- promotion record: {r.get('promotion_id')} (rule {r['rule_hash']}: {r['rule']})", "",
             "## Diagnosis", f"cause: {d.get('cause')} (confidence {d.get('confidence')})"]
    if d.get("why"):
        lines.append(f"why: {d['why']}")
    lines += [f"proposal: {p.get('kind')}: {p.get('description')}", ""]
    if p.get("check"):
        lines += ["## Discriminating check", "```json", json.dumps(p["check"], indent=1), "```", ""]
    if "guard" in s:
        lines += ["## Guard", f"ok: {s['guard']['ok']}  changed: {s['guard']['changed']}  violations: {s['guard']['violations']}", ""]
    if "regression" in s:
        lines += ["## Regression suite", f"ok: {s['regression']['ok']}", "```", s["regression"]["tail"].strip()[-600:], "```", ""]
    if "replay" in s:
        lines.append("## Replay (recorder ReplaySource)")
        for k, v in s["replay"].items():
            lines.append(f"- {k}: reproduced={v.get('reproduced')} recorded={v.get('recorded_value')} "
                         f"counterfactual on the recorded stain={[round(c['reduction'], 3) for c in v.get('counterfactual', [])]}")
        lines.append("")
    if "bench" in s:
        b = s["bench"]
        lines += ["## Sim benchmark (held-out positions, same stroke budget)", "| trial | w_true mm | incumbent | candidate | strokes i/c |",
                  "|---|---|---|---|---|"]
        for t in b["table"]:
            f = (lambda v: "unknown" if v is None else f"{v:.3f}")
            lines.append(f"| {t['trial']} | {t['w_true'] * 1000:.0f} | {f(t['incumbent'])} | {f(t['candidate'])} | {t['strokes'][0]}/{t['strokes'][1]} |")
        lines += [f"\nsummary: {b['summary']}", f"twin: {b['twin']}", ""]
    lines += ["## Next step",
              "Hardware promotion needs a human: `python -m skills.loop approve "
              f"{r.get('promotion_id')} --by <name>`" if r["decision"] == "awaiting_human_approval" else
              "Nothing to promote." + (" Run the discriminating check above." if r["decision"] == "needs_check" else "")]
    return "\n".join(lines) + "\n"


def approve(memory: Memory, promotion_id: str, approver: str) -> dict:
    """The explicit human approval step: only a candidate that passed the rule can become the incumbent."""
    rows = [p for p in memory.promotions() if p["id"] == promotion_id]
    if not rows:
        raise KeyError(promotion_id)
    p = rows[0]
    if p["decision"] != "awaiting_human_approval":
        raise PermissionError(f"{promotion_id} is {p['decision']}, not awaiting approval")
    if any(q["decision"] == "approved" and q["evidence"].get("approves") == promotion_id for q in memory.promotions()):
        raise PermissionError(f"{promotion_id} was already approved")
    memory.promote(p["candidate"], p["setup_id"], approver)
    pid = memory.record_promotion(skill=p["skill"], setup_id=p["setup_id"], candidate=p["candidate"], incumbent=p["incumbent"],
                                  rule=p["rule"], evidence={"approves": promotion_id, "run": p["evidence"].get("run_dir")},
                                  decision="approved", approved_by=approver)
    return {"approved": promotion_id, "record": pid, "incumbent": p["candidate"], "by": approver,
            "patch": p["evidence"].get("patch"), "note": "code changes in patch.diff still go through the integrator's review"}


def synthetic(loop_dir: str | Path | None = None, agent=None, repo: str | Path = REPO, keep: bool = False, **kw) -> dict:
    """End to end on the synthetic failure (overestimated tissue footprint)."""
    from .agent import HeuristicAgent
    from .bundle import synthetic_failure
    loop_dir = Path(loop_dir or Path(repo) / "skills" / ".loop")
    src = loop_dir / "synthetic" / time.strftime("%Y%m%d-%H%M%S")
    b = synthetic_failure(src)
    return run(b, agent=agent or HeuristicAgent(), repo=repo, loop_dir=loop_dir, keep=keep, **kw)


if __name__ == "__main__":
    print(json.dumps(synthetic(), indent=1, default=str)[:3000], file=sys.stdout)
