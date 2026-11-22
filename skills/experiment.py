"""Matched baseline-vs-candidate experiments with a promotion rule fixed BEFORE the trials run.

- Matched: both arms get the same stain (position, seed) and the same stroke budget.
- Counterbalanced: the arm order alternates ABBA over trials (tissue wear / board history hit both arms equally).
- Held out: test positions are chosen away from the positions the adaptation was trained on.
- Pre-registered: the rule (and its hash) is written to memory before the first trial.

Backends: SimWipeBackend (synthetic world + the real evaluator) here; a hardware backend implements the same
`run(version, trial) -> TrialResult` by sending the skill's proposals to the executor and evaluating the
overview images (see README "needs").
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from dataclasses import dataclass, field

import numpy as np

from ._contracts import Outcome, to_dict
from .model import SkillVersion
from .wipe import WipeSkill, update_width
from .wipe.planner import Region


@dataclass(frozen=True)
class PromotionRule:
    min_reduction: float = 0.90        # a trial is a success at >= 90 % contamination reduction (within the budget)
    min_success_rate: float = 1.0      # fraction of candidate trials that must succeed
    max_faults: int = 0                # no unhandled fault in any candidate trial
    max_unknown: int = 0               # outcomes with success=None (occluded / unmeasurable) allowed
    min_margin: float = 0.02           # candidate mean reduction must beat the incumbent's by at least this
    min_trials: int = 4

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def hash(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Trial:
    index: int
    region: tuple                      # stain / target region (x0, y0, x1, y1), world metres
    seed: int
    budget: int


@dataclass
class TrialResult:
    outcome: Outcome
    width: object | None
    strokes: int
    faults: list = field(default_factory=list)
    interventions: list = field(default_factory=list)
    duration_s: float = 0.0
    stats: dict = field(default_factory=dict)
    executed: list = field(default_factory=list)       # executed stroke segments


class SimWipeBackend:
    """Synthetic world with a known true tissue width; the images go through the real evaluator."""

    def __init__(self, w_true: float = 0.03, efficiency: float = 0.97, push_frac: float = 0.03, offset=(0.0, 0.0),
                 track_noise: float = 0.0015, skill: WipeSkill | None = None, out_dir=None):
        from .eval import synth
        self.synth, self.w_true, self.eff, self.push, self.offset = synth, w_true, efficiency, push_frac, offset
        self.noise, self.skill, self.out_dir = track_noise, skill or WipeSkill(), out_dir
        self.cam = synth.default_camera()
        self.H = self.cam.H_img_world()

    def run(self, version: SkillVersion, trial: Trial, execution_id: str = "") -> TrialResult:
        from .eval import evaluate
        t0 = time.time()
        sk, S = self.skill, self.synth
        plan = sk.plan(version, {"region": trial.region, "budget": trial.budget})
        val = sk.validate(plan)
        eid = execution_id or f"sim-{version.version}-t{trial.index}"
        world = S.SynthWorld().add_stain_rect(trial.region, seed=trial.seed)
        before = world.render(seed=trial.seed * 7 + 1, shadow=(trial.region[0], trial.region[1], 0.05, 0.3))
        if not val.ok:
            out = Outcome(eid, "contamination_reduction", 0.0, "fraction", False, [f"plan rejected: {val.reasons}"])
            return TrialResult(out, None, 0, [f"validation: {r}" for r in val.reasons], duration_s=time.time() - t0)
        rng = np.random.default_rng(trial.seed * 1000 + len(plan.strokes))
        executed = []                                   # "measured" path: commanded + small tracking error
        for s in plan.strokes:
            e = rng.normal(0, self.noise, 2)
            executed.append((tuple(np.add(s.start, e)), tuple(np.add(s.end, e))))
        for _ in range(plan.passes):
            world.wipe(executed, self.w_true, self.eff, self.push, self.offset)
        after = world.render(seed=trial.seed * 7 + 2)
        ev = evaluate(before, after, H=self.H, board=S.BOARD, execution_id=eid, strokes=executed, target=trial.region,
                      evidence=[f"sim:{eid}:before", f"sim:{eid}:after"], out_dir=self.out_dir)
        return TrialResult(ev.outcome, ev.width, len(plan.strokes), [], [], time.time() - t0,
                           {**ev.stats, "plan": plan.summary()}, executed)


def positions(board, size, n: int, avoid=(), min_dist: float = 0.05, margin: float = 0.03, seed: int = 0) -> list[tuple]:
    """n regions of `size` (sx, sy) on the board, at least min_dist (centre to centre) from every region in `avoid`
    and from each other (held-out positions). Deterministic."""
    rng = np.random.default_rng(seed)
    sx, sy = size
    out, centres = [], [((a[0] + a[2]) / 2, (a[1] + a[3]) / 2) for a in avoid]
    for _ in range(5000):
        if len(out) == n:
            break
        cx = rng.uniform(board.x0 + margin + sx / 2, board.x1 - margin - sx / 2)
        cy = rng.uniform(board.y0 + margin + sy / 2, board.y1 - margin - sy / 2)
        if all(math.dist((cx, cy), c) >= min_dist for c in centres):
            centres.append((cx, cy))
            out.append((round(cx - sx / 2, 4), round(cy - sy / 2, 4), round(cx + sx / 2, 4), round(cy + sy / 2, 4)))
    if len(out) < n:
        raise ValueError(f"only {len(out)} of {n} held-out positions fit on the board")
    return out


@dataclass
class ExperimentReport:
    rule: dict
    rule_hash: str
    baseline: str
    candidate: str
    rows: list
    summary: dict
    promote: bool
    reasons: list
    exp_ids: list = field(default_factory=list)

    def table(self) -> str:
        head = "| trial | region (m) | order | baseline red. | candidate red. | strokes b/c | width obs. b/c (mm) |\n|---|---|---|---|---|---|---|"
        lines = [head]
        for r in self.rows:
            def f(v):
                return "unknown" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"
            lines.append(f"| {r['trial']} | {[round(v, 3) for v in r['region']]} | {r['order']} | {f(r['baseline']['value'])}"
                         f" | {f(r['candidate']['value'])} | {r['baseline']['strokes']}/{r['candidate']['strokes']}"
                         f" | {r['baseline']['w_obs_mm']}/{r['candidate']['w_obs_mm']} |")
        s = self.summary
        lines.append(f"\nmean reduction: baseline {s['baseline_mean']:.3f}, candidate {s['candidate_mean']:.3f} "
                     f"(success {s['candidate_successes']}/{s['n']} vs {s['baseline_successes']}/{s['n']}); "
                     f"rule {self.rule_hash}: {'PROMOTE' if self.promote else 'KEEP INCUMBENT'}"
                     + ("" if self.promote else f" ({'; '.join(self.reasons)})"))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {k: v for k, v in dataclasses.asdict(self).items()} | {"table": self.table()}


def _arm(res: TrialResult) -> dict:
    w = res.width
    v = res.outcome.value
    return {"value": None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v),
            "success": res.outcome.success, "strokes": res.strokes, "faults": res.faults,
            "w_obs_mm": None if w is None or w.w_observed is None else round(w.w_observed * 1000, 1),
            "w_valid": None if w is None else w.valid, "duration_s": round(res.duration_s, 3), "evidence": res.outcome.evidence}


def decide(rule: PromotionRule, rows: list) -> tuple[bool, list, dict]:
    n = len(rows)
    b = [r["baseline"]["value"] for r in rows]
    c = [r["candidate"]["value"] for r in rows]
    unknown = sum(v is None for v in c)
    faults = sum(bool(r["candidate"]["faults"]) for r in rows)
    known = [(x, y) for x, y in zip(b, c) if x is not None and y is not None]
    bm = float(np.mean([x for x, _ in known])) if known else float("nan")
    cm = float(np.mean([y for _, y in known])) if known else float("nan")
    succ = sum(r["candidate"]["success"] is True for r in rows)
    summary = {"n": n, "baseline_mean": bm, "candidate_mean": cm, "candidate_successes": succ,
               "baseline_successes": sum(r["baseline"]["success"] is True for r in rows), "unknown": unknown,
               "faults": faults, "over_budget": sum(r["candidate"]["strokes"] > r["budget"] for r in rows)}
    reasons = []
    if n < rule.min_trials:
        reasons.append(f"{n} trials < {rule.min_trials}")
    if faults > rule.max_faults:
        reasons.append(f"{faults} candidate trials with a fault")
    if unknown > rule.max_unknown:
        reasons.append(f"{unknown} candidate outcomes unknown")
    if summary["over_budget"]:
        reasons.append("candidate exceeded the stroke budget")
    if n and succ / n < rule.min_success_rate:
        reasons.append(f"candidate success {succ}/{n} below {rule.min_success_rate:.0%} at >= {rule.min_reduction:.0%} reduction")
    if not known or not cm >= bm + rule.min_margin:
        reasons.append(f"candidate mean {cm:.3f} not better than incumbent {bm:.3f} + {rule.min_margin}")
    return not reasons, reasons, summary


def run_matched(baseline: SkillVersion, candidate: SkillVersion, trials: list[Trial], backend,
                rule: PromotionRule = PromotionRule(), memory=None, setup_id: str = "sim", change: str = "",
                versions: dict | None = None) -> ExperimentReport:
    from .memory import code_version
    versions = {"code": code_version(), "skill": {"baseline": baseline.version, "candidate": candidate.version},
                "setup": setup_id, "model": None, **(versions or {})}
    exp_ids = []
    if memory is not None:                          # pre-registration: the rule is on record before any trial
        exp_ids.append(memory.record_experiment(setup_id=setup_id, skill=candidate.skill, version=candidate.version,
                                                kind="preregistration", initial={"trials": [dataclasses.asdict(t) for t in trials]},
                                                versions=versions, change=change, outcome={"rule": rule.to_dict(), "rule_hash": rule.hash()},
                                                timing={"t": time.time()}, interventions=[]))
    rows = []
    for t in trials:
        order = ("baseline", "candidate") if t.index % 4 in (0, 3) else ("candidate", "baseline")   # ABBA
        row = {"trial": t.index, "region": list(t.region), "seed": t.seed, "budget": t.budget, "order": "".join(o[0].upper() for o in order)}
        for arm in order:
            v = baseline if arm == "baseline" else candidate
            res = backend.run(v, t, execution_id=f"{arm}-t{t.index}")
            row[arm] = _arm(res)
            if memory is not None:
                exp_ids.append(memory.record_experiment(
                    setup_id=setup_id, skill=v.skill, version=v.version, kind=f"matched:{arm}",
                    initial={"trial": dataclasses.asdict(t), "order": row["order"]}, versions=versions, change=change,
                    outcome={**to_dict(res.outcome), "stats": {k: v2 for k, v2 in res.stats.items() if k != "plan"}},
                    timing={"duration_s": res.duration_s}, interventions=res.interventions))
        rows.append(row)
    ok, reasons, summary = decide(rule, rows)
    return ExperimentReport(rule.to_dict(), rule.hash(), baseline.version, candidate.version, rows, summary, ok, reasons, exp_ids)


def train_adaptation(skill: WipeSkill, version: SkillVersion, trials: list[Trial], backend, memory=None,
                     setup_id: str = "sim", alpha: float | None = None) -> tuple[SkillVersion, list[dict]]:
    """Level-1 loop: run, evaluate, update w_hat from valid visible measurements. Returns the final version + records."""
    from .wipe.adapt import ALPHA
    records = []
    for t in trials:
        res = backend.run(version, t, execution_id=f"train-{version.version}-t{t.index}")
        visible = res.outcome.success is not None
        new, rec = update_width(version, skill.spec["w_hat"], res.width or _NoWidth, visible, alpha or ALPHA)
        rec.update(trial=t.index, reduction=res.outcome.value, outcome_evidence=res.outcome.evidence)
        records.append(rec)
        if memory is not None:
            memory.record_adaptation(skill.name, setup_id, rec)
            if new is not None:
                memory.add_version(new, setup_id, "candidate")
        if new is not None:
            version = new
    return version, records


class _NoWidth:
    w_observed, valid, why, interval, evidence = None, False, "no width measurement", None, []
