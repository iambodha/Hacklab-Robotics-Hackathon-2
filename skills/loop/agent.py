"""Improvement agents. Each gets a trace bundle and an isolated sandbox worktree, writes `diagnosis.json` into the
run folder, and may edit files in the sandbox (the guard decides afterwards what was allowed).

diagnosis.json:
  {"agent", "cause": text or "unknown", "confidence": 0..1, "evidence": [...],
   "proposal": {"kind": "param_patch" | "code_patch" | "check" | "none", "description", "changes": {...},
                "check": {...}  # for kind=check: one discriminating experiment (with executor proposals)}}

HeuristicAgent  deterministic rules on the evaluator's numbers; used in tests and as a fallback
CommandAgent    an LLM coding agent in headless mode: Claude Code (`claude -p`) or Devin (`devin -p`)
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .bundle import load
from .guard import ADD_ONLY, ALLOW, FORBID


def _write(run_dir: Path, d: dict) -> dict:
    (Path(run_dir) / "diagnosis.json").write_text(json.dumps(d, indent=1))
    return d


def width_probe(trace: dict) -> dict:
    """Discriminating check: ONE stroke through the middle of a stain wider than any plausible tissue, so both band
    edges are visible and the width is identifiable. Emitted as executor proposals (wipe + park)."""
    from ..wipe import WipeSkill
    sk = WipeSkill()
    x0, y0, x1, y1 = trace["task"]["region"]
    v = sk.seed_version(**{k: trace["incumbent"]["params"][k] for k in sk.spec})
    from ..wipe.planner import Region, plan_coverage
    cy = (y0 + y1) / 2
    plan = plan_coverage(Region(x0, cy - 0.001, x1, cy + 0.001), v.params["w_hat"], v.params["beta"], 1, sk.limits,
                         axis="x", overshoot=v.params["overshoot"])
    props = sk.execute(plan, based_on_obs="<fresh observation>", setup_id=trace["setup_id"], now=0.0, source="loop",
                       ttl_s=60.0, version=v.version, plan_id="width-probe")
    return {"name": "single-stroke width probe", "needs": "a fresh stain covering the whole region (>= 8 cm across the stroke)",
            "measure": "evaluator width on the probe (valid when both band edges are visible)",
            "decides": "whether the tissue footprint differs from w_hat (cause of the residual) or not",
            "proposals": [{"skill": p.skill, "params": {k: v for k, v in p.params.items() if k != "limits"}} for p in props]}


class HeuristicAgent:
    name = "heuristic"

    def run(self, bundle: Path, wt: Path, run_dir: Path) -> dict:
        tr = load(bundle)
        out, ev, inc = tr["outcome"], tr.get("evaluation") or {}, tr["incumbent"]["params"]
        w = ev.get("width") or {}
        base = {"agent": self.name, "evidence": [f"outcome {out['metric']}={out['value']}", f"evaluation {json.dumps(w)}"]}
        if out["success"] is None:
            return _write(run_dir, {**base, "cause": "unknown", "confidence": 0.0,
                                    "why": f"the outcome was not measurable: {ev.get('unknown')}",
                                    "proposal": {"kind": "check", "description": "park the arm and take a new inspection image",
                                                 "check": {"name": "re-inspect", "proposals": [{"skill": "park", "params": {}}]}}})
        if out["success"]:
            return _write(run_dir, {**base, "cause": "no failure", "confidence": 1.0, "proposal": {"kind": "none"}})
        if not w.get("valid"):
            return _write(run_dir, {**base, "cause": "unknown", "confidence": 0.0,
                                    "why": f"residual {1 - out['value']:.0%}, but the tissue width could not be measured "
                                           f"({w.get('why', 'no width')}); footprint, pressure and dirt pushed out are not separable",
                                    "proposal": {"kind": "check", "description": "measure the footprint directly",
                                                 "check": width_probe(tr)}})
        obs, w_hat = w["w_observed_mm"] / 1000, float(inc["w_hat"])
        inside, outside = ev.get("dirt_after_inside_px", 0), ev.get("dirt_after_outside_px", 0)
        if obs < 0.85 * w_hat:
            bounds = json.loads((Path(wt) / "skills/wipe/bounds.json").read_text())["w_hat"]
            new = round(min(max(obs, bounds["lo"]), bounds["hi"]), 4)
            p = Path(wt) / "skills/wipe/defaults.json"
            d = json.loads(p.read_text())
            d["w_hat"] = new
            p.write_text(json.dumps(d, indent=2) + "\n")
            return _write(run_dir, {**base, "confidence": 0.8,
                                    "cause": f"tissue footprint overestimated: it cleans a {obs * 1000:.0f} mm band "
                                             f"(interval {w.get('interval_mm')} mm) but the skill assumes {w_hat * 1000:.0f} mm, "
                                             f"so lanes are spaced too wide and strips along the region edges are missed "
                                             f"(residual {inside} px inside the target, {outside} px outside)",
                                    "proposal": {"kind": "param_patch", "description": f"seed w_hat {w_hat} -> {new} m "
                                                 "(smaller lane spacing, one more lane within the same budget)",
                                                 "changes": {"w_hat": new}}})
        if outside > inside:
            return _write(run_dir, {**base, "cause": "unknown", "confidence": 0.3,
                                    "why": f"most residual is outside the target ({outside} vs {inside} px): dirt pushed past the "
                                           "stroke ends is the likely but unconfirmed cause",
                                    "proposal": {"kind": "check", "description": "repeat with overshoot 0 and compare the piles",
                                                 "check": {"name": "overshoot A/B", "params": {"overshoot": [inc.get("overshoot"), 0.0]}}}})
        return _write(run_dir, {**base, "cause": "unknown", "confidence": 0.0,
                                "why": f"width {obs * 1000:.0f} mm matches w_hat {w_hat * 1000:.0f} mm; residual not explained",
                                "proposal": {"kind": "check", "description": "a second pass per lane on a fresh stain",
                                             "check": {"name": "passes A/B", "params": {"passes": [1, 2]}}}})


PROMPT = """You are the improvement agent of a robot skill system. A wipe run failed; diagnose it and, only if the
evidence supports it, propose ONE change.

Trace bundle (read it first): {bundle}/trace.json (evaluator stats, width estimate, executed strokes, Outcome) and
{bundle}/evidence/ (before | after | mask image). The recording is at the path in trace.json (recorder format).
Your working copy is the current directory: an isolated git worktree. You may edit ONLY: {allow}; you may ADD new
test files matching {add_only}. NEVER edit {forbid} - the evaluator, safety limits, parameter bounds, promotion
rule, tests and their thresholds are protected; a change there rejects your candidate. Do not move any robot or
contact any server. You may run `python -m pytest skills/tests -q`.

Write your result as JSON to {run_dir}/diagnosis.json with keys: agent, cause ("unknown" is an acceptable answer
when the evidence does not separate the explanations), confidence (0-1), evidence (list), proposal {{kind:
param_patch|code_patch|check|none, description, changes}}. If the cause is unknown, propose ONE discriminating check
(an experiment whose result separates the explanations) instead of a patch."""


class CommandAgent:
    """Headless coding agent. `argv` gets {prompt}, {prompt_file}, {run_dir}, {bundle} substituted."""

    PRESETS = {
        "claude": ["claude", "-p", "{prompt}", "--permission-mode", "acceptEdits", "--add-dir", "{run_dir}",
                   "--allowedTools", "Read,Edit,Write,Glob,Grep,Bash(python -m pytest *)", "--output-format", "json"],
        "devin": ["devin", "-p", "--prompt-file", "{prompt_file}"],
    }

    def __init__(self, preset: str = "claude", argv: list[str] | None = None, timeout: float = 1200):
        self.name = preset
        self.argv, self.timeout = argv or self.PRESETS[preset], timeout

    def run(self, bundle: Path, wt: Path, run_dir: Path) -> dict:
        prompt = PROMPT.format(bundle=bundle, run_dir=run_dir, allow=", ".join(ALLOW), add_only=", ".join(ADD_ONLY),
                               forbid=", ".join(FORBID))
        pf = Path(run_dir) / "agent_prompt.txt"
        pf.write_text(prompt)
        argv = [a.format(prompt=prompt, prompt_file=pf, run_dir=run_dir, bundle=bundle) for a in self.argv]
        try:
            r = subprocess.run(argv, cwd=wt, capture_output=True, text=True, timeout=self.timeout)
            (Path(run_dir) / "agent_log.txt").write_text(r.stdout[-20000:] + "\n--- stderr ---\n" + r.stderr[-5000:])
        except (OSError, subprocess.TimeoutExpired) as e:
            (Path(run_dir) / "agent_log.txt").write_text(f"agent failed: {e}")
        p = Path(run_dir) / "diagnosis.json"
        if p.exists():
            try:
                return {"agent": self.name, **json.loads(p.read_text())}
            except json.JSONDecodeError:
                pass
        return _write(run_dir, {"agent": self.name, "cause": "unknown", "confidence": 0.0,
                                "why": "the agent wrote no readable diagnosis.json", "proposal": {"kind": "none"}})


def make(name: str):
    return HeuristicAgent() if name == "heuristic" else CommandAgent(name)
