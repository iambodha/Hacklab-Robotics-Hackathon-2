"""Attempt records for the stir skill, shaped like the bottle's (learning/episodes.py): one append-only JSONL of
episode_begin | phase | judge | episode_end records, a folder per attempt, and a SUMMARY.md regenerated on every end.

    <root>/episodes.jsonl
    <root>/attempt_<N>/      stir_*.jpg (the human-side side view), path.json (the bowl path), report.md
    <root>/SUMMARY.md

Every attempt carries its parameters (the full set and what changed against the previous attempt), the hypothesis
behind the change, the skill's phases with their measurements, the twin judge's verdict and evidence, the outcome and
the lesson. Pure standard library.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

PHASES = ("locate", "grasp", "straighten", "lead_in", "stir", "release")
STATUSES = ("ok", "fail", "abort")


def _fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


class StirStore:
    def __init__(self, root, clock=time.time):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "episodes.jsonl"
        self.clock = clock

    # --- records ------------------------------------------------------------------------------------------------
    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def _append(self, rec: dict) -> dict:
        rec = {"t": round(self.clock(), 3), **rec}
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def attempts(self) -> dict[int, dict]:
        out: dict[int, dict] = {}
        for e in self.events():
            n = e.get("attempt")
            if e["type"] == "episode_begin":
                out[n] = {"begin": e, "phases": [], "judges": [], "end": None}
            elif n in out:
                key = {"phase": "phases", "judge": "judges"}.get(e["type"])
                if key:
                    out[n][key].append(e)
                elif e["type"] == "episode_end":
                    out[n]["end"] = e
        return out

    def open_attempt(self) -> int | None:
        open_ = [n for n, a in self.attempts().items() if a["end"] is None]
        return open_[-1] if open_ else None

    def attempt_dir(self, n: int) -> Path:
        d = self.root / f"attempt_{n}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def begin(self, params: dict, hypothesis: str, **extra) -> int:
        if self.open_attempt() is not None:
            raise RuntimeError(f"attempt {self.open_attempt()} is still open: end it first")
        att = self.attempts()
        n = max(att, default=0) + 1
        prev = att[max(att)]["begin"]["params"] if att else None
        changed = {k: {"before": prev.get(k), "after": v} for k, v in params.items() if prev and prev.get(k) != v}
        self._append({"type": "episode_begin", "attempt": n, "params": params, "changed": changed,
                      "hypothesis": hypothesis, **extra})
        return n

    def phase(self, n: int, phase: str, status: str = "ok", measurements: dict | None = None, note: str = "") -> dict:
        if phase not in PHASES or status not in STATUSES:
            raise ValueError(f"phase must be one of {PHASES}, status one of {STATUSES}")
        return self._append({"type": "phase", "attempt": n, "phase": phase, "status": status,
                             "measurements": measurements or {}, "note": note})

    def judge(self, n: int, verdict: dict, source: str = "twin") -> dict:
        return self._append({"type": "judge", "attempt": n, "source": source, "verdict": verdict["outcome"],
                             **{k: verdict[k] for k in ("stirred", "score", "quality", "terms", "gates", "metrics",
                                                        "evidence") if k in verdict}})

    def end(self, n: int, outcome: str, lesson: str, score: float, next_change: dict | None = None) -> dict:
        rec = self._append({"type": "episode_end", "attempt": n, "outcome": outcome, "lesson": lesson,
                            "score": round(float(score), 3), "next_change": next_change or {}})
        self.write_report(n)
        self.write_summary()
        return rec

    def save_json(self, n: int, name: str, obj) -> str:
        p = self.attempt_dir(n) / name
        p.write_text(json.dumps(obj))
        return str(p.relative_to(self.root))

    def save_image(self, n: int, name: str, jpeg: bytes) -> str:
        p = self.attempt_dir(n) / name
        p.write_bytes(jpeg)
        return str(p.relative_to(self.root))

    # --- reports ------------------------------------------------------------------------------------------------
    def write_report(self, n: int) -> str:
        a = self.attempts()[n]
        b, e = a["begin"], a["end"] or {}
        j = a["judges"][-1] if a["judges"] else {}
        lines = [f"# Stir attempt {n}", "", f"**Hypothesis:** {b['hypothesis']}", "",
                 "**Changed:** " + (", ".join(f"{k} {_fmt(c['before'])} -> {_fmt(c['after'])}"
                                              for k, c in b["changed"].items()) or "nothing (first or repeat)"), "",
                 "| phase | status | measurements |", "|---|---|---|"]
        for p in a["phases"]:
            lines.append(f"| {p['phase']} | {p['status']} | {json.dumps(p['measurements'])[:300]} |")
        lines += ["", f"**Judge (twin ground truth):** {j.get('verdict', '-')}, score {_fmt(j.get('score'))}", ""]
        lines += [f"- {x}" for x in j.get("evidence", [])]
        lines += ["", f"**Outcome:** {e.get('outcome', '-')}", "", f"**Lesson:** {e.get('lesson', '-')}", ""]
        p = self.attempt_dir(n) / "report.md"
        p.write_text("\n".join(lines))
        return str(p)

    def write_summary(self) -> str:
        rows = ["# Stir practice: attempts", "",
                "| # | changed | outcome | score | loops | radius (cm) | pace (loops/s) | wall | hypothesis | lesson |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for n, a in sorted(self.attempts().items()):
            b, e = a["begin"], a["end"] or {}
            m = (a["judges"][-1].get("metrics") if a["judges"] else None) or {}
            ch = ", ".join(f"{k.split('.')[-1]} {_fmt(c['before'])}→{_fmt(c['after'])}" for k, c in b["changed"].items())
            rows.append(f"| {n} | {ch or '-'} | {e.get('outcome', 'open')} | {_fmt(e.get('score'))} | "
                        f"{_fmt(m.get('loops'))} | {_fmt((m.get('radius_about_axis_m') or 0) * 100, 2)} | "
                        f"{_fmt(m.get('pace_loops_per_s'))} | {_fmt(m.get('wall_frac'))} | "
                        f"{b['hypothesis'][:140]} | {(e.get('lesson') or '')[:160]} |")
        p = self.root / "SUMMARY.md"
        p.write_text("\n".join(rows) + "\n")
        return str(p)
