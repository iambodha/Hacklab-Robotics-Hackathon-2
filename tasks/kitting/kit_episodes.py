"""Kitting episode records in the bottle's format (learning/episodes.py): one JSONL of episode_begin | phase | judge |
episode_end records, attempt_<N>/report.md per episode and SUMMARY.md, through a subclass of EpisodeStore (its storage,
attempt bookkeeping and image/JSON helpers are reused; learning/ is not edited).

What differs from the bottle: the params are the per-class kit.* values (tasks/kitting/profile.py), phases carry the
object they belong to, the episode outcome is about the whole kit and every object has its own outcome. The recorder
service is never called (sim episodes; RecorderClient disabled).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from learning import episodes as E
from learning import skillcfg as SC

from . import profile as P

PHASES = ("locate", "approach", "grasp", "lift", "carry", "place", "release", "retreat", "check")
OUTCOMES = ("complete", "partial", "failed", "aborted")          # the kit: all placed / some / none / not run
OBJECT_OUTCOMES = ("placed", "off_cell", "missed", "dropped", "blocked", "aborted")


def _no_recorder() -> E.RecorderClient:
    r = E.RecorderClient()
    r.enabled = False
    return r


def validate(params: dict) -> list[str]:
    """Errors of a flat {kit.<class>.<param>: value} dict against the envelopes (nothing is clamped)."""
    errs = []
    for c in P.CLASSES:
        for n, (lo, hi, _, _) in P.PARAMS.items():
            k = P.key(c, n)
            if k not in params:
                errs.append(f"missing {k}")
            elif not lo <= float(params[k]) <= hi:
                errs.append(f"{k}={params[k]} outside {lo}..{hi}")
    return errs


class KitEpisodeStore(E.EpisodeStore):
    def __init__(self, root: str | Path, clock=time.time):
        super().__init__(root, recorder=_no_recorder(), clock=clock)

    def begin(self, params: dict, hypothesis: str, based_on: list[int] | None = None, **context) -> dict:
        if self.open_attempt() is not None:
            return {"ok": False, "errors": [f"attempt {self.open_attempt()} is still open"]}
        errs = validate(params)
        if not (hypothesis or "").strip():
            errs.append("a hypothesis is required: what this configuration should do, and why")
        if errs:
            return {"ok": False, "errors": errs}
        prev = self.last_params()
        changed = [k for k in params if prev is not None and prev.get(k) != params[k]]
        att = self.attempts()
        n = (max(att) if att else 0) + 1
        cfg = SC.stamp()
        self._append({"type": "episode_begin", "attempt": n, "params": params, "hypothesis": hypothesis.strip(),
                      "based_on": list(based_on or []), "recording_id": None, "changed": changed, "skillcfg": cfg,
                      "recording_note": "twin episode (no recorder)", **context})
        return {"ok": True, "attempt": n, "changed": changed, "skillcfg": cfg}

    def phase(self, n: int, phase: str, status: str = "ok", measurements: dict | None = None, note: str = "",
              obj: str | None = None) -> dict:
        if phase not in PHASES or status not in E.STATUSES:
            raise ValueError(f"phase must be one of {PHASES}, status one of {E.STATUSES}")
        return self._append({"type": "phase", "attempt": n, "phase": phase, "status": status, "object": obj,
                             "measurements": measurements or {}, "note": note})

    def end(self, outcome: str, lesson: str, measurements: dict | None = None, next_change: dict | None = None,
            images: dict[str, bytes] | None = None, write_report: bool = True) -> dict:
        n = self.open_attempt()
        if n is None:
            return {"ok": False, "errors": ["no attempt is open"]}
        if outcome not in OUTCOMES or not (lesson or "").strip():
            return {"ok": False, "errors": [f"outcome must be one of {OUTCOMES} and a lesson is required"]}
        after = self.save_images(n, "after", images or {})
        rec = self._append({"type": "episode_end", "attempt": n, "outcome": outcome, "measurements": measurements or {},
                            "lesson": lesson.strip(), "next_change": next_change or {}, "images": after})
        if write_report:
            self.write_report(n)
        return {"ok": True, "attempt": n, "record": rec}

    # --- documents --------------------------------------------------------------------------------------------
    def write_report(self, n: int) -> str:
        a = self.attempts().get(n)
        if not a or not a["begin"]:
            return ""
        b, e = a["begin"], a["end"]
        out = [f"# Kitting episode {n}" + (f": {e['outcome']}" if e else " (in progress)"), "",
               f"Iteration {b.get('iteration', '-')}, seed {b.get('seed', '-')}, skill config {E.cfg_label(b)}.", "",
               f"**Hypothesis:** {b['hypothesis']}", "", "| class | " + " | ".join(P.PARAMS) + " |",
               "|---|" + "---|" * len(P.PARAMS)]
        for c in P.CLASSES:
            out.append(f"| {c} | " + " | ".join(E._fmt(b["params"].get(P.key(c, k))) for k in P.PARAMS) + " |")
        out += ["", "## Phases", "", "| object | phase | status | measurements | note |", "|---|---|---|---|---|"]
        for p in a["phases"]:
            ms = "; ".join(f"{k}={E._fmt(v)}" for k, v in (p.get("measurements") or {}).items() if not isinstance(v, (list, dict)))
            out.append(f"| {p.get('object') or '-'} | {p['phase']} | {p['status']} | {ms} | {p.get('note', '')} |")
        if a["judges"]:
            out += ["", "## Judge (twin truth)", "", "| object | verdict | error mm | diagnosis |", "|---|---|---|---|"]
            out += [f"| {j.get('object')} | {j['verdict']} | {E._fmt(j.get('error_mm'))} | {', '.join(j.get('evidence') or [])} |"
                    for j in a["judges"]]
        if e:
            out += ["", "## Outcome", "", f"**{e['outcome']}**: " + json.dumps({k: v for k, v in e["measurements"].items()
                                                                               if not isinstance(v, dict)}), "",
                    f"**Lesson:** {e['lesson']}"]
        d = self.attempt_dir(n)
        (d / "report.md").write_text("\n".join(out) + "\n")
        return str(d / "report.md")

    def write_summary(self, curve: list[dict] | None = None) -> str:
        att = self.attempts()
        out = ["# Kitting in the twin: episodes", "", f"{len(att)} episode(s); updated {time.strftime('%Y-%m-%d %H:%M:%S')}. "
               "Source: `episodes.jsonl`.", ""]
        if curve:
            out += ["## Learning curve", "", "| iteration | cfg | objects placed | kits complete | mean error mm (placed) |"
                    " screw | cap | box | changes applied after |", "|---|---|---|---|---|---|---|---|---|"]
            for r in curve:
                pc = r["per_class"]
                out.append(f"| {r['iteration']} | v{r['version']} | {r['success_rate']:.0%} | {r['kits_complete']}/{r['episodes']} | "
                           f"{E._fmt(r['mean_error_mm'])} | " + " | ".join(f"{pc[c]['success_rate']:.0%}" for c in P.CLASSES)
                           + f" | {'; '.join(r.get('applied') or []) or '-'} |")
            out.append("")
        out += ["## Episodes", "", "| # | iter | cfg | outcome | placed | mean error mm | lesson |", "|---|---|---|---|---|---|---|"]
        for n, a in att.items():
            b, e = a["begin"] or {}, a["end"] or {}
            m = e.get("measurements") or {}
            out.append(f"| [{n}](attempt_{n}/report.md) | {b.get('iteration', '-')} | {E.cfg_label(b, short=True)} | "
                       f"{e.get('outcome', 'open')} | {m.get('placed', '-')}/{m.get('objects', '-')} | {E._fmt(m.get('mean_error_mm'))} | "
                       f"{(e.get('lesson') or '').replace('|', '/')[:160]} |")
        (self.root / "SUMMARY.md").write_text("\n".join(out) + "\n")
        return str(self.root / "SUMMARY.md")


def record_episode(store: KitEpisodeStore, ep: dict, params_flat: dict, hypothesis: str, **context) -> int:
    """Write one sim_env.run_episode result as an attempt: begin, the per-object phases, a twin judge record per
    object, end (outcome + lesson from the diagnoses)."""
    r = store.begin(params_flat, hypothesis, **context, dr_truth=ep["truth"])
    if not r["ok"]:
        raise ValueError(r["errors"])
    n = r["attempt"]
    store.phase(n, "locate", "ok" if not ep.get("aborted") else "abort", {k: v for k, v in (ep.get("locate") or {}).items()},
                note=f"sheet located from the overview; error vs truth {ep.get('sheet_error_mm')} mm")
    objs = ep["objects"]
    for o in objs:
        for p in o.phases:
            store.phase(n, p["phase"], p["status"], p["measurements"], p.get("note", ""), obj=o.name)
        store.judge(n, "twin", o.outcome, o.diagnosis, object=o.name, cls=o.cls, error_mm=o.truth.get("error_mm"),
                    truth=o.truth)
    m = summarize(objs)
    if ep.get("aborted"):
        outcome, lesson = "aborted", ep["aborted"]
    else:
        outcome = "complete" if m["placed"] == m["objects"] else "failed" if m["placed"] == 0 else "partial"
        lesson = lesson_text(objs)
    store.end(outcome, lesson, m)
    return n


def summarize(objs) -> dict:
    placed = [o for o in objs if o.outcome == "placed"]
    errs = [o.truth["error_mm"] for o in placed]
    out = {"objects": len(objs), "placed": len(placed),
           "mean_error_mm": round(sum(errs) / len(errs), 1) if errs else None,
           "outcomes": {k: sum(1 for o in objs if o.outcome == k) for k in OBJECT_OUTCOMES if any(o.outcome == k for o in objs)}}
    out["per_class"] = {c: {"placed": sum(1 for o in placed if o.cls == c), "of": sum(1 for o in objs if o.cls == c)}
                        for c in P.CLASSES}
    return out


def lesson_text(objs) -> str:
    bad = {}
    for o in objs:
        for t in o.diagnosis:
            bad.setdefault((o.cls, t), []).append(o.name)
    if not bad:
        return "every object placed cleanly."
    return "; ".join(f"{c}: {t} ({len(v)})" for (c, t), v in sorted(bad.items(), key=lambda kv: -len(kv[1])))
