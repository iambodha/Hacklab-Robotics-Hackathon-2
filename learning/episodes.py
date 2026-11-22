"""Episode store for the bottle-opening demo: the single source of truth of prompts/bottle-contract.md.

    LEARN_DIR/episodes.jsonl       append-only: episode_begin | phase | judge | episode_end records
    LEARN_DIR/attempt_<N>/         before_*.jpg, after_*.jpg, check_*.jpg, twist_trace.json, report.md
    LEARN_DIR/SUMMARY.md           table of all attempts + trend + lessons (regenerated on every end)

LEARN_DIR = $BOTTLE_LEARN_DIR or <hackathon root>/logs/learning/bottle. Pure standard library: the real-arm MCP
server runs it without extra packages. The recorder service (:8900) gets one session per attempt with a marker per
phase; when it is down or busy the attempt still runs and the record says so.
"""
from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import skillcfg

DEFAULT_DIR = Path("/Users/bodha/Documents/b0dha/Robotics-Hackathon/logs/learning/bottle")
OUTCOMES = ("opened", "loosened", "no_progress", "slip", "misaligned", "aborted", "fault")
PHASES = ("locate", "approach", "grasp", "twist", "release", "regrasp", "check", "extract", "retreat")
STATUSES = ("ok", "fail", "abort")
MAX_CHANGES = 3            # parameters an attempt may change against the previous one (attempt 1: free)

# Fixed safety limits: never learnable, enforced in code by learning/twist.py.
SAFETY = {
    "max_grip_force": 0.6,          # gripper torque cap (fraction of full servo torque)
    "j7_load_hard": 500,            # wrist-roll present load (0..1000): over this the attempt aborts
    "arm_load_hard": 550,           # any other arm joint
    "below_cap_edge_m": 0.005,      # the TCP never goes lower than the cap's lower edge minus this
    "max_step_deg": 6.0,            # speed cap: degrees of twist per control step
    "min_step_dt": 0.08,            # ... and seconds per step
    "slip_ticks": 25,               # j8 closing further than this during a twist = the jaws slid on the cap
}

# Learnable parameters: (kind, lo, hi, baseline default, meaning). kind: float | int | enum (lo = the choices).
# The defaults in force come from the skill config (params.default.<name>): the learned profile
# learning/profiles/bottle_cap.json, else the baseline column here (the values before 22 Nov's attempts).
PARAMS: dict[str, tuple] = {
    "grip_close": ("float", 0.0, 1.0, 1.0,
                   "jaw target as closed fraction: 1 = close until the jaws stall on the cap"),
    "grip_torque_limit": ("float", 0.1, SAFETY["max_grip_force"], 0.4,
                          "gripper torque cap while closing and holding (fraction of full torque)"),
    "approach_z_offset_mm": ("float", -15.0, 5.0, 0.0,
                             "gripper (TCP) height at the close, relative to 10 mm below the cap top (negative = "
                             "deeper; the fingertips reach 10 mm below the TCP)"),
    "twist_deg_per_stroke": ("float", 15.0, 120.0, 45.0, "counter-clockwise rotation per stroke"),
    "strokes": ("int", 1, 24, 3, "grasp-twist-release-turn back cycles (22 Nov: up to 24, keep spinning until the cap is free)"),
    "twist_speed": ("enum", ("slow", "medium"), None, "slow", "slow = 3 deg per step, medium = 6 deg per step"),
    "lift_per_turn_mm": ("float", 0.0, 4.0, 3.0, "how far the gripper rises per full turn (follow the thread)"),
    "yaw_start_deg": ("float", -90.0, 90.0, 0.0, "jaw closing axis at the first grasp (0 = jaws close along x)"),
    "regrasp_backoff_deg": ("float", 0.0, 120.0, 0.0,
                            "how far the open jaws turn back before regrasping; 0 = the full stroke"),
    "j7_load_abort": ("float", 100.0, float(SAFETY["j7_load_hard"]), 350.0,
                      "end a stroke when the wrist-roll load passes this (0..1000)"),
}


def baseline() -> dict:
    """The code defaults (PARAMS): what attempt 1 of 22 Nov would have started from."""
    return {k: spec[3] for k, spec in PARAMS.items()}


def defaults(values: dict | None = None) -> dict:
    """The episode defaults in force: skill-config keys params.default.<name> (the learned profile, else baseline)."""
    v = skillcfg.active()["values"] if values is None else values
    return {k: v.get(f"params.default.{k}", spec[3]) for k, spec in PARAMS.items()}


def params_table() -> dict[str, tuple]:
    """PARAMS with the bounds and defaults of the active skill config (params.bounds.<name>.lo/hi,
    params.default.<name>; each capped by a hard envelope in skillcfg.PARAM_ENVELOPE)."""
    v = skillcfg.active()["values"]
    d = defaults(v)
    return {k: (kind, lo, hi, d[k], about) if kind == "enum" else
            (kind, v[f"params.bounds.{k}.lo"], v[f"params.bounds.{k}.hi"], d[k], about)
            for k, (kind, lo, hi, _, about) in PARAMS.items()}


def config_changes_since(begin: dict | None, values: dict | None = None) -> dict:
    """{name: {"before", "after"}} for the episode defaults (params.default.*) that changed in the skill config since
    the attempt `begin` ran (its stamped version and overrides): an applied proposal reaches the next attempt."""
    if not begin or not begin.get("skillcfg"):
        return {}
    now = defaults(values)
    then = begin.get("skill_defaults") or defaults({**skillcfg.defaults(), **(begin["skillcfg"].get("overrides") or {})})
    if begin["skillcfg"].get("version") == skillcfg.version() and then == now:
        return {}
    return {k: {"before": then[k], "after": now[k]} for k in PARAMS if then[k] != now[k]}


def bounds_text() -> str:
    out = []
    for k, (kind, lo, hi, dflt, about) in params_table().items():
        rng = "|".join(lo) if kind == "enum" else f"{lo}..{hi}"
        out.append(f"  {k}: {rng} (default {dflt}) - {about}")
    return "\n".join(out)


def validate(params: dict, base: dict | None = None) -> tuple[dict, list[str], list[str]]:
    """(full params, changed keys vs base, errors). Missing keys are inherited from base (else the defaults);
    out-of-range or unknown keys are errors: nothing is clamped."""
    base = dict(base or defaults())
    full, errors = dict(base), []
    table = params_table()
    for k, v in (params or {}).items():
        if k not in table:
            errors.append(f"unknown parameter {k!r} (learnable: {', '.join(PARAMS)})")
            continue
        kind, lo, hi = table[k][:3]
        if kind == "enum":
            if v not in lo:
                errors.append(f"{k}={v!r} must be one of {list(lo)}")
                continue
        else:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                errors.append(f"{k}={v!r} must be a number")
                continue
            if kind == "int" and float(v) != int(v):
                errors.append(f"{k}={v} must be a whole number")
                continue
            if not lo <= v <= hi:
                errors.append(f"{k}={v} is outside its bounds {lo}..{hi}")
                continue
            v = int(v) if kind == "int" else float(v)
        full[k] = v
    changed = [k for k in PARAMS if full.get(k) != base.get(k)]
    return full, changed, errors


class RecorderClient:
    """POST /recordings, /recordings/<id>/markers, /recordings/<id>/stop on the recorder service. Never raises."""

    def __init__(self, url: str | None = None, timeout: float = 3.0):
        self.url = (url or os.environ.get("RECORDER_URL", "http://127.0.0.1:8900")).rstrip("/")
        self.timeout = timeout
        self.enabled = os.environ.get("BOTTLE_RECORD", "1") == "1"

    def _post(self, path: str, body: dict) -> tuple[dict | None, str]:
        if not self.enabled:
            return None, "recording disabled (BOTTLE_RECORD=0)"
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read() or b"{}"), ""
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read()).get("error", "")
            except ValueError:
                msg = ""
            return None, f"recorder HTTP {e.code}: {msg}"
        except (OSError, ValueError) as e:
            return None, f"recorder unreachable at {self.url}: {type(e).__name__}"

    def start(self, name: str, note: str = "") -> tuple[str | None, str]:
        out, err = self._post("/recordings", {"name": name, "note": note})
        return (out or {}).get("id"), err

    def marker(self, rec_id: str | None, label: str, data=None) -> str:
        return self._post(f"/recordings/{rec_id}/markers", {"label": label, "data": data})[1] if rec_id else ""

    def stop(self, rec_id: str | None) -> str:
        return self._post(f"/recordings/{rec_id}/stop", {})[1] if rec_id else ""


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if v == int(v):
            return str(int(v))
        return f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}".rstrip("0").rstrip(".")
    return str(v)


class EpisodeStore:
    def __init__(self, root: str | Path | None = None, recorder: RecorderClient | None = None, clock=time.time):
        self.root = Path(root or os.environ.get("BOTTLE_LEARN_DIR") or DEFAULT_DIR)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log = self.root / "episodes.jsonl"
        self.recorder = recorder if recorder is not None else RecorderClient()
        self.clock = clock

    # --- raw records -------------------------------------------------------------------------------
    def events(self) -> list[dict]:
        if not self.log.exists():
            return []
        out = []
        for line in self.log.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue            # a torn last line after a crash: skip it, keep the rest
        return out

    def _append(self, rec: dict) -> dict:
        rec = {"type": rec.pop("type"), "attempt": rec.pop("attempt"), "t": round(self.clock(), 3), **rec}
        with self.log.open("a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        return rec

    def attempts(self) -> dict[int, dict]:
        """attempt -> {"begin", "phases", "judges", "end"} in file order."""
        out: dict[int, dict] = {}
        for e in self.events():
            n = e.get("attempt")
            if n is None:
                continue
            a = out.setdefault(n, {"begin": None, "phases": [], "judges": [], "end": None})
            kind = e.get("type")
            if kind == "episode_begin":
                a["begin"] = e
            elif kind == "phase":
                a["phases"].append(e)
            elif kind == "judge":
                a["judges"].append(e)
            elif kind == "episode_end":
                a["end"] = e
        return dict(sorted(out.items()))

    def open_attempt(self) -> int | None:
        """The attempt that has begun and not ended (None if there is none)."""
        open_ = [n for n, a in self.attempts().items() if a["begin"] and not a["end"]]
        return open_[-1] if open_ else None

    def attempt_dir(self, n: int) -> Path:
        d = self.root / f"attempt_{n}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_images(self, n: int, prefix: str, images: dict[str, bytes]) -> list[str]:
        paths = []
        for role, data in (images or {}).items():
            if data:
                p = self.attempt_dir(n) / f"{prefix}_{role}.jpg"
                p.write_bytes(data)
                paths.append(str(p))
        return paths

    def save_json(self, n: int, name: str, obj) -> str:
        p = self.attempt_dir(n) / name
        p.write_text(json.dumps(obj, indent=1, default=float))
        return str(p)

    # --- the episode lifecycle -----------------------------------------------------------------------
    def last_params(self) -> dict | None:
        done = [a["begin"]["params"] for a in self.attempts().values() if a["begin"]]
        return done[-1] if done else None

    def begin(self, params: dict, hypothesis: str, based_on: list[int] | None = None,
              images: dict[str, bytes] | None = None) -> dict:
        """Validate and record the start of an attempt. Returns {"ok", "attempt", "params", "changed", "errors",
        "recording_id", "note"}; with errors nothing is written."""
        if self.open_attempt() is not None:
            n = self.open_attempt()
            return {"ok": False, "errors": [f"attempt {n} is still open: call episode_end for it first"]}
        prev = self.last_params()
        from_cfg = {}
        if prev is not None:
            last = [a["begin"] for a in self.attempts().values() if a["begin"]][-1]
            from_cfg = config_changes_since(last)          # an applied proposal changed these defaults: take them over
            base = {**prev, **{k: c["after"] for k, c in from_cfg.items()}}
            full, _, errors = validate(params, base)
            changed = [k for k in PARAMS if full.get(k) != prev.get(k)]
            from_cfg = {k: c for k, c in from_cfg.items() if k not in (params or {}) and full[k] != prev.get(k)}
        else:
            full, changed, errors = validate(params, prev)
        if prev is not None and len(changed) > MAX_CHANGES:
            errors.append(f"{len(changed)} parameters changed ({', '.join(changed)}); change at most {MAX_CHANGES} "
                          f"per attempt so the effect can be attributed")
        if not (hypothesis or "").strip():
            errors.append("a hypothesis is required: what you expect this change to do, and why")
        if errors:
            return {"ok": False, "errors": errors, "params": full, "changed": changed}
        attempts = self.attempts()
        n = (max(attempts) if attempts else 0) + 1
        if based_on is None:
            based_on = [k for k, a in attempts.items() if a["end"]]
        cfg = skillcfg.stamp()
        rec_id, note = self.recorder.start(f"bottle-attempt-{n}", hypothesis[:200])
        self.save_images(n, "before", images or {})
        self._append({"type": "episode_begin", "attempt": n, "params": full, "hypothesis": hypothesis.strip(),
                      "based_on": list(based_on), "recording_id": rec_id, "changed": changed,
                      "skillcfg": cfg, "skill_profile": skillcfg.profile_label(), "skill_defaults": defaults(),
                      **({"from_config": from_cfg} if from_cfg else {}),
                      **({"recording_note": note} if note else {})})
        self.write_report(n)
        return {"ok": True, "attempt": n, "params": full, "changed": changed, "recording_id": rec_id, "note": note,
                "skillcfg": cfg, "from_config": from_cfg}

    def phase(self, n: int, phase: str, status: str = "ok", measurements: dict | None = None, note: str = "") -> dict:
        if phase not in PHASES or status not in STATUSES:
            raise ValueError(f"phase must be one of {PHASES}, status one of {STATUSES}")
        rec = self._append({"type": "phase", "attempt": n, "phase": phase, "status": status,
                            "measurements": measurements or {}, "note": note})
        rid = (self.attempts().get(n, {}).get("begin") or {}).get("recording_id")
        self.recorder.marker(rid, f"{phase}:{status}", {"attempt": n, "note": note[:200]})
        return rec

    def judge(self, n: int, source: str, verdict: str, evidence: list[str], **extra) -> dict:
        return self._append({"type": "judge", "attempt": n, "source": source, "verdict": verdict,
                             "evidence": list(evidence), **extra})

    def end(self, outcome: str, lesson: str, next_change: dict | None = None, images: dict[str, bytes] | None = None,
            rotation_achieved_deg: float | None = None, cap_lift_mm: float | None = None,
            measurements: dict | None = None) -> dict:
        n = self.open_attempt()
        if n is None:
            return {"ok": False, "errors": ["no attempt is open: call episode_begin first"]}
        errors = []
        if outcome not in OUTCOMES:
            errors.append(f"outcome must be one of {list(OUTCOMES)}")
        if not (lesson or "").strip():
            errors.append("a lesson is required: what this attempt showed, in one or two sentences")
        if next_change is not None:
            _, _, bad = validate(next_change, self.attempts()[n]["begin"]["params"])
            errors += [f"next_change: {b}" for b in bad]
        if errors:
            return {"ok": False, "errors": errors}
        a = self.attempts()[n]
        meas = aggregate(a["phases"])
        meas.update(measurements or {})
        rot = meas.get("rotation_achieved_deg", 0.0) if rotation_achieved_deg is None else rotation_achieved_deg
        lift = meas.get("cap_lift_mm") if cap_lift_mm is None else cap_lift_mm
        after = self.save_images(n, "after", images or {})
        rec = self._append({"type": "episode_end", "attempt": n, "outcome": outcome,
                            "rotation_achieved_deg": round(float(rot or 0.0), 1),
                            "cap_lift_mm": None if lift is None else round(float(lift), 1),
                            "measurements": meas, "lesson": lesson.strip(), "next_change": next_change or {},
                            "images": after})
        rid = a["begin"].get("recording_id")
        self.recorder.marker(rid, f"end:{outcome}", {"attempt": n})
        stop_note = self.recorder.stop(rid)
        self.write_report(n)
        self.write_summary()
        return {"ok": True, "attempt": n, "record": rec, "stop_note": stop_note,
                "report": str(self.attempt_dir(n) / "report.md"), "summary": str(self.root / "SUMMARY.md")}

    # --- what the agent reads ------------------------------------------------------------------------------
    def history_text(self, last_n: int = 10) -> str:
        att = self.attempts()
        if not att:
            return ("No attempts yet. This is attempt 1: start from the defaults or from your measurement of the cap.\n"
                    "Learnable parameters and bounds:\n" + bounds_text())
        lines = [f"{len(att)} attempt(s) so far; showing the last {min(last_n, len(att))}."]
        for n, a in list(att.items())[-max(1, int(last_n)):]:
            b, e = a["begin"] or {}, a["end"]
            ch = b.get("changed") or []
            lines.append(f"\n## attempt {n}" + ("" if e else " (STILL OPEN)"))
            lines.append("params: " + ", ".join(f"{k}={_fmt(v)}" for k, v in (b.get("params") or {}).items()))
            if ch:
                lines.append("changed vs previous: " + ", ".join(ch))
            if b.get("from_config"):
                lines.append("taken over from the skill config: " + ", ".join(
                    f"{k} {_fmt(c['before'])} -> {_fmt(c['after'])}" for k, c in b["from_config"].items()))
            if b.get("skillcfg"):
                lines.append("skill config: " + cfg_label(b))
            lines.append(f"hypothesis: {b.get('hypothesis', '')}")
            for p in a["phases"]:
                m = p.get("measurements") or {}
                brief = ", ".join(f"{k}={_fmt(v)}" for k, v in m.items() if not isinstance(v, (list, dict)))
                lines.append(f"  {p['phase']}: {p['status']}" + (f" ({brief})" if brief else "")
                             + (f" - {p['note']}" if p.get("note") else ""))
            for j in a["judges"]:
                lines.append(f"  judge[{j.get('source')}]: {j.get('verdict')}")
            if e:
                lines.append(f"OUTCOME: {e['outcome']}; rotation {_fmt(e.get('rotation_achieved_deg'))} deg; "
                             f"cap lift {_fmt(e.get('cap_lift_mm'))} mm")
                lines.append(f"lesson: {e.get('lesson', '')}")
                if e.get("next_change"):
                    lines.append(f"planned next change: {json.dumps(e['next_change'])}")
        lines.append("\n" + trend_text(att))
        return "\n".join(lines)

    # --- documentation ------------------------------------------------------------------------------------
    def write_report(self, n: int) -> str:
        a = self.attempts().get(n)
        if not a or not a["begin"]:
            return ""
        d, b, e = self.attempt_dir(n), a["begin"], a["end"]
        t0 = b["t"]
        out = [f"# Attempt {n}" + (f": {e['outcome']}" if e else " (in progress)"), "",
               f"Started {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0))}"
               + (f", took {e['t'] - t0:.0f} s" if e else "") + ".", "",
               f"**Hypothesis:** {b['hypothesis']}", "",
               f"Based on attempts: {', '.join(map(str, b.get('based_on') or [])) or 'none'}. "
               f"Changed: {', '.join(b.get('changed') or []) or 'nothing'}.", "",
               f"Skill config: {cfg_label(b)}" + (f"; learned profile {b['skill_profile']}"
                                                  if b.get("skill_profile") else "") + ".", "",
               "| parameter | value |", "|---|---|"]
        out += [f"| {k}{' *' if k in (b.get('changed') or []) else ''} | {_fmt(v)} |" for k, v in b["params"].items()]
        out += ["", "## Phases", "", "| t (s) | phase | status | measurements | note |", "|---|---|---|---|---|"]
        for p in a["phases"]:
            m = p.get("measurements") or {}
            ms = "; ".join(f"{k}={_fmt(v)}" for k, v in m.items() if not isinstance(v, (list, dict)))
            out.append(f"| {p['t'] - t0:.1f} | {p['phase']} | {p['status']} | {ms} | {p.get('note', '')} |")
        strokes = [s for p in a["phases"] if p["phase"] == "twist" for s in (p.get("measurements") or {}).get("strokes", [])]
        if strokes:
            out += ["", "## Strokes", "", "| # | commanded deg | achieved deg | peak j7 load | mean j7 load | j8 change (ticks) "
                    "| z change (mm) | stopped by |", "|---|---|---|---|---|---|---|---|"]
            for s in strokes:
                out.append(f"| {s.get('stroke')} | {_fmt(s.get('commanded_deg'))} | {_fmt(s.get('achieved_deg'))} | "
                           f"{_fmt(s.get('peak_j7_load'))} | {_fmt(s.get('mean_j7_load'))} | {_fmt(s.get('j8_change_ticks'))} | "
                           f"{_fmt(s.get('z_change_mm'))} | {s.get('abort_reason') or '-'} |")
        if a["judges"]:
            out += ["", "## Judgements", ""]
            out += [f"- **{j['source']}**: {j['verdict']}" for j in a["judges"]]
        if e:
            out += ["", "## Outcome", "", f"**{e['outcome']}**: rotation achieved {_fmt(e.get('rotation_achieved_deg'))} deg, "
                    f"cap lift {_fmt(e.get('cap_lift_mm'))} mm.", "", f"**Lesson:** {e['lesson']}", ""]
            if e.get("next_change"):
                out.append(f"**Next change:** `{json.dumps(e['next_change'])}`")
        imgs = sorted(p.name for p in d.glob("*.jpg"))
        if imgs:
            out += ["", "## Images", ""] + [f"![{i}]({i})" for i in imgs]
        if (d / "twist_trace.json").exists():
            out += ["", "Twist trace: [twist_trace.json](twist_trace.json)"]
        rid = b.get("recording_id")
        out += ["", f"Recording: {recording_link(rid) if rid else 'none (' + b.get('recording_note', 'not started') + ')'}"]
        (d / "report.md").write_text("\n".join(out) + "\n")
        return str(d / "report.md")

    def write_summary(self) -> str:
        att = self.attempts()
        out = ["# Bottle opening: attempts", "",
               f"{len(att)} attempt(s); updated {time.strftime('%Y-%m-%d %H:%M:%S')}. Source: `episodes.jsonl`.", "",
               trend_text(att), "",
               "| # | cfg | changed | twist/stroke | strokes | torque | z off | lift/turn | outcome | rotation (deg) "
               "| cap lift (mm) | lesson | links |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for n, a in att.items():
            b, e = a["begin"] or {}, a["end"] or {}
            p = b.get("params") or {}
            rid = b.get("recording_id")
            links = f"[report](attempt_{n}/report.md)" + (f" · [before](attempt_{n}/before_wrist.jpg)"
                                                          if (self.root / f"attempt_{n}/before_wrist.jpg").exists() else "") \
                + (f" · [after](attempt_{n}/after_wrist.jpg)" if (self.root / f"attempt_{n}/after_wrist.jpg").exists() else "") \
                + (f" · [recording]({recording_link(rid)})" if rid else "")
            lesson = (e.get("lesson") or "").replace("|", "/").replace("\n", " ")
            out.append(f"| {n} | {cfg_label(b, short=True)} | {', '.join(b.get('changed') or []) or '-'} | {_fmt(p.get('twist_deg_per_stroke'))} | "
                       f"{_fmt(p.get('strokes'))} | {_fmt(p.get('grip_torque_limit'))} | {_fmt(p.get('approach_z_offset_mm'))} | "
                       f"{_fmt(p.get('lift_per_turn_mm'))} | {e.get('outcome', 'open')} | {_fmt(e.get('rotation_achieved_deg'))} | "
                       f"{_fmt(e.get('cap_lift_mm'))} | {lesson} | {links} |")
        lessons = [(n, a["end"]["lesson"]) for n, a in att.items() if a["end"]]
        if lessons:
            out += ["", "## Lessons, in order", ""] + [f"{n}. {l}" for n, l in lessons]
        per = by_config(att)
        if len(per) > 1 or any(k != "-" for k in per):
            out += ["", "## Outcomes per skill-config version", "", "| cfg | attempts | outcomes | mean rotation (deg) |",
                    "|---|---|---|---|"]
            out += [f"| {k} | {', '.join(map(str, v['attempts']))} | {', '.join(v['outcomes'])} | {v['mean_rotation']:.0f} |"
                    for k, v in per.items()]
        (self.root / "SUMMARY.md").write_text("\n".join(out) + "\n")
        return str(self.root / "SUMMARY.md")


def cfg_label(begin: dict, short: bool = False) -> str:
    """'twin v3' (+ the keys that differ from the code defaults) for an episode_begin record; '-' before skillcfg."""
    c = begin.get("skillcfg")
    if not c:
        return "-"
    lab = f"{c['target']} v{c['version']}"
    if short or not c.get("overrides"):
        return lab + ("" if c.get("overrides") or short else " (code defaults)")
    return lab + " (" + ", ".join(f"{k}={json.dumps(v)}" for k, v in c["overrides"].items()) + ")"


def by_config(att: dict[int, dict]) -> dict[str, dict]:
    """Finished attempts grouped by the skill-config version they ran with, in first-use order."""
    out: dict[str, dict] = {}
    for n, a in att.items():
        if not (a["begin"] and a["end"]):
            continue
        g = out.setdefault(cfg_label(a["begin"], short=True), {"attempts": [], "outcomes": [], "rotations": []})
        g["attempts"].append(n)
        g["outcomes"].append(a["end"]["outcome"])
        g["rotations"].append(float(a["end"].get("rotation_achieved_deg") or 0))
    for g in out.values():
        g["mean_rotation"] = sum(g.pop("rotations")) / len(g["attempts"])
    return out


def recording_link(rid: str) -> str:
    url = os.environ.get("RECORDER_URL", "http://127.0.0.1:8900").rstrip("/")
    return f"{url}/recordings/{rid}/viewer"


def aggregate(phases: list[dict]) -> dict:
    """Attempt-level measurements from the phase records: total achieved rotation, cap lift, slips, aborts."""
    strokes = [s for p in phases if p["phase"] == "twist" for s in (p.get("measurements") or {}).get("strokes", [])]
    out: dict = {"strokes_done": len(strokes),
                 "rotation_achieved_deg": round(sum(max(0.0, float(s.get("achieved_deg") or 0)) for s in strokes), 1)}
    if strokes:
        out["peak_j7_load"] = max(float(s.get("peak_j7_load") or 0) for s in strokes)
        out["slip_strokes"] = sum(1 for s in strokes if s.get("slip"))
        out["z_change_mm"] = round(sum(float(s.get("z_change_mm") or 0) for s in strokes), 1)
    for p in phases:
        m = p.get("measurements") or {}
        if p["phase"] in ("check", "extract") and m.get("cap_lift_mm") is not None:
            out["cap_lift_mm"] = m["cap_lift_mm"]                 # the later lift (extract after check) counts
        if p["phase"] in ("check", "extract") and m.get("cap_free") is not None:
            out["cap_free"] = m["cap_free"]
        if p["phase"] == "twist" and m.get("summary"):
            out.update({k: m[k] for k in ("stopped_by", "strokes_needed", "free_signal") if k in m})
        if p["status"] == "abort":
            out.setdefault("aborts", []).append(f"{p['phase']}: {p.get('note', '')}"[:160])
    return out


def trend_text(att: dict[int, dict]) -> str:
    done = [(n, a["end"]) for n, a in att.items() if a["end"]]
    if not done:
        return "Trend: no finished attempts yet."
    rots = [f"{_fmt(e.get('rotation_achieved_deg'))}" for _, e in done]
    outs = [e["outcome"] for _, e in done]
    best_n, best = max(done, key=lambda ne: (ne[1]["outcome"] == "opened", ne[1].get("rotation_achieved_deg") or 0))
    txt = (f"Trend: rotation per attempt (deg) {' -> '.join(rots)}; outcomes {' -> '.join(outs)}. "
           f"Best so far: attempt {best_n} ({best['outcome']}, {_fmt(best.get('rotation_achieved_deg'))} deg).")
    if len(done) >= 3:
        r = [float(e.get("rotation_achieved_deg") or 0) for _, e in done]
        half = len(r) // 2
        early, late = sum(r[:half]) / half, sum(r[half:]) / (len(r) - half)
        txt += f" Mean rotation: first half {early:.0f} deg, second half {late:.0f} deg" + \
               (" (improving)." if late > early + 5 else " (getting worse)." if late < early - 5 else " (flat).")
    return txt
