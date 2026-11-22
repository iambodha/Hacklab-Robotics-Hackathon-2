"""Read-only views over the bottle-learning records (see prompts/bottle-contract.md).

Every reader is cheap to call on each poll: files are re-parsed only when (mtime, size) changes, and
missing or half-written files yield empty results instead of errors.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HACK = Path("/Users/bodha/Documents/b0dha/Robotics-Hackathon")
LEARN_DIR = Path(os.environ.get("BOTTLE_LEARN_DIR", HACK / "logs/learning/bottle"))
ACTIONS_LOG = Path(os.environ.get("ACTIONS_LOG", HACK / "logs/real_sessions_home/actions.jsonl"))
OPERATOR_LOG = os.environ.get("OPERATOR_LOG", "")

LOAD_KEY = re.compile(r"j7.*load|load.*j7", re.I)
SLIP_KEY = re.compile(r"slip", re.I)


def _stat(p: Path):
    try:
        st = p.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


class Cached:
    """Re-run `load(path)` only when the file's (mtime, size) changes."""

    def __init__(self, load):
        self.load, self.key, self.value = load, None, None

    def get(self, path: Path):
        k = (str(path), _stat(path))
        if k != self.key:
            self.key, self.value = k, (self.load(path) if k[1] else None)
        return self.value


def read_jsonl(path: Path, tail_bytes: int | None = None) -> list[dict]:
    try:
        with open(path, "rb") as f:
            if tail_bytes:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - tail_bytes))
                if size > tail_bytes:
                    f.readline()
            raw = f.read()
    except OSError:
        return []
    out = []
    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _find_max(meas: dict | None, pat: re.Pattern):
    vals = [abs(v) for k, v in (meas or {}).items() if pat.search(k) and _num(v) is not None]
    return max(vals) if vals else None


def trace_columns(trace) -> dict[str, list]:
    """Normalise a twist_trace.json (list of rows, dict of columns, or {samples|rows|data: ...}) to columns."""
    if isinstance(trace, dict):
        for k in ("samples", "rows", "data", "trace", "series"):
            if k in trace and isinstance(trace[k], (list, dict)):
                return trace_columns(trace[k])
        return {k: v for k, v in trace.items() if isinstance(v, list)}
    if isinstance(trace, list) and trace and isinstance(trace[0], dict):
        keys: list[str] = []
        for row in trace:
            keys += [k for k in row if k not in keys]
        return {k: [r.get(k) if isinstance(r, dict) else None for r in trace] for k in keys}
    return {}


def _trace_peak_load(cols: dict) -> float | None:
    vals = [abs(v) for k, col in cols.items() if LOAD_KEY.search(k) for v in col if _num(v) is not None]
    return max(vals) if vals else None


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def build_attempts(events: list[dict]) -> list[dict]:
    by: dict[int, dict] = {}
    for e in events:
        n = e.get("attempt")
        if not isinstance(n, int):
            continue
        a = by.setdefault(n, {"attempt": n, "begin": None, "phases": [], "end": None, "judges": []})
        typ = e.get("type")
        if typ == "episode_begin":
            a["begin"] = e
        elif typ == "phase":
            a["phases"].append(e)
        elif typ == "episode_end":
            a["end"] = e
        elif typ == "judge":
            a["judges"].append(e)
    out, prev_params = [], None
    for n in sorted(by):
        a = by[n]
        b, end = a["begin"] or {}, a["end"] or {}
        given = b.get("params") if isinstance(b.get("params"), dict) else {}
        # params may be complete or only the deltas: the effective set carries earlier values forward
        params = {**(prev_params or {}), **given}
        changed = sorted(k for k in given if prev_params is not None and given[k] != prev_params.get(k))
        meas = end.get("measurements") if isinstance(end.get("measurements"), dict) else {}
        last_phase = a["phases"][-1] if a["phases"] else None
        slips = meas.get("slip_events")
        if _num(slips) is None:
            slips = sum(1 for p in a["phases"] if SLIP_KEY.search(json.dumps(p.get("measurements", {})) + str(p.get("note", ""))))
        out.append({
            "attempt": n,
            "t_begin": b.get("t"), "t_end": end.get("t") if end else None,
            "hypothesis": b.get("hypothesis"), "based_on": b.get("based_on") or [],
            "recording_id": b.get("recording_id"),
            "params": params, "params_given": sorted(given), "changed": changed,
            "phases": a["phases"],
            "phase": last_phase.get("phase") if last_phase else None,
            "phase_status": last_phase.get("status") if last_phase else None,
            "done": bool(a["end"]),
            "outcome": end.get("outcome"),
            "rotation_achieved_deg": _num(end.get("rotation_achieved_deg")),
            "cap_lift_mm": _num(end.get("cap_lift_mm")),
            "measurements": meas,
            "peak_j7_load": _find_max(meas, LOAD_KEY),
            "slip_events": slips,
            "lesson": end.get("lesson"), "next_change": end.get("next_change"),
            "judges": a["judges"],
        })
        if params:
            prev_params = params
    return out


class LearnStore:
    def __init__(self, learn_dir: Path = LEARN_DIR):
        self.dir = Path(learn_dir)
        self._episodes = Cached(read_jsonl)
        self._traces: dict[int, Cached] = {}

    @property
    def episodes_path(self) -> Path:
        return self.dir / "episodes.jsonl"

    def version(self) -> str:
        """Changes whenever anything the dashboard shows from LEARN_DIR may have changed."""
        parts = [_stat(self.episodes_path), _stat(self.dir / "SUMMARY.md")]
        try:
            for d in sorted(self.dir.glob("attempt_*")):
                parts.append((d.name, _stat(d), _stat(d / "twist_trace.json"), _stat(d / "report.md")))
        except OSError:
            pass
        return str(hash(tuple(parts)))

    def files(self, n: int) -> list[str]:
        d = self.dir / f"attempt_{n}"
        try:
            return sorted(p.name for p in d.iterdir() if p.is_file())
        except OSError:
            return []

    def trace(self, n: int):
        c = self._traces.setdefault(n, Cached(_load_json))
        return c.get(self.dir / f"attempt_{n}" / "twist_trace.json")

    def state(self) -> dict:
        attempts = build_attempts(self._episodes.get(self.episodes_path) or [])
        for a in attempts:
            a["files"] = self.files(a["attempt"])
            if a["peak_j7_load"] is None and "twist_trace.json" in a["files"]:
                a["peak_j7_load"] = _trace_peak_load(trace_columns(self.trace(a["attempt"])))
        reports = sorted((int(m.group(1)) for d in self._attempt_dirs() if (m := re.fullmatch(r"attempt_(\d+)", d.name))
                          and (d / "report.md").is_file()))
        return {"learn_dir": str(self.dir), "exists": self.dir.is_dir(), "attempts": attempts,
                "summary": (self.dir / "SUMMARY.md").is_file(), "reports": reports, "version": self.version()}

    def _attempt_dirs(self):
        try:
            return [d for d in self.dir.glob("attempt_*") if d.is_dir()]
        except OSError:
            return []

    def resolve(self, rel: str) -> Path | None:
        """Map a request path to a file inside LEARN_DIR, refusing anything that escapes it."""
        base = self.dir.resolve()
        try:
            p = (base / rel).resolve()
        except (OSError, ValueError):
            return None
        return p if p.is_file() and base in p.parents else None


def _short(v, n: int):
    s = v if isinstance(v, str) else json.dumps(v, default=str)
    return s if len(s) <= n else s[:n] + f"... [{len(s) - n} more chars]"


class AgentStore:
    """Tail of the real-arm MCP action log plus the latest assistant text from OPERATOR_LOG."""

    def __init__(self, actions: Path = ACTIONS_LOG, operator_log: str = OPERATOR_LOG):
        self.actions = Path(actions)
        self.operator = Path(operator_log) if operator_log else None
        self._actions = Cached(lambda p: read_jsonl(p, tail_bytes=512_000))
        self._op = Cached(_read_operator)

    def state(self, limit: int = 60) -> dict:
        rows = (self._actions.get(self.actions) or [])[-limit:]
        calls = [{"t": r.get("t"), "tool": r.get("tool"), "args": _short(r.get("args", {}), 400), "s": r.get("s"),
                  "error": r.get("error"), "result": _short(r.get("result") or "", 1800)} for r in rows]
        op = self._op.get(self.operator) if self.operator else None
        return {"actions_path": str(self.actions), "actions": calls,
                "operator_path": str(self.operator) if self.operator else None, "operator": op,
                "version": str(hash((_stat(self.actions), _stat(self.operator) if self.operator else None)))}


ROLE = re.compile(r"^\s*(?:#{1,6}\s*|\*\*|>\s*)?(assistant|user|human|devin|tool|system)\b\s*[:*]*\s*", re.I | re.M)


def _read_operator(path: Path) -> dict | None:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 200_000))
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    marks = list(ROLE.finditer(text))
    latest = None
    for i, m in enumerate(marks):
        if m.group(1).lower() in ("assistant", "devin"):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
            body = text[m.end():end].strip()
            if body:
                latest = body
    if latest is None:
        latest = text[-4000:].strip()
    return {"latest": latest[-6000:], "tail": text[-1500:], "bytes": size}
