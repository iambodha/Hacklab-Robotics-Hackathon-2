"""Versioned skill configuration: the strategy settings a skill reads at call time, changeable without a code edit or
a restart, through proposals that are validated, checked offline and recorded (learning/SELF_TUNING.md).

    SKILLCFG_DIR/changes.jsonl        append-only: one record per applied version (and per rejected proposal)
    SKILLCFG_DIR/<target>.json        current config of a target: {"target", "version", "values": {key: value}}
    SKILLCFG_DIR/proposals/<id>.json  one proposal: changes (before/after), reason, evidence, checks, status

Targets: "real" (the real arm: a version only after a human approves, `python -m learning.skillcfg approve <id>`) and
"twin" (MuJoCo: proposals made there are applied automatically when their checks pass). A process reads one target:
set_target(), else env SKILLCFG_TARGET, else "real" (so a server that forgets to choose never runs unapproved values).

Keys have a class: "strategy" (changeable within the key's envelope) or "safety" (listed, never changeable here).
Values not in the file are the code defaults: with no file every reader behaves exactly as before. Reads go through an
mtime cache (as sim/real_joints.py), so a running server sees a new version on its next call. Pure standard library:
the real-arm MCP server and episodes.py import it.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import sys
import time
import uuid
from pathlib import Path

DEFAULT_DIR = Path("/Users/bodha/Documents/b0dha/Robotics-Hackathon/logs/skillcfg")
REPO = Path(__file__).resolve().parents[1]
TARGETS = ("real", "twin")
SEARCHES = ("least_lean", "near_first")
_target: str | None = None

# Code defaults of the twist_cap strategy (learning/twist.py takes its module constants from here).
TWIST_DEFAULTS = {
    "twist.approach": "auto",
    "twist.side_lean_deg": 90.0,      # a grasp pose leaning more than this is approached from the side, not from above.
                                      # 22 Nov: always from above (hover + guarded vertical descent); with the least-lean
                                      # IK the real cap is reached at ~35 deg, and the side approach slid the jaws up onto
                                      # the cap's bevel
    "twist.hovers_m": [0.05, 0.035, 0.025],   # m above the cap top before the descent: the highest one reachable
    "twist.max_lean_deg": 75.0,       # deg from vertical: steeper poses are not used for the cap
    "twist.grasp_below_top_m": 0.010,  # nominal TCP height below the cap top (approach_z_offset_mm is added)
    "twist.release_open": 0.25,       # how much the jaws open (norm) to release the cap between strokes
}


# --- the registry ----------------------------------------------------------------------------------------------
def _profile_ik() -> dict:
    """The robot profile's ik section (sim/robots/<ROBOT_PROFILE or elrobot>.json): source of the IK defaults."""
    choice = os.environ.get("ROBOT_PROFILE") or "elrobot"
    p = Path(choice)
    if not p.exists():
        p = REPO / "sim" / "robots" / (choice if choice.endswith(".json") else choice + ".json")
    return dict(_ik_of(str(p), p.stat().st_mtime_ns))


@functools.lru_cache(maxsize=8)
def _ik_of(path: str, mtime_ns: int) -> tuple:
    return tuple(json.loads(Path(path).read_text()).get("ik", {}).items())


# hard envelopes of the episode parameters' bounds: a bound may move inside these, never outside
PARAM_ENVELOPE = {
    "grip_close": (0.0, 1.0),
    "grip_torque_limit": (0.05, "max_grip_force"),
    "approach_z_offset_mm": (-20.0, 10.0),
    "twist_deg_per_stroke": (5.0, 180.0),
    "strokes": (1, 48),
    "lift_per_turn_mm": (0.0, 6.0),
    "yaw_start_deg": (-180.0, 180.0),
    "regrasp_backoff_deg": (0.0, 180.0),
    "j7_load_abort": (50.0, "j7_load_hard"),
}


def _key(kind, default, about, readers, cls="strategy", lo=None, hi=None, choices=None, item=None, length=None):
    return {"kind": kind, "default": default, "about": about, "readers": readers, "class": cls, "lo": lo, "hi": hi,
            "choices": choices, "item": item, "length": length}


@functools.lru_cache(maxsize=4)
def _registry(profile_key: str) -> dict:
    from . import episodes as E
    T = TWIST_DEFAULTS
    ik = _profile_ik()
    reg = {
        "ik.leans": _key("list", [int(v) for v in ik.get("leans_small", []) + ik.get("leans_large", [])],
                         "gripper leans (deg from vertical; + = away from the base) the jaws-down IK may use, tried in "
                         "order of size", "sim/real_arm_mcp._solve_tcp, sim/build_scene.ArmIK.solve_least_lean, "
                         "executor/kinematics.Kinematics.solve", item=("int", -80, 80), length=(1, 32)),
        "ik.search": _key("enum", "least_lean", "least_lean: the least lean reachable anywhere wins (random restarts "
                          "for smaller leans than the one found near the current joints); near_first: the first lean "
                          "reachable from the current joints wins (pre-22 Nov order)",
                          "sim/real_arm_mcp._solve_tcp, sim/build_scene.ArmIK.solve_least_lean", choices=SEARCHES),
        "ik.restarts": _key("int", int(ik.get("restarts", 12)), "random restarts per lean in the IK search",
                            "real_arm_mcp, build_scene, executor", lo=0, hi=32),
        "executor.ik.search": _key("enum", "near_first", "the executor's IK search order (near_first plans ~50x "
                                   "faster for high lifts close to the base; least_lean as the MCP servers)",
                                   "executor/kinematics.Kinematics.solve", choices=SEARCHES),
        "twist.approach": _key("enum", T["twist.approach"], "how twist_cap reaches the cap: top = hover + guarded vertical descent, "
                               "side = along the leaning jaws, auto = side only above twist.side_lean_deg",
                               "learning/twist.py", choices=("auto", "top", "side")),
        "twist.side_lean_deg": _key("float", T["twist.side_lean_deg"], "auto approach: a grasp pose leaning more than "
                                    "this is approached from the side", "learning/twist.py", lo=0.0, hi=90.0),
        "twist.hovers_m": _key("list", list(T["twist.hovers_m"]), "heights above the cap top for the hover before the descent "
                               "(the first reachable is used)", "learning/twist.py", item=("float", 0.015, 0.10),
                               length=(1, 6)),
        "twist.max_lean_deg": _key("float", T["twist.max_lean_deg"], "steeper grasp/hover poses are refused",
                                   "learning/twist.py", lo=10.0, hi=80.0),
        "twist.grasp_below_top_m": _key("float", T["twist.grasp_below_top_m"], "nominal TCP height below the cap top at "
                                        "the close (approach_z_offset_mm is added; the safety floor still applies)",
                                        "learning/twist.py", lo=0.0, hi=0.02),
        "twist.release_open": _key("float", T["twist.release_open"], "how far the jaws open (norm) to release the cap "
                                   "between strokes", "learning/twist.py", lo=0.05, hi=0.6),
    }
    for name, spec in E.PARAMS.items():
        kind, lo, hi = spec[:3]
        if kind == "enum":
            continue
        elo, ehi = PARAM_ENVELOPE[name]
        ehi = E.SAFETY[ehi] if isinstance(ehi, str) else ehi
        for side, v in (("lo", lo), ("hi", hi)):
            reg[f"params.bounds.{name}.{side}"] = _key(kind, v, f"{'lower' if side == 'lo' else 'upper'} bound of the "
                                                       f"episode parameter {name}", "learning/episodes.validate",
                                                       lo=elo, hi=ehi)
    for k, v in E.SAFETY.items():
        reg[f"safety.{k}"] = _key("fixed", v, "episodes.SAFETY: enforced in learning/twist.py", "learning/twist.py",
                                  cls="safety")
    for k in ("min_tcp_z", "max_tcp_z", "max_radius"):
        reg[f"workspace.{k}"] = _key("fixed", _profile_raw().get("workspace", {}).get(k), "robot profile workspace",
                                     "sim/real_arm_mcp, executor", cls="safety")
    reg["motion.load_abort"] = _key("fixed", _profile_raw().get("motion", {}).get("load_abort"),
                                    "servo load that aborts a move (robot profile)", "sim/real_arm_mcp", cls="safety")
    return reg


def _profile_raw() -> dict:
    choice = os.environ.get("ROBOT_PROFILE") or "elrobot"
    p = Path(choice)
    if not p.exists():
        p = REPO / "sim" / "robots" / (choice if choice.endswith(".json") else choice + ".json")
    return json.loads(p.read_text())


def registry() -> dict:
    return _registry(os.environ.get("ROBOT_PROFILE") or "elrobot")


def defaults() -> dict:
    return {k: s["default"] for k, s in registry().items()}


def _check_value(spec: dict, v):
    """(normalised value, error or None)."""
    def num(kind, v, lo, hi):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None, f"{v!r} must be a number"
        if kind == "int" and float(v) != int(v):
            return None, f"{v} must be a whole number"
        if not lo <= v <= hi:
            return None, f"{v} is outside the envelope {lo}..{hi}"
        return (int(v) if kind == "int" else float(v)), None
    kind = spec["kind"]
    if kind == "fixed":
        return None, "is a safety limit: not changeable through skill config"
    if kind == "enum":
        return (v, None) if v in spec["choices"] else (None, f"{v!r} must be one of {list(spec['choices'])}")
    if kind == "list":
        if not isinstance(v, (list, tuple)):
            return None, f"{v!r} must be a list"
        n0, n1 = spec["length"]
        if not n0 <= len(v) <= n1:
            return None, f"needs {n0}..{n1} entries, got {len(v)}"
        out = []
        for x in v:
            x, err = num(*spec["item"][:1], x, *spec["item"][1:])
            if err:
                return None, "entry " + err
            out.append(x)
        if len(set(out)) != len(out):
            return None, "entries must be unique"
        return out, None
    return num(kind, v, spec["lo"], spec["hi"])


def validate(changes: dict, base: dict | None = None) -> tuple[dict, list[str]]:
    """(the full config after the changes, errors). Errors: unknown key, safety key, type, envelope, and the
    cross-key rules (a bound pair lo <= hi containing the parameter's default)."""
    reg = registry()
    full, errors = dict(base if base is not None else defaults()), []
    for k, v in (changes or {}).items():
        if k not in reg:
            errors.append(f"unknown key {k!r} (see skill_config())")
            continue
        v, err = _check_value(reg[k], v)
        if err:
            errors.append(f"{k} {err}")
            continue
        full[k] = v
    from . import episodes as E
    for name, spec in E.PARAMS.items():
        lo, hi = full.get(f"params.bounds.{name}.lo"), full.get(f"params.bounds.{name}.hi")
        if lo is None:
            continue
        if lo > hi:
            errors.append(f"params.bounds.{name}: lo {lo} > hi {hi}")
        elif not lo <= spec[3] <= hi:
            errors.append(f"params.bounds.{name}: {lo}..{hi} must contain the default {spec[3]}")
    return full, errors


# --- storage and the read path -----------------------------------------------------------------------------------
def root() -> Path:
    return Path(os.environ.get("SKILLCFG_DIR") or DEFAULT_DIR)


def set_target(target: str) -> None:
    global _target
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    _target = target


def target() -> str:
    t = _target or os.environ.get("SKILLCFG_TARGET") or "real"
    return t if t in TARGETS else "real"


def _current_path(t: str) -> Path:
    return root() / f"{t}.json"


@functools.lru_cache(maxsize=16)
def _load(path: str, mtime_ns: int) -> tuple:
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        print(f"skillcfg: cannot read {path} ({e}); using the code defaults", file=sys.stderr)
        return 0, ()
    return int(d.get("version", 0)), tuple((k, json.dumps(v)) for k, v in (d.get("values") or {}).items())


def active(t: str | None = None) -> dict:
    """{"target", "version", "values": full config, "overrides": keys that differ from the code defaults}. A stored
    value that no longer passes its envelope is ignored (the default is used) and listed under "ignored"."""
    t = t or target()
    p = _current_path(t)
    version, stored = (_load(str(p), p.stat().st_mtime_ns) if p.exists() else (0, ()))
    reg, vals, ignored = registry(), defaults(), []
    for k, raw in stored:
        if k in reg:
            v, err = _check_value(reg[k], json.loads(raw))
            if err is None:
                vals[k] = v
                continue
        ignored.append(k)
    d = defaults()
    return {"target": t, "version": version, "values": vals,
            "overrides": {k: v for k, v in vals.items() if v != d[k]}, "ignored": ignored}


def get(key: str, t: str | None = None):
    return active(t)["values"][key]


def version(t: str | None = None) -> int:
    return active(t)["version"]


def stamp(t: str | None = None) -> dict:
    """What an attempt record carries: which config it ran with."""
    a = active(t)
    return {"target": a["target"], "version": a["version"], "overrides": a["overrides"]}


@contextlib.contextmanager
def _locked():
    import fcntl
    root().mkdir(parents=True, exist_ok=True)
    with open(root() / ".lock", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, p)


def _append_log(rec: dict) -> None:
    root().mkdir(parents=True, exist_ok=True)
    with (root() / "changes.jsonl").open("a") as f:
        f.write(json.dumps({"t": round(time.time(), 3), **rec}, default=str) + "\n")


def changes_log(t: str | None = None) -> list[dict]:
    p = root() / "changes.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        with contextlib.suppress(ValueError):
            r = json.loads(line)
            if t is None or r.get("target") == t:
                out.append(r)
    return out


def _commit(t: str, values: dict, rec: dict) -> int:
    """Write a new version of target t (values = full config). Call under _locked()."""
    cur = active(t)
    v = cur["version"] + 1
    d = defaults()
    overrides = {k: x for k, x in values.items() if k in d and x != d[k]}
    changed = {k: {"before": cur["values"][k], "after": values[k]} for k in d if values.get(k) != cur["values"][k]}
    _write_json(_current_path(t), {"target": t, "version": v, "values": overrides, "t": round(time.time(), 3)})
    _append_log({"type": "version", "target": t, "version": v, "previous": cur["version"], "changes": changed,
                 "values": overrides, **rec})
    return v


# --- proposals ---------------------------------------------------------------------------------------------------
def _ppath(pid: str) -> Path:
    return root() / "proposals" / f"{pid}.json"


def proposal(pid: str) -> dict | None:
    p = _ppath(pid)
    return json.loads(p.read_text()) if p.exists() else None


def proposals(status: str | None = None, t: str | None = None) -> list[dict]:
    d = root() / "proposals"
    out = [json.loads(p.read_text()) for p in sorted(d.glob("cfg-*.json"))] if d.exists() else []
    out = [p for p in out if (status is None or p["status"] == status) and (t is None or p["target"] == t)]
    return sorted(out, key=lambda p: p["t"])


def update_proposal(pid: str, fn) -> dict:
    """Read-modify-write a proposal under the lock (the async twin check writes its results this way)."""
    with _locked():
        p = proposal(pid)
        fn(p)
        _write_json(_ppath(pid), p)
        return p


def propose(changes: dict, reason: str, evidence=(), author: str = "agent", t: str | None = None) -> dict:
    """Validate a change request against the target's current config. Returns {"ok": False, "errors"} (nothing
    stored) or {"ok": True, "proposal": record} with status "pending" and empty checks."""
    t = t or target()
    if isinstance(evidence, str):
        evidence = [evidence] if evidence.strip() else []
    errors = []
    if not (reason or "").strip():
        errors.append("a reason is required: what the evidence shows and what the change should do")
    if not evidence:
        errors.append("evidence is required: episode records (e.g. 'attempt 7 approach lean_deg=60'), tool notes, "
                      "files or check results")
    if not changes:
        errors.append("no changes given")
    cur = active(t)
    full, bad = validate(changes, cur["values"])
    errors += bad
    if errors:
        return {"ok": False, "errors": errors}
    diff = {k: {"before": cur["values"][k], "after": full[k]} for k in changes if full[k] != cur["values"][k]}
    if not diff:
        return {"ok": False, "errors": ["every value is already the current one: nothing to change"]}
    rec = {"id": "cfg-" + time.strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:4], "target": t, "status": "pending",
           "t": round(time.time(), 3), "author": author, "reason": reason.strip(), "evidence": list(evidence),
           "base_version": cur["version"], "changes": diff, "checks": {}}
    with _locked():
        _write_json(_ppath(rec["id"]), rec)
    return {"ok": True, "proposal": rec}


def checks_failed(p: dict) -> list[str]:
    return [f"{name}: {c.get('summary', '')}" for name, c in (p.get("checks") or {}).items() if c.get("result") == "fail"]


def apply(pid: str, by: str, human: bool = False, force: bool = False) -> dict:
    """Apply a pending proposal as a new version of its target. The real target needs human=True (only the CLI sets
    it). Refused when a check failed (force=True overrides, human only) or when the target moved on since the proposal
    and the keys it changes no longer have the proposal's 'before' values."""
    with _locked():
        p = proposal(pid)
        if p is None:
            return {"ok": False, "errors": [f"no proposal {pid}"]}
        errs = []
        if p["status"] != "pending":
            errs.append(f"{pid} is {p['status']}, not pending")
        if p["target"] == "real" and not human:
            errs.append("the real-arm config changes only with a human approval: python -m learning.skillcfg approve "
                        f"{pid} --by <name>")
        if checks_failed(p) and not (force and human):
            errs.append("checks failed: " + "; ".join(checks_failed(p)))
        cur = active(p["target"])
        moved = [k for k, c in p["changes"].items() if cur["values"][k] != c["before"]]
        if moved:
            errs.append(f"the config changed since this proposal (now version {cur['version']}): {', '.join(moved)}; "
                        f"propose again")
        full, bad = validate({k: c["after"] for k, c in p["changes"].items()}, cur["values"])
        errs += bad
        if errs:
            return {"ok": False, "errors": errs}
        v = _commit(p["target"], full, {"author": p["author"], "approved_by": by, "reason": p["reason"],
                                        "evidence": p["evidence"], "proposal": pid,
                                        "checks": {k: c.get("summary") for k, c in p["checks"].items()}})
        p.update(status="applied", version=v, applied_by=by, applied_t=round(time.time(), 3))
        _write_json(_ppath(pid), p)
    return {"ok": True, "target": p["target"], "version": v}


def reject(pid: str, by: str, why: str = "") -> dict:
    p = proposal(pid)
    if p is None or p["status"] != "pending":
        return {"ok": False, "errors": [f"{pid} is not a pending proposal"]}
    update_proposal(pid, lambda r: r.update(status="rejected", rejected_by=by, rejected_why=why))
    _append_log({"type": "rejected", "target": p["target"], "proposal": pid, "by": by, "why": why})
    return {"ok": True}


def values_at(t: str, v: int) -> dict | None:
    """The full config of target t at version v (0 = the code defaults); None if there is no such version."""
    if v == 0:
        return defaults()
    rec = next((r for r in changes_log(t) if r.get("type") == "version" and r.get("version") == v), None)
    if rec is None:
        return None
    full, _ = validate({}, {**defaults(), **rec["values"]})
    return full


def revert(t: str, to_version: int, by: str, reason: str, human: bool = False) -> dict:
    """A NEW version of target t whose values equal those of version to_version (history is never rewritten)."""
    if t == "real" and not human:
        return {"ok": False, "errors": ["reverting the real-arm config is a human step (the CLI)"]}
    with _locked():
        vals = values_at(t, int(to_version))
        if vals is None:
            return {"ok": False, "errors": [f"{t} has no version {to_version}"]}
        full, bad = validate({}, vals)
        if bad:
            return {"ok": False, "errors": bad}
        v = _commit(t, full, {"author": by, "approved_by": by, "reason": reason or f"revert to version {to_version}",
                              "evidence": [], "revert_to": int(to_version)})
    return {"ok": True, "target": t, "version": v}


# --- the shared lean search (real arm, twin and executor) -----------------------------------------------------------
def lean_search(try_lean, leans=None, restarts=None, search=None, near=None, t: str | None = None):
    """(q, lean) of the jaws-down IK by the configured policy; (None, None) if unreachable.
    try_lean(lean_deg, restarts) -> q or None: one damped-least-squares solve (restarts 0 = from the current joints
    only). near(q) -> bool optionally rejects a no-restart solution that jumps to another arm branch.
    least_lean: pass 1 from the current joints, leans by size, the first hit; pass 2 random restarts only for smaller
    leans than that hit, so the least lean wins. near_first: the first lean reachable from the current joints wins,
    restarts only if none is."""
    a = active(t)["values"] if None in (leans, restarts, search) else {}
    leans = list(a["ik.leans"] if leans is None else leans)
    restarts = int(a["ik.restarts"] if restarts is None else restarts)
    search = a["ik.search"] if search is None else search
    ok = near or (lambda q: True)
    if search == "near_first":
        for rs in (0, restarts):
            for lean in leans:
                q = try_lean(lean, rs)
                if q is not None and (rs or ok(q)):
                    return q, lean
        return None, None
    leans = sorted(leans, key=abs)
    first = (None, None)
    for lean in leans:
        q = try_lean(lean, 0)
        if q is not None and ok(q):
            first = (q, lean)
            break
    if restarts:
        for lean in leans:
            if first[1] is not None and abs(lean) >= abs(first[1]):
                break
            q = try_lean(lean, restarts)
            if q is not None:
                return q, lean
    return first


# --- text for the tools and the CLI --------------------------------------------------------------------------------
def _short(v) -> str:
    return json.dumps(v) if isinstance(v, (list, dict)) else str(v)


def envelope_text(spec: dict) -> str:
    if spec["kind"] == "fixed":
        return "fixed"
    if spec["kind"] == "enum":
        return "|".join(spec["choices"])
    if spec["kind"] == "list":
        k, lo, hi = spec["item"]
        return f"list of {spec['length'][0]}..{spec['length'][1]} unique {k}s in {lo}..{hi}"
    return f"{spec['lo']}..{spec['hi']}"


def proposal_text(p: dict) -> str:
    lines = [f"{p['id']} [{p['status']}] target={p['target']} by {p['author']} (base version {p['base_version']})"
             + (f" -> version {p['version']}" if p.get("version") else "")]
    lines += [f"  {k}: {_short(c['before'])} -> {_short(c['after'])}" for k, c in p["changes"].items()]
    lines.append(f"  reason: {p['reason']}")
    lines += [f"  evidence: {e}" for e in p["evidence"]]
    for name, c in (p.get("checks") or {}).items():
        lines.append(f"  check {name}: {c.get('result')} - {c.get('summary', '')}")
    return "\n".join(lines)


def config_text(t: str | None = None, history: int = 5) -> str:
    a, reg = active(t), registry()
    lines = [f"skill config: target {a['target']}, version {a['version']}"
             + (f" (changed vs code defaults: {', '.join(a['overrides'])})" if a["overrides"] else " (code defaults)"),
             "", "strategy keys (propose_skill_change may change them inside the envelope; * = not the code default):"]
    for k, s in reg.items():
        if s["class"] == "strategy":
            star = " *" if k in a["overrides"] else ""
            dflt = f"; default {_short(s['default'])}" if star else ""
            lines.append(f"  {k}{star} = {_short(a['values'][k])}   [{envelope_text(s)}{dflt}] {s['about']}")
    lines += ["", "safety keys (read-only, never changeable here): "
              + ", ".join(f"{k}={_short(s['default'])}" for k, s in reg.items() if s["class"] == "safety")]
    if a["ignored"]:
        lines.append(f"IGNORED stored values (outside their envelope now): {', '.join(a['ignored'])}")
    pend = proposals("pending", a["target"])
    lines += ["", f"pending proposals for {a['target']}: {len(pend)}"] + [proposal_text(p) for p in pend]
    log = [r for r in changes_log(a["target"]) if r.get("type") == "version"][-history:]
    if log:
        lines += ["", "recent versions:"] + [
            f"  v{r['version']} {time.strftime('%m-%d %H:%M', time.localtime(r['t']))} by {r.get('approved_by')}: "
            + ", ".join(f"{k} {_short(c['before'])}->{_short(c['after'])}" for k, c in r["changes"].items())
            + f" ({r.get('reason', '')[:100]})" for r in log]
    return "\n".join(lines)


# --- CLI ---------------------------------------------------------------------------------------------------------
def _parse_value(s: str):
    try:
        return json.loads(s)
    except ValueError:
        return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m learning.skillcfg", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show", help="current config, pending proposals, recent versions")
    s.add_argument("--target", choices=TARGETS, default=None)
    sub.add_parser("proposals", help="all proposals").add_argument("--status", default=None)
    sub.add_parser("proposal", help="one proposal").add_argument("id")
    s = sub.add_parser("propose", help="propose a change (KEY=JSON ...)")
    s.add_argument("changes", nargs="+")
    s.add_argument("--target", choices=TARGETS, required=True)
    s.add_argument("--reason", required=True)
    s.add_argument("--evidence", action="append", default=[])
    s.add_argument("--by", default="cli")
    s.add_argument("--no-check", action="store_true")
    s = sub.add_parser("check", help="run the offline checks of a proposal")
    s.add_argument("id")
    s.add_argument("--twin", action="store_true", help="also the twin before/after run (minutes)")
    for name in ("approve", "reject"):
        s = sub.add_parser(name, help=f"{name} a pending proposal (human step)")
        s.add_argument("id")
        s.add_argument("--by", required=True, help="your name")
        if name == "approve":
            s.add_argument("--force", action="store_true", help="apply although a check failed")
        else:
            s.add_argument("--why", default="")
    s = sub.add_parser("revert", help="new version restoring the values of an earlier version (0 = code defaults)")
    s.add_argument("target", choices=TARGETS)
    s.add_argument("version", type=int)
    s.add_argument("--by", required=True)
    s.add_argument("--reason", default="")
    s = sub.add_parser("log", help="the change log")
    s.add_argument("--target", choices=TARGETS, default=None)
    a = ap.parse_args(argv)

    def out(r):
        if not r.get("ok", True):
            print("REFUSED:\n- " + "\n- ".join(r["errors"]))
            return 1
        return 0
    if a.cmd == "show":
        print(config_text(a.target))
    elif a.cmd == "proposals":
        print("\n\n".join(proposal_text(p) for p in proposals(a.status)) or "no proposals")
    elif a.cmd == "proposal":
        p = proposal(a.id)
        print(proposal_text(p) if p else f"no proposal {a.id}")
    elif a.cmd == "propose":
        ch = dict(c.split("=", 1) for c in a.changes)
        r = propose({k: _parse_value(v) for k, v in ch.items()}, a.reason, a.evidence, a.by, a.target)
        if out(r):
            return 1
        pid = r["proposal"]["id"]
        if not a.no_check:
            from . import skillcheck
            skillcheck.run_checks(pid)
        print(proposal_text(proposal(pid)))
    elif a.cmd == "check":
        from . import skillcheck
        skillcheck.run_checks(a.id, twin="sync" if a.twin else False)
        print(proposal_text(proposal(a.id)))
    elif a.cmd == "approve":
        p = proposal(a.id)
        if p:
            print(proposal_text(p))
        r = apply(a.id, a.by, human=True, force=a.force)
        if out(r):
            return 1
        print(f"APPLIED: {r['target']} config is now version {r['version']}")
    elif a.cmd == "reject":
        return out(reject(a.id, a.by, a.why))
    elif a.cmd == "revert":
        r = revert(a.target, a.version, a.by, a.reason, human=True)
        if out(r):
            return 1
        print(f"REVERTED: {r['target']} config is now version {r['version']} (= the values of version {a.version})")
    elif a.cmd == "log":
        for r in changes_log(a.target):
            print(json.dumps(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
