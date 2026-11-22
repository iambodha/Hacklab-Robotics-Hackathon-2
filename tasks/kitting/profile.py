"""Per-object-class kitting skill parameters as versioned skill config (learning/skillcfg.py), and the exported profile
the real arm uses.

Keys: kit.<class>.<param> for class in screw | cap | box and the params below. They are registered into the skillcfg
registry at runtime (register()), because the registry in learning/skillcfg.py is closed and is owned by another agent
(r5-skills). Everything else is the real skillcfg flow: propose -> checks -> apply (twin: automatic when the checks
pass) -> a new version in changes.jsonl; revert makes a new version with old values. The kitting loop runs in its own
SKILLCFG_DIR so the bottle's config is never touched.

The exported profile (tasks/kitting/profiles/kit_learned.json) is a plain JSON file: per-class values + provenance (which
twin version, which iterations, the measured success and placement error). tasks/kitting/run_real.py reads it as-is.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILE = HERE / "profiles" / "kit_learned.json"
CLASSES = ("screw", "cap", "box")
READERS = "tasks/kitting/plan.plan_pick_place (sim: tasks/kitting/sim_env.py, real: tasks/kitting/run_real.py)"

# name: (lo, hi, generic default, meaning). Envelopes: the physical / safety range; the generic default is ONE setting
# for every class, the place the learning starts from (the values an engineer would type before any trial).
PARAMS: dict[str, tuple[float, float, float, str]] = {
    "grasp_z_mm": (10.0, 40.0, 18.0,
                   "TCP height above the table when the jaws close; the fingertips reach 10 mm below the TCP. Lower "
                   "bound 10 mm = fingertips on the table (profile touch.nominal_table_tcp_z; only a guarded descent "
                   "goes below workspace.min_tcp_z 12 mm, it stops on contact). Generic 18 mm: the executor pick "
                   "height for its default 20 mm object (table + h/2 + pick.grasp_above_mid 8 mm)"),
    "grip_force": (0.1, 0.6, 0.35,
                   "jaw torque limit while closing (fraction of the jaw servo torque). Generic = profile "
                   "gripper.force_default 0.35; upper bound = episodes.SAFETY max_grip_force 0.6"),
    "yaw_deg": (-90.0, 90.0, 0.0,
                "jaw closing axis relative to the object's long axis (0 = along it, 90 = across it); round objects "
                "ignore it. Generic 0 = no knowledge of which side is short"),
    "place_z_mm": (12.0, 80.0, 40.0,
                   "TCP height above the sheet when the jaws open (the object's bottom is then place_z - grasp_z above "
                   "the paper). Generic 40 mm: release from above, as the bin drop of tasks/blue_to_bin"),
    "release_open": (0.3, 1.0, 1.0, "how far the jaws open to release (norm, 1 = fully open)"),
}


def key(cls: str, name: str) -> str:
    return f"kit.{cls}.{name}"


def skillcfg():
    from learning import skillcfg as SC
    return SC


def register() -> list[str]:
    """Add the kit.* keys to the skillcfg registry (idempotent). Returns the keys."""
    SC = skillcfg()
    reg = SC.registry()
    keys = []
    for cls in CLASSES:
        for name, (lo, hi, dflt, about) in PARAMS.items():
            k = key(cls, name)
            keys.append(k)
            if k not in reg:
                reg[k] = SC._key("float", float(dflt), f"{cls}: {about}", READERS, lo=float(lo), hi=float(hi))
    return keys


def use_dir(path: str | Path, target: str = "twin") -> Path:
    """Point skillcfg at a kitting-only config directory and target (this process only)."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    os.environ["SKILLCFG_DIR"] = str(p)
    skillcfg().set_target(target)
    register()
    return p


def generic() -> dict[str, dict[str, float]]:
    return {c: {n: float(v[2]) for n, v in PARAMS.items()} for c in CLASSES}


def from_values(values: dict) -> dict[str, dict[str, float]]:
    return {c: {n: float(values[key(c, n)]) for n in PARAMS} for c in CLASSES}


def current(t: str | None = None) -> tuple[dict[str, dict[str, float]], int]:
    """(per-class params, skillcfg version) of the active config of target t."""
    register()
    a = skillcfg().active(t)
    return from_values(a["values"]), int(a["version"])


def flat(params: dict[str, dict[str, float]]) -> dict[str, float]:
    return {key(c, n): v for c, d in params.items() for n, v in d.items()}


def export(path: Path = PROFILE, provenance: dict | None = None, t: str | None = None) -> Path:
    """Write the active per-class values (+ provenance) as the profile the real run reads."""
    params, version = current(t)
    out = {"name": "kitting", "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "source": f"skillcfg target {skillcfg().target() if t is None else t} version {version}",
           "_about": "Learned per-class kitting parameters (tasks/kitting/profile.py: PARAMS gives meanings and envelopes). "
                     "Read as-is by tasks/kitting/run_real.py. Regenerate with python -m tasks.kitting.learn.",
           "classes": params, "generic": generic(), "provenance": provenance or {}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    return path


def load(path: Path = PROFILE) -> dict[str, dict[str, float]]:
    """Per-class params of an exported profile, checked against the envelopes (out of range = refused, not clamped)."""
    d = json.loads(Path(path).read_text())["classes"]
    bad = [f"{c}.{n}={d[c][n]}" for c in CLASSES for n, (lo, hi, _, _) in PARAMS.items()
           if not lo <= float(d[c][n]) <= hi]
    if bad:
        raise ValueError(f"{path}: outside the envelopes: {', '.join(bad)}")
    return {c: {n: float(d[c][n]) for n in PARAMS} for c in CLASSES}
