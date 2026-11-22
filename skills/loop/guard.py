"""PROTECTED. What the improvement agent may change, checked on the sandbox's diff after the agent ran.

A changed path is allowed only if it matches ALLOW and none of FORBID. Additions-only areas accept new files but
no edits to existing ones. Semantic checks follow: seed parameters must stay inside the protected bounds.
The orchestrator runs this from the incumbent checkout (never the sandbox's copy), before any test is run.
"""
from __future__ import annotations

import fnmatch
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# editable by the agent: the wipe skill's planning/adaptation code and seed values, new candidate tests
ALLOW = [
    "skills/wipe/defaults.json",
    "skills/wipe/planner.py",
    "skills/wipe/skill.py",
    "skills/wipe/adapt.py",
    "skills/wipe/__init__.py",
]
ADD_ONLY = ["skills/tests/candidate/*"]          # new test files are welcome; existing tests are not editable

# never, whatever ALLOW says: the evaluator, safety limits, bounds, promotion rule, executor validation,
# tests and their thresholds, memory, the loop itself, robot profiles, Station/MCP motion code
FORBID = [
    "skills/eval/*",
    "skills/wipe/limits.py",
    "skills/wipe/bounds.json",
    "skills/experiment.py",
    "skills/memory.py",
    "skills/_contracts.py",
    "skills/model.py",
    "skills/stir_preflight.py",
    "skills/loop/*",
    "skills/tests/test_*",
    "skills/tests/conftest.py",
    "executor/*",
    "sim/*",
    "arm_mcp/*",
    "robot_mcp/*",
    "recorder/*",
    "calibration/*",
    "perception/*",
    ".gitignore",
    "*.sh",
]


def _match(path: str, pats) -> bool:
    return any(fnmatch.fnmatch(path, p) for p in pats)


@dataclass
class GuardReport:
    ok: bool
    changed: list = field(default_factory=list)       # (status, path)
    violations: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "changed": self.changed, "violations": self.violations}


def changed_files(wt: Path, base: str) -> list[tuple[str, str]]:
    """(status, path) for every change in the sandbox vs `base`: committed, staged, unstaged and untracked."""
    out = {}
    r = subprocess.run(["git", "diff", "--name-status", "--no-renames", base], cwd=wt, capture_output=True, text=True, check=True)
    for line in r.stdout.splitlines():
        st, p = line.split("\t", 1)
        out[p] = st[0]
    r = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], cwd=wt, capture_output=True, text=True, check=True)
    for p in r.stdout.splitlines():
        out[p] = "A"
    r = subprocess.run(["git", "ls-files", "--others", "--ignored", "--exclude-standard"], cwd=wt, capture_output=True,
                       text=True, check=True)
    for p in r.stdout.splitlines():                   # ignored files too (an agent could hide a change behind .gitignore)
        if "__pycache__" not in p and not p.endswith(".pyc") and not p.startswith(".pytest_cache"):
            out[p] = "A"
    return sorted((s, p) for p, s in out.items())


def check_paths(changes: list[tuple[str, str]]) -> list[str]:
    bad = []
    for st, p in changes:
        if _match(p, FORBID):
            bad.append(f"{p}: protected path ({st})")
        elif _match(p, ADD_ONLY):
            if st != "A":
                bad.append(f"{p}: only new files may be added here ({st})")
        elif not _match(p, ALLOW):
            bad.append(f"{p}: not on the allowlist ({st})")
    return bad


def check_params(wt: Path, bounds_from: Path) -> list[str]:
    """Seed values in the sandbox's defaults.json must lie inside the INCUMBENT's protected bounds.json."""
    try:
        d = json.loads((wt / "skills/wipe/defaults.json").read_text())
    except (OSError, json.JSONDecodeError) as e:
        return [f"skills/wipe/defaults.json unreadable: {e}"]
    b = json.loads(Path(bounds_from).read_text())
    bad = []
    for k, v in b.items():
        if k.startswith("_"):
            continue
        if k not in d:
            bad.append(f"defaults.json lacks {k}")
        elif not isinstance(d[k], (int, float)) or not v["lo"] <= d[k] <= v["hi"]:
            bad.append(f"defaults.json {k}={d[k]!r} outside protected bounds [{v['lo']}, {v['hi']}]")
    bad += [f"defaults.json has unknown key {k}" for k in d if not k.startswith("_") and k not in b]
    return bad


def check(wt: str | Path, base: str, bounds_from: str | Path) -> GuardReport:
    wt = Path(wt)
    changes = changed_files(wt, base)
    bad = check_paths(changes)
    bad += check_params(wt, Path(bounds_from))
    return GuardReport(not bad, [list(c) for c in changes], bad)
