"""Where things live. Generaloid wraps the existing top-level packages (sim/, learning/, arm_mcp/, ...) in place
until the release restructure, so it needs the repo root (packages) and sim/ (script-style modules) on sys.path."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SIM = REPO / "sim"
GENERATED = Path(os.environ.get("GENERALOID_GENERATED", REPO / "generaloid" / ".generated"))   # twin scenes (small XML)


def ensure_repo_on_path() -> None:
    """Repo root first-class (packages: learning, arm_mcp, recorder, ...)."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def ensure_sim_on_path() -> None:
    """sim/ holds script-style modules (sim_arm_mcp, build_scene, robot_profile) imported by bare name. Appended, not
    put in front: sim/recorder.py would otherwise shadow the repo-level recorder/ package for everyone in this process.
    (sim_arm_mcp and real_arm_mcp put sim/ in front themselves when they are imported; that is why mounting them
    happens once per process, in the server process or a test subprocess.)"""
    ensure_repo_on_path()
    if str(SIM) not in sys.path:
        sys.path.append(str(SIM))
