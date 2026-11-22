"""The modules around the Generaloid core, wrapped in place (the release restructure moves them; see
docs/generaloid/MIGRATION.md). Nothing is imported until asked for: several of them pull heavy dependencies
(MuJoCo, OpenCV, google-genai) and some must not share a process (sim/recorder.py vs the recorder/ package).

    from generaloid import modules
    learning = modules.load("learning")          # the `learning` package, as is
    modules.MODULES["twin"].entry                # how to run it
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass

from generaloid.core.paths import ensure_repo_on_path, ensure_sim_on_path


@dataclass(frozen=True)
class Module:
    name: str
    path: str          # where it lives today
    role: str
    imports: str       # importable name (bare names live in sim/)
    entry: str         # how a human runs it
    station: bool      # talks to NormaCore Station (directly or through the camera relay)


MODULES = {m.name: m for m in [
    Module("station", "repo/norma-core, station-bin", "NormaCore Station: motor buses, cameras, commands (TCP :8888)",
           "generaloid.core.station", "scripts/run-station.sh", True),
    Module("twin", "sim/", "MuJoCo digital twin of the calibrated table (scene builder, twin MCP server)",
           "build_scene", "python -m generaloid serve --connector elrobot --mode sim", False),
    Module("splat", "sim/splat/", "Gaussian-splat twin from plain cameras + ArUco markers (no depth sensor)",
           "", "bash sim/splat/prepare.sh (see sim/splat/README.md)", False),
    Module("learning", "learning/", "learning loop: episode store, versioned skill config, proposals, twist skill",
           "learning", "python -m learning.sim_loop", False),
    Module("skills", "skills/", "skill library and its learning experiments", "skills", "see skills/", False),
    Module("executor", "executor/", "one async motion owner; advisors and agents submit proposals",
           "executor", "python -m executor", True),
    Module("advisor", "executor/advisor/, models/", "optional model advisors (Gemini Robotics ER, Claude) as add-ons",
           "executor.advisor", "python -m executor.advisor.mcp_server", False),
    Module("recorder", "recorder/", "records every camera frame and motor state from Station (:8900)",
           "recorder", "python -m recorder", True),
    Module("dashboard", "dashboard/", "live dashboard: episodes, skill config, proposals, human approval (:8920)",
           "dashboard", "python -m dashboard", False),
    Module("perception", "perception/", "ArUco scene calibration, pixel -> table, object location",
           "perception", "see perception/README.md", True),
    Module("calibration", "calibration/, camera_pose/", "camera intrinsics, scene calibration, wrist camera mount",
           "camera_pose", "see camera_pose/HANDOFF.md", False),
]}


def load(name: str):
    """Import a module's package as is (twin/sim modules by their bare name from sim/)."""
    m = MODULES[name]
    if not m.imports:
        raise ValueError(f"{name} is a pipeline of scripts ({m.entry}), not an importable module")
    ensure_sim_on_path() if m.path == "sim/" else ensure_repo_on_path()
    return importlib.import_module(m.imports)
