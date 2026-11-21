"""Environment-driven configuration for robot_mcp."""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
DATA_DIR = PACKAGE_DIR / "data"

def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


STATION_HOST = os.environ.get("STATION_HOST", "localhost")

# This year's follower arm bus (the arm we command). The leader bus
# (5B61035034) is never commanded. Auto-discovered if unset/empty.
BUS_SERIAL = os.environ.get("ROBOT_BUS_SERIAL", "5B61033301") or None

# Path to a norma-core checkout (provides station_py + generated protobufs).
NORMA_CORE_REPO = Path(os.environ.get("NORMA_CORE_REPO", REPO_ROOT / "norma-core"))

# Optional explicit camera queues ("usbvideo/<hash>,<hash>"). When unset we
# first try the inference/normvla combined frame, then give up gracefully.
CAM_QUEUES = [q.strip() for q in os.environ.get("ROBOT_CAM_QUEUES", "").split(",") if q.strip()]

# Safety envelope.
MAX_STEP = _f("ROBOT_MAX_STEP", 0.25)          # max normalized move per single call
CREEP_STEP = _f("ROBOT_CREEP_STEP", 0.03)      # per-iteration step inside skills
CREEP_DT = _f("ROBOT_CREEP_DT", 0.5)           # pause between creep iterations (s)

# Downscale observed frames for token budget (requires pillow).
IMG_MAX_W = int(os.environ.get("ROBOT_IMG_MAX_W", "512"))

# Enable a software mock robot ("mock") for development without hardware.
ENABLE_MOCK = os.environ.get("ROBOT_MCP_ENABLE_MOCK", "") == "1"
