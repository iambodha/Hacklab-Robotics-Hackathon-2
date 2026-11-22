import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def run_json(args: list[str], env_extra: dict | None = None, timeout: int = 300) -> dict:
    """Run `python <args>` from the repo root in a fresh process (servers that register tools at import, or put sim/
    in front of sys.path, get their own process) and return its last stdout line as JSON. The child inherits the
    test guard (STATION_HOST unresolvable, PYTEST_CURRENT_TEST set)."""
    env = {**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "cgl"), "ROBOT_SIM_RECORD": "0",
           "BOTTLE_GEMINI_JUDGE": "0", "PYTHONPATH": str(REPO), **(env_extra or {})}
    p = subprocess.run([sys.executable, *args], cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)
    assert p.returncode == 0, p.stderr[-3000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


@pytest.fixture
def needs_twin():
    pytest.importorskip("mujoco")
    pytest.importorskip("mcp")
