"""pytest wrapper for sim/selftest.py: venue pick-and-place and the home-table marker/wrist-view check (~25 s + ~40 s).

  MUJOCO_GL=cgl python -m pytest -q sim/test_selftest.py
selftest.py picks its layout at import (sim_arm_mcp reads SIM_LAYOUT), so each layout runs in its own process.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("mcp")
pytest.importorskip("cv2")
HERE = Path(__file__).resolve().parent


@pytest.mark.parametrize("layout", ["venue", "home"])
def test_selftest(layout):
    env = {**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl"),
           "ROBOT_SIM_RECORD": "0"}   # no session dir, no dashboard port (8891) to collide with a running sim
    p = subprocess.run([sys.executable, "selftest.py", "--layout", layout], cwd=HERE, env=env,
                       capture_output=True, text=True, timeout=300)
    out = p.stdout
    report = json.loads(out[out.index("{"):out.rindex("}") + 1])
    assert p.returncode == 0 and report["success"], out[-3000:] + p.stderr[-2000:]
