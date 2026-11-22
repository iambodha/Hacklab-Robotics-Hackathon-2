"""The MuJoCo twin MCP server (sim/sim_arm_mcp.py, unchanged) with the comms tools added: what a Generaloid sim
server with comms looks like, before generaloid's own server assembly lands.

    python -m generaloid.comms twin --robot-id elrobot-twin-a     # stdio MCP: every twin tool + comms_*

Twin only. This never imports the real-arm server and never opens a Station client.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]

# What the elrobot twin advertises in its hello. Skills = the task-level tools of sim_arm_mcp.
ELROBOT_TWIN = SimpleNamespace(
    name="elrobot",
    display_name="ElRobot (MuJoCo twin)",
    skills=["twist_cap", "check_cap", "stir_spoon", "move_tcp", "gripper", "look"],
    capabilities=["7dof-arm", "parallel-gripper", "wrist-camera", "overview-camera", "twin"],
    frames={"world": "table", "units": "m",
            "convention": "sim base frame: +x ahead of the arm, +y left, z up; cross-robot alignment requires calibration"},
)


def main(robot_id: str | None = None):
    os.environ.setdefault("ROBOT_SIM_RECORD", "0")          # no recorder/dashboard port per twin process
    if os.environ.get("ROBOT_LIVE_TESTS") != "1":
        os.environ["STATION_HOST"] = "station-disabled-in-tests.invalid"
    sys.path.insert(0, str(ROOT / "sim"))
    sys.path.insert(0, str(ROOT))
    import sim_arm_mcp                                       # builds the twin and its FastMCP server
    from generaloid.comms import register
    register(sim_arm_mcp.server, ELROBOT_TWIN, robot_id=robot_id)
    sim_arm_mcp.server.run()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
