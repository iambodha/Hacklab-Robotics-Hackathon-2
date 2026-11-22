"""The digital twin as a Generaloid module.

  sim/build_scene.py, sim/sim_arm_mcp.py   the calibrated ElRobot twin (wrapped by the elrobot connector)
  sim/splat/                               Gaussian-splat twin from plain cameras + ArUco markers (no depth sensor)
  generaloid.twin.generic                  connector-driven twin + tool surface for every other arm (so101, ...)
  generaloid.twin.ik                       least-lean jaws-down IK for any n-joint arm
"""
from generaloid.core.paths import SIM

SPLAT = SIM / "splat"
