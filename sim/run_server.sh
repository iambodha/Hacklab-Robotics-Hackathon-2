#!/usr/bin/env bash
# Start the simulated-arm MCP server (stdio). Python needs mujoco, numpy, pillow and mcp.
# Env: ROBOT_SIM_PYTHON (default python3), NORMA_CORE_REPO (full norma-core clone with hardware/),
#      ROBOT_SIM_GUI=1 for the MuJoCo viewer, plus the ROBOT_* options in sim_arm_mcp.py.
set -euo pipefail
cd "$(dirname "$0")"
if [ "${ROBOT_SIM_GUI:-0}" = 1 ]; then export MUJOCO_GL=glfw; else export MUJOCO_GL=osmesa; fi
exec "${ROBOT_SIM_PYTHON:-python3}" sim_arm_mcp.py "$@"
