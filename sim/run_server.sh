#!/usr/bin/env bash
# Start the simulated-arm MCP server (stdio). Python needs mujoco, numpy, pillow and mcp.
# Env: ROBOT_SIM_PYTHON (default python3), NORMA_CORE_REPO (full norma-core clone with hardware/),
#      ROBOT_SIM_GUI=1 for the MuJoCo viewer, plus the ROBOT_* options in sim_arm_mcp.py.
set -euo pipefail
cd "$(dirname "$0")"
py="${ROBOT_SIM_PYTHON:-}"
if [ -z "$py" ] && [ -x ../.venv/bin/python ]; then py=../.venv/bin/python; fi
py="${py:-python3}"
if [ "$(uname)" = Darwin ]; then
  # macOS: offscreen cameras on CGL; the MuJoCo viewer must run under mjpython (needs libpython on the dyld path).
  export MUJOCO_GL=cgl
  if [ "${ROBOT_SIM_GUI:-0}" = 1 ] && [ -x "$(dirname "$py")/mjpython" ]; then
    DYLD_LIBRARY_PATH="$("$py" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    export DYLD_LIBRARY_PATH
    py="$(dirname "$py")/mjpython"
  fi
elif [ "${ROBOT_SIM_GUI:-0}" = 1 ]; then export MUJOCO_GL=glfw; else export MUJOCO_GL=osmesa; fi
exec "$py" sim_arm_mcp.py "$@"
