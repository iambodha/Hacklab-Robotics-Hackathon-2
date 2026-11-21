#!/usr/bin/env bash
# Start the System 2 MCP server (stdio) — the only tools a planner gets. It spawns the arm server
# itself (default: the MuJoCo twin, sim/run_server.sh; override with S2_ARM_CMD).
# Env: S2_ARM (sim | real), S2_POLICY (geometric | jev | shadow), S2_JEV_URL, ROBOT_SIM_GUI=1 (MuJoCo window),
#      ROBOT_SIM_RECORD_DIR (session recordings + live dashboard), see s2/config.py for the rest.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
S2_HOME="${S2_HOME:-$HOME/.local/share/elrobot-system2}"
export ROBOT_SIM_PYTHON="${ROBOT_SIM_PYTHON:-$S2_HOME/sim-venv/bin/python}"
if [ -z "${NORMA_CORE_REPO:-}" ] && [ -d "$here/../norma-core/hardware/elrobot" ]; then
  export NORMA_CORE_REPO="$here/../norma-core"
fi
if [ "${S2_ARM:-sim}" = real ]; then
  # The real follower through elrobot-real (sim/REAL_ARM.md: Station + cam_stream running, ROBOT_WRIST_CAM etc. set).
  export S2_ARM_CMD="${S2_ARM_CMD:-bash $(dirname "$here")/sim/run_real_server.sh}"
  export S2_GRIP_MIN_OPEN="${S2_GRIP_MIN_OPEN:-0.045}" S2_JAW_STEP="${S2_JAW_STEP:-1.0}"
fi
cd "$here"
exec "${S2_PYTHON:-$S2_HOME/system2-venv/bin/python}" -m s2.server "$@"
