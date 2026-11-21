#!/usr/bin/env bash
# Start the SO-101 simulated-arm MCP server (`so101-sim`, stdio).
set -euo pipefail
cd "$(dirname "$0")"
export ROBOT_MODEL=so101
if [ "${ROBOT_SIM_GUI:-0}" = 1 ]; then export MUJOCO_GL=glfw; else export MUJOCO_GL="${MUJOCO_GL:-egl}"; fi
if [ -n "${ROBOT_SIM_PYTHON:-}" ]; then exec "$ROBOT_SIM_PYTHON" so101_sim_mcp.py "$@"; fi
exec uv run -q --no-project --with 'mcp>=1.12,<2' --with pillow --with numpy --with mujoco python3 so101_sim_mcp.py "$@"
