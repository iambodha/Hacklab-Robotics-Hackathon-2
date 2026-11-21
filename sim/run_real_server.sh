#!/usr/bin/env bash
# Start the REAL-arm MCP server (stdio), same tools as the sim. Needs Station (:8888) and the camera relay
# (scripts/cam_stream.py, :8890) running on this machine, and teleop OFF.
# Python: ROBOT_REAL_PYTHON if set (needs mujoco, numpy, pillow, mcp<2), else uv with those packages.
set -euo pipefail
cd "$(dirname "$0")"
if [ "$(uname)" = Darwin ]; then export MUJOCO_GL="${MUJOCO_GL:-cgl}"; else export MUJOCO_GL="${MUJOCO_GL:-egl}"; fi
if [ -n "${ROBOT_REAL_PYTHON:-}" ]; then exec "$ROBOT_REAL_PYTHON" real_arm_mcp.py "$@"; fi
exec uv run -q --no-project --with 'mcp>=1.12,<2' --with pillow --with numpy --with mujoco python real_arm_mcp.py "$@"
