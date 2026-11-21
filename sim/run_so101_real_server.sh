#!/usr/bin/env bash
# Start the REAL SO-101 MCP server (`so101-real`, stdio). Connects automatically to Station WebSocket (:8000) or TCP (:8888).
set -euo pipefail
cd "$(dirname "$0")"
export ROBOT_MODEL=so101
if [ "$(uname)" = Darwin ]; then export MUJOCO_GL="${MUJOCO_GL:-cgl}"; else export MUJOCO_GL="${MUJOCO_GL:-egl}"; fi
if [ -n "${ROBOT_REAL_PYTHON:-}" ]; then exec "$ROBOT_REAL_PYTHON" so101_real_mcp.py "$@"; fi
exec uv run -q --no-project --with 'mcp>=1.12,<2' --with pillow --with numpy --with mujoco python3 so101_real_mcp.py "$@"
