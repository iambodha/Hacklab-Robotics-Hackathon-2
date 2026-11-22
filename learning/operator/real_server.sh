#!/bin/bash
# elrobot-real MCP server (stdio) from THIS worktree, configured like scripts_integrator/real_server_home.sh (home
# table), plus the bottle tools (learning/). Used by learning/operator/.devin/mcp_config.json.
ROOT=/Users/bodha/Documents/b0dha/Robotics-Hackathon
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
export NORMA_CORE_REPO=$ROOT/repo/norma-core
export ROBOT_WRIST_CAM=28ff0ca8 ROBOT_OVERVIEW_CAM=9ea4cacd ROBOT_CAM_URL=http://127.0.0.1:8890
export ROBOT_REAL_LOG_DIR=$ROOT/logs/real_sessions_home
export ROBOT_OVERVIEW_CAL="${ROBOT_OVERVIEW_CAL:-$ROOT/ws/integration/sim/real_overview_camera.json}"
export BOTTLE_LEARN_DIR="${BOTTLE_LEARN_DIR:-$ROOT/logs/learning/bottle}"
mkdir -p "$ROBOT_REAL_LOG_DIR" "$BOTTLE_LEARN_DIR"
exec bash "$HERE/sim/run_real_server.sh" "$@"
