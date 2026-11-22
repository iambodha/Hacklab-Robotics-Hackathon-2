#!/usr/bin/env bash
# MCP server from THIS worktree. SITE defaults to home; explicit env overrides win.
set -euo pipefail
ROOT=/Users/bodha/Documents/b0dha/Robotics-Hackathon
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
export NORMA_CORE_REPO="${NORMA_CORE_REPO:-$ROOT/repo/norma-core}"
export ROBOT_CAM_URL="${ROBOT_CAM_URL:-http://127.0.0.1:8890}"
exec bash "$HERE/sim/run_real_server.sh" "$@"
