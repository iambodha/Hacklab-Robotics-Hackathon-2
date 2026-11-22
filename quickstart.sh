#!/usr/bin/env bash
# Offline simulation only. Never starts Station or a real-arm server.
set -euo pipefail
cd "$(dirname "$0")"
export STATION_HOST=station-disabled-in-tests.invalid
export ROBOT_CAM_URL=http://camera-relay-disabled-in-tests.invalid:8890
export EXECUTOR_REAL_MOTION=0
export ROBOT_SIM_RECORD=0
export BOTTLE_GEMINI_JUDGE=0
export DATA_ROOT="${DATA_ROOT:-$PWD/.local/data}"
export NORMA_CORE_REPO="${NORMA_CORE_REPO:-$PWD/norma-core}"
export SKILLCFG_DIR="$DATA_ROOT/quickstart/skillcfg"
export BOTTLE_LEARN_DIR="$DATA_ROOT/quickstart/episodes"
export BOTTLE_SIM_LEARN_DIR="$DATA_ROOT/quickstart/episodes"
export MODELS_SECRETS_FILE=/dev/null
unset GEMINI_API_KEY GOOGLE_API_KEY ROBOT_LIVE_TESTS
if ! command -v uv >/dev/null; then
  echo 'Install uv (https://docs.astral.sh/uv/), then rerun: bash quickstart.sh' >&2
  exit 1
fi
exec uv run --extra sim python scripts/release_quickstart.py "$@"
