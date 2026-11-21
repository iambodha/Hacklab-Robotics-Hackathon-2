#!/usr/bin/env bash
# Read-only health check of a System 2 install. Exit code 0 = ready for the simulator.
#   bash system2/doctor.sh
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(dirname "$here")"
S2_HOME="${S2_HOME:-$HOME/.local/share/elrobot-system2}"
ok=0
check() { if eval "$2" >/dev/null 2>&1; then echo "  ok    $1"; else echo "  FAIL  $1${3:+ -> $3}"; ok=1; fi; }
echo "System 2 install ($S2_HOME)"
check "system2 venv (mcp, numpy, opencv)" "$S2_HOME/system2-venv/bin/python -c 'import mcp, numpy, cv2, httpx'" "bash system2/setup/install.sh"
check "sim venv (mujoco)" "${ROBOT_SIM_PYTHON:-$S2_HOME/sim-venv/bin/python} -c 'import mujoco, mcp, PIL'" "bash system2/setup/install.sh"
urdf="${NORMA_CORE_REPO:-$repo/norma-core}/hardware/elrobot/simulation/elrobot_follower.urdf"
check "ElRobot URDF ($urdf)" "test -f '$urdf'" "set NORMA_CORE_REPO or run system2/setup/install.sh"
check "unit tests" "cd '$here' && $S2_HOME/system2-venv/bin/python -m pytest -q tests -p no:cacheprovider"
check "claude CLI (System 2 planner)" "command -v claude" "npm install -g @anthropic-ai/claude-code (or use the Windows launcher)"
echo "Optional: Open-Jev System 1"
if [ -f "$S2_HOME/models/Open-Jev-2B/package/checkpoint/model.json" ]; then
  echo "  ok    Open-Jev-2B package installed"
  if curl -sf --max-time 3 "${S2_JEV_URL:-http://127.0.0.1:8791}/health" >/dev/null; then
    echo "  ok    Open-Jev service is up at ${S2_JEV_URL:-http://127.0.0.1:8791}"
  else
    echo "  --    Open-Jev service not running (start it: bash system2/open_jev.sh)"
  fi
else
  echo "  --    not installed (bash system2/setup/install_open_jev.sh; needs an NVIDIA GPU with ~5 GB free)"
fi
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader | sed 's/^/  gpu   /'
exit $ok
