#!/usr/bin/env bash
# Launch Claude Code as the arm operator with ONLY the simulated-robot MCP tools:
# no shell, no file access, no other MCP servers, so it cannot read the scene layout or sim state.
#   bash sim/run_agent.sh                 # interactive
#   bash sim/run_agent.sh -p "Start."     # one-shot (headless)
#   ROBOT_SIM_LAYOUT=$PWD/sim/layout_stir.json ROBOT_SIM_PROMPT=$PWD/sim/OPERATOR_STIR.md bash sim/run_agent.sh  # stir task
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
export ROBOT_SIM_GUI="${ROBOT_SIM_GUI:-1}"
export ROBOT_SIM_TRUTH_LOG="${ROBOT_SIM_TRUTH_LOG:-$HOME/.cache/elrobot-sim/truth.jsonl}"
mkdir -p "$(dirname "$ROBOT_SIM_TRUTH_LOG")"
config="$(mktemp --suffix=.json)"
trap 'rm -f "$config"' EXIT
cat > "$config" <<EOF
{"mcpServers": {"elrobot-sim": {"command": "bash", "args": ["$here/run_server.sh"]}}}
EOF
export ROBOT_SIM_RECORD_DIR="${ROBOT_SIM_RECORD_DIR:-$here/sessions}"
workdir="$(mktemp -d)"  # empty cwd: no CLAUDE.md or repo files in context
cd "$workdir"
status=0
claude --tools "" --strict-mcp-config --mcp-config "$config" \
  --allowedTools "mcp__elrobot-sim" \
  --append-system-prompt "$(cat "${ROBOT_SIM_PROMPT:-$here/OPERATOR.md}")" "$@" || status=$?
sleep 2  # Claude kills the server on exit; build the replay for the newest session here
session="$(ls -1d "$ROBOT_SIM_RECORD_DIR"/*/ 2>/dev/null | sort | tail -1)"
if [ -n "$session" ]; then "${ROBOT_SIM_PYTHON:-python3}" "$here/make_replay.py" "$session" || true; fi
exit $status
