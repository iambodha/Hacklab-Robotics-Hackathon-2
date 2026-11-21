#!/usr/bin/env bash
# Claude Code as System 2, with ONLY the elrobot-system2 tools: no shell, no files, no other MCP
# servers, an empty working directory. The MuJoCo window and the live dashboard open by default.
#   bash system2/run_agent.sh                                  # interactive
#   bash system2/run_agent.sh -p "Put the cube in the box."    # one-shot
#   S2_POLICY=jev bash system2/run_agent.sh ...                # Open-Jev as System 1 (started if needed)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
policy="${S2_POLICY:-geometric}"
if [ "$policy" != geometric ]; then bash "$here/open_jev.sh" --background; fi
export ROBOT_SIM_GUI="${ROBOT_SIM_GUI:-1}"
export ROBOT_SIM_RECORD_DIR="${ROBOT_SIM_RECORD_DIR:-$(dirname "$here")/sim/sessions}"
export ROBOT_SIM_TRUTH_LOG="${ROBOT_SIM_TRUTH_LOG:-$HOME/.cache/elrobot-sim/truth.jsonl}"   # human-side judge only
mkdir -p "$(dirname "$ROBOT_SIM_TRUTH_LOG")"
config="$(mktemp --suffix=.json)"
trap 'rm -f "$config"' EXIT
cat > "$config" <<EOF
{"mcpServers": {"elrobot-system2": {"command": "bash", "args": ["$here/run_server.sh"], "env": {"S2_POLICY": "$policy"}}}}
EOF
cd "$(mktemp -d)"
export MCP_TIMEOUT="${MCP_TIMEOUT:-120000}"  # first start builds the scene and loads MuJoCo
exec claude --tools "" --strict-mcp-config --mcp-config "$config" --allowedTools "mcp__elrobot-system2" \
  --append-system-prompt "$(cat "$here/PLANNER.md")" "$@"
