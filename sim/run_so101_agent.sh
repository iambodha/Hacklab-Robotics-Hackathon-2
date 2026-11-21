#!/usr/bin/env bash
# Launch an MCP operator agent on the SO-101 MuJoCo twin (`so101-sim`).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
export ROBOT_MODEL=so101
export ROBOT_SIM_GUI="${ROBOT_SIM_GUI:-0}"
export ROBOT_SIM_TRUTH_LOG="${ROBOT_SIM_TRUTH_LOG:-$HOME/.cache/so101-sim/truth.jsonl}"
mkdir -p "$(dirname "$ROBOT_SIM_TRUTH_LOG")"
config="$(mktemp --suffix=.json)"
trap 'rm -f "$config"' EXIT
cat > "$config" <<EOF
{"mcpServers": {"so101-sim": {"command": "bash", "args": ["$here/run_so101_server.sh"]}}}
EOF
export ROBOT_SIM_RECORD_DIR="${ROBOT_SIM_RECORD_DIR:-$here/sessions_so101}"
workdir="$(mktemp -d)"
cd "$workdir"
status=0
if command -v claude >/dev/null 2>&1; then
  claude --tools "" --strict-mcp-config --mcp-config "$config" \
    --allowedTools "mcp__so101-sim" \
    --append-system-prompt "$(cat "$here/OPERATOR_SO101.md")" "$@" || status=$?
elif [ -x "/usr/share/devin-desktop/resources/app/extensions/windsurf/devin/bin/devin" ]; then
  mkdir -p "$workdir/.devin"
  cp "$config" "$workdir/.devin/mcp_config.json"
  /usr/share/devin-desktop/resources/app/extensions/windsurf/devin/bin/devin \
    --permission-mode dangerous --respect-workspace-trust false \
    --prompt-file "$here/OPERATOR_SO101.md" "$@" || status=$?
else
  echo "Neither claude nor devin CLI found in PATH; use Devin Desktop with .devin/mcp_config.json or sim/mcp_daemon.py." >&2
  exit 1
fi
sleep 1
session="$(ls -1d "$ROBOT_SIM_RECORD_DIR"/*/ 2>/dev/null | sort | tail -1 || true)"
if [ -n "$session" ]; then python3 "$here/make_replay.py" "$session" || true; fi
exit $status
