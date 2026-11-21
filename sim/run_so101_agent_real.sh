#!/usr/bin/env bash
# Launch an MCP operator agent on the REAL SO-101 arm (`so101-real`).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
export ROBOT_MODEL=so101
config="$(mktemp "${TMPDIR:-/tmp}/so101-real.XXXXXX.json")"
trap 'rm -f "$config"' EXIT
cat > "$config" <<EOF
{"mcpServers": {"so101-real": {"command": "bash", "args": ["$here/run_so101_real_server.sh"]}}}
EOF
workdir="$(mktemp -d)"
cd "$workdir"
if command -v claude >/dev/null 2>&1; then
  exec claude --tools "" --strict-mcp-config --mcp-config "$config" \
    --allowedTools "mcp__so101-real" \
    --append-system-prompt "$(cat "$here/OPERATOR_SO101_REAL.md")" "$@"
elif [ -x "/usr/share/devin-desktop/resources/app/extensions/windsurf/devin/bin/devin" ]; then
  mkdir -p "$workdir/.devin"
  cp "$config" "$workdir/.devin/mcp_config.json"
  exec /usr/share/devin-desktop/resources/app/extensions/windsurf/devin/bin/devin \
    --permission-mode dangerous --respect-workspace-trust false \
    --prompt-file "$here/OPERATOR_SO101_REAL.md" "$@"
else
  echo "Neither claude nor devin CLI found in PATH; use Devin Desktop with .devin/mcp_config.json or sim/mcp_daemon.py." >&2
  exit 1
fi
