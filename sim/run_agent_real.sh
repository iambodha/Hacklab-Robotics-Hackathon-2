#!/usr/bin/env bash
# Claude Code as the operator of the REAL arm with ONLY the elrobot-real tools (no shell, no files, no other
# MCP servers), the same way run_agent.sh runs the simulator. A human must watch the arm and be ready to stop it
# (Ctrl-C here, then: the servos hold their last goal).
#   bash sim/run_agent_real.sh                       # interactive
#   bash sim/run_agent_real.sh -p "Start." --model sonnet    # one-shot
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
config="$(mktemp "${TMPDIR:-/tmp}/elrobot-real.XXXXXX")"
trap 'rm -f "$config"' EXIT
cat > "$config" <<EOF
{"mcpServers": {"elrobot-real": {"command": "bash", "args": ["$here/run_real_server.sh"]}}}
EOF
workdir="$(mktemp -d)"   # empty cwd: no CLAUDE.md or repo files in context
cd "$workdir"
exec claude --tools "" --strict-mcp-config --mcp-config "$config" \
  --allowedTools "mcp__elrobot-real" \
  --append-system-prompt "$(cat "$here/OPERATOR_REAL.md")" "$@"
