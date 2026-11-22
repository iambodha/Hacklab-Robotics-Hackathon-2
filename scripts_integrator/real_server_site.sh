#!/usr/bin/env bash
# Explicit operator launcher. Never invoke this during offline tests.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
export SITE="${1:-${SITE:-home}}"
if [ "$#" -gt 0 ]; then shift; fi
# Python registry in the MCP server supplies all site defaults; explicit overrides win.
exec bash "$HERE/sim/run_real_server.sh" "$@"
