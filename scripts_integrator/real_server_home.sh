#!/usr/bin/env bash
# Preserve the named home entry point even when the operator's shell selects office.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/real_server_site.sh" home "$@"
