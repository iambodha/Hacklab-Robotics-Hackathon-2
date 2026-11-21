#!/usr/bin/env bash
# Operator on the REAL arm driven by an OpenAI-compatible model endpoint (Gemma etc.), see run_agent_openai.py.
# Settings come from the environment or from sim/model_endpoint.env (copy model_endpoint.env.example).
#   bash sim/run_agent_model.sh --check
#   bash sim/run_agent_model.sh -p "Sort the loose items: garbage into the white box, the rest into the grey box."
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$here/model_endpoint.env" ]; then set -a; . "$here/model_endpoint.env"; set +a; fi
if [ -z "${NORMA_CORE_REPO:-}" ]; then
  for c in "$here/../norma-core" "$here/../../norma-core"; do
    if [ -d "$c/hardware/elrobot/simulation" ]; then export NORMA_CORE_REPO="$(cd "$c" && pwd)"; break; fi
  done
fi
cd "$here/.."
exec uv run -q --no-project --with 'mcp>=1.12,<2' python "$here/run_agent_openai.py" "$@"
