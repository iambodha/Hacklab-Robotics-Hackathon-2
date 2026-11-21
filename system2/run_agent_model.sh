#!/usr/bin/env bash
# System 2 = any OpenAI-compatible vision model (e.g. Gemma on a friend's machine behind a Cloudflare tunnel),
# through the same runner as sim/run_agent_model.sh, but with the elrobot-system2 tools and PLANNER.md.
# Endpoint settings: MODEL_BASE_URL, MODEL_NAME, ... or sim/model_endpoint.env (see model_endpoint.env.example).
#   bash system2/run_agent_model.sh --check
#   bash system2/run_agent_model.sh -p "Put the black cube in the grey box."
#   S2_ARM=real bash system2/run_agent_model.sh -p "..."      # the real follower (sim/REAL_ARM.md first)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
if [ "${S2_POLICY:-geometric}" != geometric ]; then bash "$here/open_jev.sh" --background; fi
export OPERATOR_SERVER_SH="$here/run_server.sh" OPERATOR_PROMPT="$here/PLANNER.md"
export ROBOT_SIM_GUI="${ROBOT_SIM_GUI:-1}"
exec bash "$(dirname "$here")/sim/run_agent_model.sh" "$@"
