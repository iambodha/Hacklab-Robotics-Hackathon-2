#!/usr/bin/env bash
# Install the System 2 goal server venv and, unless S2_SKIP_SIM=1, the simulator venv.
#   bash system2/setup/install.sh
# S2_HOME (default ~/.local/share/elrobot-system2) holds the venvs. Re-running is safe.
# Open-Jev (the optional learned System 1) is separate: bash system2/setup/install_open_jev.sh
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(dirname "$here")"
S2_HOME="${S2_HOME:-$HOME/.local/share/elrobot-system2}"
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1; }
mkdir -p "$S2_HOME"

[ -x "$S2_HOME/system2-venv/bin/python" ] || uv venv --python 3.12 "$S2_HOME/system2-venv"
uv pip install --python "$S2_HOME/system2-venv/bin/python" -r "$here/requirements.txt"

if [ "${S2_SKIP_SIM:-0}" != 1 ]; then
  [ -x "$S2_HOME/sim-venv/bin/python" ] || uv venv --python 3.12 "$S2_HOME/sim-venv"
  uv pip install --python "$S2_HOME/sim-venv/bin/python" -r "$here/setup/sim-requirements.txt"
  if [ -z "${NORMA_CORE_REPO:-}" ] && [ ! -f "$repo/norma-core/hardware/elrobot/simulation/elrobot_follower.urdf" ]; then
    echo "Fetching the ElRobot URDF + meshes (norma-core, sparse checkout) into $repo/norma-core"
    git clone --filter=blob:none --sparse https://github.com/norma-core/norma-core.git "$repo/norma-core"
    git -C "$repo/norma-core" sparse-checkout set hardware/elrobot
    git -C "$repo/norma-core" checkout --quiet 6b494f77fac366a878ae71222458b196f8400dee
  fi
fi
echo "System 2 installed under $S2_HOME. Check it with: bash system2/doctor.sh"
