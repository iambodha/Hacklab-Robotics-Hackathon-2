#!/usr/bin/env bash
# Install the Open-Jev System 1 service (Linux or WSL2 with an NVIDIA GPU), fully pinned.
#   bash system2/setup/install_open_jev.sh
# Everything lands under S2_HOME (default ~/.local/share/elrobot-system2): the loader source,
# its own venv, the released 2B package and the pinned Qwen base weights (~10 GB in total).
# Re-running is safe: finished steps are skipped.
set -euo pipefail
S2_HOME="${S2_HOME:-$HOME/.local/share/elrobot-system2}"
export HF_HOME="${S2_HF_HOME:-$S2_HOME/hf}"
JEV_SRC="$S2_HOME/Open-Jev"
VENV="$S2_HOME/open-jev-venv"
PKG="$S2_HOME/models/Open-Jev-2B"
# Pins from Open-Jev docs/deployment.md (the tested server implementation and release).
JEV_COMMIT=80ca8e81d08992cb2a4cbb6a0caa1355ad3e5aee
JEV_MODEL=ZefanCai/Open-Jev-2B
JEV_MODEL_REV=0c7aa498b1627be8da4acf34c863ff0ee0a92785
BASE_MODEL=Qwen/Qwen3.5-2B
BASE_REV=15852e8c16360a2fea060d615a32b45270f8a8fc
TORCH_SPEC="torch==2.9.0"
TORCH_INDEX="${S2_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1; }
command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
mkdir -p "$S2_HOME/models" "$HF_HOME"

if [ ! -d "$JEV_SRC/.git" ]; then
  git clone --filter=blob:none https://github.com/Zefan-Cai/Open-Jev.git "$JEV_SRC"
fi
git -C "$JEV_SRC" fetch --quiet origin "$JEV_COMMIT" || true
git -C "$JEV_SRC" checkout --quiet "$JEV_COMMIT"
echo "Open-Jev loader at $(git -C "$JEV_SRC" rev-parse HEAD)"

[ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" "$TORCH_SPEC" --index-url "$TORCH_INDEX"
uv pip install --python "$VENV/bin/python" -e "$JEV_SRC[train]"

if [ ! -f "$PKG/package/checkpoint/model.json" ]; then
  "$VENV/bin/hf" download "$JEV_MODEL" --revision "$JEV_MODEL_REV" --local-dir "$PKG"
fi
"$VENV/bin/python" - "$PKG/package/checkpoint/model.json" "$BASE_MODEL" "$BASE_REV" <<'EOF'
import json, sys
from huggingface_hub import snapshot_download
cfg = json.load(open(sys.argv[1]))
if (cfg["model_id"], cfg["revision"]) != (sys.argv[2], sys.argv[3]):
    sys.exit(f"checkpoint expects base {cfg['model_id']}@{cfg['revision']}, installer pins {sys.argv[2]}@{sys.argv[3]}")
print("base weights:", snapshot_download(sys.argv[2], revision=sys.argv[3]))
EOF
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "Open-Jev installed. Start it with: bash system2/open_jev.sh"
