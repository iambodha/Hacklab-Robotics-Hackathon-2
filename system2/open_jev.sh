#!/usr/bin/env bash
# Start the Open-Jev System 1 service (installed by setup/install_open_jev.sh), loopback only.
#   bash system2/open_jev.sh            # then: curl http://127.0.0.1:8791/health
# Env: S2_JEV_PORT (8791), S2_JEV_DEVICE (cuda:0; "cpu" works but is slow), S2_JEV_BATCH (candidate
# prompts per forward pass, 4), S2_JEV_MAX_LENGTH (4096).
#   bash system2/open_jev.sh --background   # start detached (log: S2_JEV_LOG) and wait until it is ready
set -euo pipefail
S2_HOME="${S2_HOME:-$HOME/.local/share/elrobot-system2}"
url="http://127.0.0.1:${S2_JEV_PORT:-8791}/health"
if [ "${1:-}" = --background ]; then
  if curl -sf --max-time 3 "$url" >/dev/null; then echo "Open-Jev already running"; exit 0; fi
  nohup bash "$0" > "${S2_JEV_LOG:-/tmp/open-jev.log}" 2>&1 &
  echo "starting Open-Jev (loading takes 1-3 min; log ${S2_JEV_LOG:-/tmp/open-jev.log}) ..."
  for _ in $(seq 1 150); do
    sleep 2
    if curl -sf --max-time 3 "$url" >/dev/null; then echo "Open-Jev ready"; exit 0; fi
    kill -0 $! 2>/dev/null || { echo "Open-Jev exited; see the log" >&2; exit 1; }
  done
  echo "Open-Jev did not become ready in 5 min" >&2; exit 1
fi
export HF_HOME="${S2_HF_HOME:-$S2_HOME/hf}"
export HF_HUB_OFFLINE=1   # everything was downloaded by the installer; never fetch at serve time
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ckpt="$S2_HOME/models/Open-Jev-2B/package/checkpoint"
[ -f "$ckpt/model.json" ] || { echo "Open-Jev is not installed: bash system2/setup/install_open_jev.sh" >&2; exit 1; }
cd "$S2_HOME/Open-Jev"
exec "$S2_HOME/open-jev-venv/bin/python" -m jev.server --checkpoint "$ckpt" \
  --device "${S2_JEV_DEVICE:-cuda:0}" --max-length "${S2_JEV_MAX_LENGTH:-4096}" \
  --batch-size "${S2_JEV_BATCH:-4}" --no-prefix-cache --host 127.0.0.1 --port "${S2_JEV_PORT:-8791}"
