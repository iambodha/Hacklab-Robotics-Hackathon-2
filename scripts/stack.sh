#!/bin/bash
# One-command demo stack: recorder (:8900), executor console (:8910), twin mirror (:8892).
#
#   scripts/stack.sh up [--site home|office] [--backend fake|sim|real] [--real-motion] [--station]
#   scripts/stack.sh status
#   scripts/stack.sh down [--station]
#
# up      starts whatever is not already listening (idempotent) and prints the URLs. Station is only checked;
#         it is started only with --station (scripts/run-station.sh). The twin mirror needs Station and is skipped
#         without it. --real-motion starts the console with EXECUTOR_REAL_MOTION=1, which allows (does not do)
#         real motion: the arm still has to be enabled in the console or with `task --arm-enabled`.
# down    stops only what this script started (pid files in logs/); Station only with --station.
# Site: --site or SITE (default home, sites/<name>/site.json) is exported to every service it starts: calibration,
#         twin layout, camera queues, action logs and recordings follow it. A service already running for another
#         site is left alone with a warning: `down`, then `up` to switch.
# Logs: logs/<service>.log. Recordings: RECORDER_DIR (default "$DATA_ROOT/<site recordings dir>" on the external
#         drive: recordings/ for home, recordings/office/ for office).
# The console's twin uses SIM_LAYOUT (default: the site's layout; a name or a path); the mirror always uses the site's.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$REPO/logs"
DATA_ROOT="${DATA_ROOT:-/Volumes/Dell HDD/Robotics-Hackathon}"
SITE="${SITE:-home}"
BACKEND=sim
REAL_MOTION=0
WITH_STATION=0
CMD="${1:-status}"; shift || true
case "$CMD" in -h|--help|help) sed -n '2,19p' "$0"; exit 0 ;; esac
while [ $# -gt 0 ]; do
  case "$1" in
    --site) SITE="$2"; shift ;;
    --backend) BACKEND="$2"; shift ;;
    --real-motion) REAL_MOTION=1 ;;
    --station) WITH_STATION=1 ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)"; exit 2 ;;
  esac
  shift
done

export SITE
SITE_RECORDINGS="$(cd "$REPO" && python3 -c 'import sys
from sites import get_site
try: print(get_site().manifest["directories"]["recordings"])
except ValueError as e: sys.exit(str(e))')" || exit 2
if [ -n "${RECORDER_DIR:-}" ]; then REC_DRIVE="$(dirname "$RECORDER_DIR")"
else RECORDER_DIR="$DATA_ROOT/$SITE_RECORDINGS"; REC_DRIVE="$DATA_ROOT"; fi

if [ -z "${NORMA_CORE_REPO:-}" ]; then
  for c in "$REPO/norma-core" "$REPO/../repo/norma-core" "$REPO/../../repo/norma-core"; do
    [ -d "$c" ] && NORMA_CORE_REPO="$(cd "$c" && pwd)" && break
  done
fi
export NORMA_CORE_REPO="${NORMA_CORE_REPO:-}" MUJOCO_GL="${MUJOCO_GL:-cgl}"
PYDEPS=(--with numpy --with scipy --with pillow --with mujoco --with opencv-contrib-python --with 'mcp>=1.12,<2' --with httpx)

mkdir -p "$LOGS"
[ -f "$LOGS/.gitignore" ] || printf '*\n' > "$LOGS/.gitignore"

listening_pid() { lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null | head -1; }
port_of() { case "$1" in station) echo 8888 ;; recorder) echo 8900 ;; console) echo 8910 ;; mirror) echo 8892 ;; esac; }
url_of() { case "$1" in station) echo "http://127.0.0.1:8889 (web UI), tcp 127.0.0.1:8888" ;; recorder) echo "http://127.0.0.1:8900" ;;
  console) echo "http://127.0.0.1:8910" ;; mirror) echo "http://127.0.0.1:8892" ;; esac; }

wait_port() {  # name port seconds
  local i=0
  while [ $i -lt $(( $3 * 4 )) ]; do
    [ -n "$(listening_pid "$2")" ] && return 0
    kill -0 "$(cat "$LOGS/$1.pid" 2>/dev/null)" 2>/dev/null || return 1
    sleep 0.25; i=$((i + 1))
  done
  return 1
}

start() {  # name cmd...  (runs in REPO, log + pid under logs/)
  local name="$1"; shift
  local port; port="$(port_of "$name")"
  local pid; pid="$(listening_pid "$port")"
  if [ -n "$pid" ]; then
    printf '  %-9s already running (pid %s)   %s\n' "$name" "$pid" "$(url_of "$name")"
    local was; was="$(cat "$LOGS/$name.site" 2>/dev/null)"
    if [ "$name" != station ] && [ "$was" != "$SITE" ]; then
      echo "            WARNING: it runs for SITE=${was:-unknown}, not $SITE: stack.sh down, then up, to switch"
    fi
    return 0
  fi
  ( cd "$REPO" && exec nohup "$@" >>"$LOGS/$name.log" 2>&1 ) &
  echo $! > "$LOGS/$name.pid"
  echo "$SITE" > "$LOGS/$name.site"
  printf '  %-9s starting (log logs/%s.log) ... ' "$name" "$name"
  if wait_port "$name" "$port" "${WAIT:-90}"; then echo "up   $(url_of "$name")"
  else echo "FAILED: see logs/$name.log"; tail -5 "$LOGS/$name.log" | sed 's/^/      /'; return 1; fi
}

stop() {  # name
  local f="$LOGS/$1.pid"
  [ -f "$f" ] || { printf '  %-9s not started by this script: left alone\n' "$1"; return 0; }
  local pid; pid="$(cat "$f")"
  if kill -0 "$pid" 2>/dev/null; then
    pkill -TERM -P "$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null
    for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$pid" 2>/dev/null || break; sleep 0.3; done
    kill -0 "$pid" 2>/dev/null && { pkill -KILL -P "$pid" 2>/dev/null; kill -KILL "$pid" 2>/dev/null; }
    printf '  %-9s stopped (pid %s)\n' "$1" "$pid"
  else
    printf '  %-9s was not running\n' "$1"
  fi
  rm -f "$f" "$LOGS/$1.site"
}

status() {
  for s in station recorder console mirror; do
    local pid; pid="$(listening_pid "$(port_of "$s")")"
    local own=""; [ -f "$LOGS/$s.pid" ] && own=" (started by stack.sh, SITE=$(cat "$LOGS/$s.site" 2>/dev/null || echo unknown))"
    if [ -n "$pid" ]; then printf '  %-9s up    pid %-6s %s%s\n' "$s" "$pid" "$(url_of "$s")" "$own"
    elif [ -n "$own" ] && kill -0 "$(cat "$LOGS/$s.pid")" 2>/dev/null; then printf '  %-9s not listening (process alive: starting or restarting, see logs/%s.log)\n' "$s" "$s"
    elif [ -n "$own" ]; then printf '  %-9s DIED: see logs/%s.log\n' "$s" "$s"
    else printf '  %-9s down\n' "$s"; fi
  done
}

case "$CMD" in
  up)
    echo "stack up (site $SITE, backend $BACKEND$([ $REAL_MOTION = 1 ] && echo ', real motion ALLOWED'))"
    if [ -z "$(listening_pid 8888)" ]; then
      if [ $WITH_STATION = 1 ]; then WAIT=20 start station "$REPO/scripts/run-station.sh"
      else echo "  station   not running: start it (scripts/run-station.sh) or pass --station; real backend + mirror need it"; fi
    else printf '  %-9s up   %s\n' station "$(url_of station)"; fi
    if [ -z "$(listening_pid 8900)" ] && [ ! -d "$REC_DRIVE" ]; then
      echo "  recorder  NOT started: $REC_DRIVE is missing (mount the external drive or set RECORDER_DIR)"
    else
      mkdir -p "$RECORDER_DIR" 2>/dev/null
      RECORDER_DIR="$RECORDER_DIR" start recorder uv run --no-project -q python -m recorder serve
    fi
    ROBOT_SIM_LAYOUT="${SIM_LAYOUT:-$SITE}" EXECUTOR_REAL_MOTION=$([ $REAL_MOTION = 1 ] && echo 1 || echo 0) \
      start console uv run --no-project -q "${PYDEPS[@]}" python -m executor serve --backend "$BACKEND"
    if [ -n "$(listening_pid 8888)" ]; then
      # restarted if it exits (it is read-only); every restart is logged
      WAIT=60 start mirror bash -c 'while :; do "$@"; echo "=== mirror exited ($?) at $(date +%T); restarting in 3 s"; sleep 3; done' _ \
        uv run --no-project -q "${PYDEPS[@]}" python sim/mirror_real.py --layout "$SITE" --port 8892
    else echo "  mirror    skipped: needs Station"; fi
    echo
    echo "  open the console:  http://127.0.0.1:8910"
    echo "  run from a shell:  python -m executor task \"Put the blue object in the bin\" --backend sim"
    ;;
  down)
    echo "stack down"
    stop mirror; stop console; stop recorder
    if [ $WITH_STATION = 1 ]; then stop station; fi
    ;;
  status) status ;;
  *) echo "usage: scripts/stack.sh up|down|status [--site home|office] [--backend fake|sim|real] [--real-motion] [--station]"; exit 2 ;;
esac
