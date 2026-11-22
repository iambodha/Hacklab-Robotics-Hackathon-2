#!/bin/bash
# Serves the intro animation locally (needs no internet) and opens it.
# Keys: F fullscreen · space pause · R restart · ←/→ seek 2s.  URL params: ?noloop  ?t=SECONDS
cd "$(dirname "$0")"
PORT=${1:-8765}
python3 -m http.server "$PORT" --bind 127.0.0.1 >/dev/null 2>&1 &
sleep 0.5; open "http://127.0.0.1:$PORT/?noloop"
wait
