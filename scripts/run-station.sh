#!/bin/zsh
# Starts NormaCore Station. Web UI: http://localhost:8889  ·  TCP API: localhost:8888
# Bound to loopback so nobody else on the venue network can drive the arm.
# Pass --lan to expose it (e.g. to view cameras from a phone).
cd "$(dirname "$0")/../station-run"
host=127.0.0.1; [[ "$1" == "--lan" ]] && host=0.0.0.0
exec ../station-bin/station --tcp $host:8888 --web $host:8889
