"""python -m dashboard [--port 8920] [--learn-dir DIR] [--actions FILE] [--operator-log FILE]"""
from __future__ import annotations

import argparse

from . import data
from .data import AgentStore, LearnStore
from .server import RECORDER_URL, RELAY_URL, App, make_server


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dashboard", description="Live bottle-opening learning dashboard")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8920)
    ap.add_argument("--learn-dir", default=str(data.LEARN_DIR), help="episode records (env BOTTLE_LEARN_DIR)")
    ap.add_argument("--actions", default=str(data.ACTIONS_LOG), help="real-arm MCP action log (env ACTIONS_LOG)")
    ap.add_argument("--operator-log", default=data.OPERATOR_LOG, help="Devin export/log (env OPERATOR_LOG)")
    ap.add_argument("--recorder", default=RECORDER_URL)
    ap.add_argument("--relay", default=RELAY_URL)
    a = ap.parse_args(argv)
    app = App(LearnStore(a.learn_dir), AgentStore(a.actions, a.operator_log), a.recorder, a.relay)
    srv = make_server(app, a.host, a.port)
    print(f"dashboard on http://{a.host}:{a.port}  learn_dir={a.learn_dir}  actions={a.actions}  "
          f"operator_log={a.operator_log or '-'}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
