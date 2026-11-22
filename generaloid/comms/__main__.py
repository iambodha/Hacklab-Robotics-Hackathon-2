"""Standalone runner, no Generaloid needed.

  python -m generaloid.comms hub    [--port 8930] [--token-file F] [--peer-keys keys.json] [--log hub.jsonl]
                                    [--trust-proxy]
  python -m generaloid.comms client --url URL --robot-id human-ops listen
  python -m generaloid.comms client --url URL --robot-id human-ops send --to so101-a --kind message \
                                    --payload '{"text": "hi"}' [--re <request id>]
  python -m generaloid.comms client --url URL --robot-id human-ops peers | requests | log
  python -m generaloid.comms mcp    [--robot-id ID] [--connector NAME]   # stdio MCP server with only comms_* tools
  python -m generaloid.comms twin   [--robot-id ID]                      # MuJoCo twin MCP + comms_* tools

The join token comes from $GENERALOID_COMMS_TOKEN or --token-file, never from the command line (other users of the
machine can read command lines in `ps`). `hub` generates one if neither is set and writes it to --token-file (0600)
or prints it once.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def _token(args, create: bool) -> str:
    tok = os.environ.get("GENERALOID_COMMS_TOKEN", "").strip()
    if tok:
        return tok
    f = Path(args.token_file).expanduser() if getattr(args, "token_file", None) else None
    if f and f.exists():
        return f.read_text().strip()
    if not create:
        sys.exit("no token: set GENERALOID_COMMS_TOKEN or pass --token-file")
    tok = secrets.token_urlsafe(24)
    if f:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(tok + "\n")
        f.chmod(0o600)
        print(f"join token written to {f} (mode 600)", file=sys.stderr)
    else:
        print(f"join token (give it privately to the robots that should join): {tok}", file=sys.stderr)
    return tok


def cmd_hub(args):
    import uvicorn
    from .hub import Hub, HubConfig, create_app
    peer_keys = json.loads(Path(args.peer_keys).read_text()) if args.peer_keys else None
    hub = Hub(token=None if peer_keys else _token(args, create=True), peer_keys=peer_keys, log_path=args.log,
              config=HubConfig(trust_proxy=args.trust_proxy))
    if args.bind not in ("127.0.0.1", "localhost", "::1") and os.environ.get("GENERALOID_COMMS_ALLOW_LAN") != "1":
        sys.exit("refusing to bind beyond localhost without GENERALOID_COMMS_ALLOW_LAN=1 (use cloudflared instead)")
    print(f"generaloid-comms hub on http://{args.bind}:{args.port}  "
          f"(auth: {'per-peer keys' if peer_keys else 'shared token'}; log: {args.log or 'memory only'})",
          file=sys.stderr)
    print(f"public access: cloudflared tunnel --no-autoupdate --url http://127.0.0.1:{args.port}", file=sys.stderr)
    uvicorn.run(create_app(hub), host=args.bind, port=args.port, log_level="warning", access_log=False)


def cmd_client(args):
    from .client import CommsClient
    hello = {"robot_id": args.robot_id, "connector": args.connector, "display_name": args.robot_id,
             "capabilities": ["human-operator"] if args.connector == "human" else []}
    c = CommsClient(args.url, _token(args, create=False), hello)
    c.join(start_polling=args.action == "listen")
    try:
        if args.action == "listen":
            n = 0
            while True:
                for m in c.inbox(n, wait=30):
                    n = m["n"]
                    print(json.dumps(m), flush=True)
        elif args.action == "send":
            print(json.dumps(c.send(args.to, args.kind, json.loads(args.payload or "{}"), re=args.re), indent=1))
        elif args.action == "peers":
            print(json.dumps(c.peers(), indent=1))
        elif args.action == "requests":
            print(json.dumps(c.requests(), indent=1))
        elif args.action == "log":
            print(json.dumps(c.log(), indent=1))
    except KeyboardInterrupt:
        pass
    finally:
        c.leave()


def cmd_mcp(args):
    from mcp.server.fastmcp import FastMCP
    from . import register
    server = FastMCP("generaloid-comms")
    register(server, SimpleNamespace(name=args.connector, skills=[], capabilities=[]), robot_id=args.robot_id)
    server.run()


def cmd_twin(args):
    from .twin_server import main
    main(args.robot_id)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m generaloid.comms", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hub", help="run a hub (one robot hosts it)")
    h.add_argument("--port", type=int, default=int(os.environ.get("GENERALOID_COMMS_PORT", "8930")))
    h.add_argument("--bind", default="127.0.0.1")
    h.add_argument("--token-file")
    h.add_argument("--peer-keys", help="JSON {robot_id: key}; replaces the shared token")
    h.add_argument("--log", help="append every message/event as JSONL here")
    h.add_argument("--trust-proxy", action="store_true", help="behind cloudflared: rate-limit by CF-Connecting-IP")
    h.set_defaults(fn=cmd_hub)
    c = sub.add_parser("client", help="a human/scripted participant")
    c.add_argument("--url", required=True)
    c.add_argument("--robot-id", default="human-ops")
    c.add_argument("--connector", default="human")
    c.add_argument("--token-file")
    c.add_argument("action", choices=["listen", "send", "peers", "requests", "log"])
    c.add_argument("--to")
    c.add_argument("--kind", default="message")
    c.add_argument("--payload")
    c.add_argument("--re")
    c.set_defaults(fn=cmd_client)
    m = sub.add_parser("mcp", help="stdio MCP server with only the comms tools")
    m.add_argument("--robot-id")
    m.add_argument("--connector", default="robot")
    m.set_defaults(fn=cmd_mcp)
    t = sub.add_parser("twin", help="MuJoCo twin MCP server + comms tools")
    t.add_argument("--robot-id")
    t.set_defaults(fn=cmd_twin)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
