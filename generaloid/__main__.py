"""Generaloid CLI.

    python -m generaloid serve --connector elrobot --mode sim [--layout home] [--no-addons]   # MCP server on stdio
    python -m generaloid serve --connector elrobot --mode real          # the real arm through NormaCore Station
    python -m generaloid connectors                                     # built-in connectors
    python -m generaloid describe --connector so101                     # a connector's declarations (JSON)
    python -m generaloid tools --connector so101 --mode sim             # mount, list the MCP tools, exit
    python -m generaloid modules                                        # the modules around the core
    python -m generaloid station                                        # Station settings (never connects)

stdout belongs to the MCP protocol in `serve`; everything else Generaloid says goes to stderr.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _env_for(args) -> None:
    if getattr(args, "layout", None):
        os.environ["SIM_LAYOUT"] = os.environ["ROBOT_SIM_LAYOUT"] = args.layout


def main(argv: list[str] | None = None) -> int:
    from generaloid.core.paths import ensure_repo_on_path
    ensure_repo_on_path()
    ap = argparse.ArgumentParser(prog="python -m generaloid", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("serve", "tools"):
        p = sub.add_parser(cmd)
        p.add_argument("--connector", required=True, help="built-in name (elrobot, so101) or module:attr")
        p.add_argument("--mode", choices=("sim", "real"), default="sim")
        p.add_argument("--layout", help="twin table layout: a path or a name (sim modes)")
        p.add_argument("--no-addons", action="store_true", help="do not call add-on hooks (generaloid.comms)")
    sub.add_parser("connectors")
    sub.add_parser("describe").add_argument("--connector", required=True)
    sub.add_parser("modules")
    sub.add_parser("station")
    args = ap.parse_args(argv)

    from generaloid import connectors
    if args.cmd == "connectors":
        for n in connectors.available():
            c = connectors.get(n)
            print(f"{n:10} {c.title}  modes={','.join(c.modes)}  hardware_verified={c.hardware_verified}")
        return 0
    if args.cmd == "describe":
        print(json.dumps(connectors.get(args.connector).describe(), indent=2))
        return 0
    if args.cmd == "modules":
        from generaloid import modules
        for m in modules.MODULES.values():
            print(f"{m.name:12} {m.path:28} {m.role}")
        return 0
    if args.cmd == "station":
        from generaloid.core.station import Station
        print(json.dumps(Station().describe(), indent=2))
        return 0

    _env_for(args)
    conn = connectors.get(args.connector)
    from generaloid.core import server as S
    if args.cmd == "tools":
        srv, mount = S.build_server(conn, args.mode, addons=not args.no_addons)
        try:
            print(json.dumps({"server": srv.name, "tools": sorted(S.tool_names(srv)), "addons": mount.addons}))
        finally:
            mount.close()
        return 0
    if args.mode == "real":
        print(f"generaloid: REAL mode: {conn.title} through NormaCore Station at "
              f"{os.environ.get('STATION_HOST') or 'localhost'}", file=sys.stderr, flush=True)
    S.serve(conn, args.mode, addons=not args.no_addons)
    return 0


if __name__ == "__main__":
    sys.exit(main())
