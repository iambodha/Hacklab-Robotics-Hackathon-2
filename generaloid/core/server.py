"""MCP server assembly: connector + mode -> one FastMCP server that any MCP client (Claude, Gemini, Devin, ...) drives.

    server, mount = build_server(get("elrobot"), "sim")
    server.run()                                  # stdio; mount.close() afterwards

The connector mounts its robot (twin or real, through the Station) and hands back an MCP tool surface; the server
adopts those tools unchanged (same functions, schemas, descriptions, annotations), then runs the add-on hooks
(generaloid.comms). Nothing robot-specific lives here.
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

from .hooks import register_comms
from .station import Station

if TYPE_CHECKING:
    from generaloid.connectors.base import Connector, Mount

MODES = ("sim", "real")


def tool_names(server: FastMCP) -> list[str]:
    return [t.name for t in server._tool_manager.list_tools()]


def adopt_tools(dst: FastMCP, src: FastMCP) -> list[str]:
    """Put every tool registered on `src` onto `dst` as the same Tool object (no re-wrapping: parity by construction).
    FastMCP has no public "copy a tool" call; ToolManager keeps its tools in `_tools` (mcp 1.x)."""
    names = []
    for tool in src._tool_manager.list_tools():
        dst._tool_manager._tools[tool.name] = tool
        names.append(tool.name)
    return names


def build_server(connector: "Connector", mode: str, station: Station | None = None, name: str | None = None,
                 addons: bool = True) -> tuple[FastMCP, "Mount"]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    if mode not in connector.modes:
        raise ValueError(f"connector {connector.name!r} supports modes {connector.modes}, not {mode!r}")
    mount = connector.mount(mode, station or Station())
    server = FastMCP(name or f"generaloid-{connector.name}-{mode}",
                     instructions=f"Generaloid robot server: {connector.title} ({mode}). {connector.frame_text}")
    adopt_tools(server, mount.tools)
    mount.addons["comms"] = register_comms(server, connector) if addons else "disabled"
    return server, mount


def serve(connector: "Connector", mode: str, station: Station | None = None, addons: bool = True) -> None:
    server, mount = build_server(connector, mode, station, addons=addons)
    print(f"generaloid: serving {connector.name} ({mode}) with {len(tool_names(server))} tools; "
          f"add-ons {mount.addons}", file=sys.stderr, flush=True)
    try:
        server.run()
    finally:
        mount.close()
