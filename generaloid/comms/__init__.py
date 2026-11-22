"""generaloid.comms: robots that talk to each other.

One robot's MCP server hosts a hub; other robots (other agents, each driving its own arm) join it, locally or across
the internet behind a Cloudflare tunnel, and exchange requests, help calls, progress, results, observations and
status. Docs: docs/comms/.

Core principle: a message from another robot never moves this robot's arm. It arrives as a request that the local
agent (or a human) decides on, under this robot's own safety envelope.

Hook for Generaloid: `register(server, connector)` adds the comms_* MCP tools to a FastMCP server. Generaloid calls
it when this package imports and skips it otherwise. Standalone: `python -m generaloid.comms hub|client|mcp|twin`.
"""
from .protocol import PROTOCOL, VERSION


def register(server, connector=None, **kw):
    """Add comms_host / comms_join / comms_peers / comms_send / comms_inbox / comms_reply / comms_requests /
    comms_leave to `server` (a FastMCP). `connector` is only read for the robot's identity (name, skills,
    capabilities, frames). Returns the CommsTools state object."""
    from .tools import register as _register
    return _register(server, connector, **kw)


__all__ = ["register", "PROTOCOL", "VERSION"]
