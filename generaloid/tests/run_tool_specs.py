"""Subprocess body of test_tool_parity.py: the MCP tool specs of an existing server module, loaded directly.

    python -m generaloid.tests.run_tool_specs sim_arm_mcp|real_arm_mcp      -> one JSON line {name: spec}
"""
from __future__ import annotations

import asyncio
import importlib
import json
import sys

from generaloid.core.paths import ensure_sim_on_path

if __name__ == "__main__":
    ensure_sim_on_path()
    mod = importlib.import_module(sys.argv[1])
    tools = asyncio.run(mod.server.list_tools())
    print(json.dumps({t.name: {"description": t.description, "inputSchema": t.inputSchema} for t in tools}))
    if hasattr(mod, "arm"):
        mod.arm.running = False
