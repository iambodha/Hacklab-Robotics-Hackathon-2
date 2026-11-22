"""Generaloid: a hardware-agnostic, model-agnostic robot MCP platform built around NormaCore Station.

    NormaCore Station (motor buses, cameras, commands, recording)      generaloid.core.station
        ^ every connector's I/O goes through it
    connectors (one per robot: elrobot, so101, ...)                     generaloid.connectors
        ^ a connector declares kinematics, joints, gripper, cameras, safety envelope, twin, skills
    MCP server assembly + add-on hooks (comms)                          generaloid.core.server / .hooks
        ^ any MCP client drives it: Claude, Gemini, Devin, ...
    surrounding modules: twin, splat, learning loop, executor,
    recorder, dashboard, perception, calibration                        generaloid.modules

    python -m generaloid serve --connector elrobot --mode sim

See docs/generaloid/ for the architecture, the connector guide and the migration plan.
"""
from __future__ import annotations

__version__ = "0.1.0"
