"""Generaloid core: the NormaCore Station I/O layer, MCP server assembly and add-on hooks."""
from .hooks import COMMS_MODULE, CommsRegister, register_comms  # noqa: F401
from .station import Station, StationRefused, check_live_allowed, connect, live_allowed, station_host  # noqa: F401
