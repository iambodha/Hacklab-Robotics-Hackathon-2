"""Add-on hooks: optional modules that extend a running Generaloid server without the server knowing them.

The communication hook (inter-robot messaging, built by agent r5-comms in generaloid/comms/):

    generaloid.comms.register(server: FastMCP, connector: Connector) -> None

is called once, after the connector's tools are on the server and before it starts serving. It may add tools or
resources to `server` and read anything public on `connector` (name, joints, cameras, skills, describe()). If
generaloid.comms is not installed, the server starts without it and says nothing on stdout (stdout is the MCP
stdio channel). If it is installed but broken (its own import fails, or register raises), the robot server still
starts: robot tools never depend on an add-on. The reason goes to stderr and into the returned status.
"""
from __future__ import annotations

import importlib
import sys
import traceback
from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    from generaloid.connectors.base import Connector

COMMS_MODULE = "generaloid.comms"
COMMS_ENTRY = "register"


class CommsRegister(Protocol):
    """Signature of generaloid.comms.register."""

    def __call__(self, server: Any, connector: "Connector") -> None: ...


def _load(module: str) -> tuple[Any | None, str]:
    try:
        return importlib.import_module(module), "loaded"
    except ModuleNotFoundError as e:
        if e.name and (e.name == module or module.startswith(e.name + ".")):
            return None, "absent"                          # the add-on itself is not there: skip quietly
        return None, f"import failed: {type(e).__name__}: {e}"
    except Exception as e:                                 # noqa: BLE001  (an add-on bug must not stop the robot)
        return None, f"import failed: {type(e).__name__}: {e}"


def register_comms(server: Any, connector: "Connector", module: str = COMMS_MODULE) -> str:
    """Call `<module>.register(server, connector)` if that module imports. Returns a status:
    'absent' | 'registered' | 'no register()' | 'import failed: ...' | 'register failed: ...'."""
    mod, status = _load(module)
    if mod is None:
        if status != "absent":
            print(f"generaloid: comms add-on skipped ({status})", file=sys.stderr, flush=True)
        return status
    fn: Callable | None = getattr(mod, COMMS_ENTRY, None)
    if not callable(fn):
        print(f"generaloid: {module} has no {COMMS_ENTRY}(server, connector); skipped", file=sys.stderr, flush=True)
        return f"no {COMMS_ENTRY}()"
    try:
        fn(server, connector)
    except Exception as e:                                 # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        return f"register failed: {type(e).__name__}: {e}"
    return "registered"
