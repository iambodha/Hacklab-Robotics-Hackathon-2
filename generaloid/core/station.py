"""NormaCore Station: the I/O layer at the centre of Generaloid.

Every motor bus, camera, command and recording of a real robot goes through one NormaCore Station (TCP :8888,
repo/norma-core + repo/station-bin). Connectors never open a serial port or a camera device themselves: they get a
`Station` and use its motor-bus and camera queues. In the twin, the simulator stands in for the Station.

    st = Station()                    # host from STATION_HOST (default localhost)
    cli = await st.connect()          # refuses under pytest unless ROBOT_LIVE_TESTS=1
    st.install()                      # route the existing Station users (arm_mcp.station, sim/station_io) through it
    st.bus()                          # motor bus (st3215/inference reads, sync writes): arm_mcp/station.py
    st.cameras()                      # camera queues (usbvideo/<hash> JPEGs, joint follower): sim/station_io.py

Safety (owned here): `connect()` is the one shared Station connect helper. On 22 Nov a test run opened dozens of
Station connections per second while an operator agent drove the real arm, and its motor reads stalled. The repo-root
conftest.py points STATION_HOST at an unresolvable name during tests; `check_live_allowed()` adds a hard refusal
that also covers subprocesses started by a test (they inherit PYTEST_CURRENT_TEST).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from .paths import REPO, ensure_repo_on_path

DEFAULT_HOST = "localhost"
STATION_PORT = 8888                 # NormaCore Station TCP API (station-bin default); station_py picks it
LIVE_TESTS_ENV = "ROBOT_LIVE_TESTS"
RECORDER_URL = os.environ.get("RECORDER_URL", "http://localhost:8900")   # recorder/: records every frame + motor state
_log = logging.getLogger("generaloid.station")


class StationRefused(ConnectionRefusedError):
    """A Station connection was refused by Generaloid's guard (not by the Station). A ConnectionError, so code that
    already handles an unreachable Station handles this too."""


def station_host() -> str:
    """The Station host every client uses: env STATION_HOST, else localhost. Read at call time."""
    return os.environ.get("STATION_HOST") or DEFAULT_HOST


def under_pytest() -> bool:
    """True inside a pytest run, including subprocesses a test starts (they inherit PYTEST_CURRENT_TEST)."""
    return "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules or "_pytest" in sys.modules


def live_allowed() -> bool:
    return not under_pytest() or os.environ.get(LIVE_TESTS_ENV) == "1"


def check_live_allowed(host: str | None = None) -> None:
    """Raise StationRefused when a live Station connection is not allowed (under pytest without ROBOT_LIVE_TESTS=1)."""
    if not live_allowed():
        raise StationRefused(f"refusing to connect to NormaCore Station at {host or station_host()}: running under "
                             f"pytest and {LIVE_TESTS_ENV} != 1 (tests never talk to the live arm)")


def norma_core() -> Path | None:
    """The norma-core checkout that provides station_py and the generated protobuf readers."""
    cands = [os.environ.get("NORMA_CORE_REPO"), REPO / "norma-core", REPO.parent / "norma-core",
             REPO.parent.parent / "repo" / "norma-core"]
    return next((Path(p) for p in cands if p and (Path(p) / "software/station/shared/station_py").exists()), None)


def _station_py():
    nc = norma_core()
    if nc is None:
        raise RuntimeError("norma-core checkout not found: set NORMA_CORE_REPO")
    for p in (nc, nc / "software/station/shared"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import station_py
    return station_py


async def connect(log: logging.Logger | None = None, host: str | None = None):
    """The one shared Station connect helper: guard, then station_py.new_station_client(host or STATION_HOST)."""
    host = host or station_host()
    check_live_allowed(host)
    return await _station_py().new_station_client(host, log or _log)


class Station:
    """One NormaCore Station as an I/O layer object handed to connectors (see the module docstring).

    Subclass it to stand in for a Station (generaloid.testing.FakeStation): a connector only uses connect(), bus(),
    cameras() and install(), so a fake bus is a drop-in."""

    def __init__(self, host: str | None = None):
        self._host = host

    @property
    def host(self) -> str:
        return self._host or station_host()

    async def connect(self, log: logging.Logger | None = None):
        return await connect(log, self.host)

    def bus(self):
        """Motor bus access (arm_mcp/station.py: read_state, sync_write, bus_frame, BUS, NAMES, IDS): the module object
        arm_mcp/server.py uses (it imports it by the bare name `station`, so `arm_mcp.station` would be a second copy).
        Import after ROBOT_BUS_SERIAL is set: the module reads it at import (real_arm_mcp sets it from the profile)."""
        ensure_repo_on_path()
        from arm_mcp import server as am
        return am.stn

    def cameras(self):
        """Camera queues and the joint follower (sim/station_io.py: grab_jpegs, JointFollower, snapshot)."""
        from .paths import ensure_sim_on_path
        ensure_sim_on_path()
        import station_io
        return station_io

    def install(self) -> "Station":
        """Route the existing Station users through this object's connect() (same host, same guard). Idempotent."""
        conn = self.connect

        async def _connect():
            return await conn(logging.getLogger("generaloid.station.bus"))

        bus = self.bus()
        bus.connect = _connect
        try:
            self.cameras().connect = _connect
        except ImportError:                                   # cameras module needs numpy only; bus already routed
            pass
        return self

    def describe(self) -> dict:
        return {"host": self.host, "port": STATION_PORT, "live_allowed": live_allowed(), "recorder": RECORDER_URL,
                "norma_core": str(norma_core()) if norma_core() else None}
