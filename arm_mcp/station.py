"""Shared Station access for arm_mcp: norma-core imports, live motor-bus reads, sync writes."""
from __future__ import annotations

import asyncio
import logging
import os
import struct
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CANDIDATES = [os.environ.get("NORMA_CORE_REPO"), _HERE.parent / "norma-core", _HERE.parent.parent / "norma-core"]
NORMA_CORE = next((Path(p) for p in _CANDIDATES if p and (Path(p) / "software/station/shared/station_py").exists()), None)
if NORMA_CORE is None:
    raise SystemExit("norma-core checkout not found: set NORMA_CORE_REPO or place it next to this repo")
for p in (NORMA_CORE, NORMA_CORE / "software/station/shared"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if str(_HERE.parent) not in sys.path:
    sys.path.append(str(_HERE.parent))

from perception.latest import Latest  # noqa: E402  (one shared subscription per client and queue)
from station_py import new_station_client, send_commands  # noqa: E402
from target.gen_python.protobuf.drivers.st3215 import st3215  # noqa: E402
from target.gen_python.protobuf.station import commands, drivers  # noqa: E402

BUS = os.environ.get("ROBOT_BUS_SERIAL", "5B61033301")
STATION_HOST = os.environ.get("STATION_HOST", "localhost")
ST3215_QUEUE = "st3215/inference"
NAMES = ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "j8"]
IDS = [1, 2, 3, 4, 5, 6, 7, 8]
_log = logging.getLogger("arm_mcp")


def as_str(v) -> str:
    return bytes(v).decode("utf-8", "replace") if isinstance(v, (bytes, bytearray, memoryview)) else str(v)


def u16(s: bytes, a: int) -> int:
    return struct.unpack("<H", s[a:a + 2])[0]


def norm_raw(raw: int) -> int:
    return (4096 - (raw & 4095)) & 4095 if raw & 0x8000 else raw & 4095


async def connect():
    return await new_station_client(STATION_HOST, _log)


async def bus_frame(cli):
    """One fresh raw st3215/inference frame for BUS: list of motor readers, or None if the bus is absent.
    Uses the client's single long-lived subscription (station_py has no unfollow)."""
    e = await Latest.of(cli, ST3215_QUEUE).get(fresh=True, timeout=5)
    st = st3215.InferenceStateReader(memoryview(bytes(e.Data)))
    for bs in st.get_buses() or []:
        if as_str(bs.get_bus().get_serial_number()) == BUS:
            return bs.get_motors() or []
    return None


async def read_state(cli) -> dict:
    """Live per-motor state keyed by motor id. Retries, then raises if any motor read failed."""
    for _ in range(5):
        motors = await bus_frame(cli)
        out = {}
        for m in motors or []:
            s, err = bytes(m.get_state()), m.get_error()
            if len(s) < 0x46 or (err and as_str(err.get_description())):
                out = None
                break
            lo, hi = int(m.get_range_min()), int(m.get_range_max())
            raw = norm_raw(u16(s, 0x38))
            out[int(m.get_id())] = {"raw": raw, "goal": u16(s, 0x2A), "lo": lo, "hi": hi,
                                    "norm": round((raw - lo) / (hi - lo), 4) if hi > lo else None,
                                    "in_range": lo <= raw <= hi, "torque": s[0x28], "status": s[0x41],
                                    "temp": s[0x3F], "volt": s[0x3E] / 10, "load": u16(s, 0x3C) & 0x3FF}
        if out and len(out) == len(IDS):
            return out
        await asyncio.sleep(0.3)
    raise RuntimeError(f"motor bus {BUS}: no complete error-free read (bus desync? restart Station)")


async def sync_write(cli, address: int, values: dict[int, int], width: int) -> None:
    motors = [st3215.ST3215SyncWriteCommand_MotorWrite(motor_id=mid, value=int(v).to_bytes(width, "little"))
              for mid, v in values.items()]
    cmd = st3215.Command(target_bus_serial=BUS,
                         sync_write=st3215.ST3215SyncWriteCommand(address=address, motors=motors))
    await send_commands(cli, [commands.DriverCommand(type=drivers.StationCommandType.STC_ST3215_COMMAND,
                                                     body=cmd.encode())])
