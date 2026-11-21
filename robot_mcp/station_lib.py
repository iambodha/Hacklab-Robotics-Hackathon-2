"""station_lib — thin control layer over the NormaCore Station for the ElRobot.

Ported from last year's proven `claude-supervisor/robot_lib.py`, adapted to
this year's buses and to lazily import the norma-core checkout (so the rest of
the package imports fine on machines without it).

Protocol (verified on hardware both years):
  - Move:  ST3215SyncWriteCommand{address: 0x2A, motors:[{motor_id, value}]}
           wrapped in DriverCommand{STC_ST3215_COMMAND} via send_commands.
           0x2A = target-position register.
  - Read:  follow "st3215/inference" -> InferenceStateReader -> per-bus motors;
           present-position register = 0x38 (sign-normalized by _norm_raw).
           This read is camera-independent and never freezes on a camera drop.
  - Images: tail-read "inference/normvla" (combined frame w/ JPEGs), or follow
           "usbvideo/<hash>" queues when named explicitly.
"""
from __future__ import annotations

import asyncio
import logging
import os
import struct
import sys
from typing import Optional

from . import config

_log = logging.getLogger("robot_mcp.station")

NORMVLA_QUEUE = "inference/normvla"
ST3215_QUEUE = "st3215/inference"
ST3215_GOAL_REGISTER = 0x2A        # write target position
ST3215_POS_REGISTER = 0x38         # read present position

# ElRobot follower is 8-DOF; motor ids 1..8 map to joints j1..j8.
MOTOR_IDS = list(range(1, 9))
JOINT_NAMES = [f"j{i}" for i in MOTOR_IDS]

# Learned the hard way on last year's identical arm:
#   j8 = gripper JAWS (open ~0.78, close until it stalls ~0.22-0.34 = a grip)
#   j7 = gripper ROTATION (~0.45 at grasp), NOT the jaws.
JOINT_ROLES = {
    "j1": "base rotation", "j2": "shoulder", "j3": "elbow",
    "j4": "wrist pitch", "j5": "wrist", "j6": "wrist",
    "j7": "gripper rotation", "j8": "gripper jaws",
}
GRIPPER_JOINT = "j8"


# ---------------------------------------------------------------------------
# Lazy import of the norma-core checkout

_loaded = False


def _norma():
    """Make the norma-core checkout importable and return its modules.
    Raises a helpful error if the checkout is missing."""
    global _loaded
    repo = config.NORMA_CORE_REPO
    if not (repo / "software" / "station" / "shared" / "station_py").exists():
        raise RuntimeError(
            f"norma-core checkout not found at {repo}. Clone "
            f"https://github.com/norma-core/norma-core next to this repo or set "
            f"NORMA_CORE_REPO."
        )
    for p in (str(repo), str(repo / "software" / "station" / "shared")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import station_py                                                   # type: ignore
    from target.gen_python.protobuf.drivers.st3215 import st3215          # type: ignore
    from target.gen_python.protobuf.station import commands, drivers      # type: ignore
    try:
        from target.gen_python.protobuf.drivers.inferences import normvla  # type: ignore
    except ImportError:
        normvla = None
    _loaded = True
    return station_py, st3215, commands, drivers, normvla


async def connect(server: str | None = None):
    """Connect to the station normfs server (default localhost:8888)."""
    station_py, *_ = _norma()
    return await station_py.new_station_client(server or config.STATION_HOST, _log)


def _as_str(v) -> str:
    if isinstance(v, (bytes, bytearray, memoryview)):
        return bytes(v).decode("utf-8", "replace")
    return str(v)


_SIGN = 0x8000
_MAXA = 4095


def _norm_raw(raw: int) -> int:
    if raw & _SIGN:
        return (_MAXA + 1 - (raw & _MAXA)) & _MAXA
    return raw & _MAXA


# ---------------------------------------------------------------------------
# Joint state (camera-independent — the only read motion control should use)

async def list_buses(cli, timeout: float = 5.0) -> list[dict]:
    """Every ST3215 bus the station sees. With leader+follower there are two."""
    _, st3215, _, _, _ = _norma()
    q: asyncio.Queue = asyncio.Queue()
    cli.follow(ST3215_QUEUE, q)
    entry = await asyncio.wait_for(q.get(), timeout)
    if entry is None:
        raise RuntimeError(f"{ST3215_QUEUE} closed before delivering a bus state.")
    state = st3215.InferenceStateReader(memoryview(bytes(entry.Data)))
    buses = []
    for bus_state in (state.get_buses() or []):
        info = bus_state.get_bus()
        if not info:
            continue
        motors = bus_state.get_motors() or []
        buses.append({
            "serial": _as_str(info.get_serial_number()),
            "port": _as_str(info.get_port_name()),
            "motor_ids": [int(m.get_id()) for m in motors],
            "n_motors": len(motors),
        })
    return buses


async def discover_bus_serial(cli, timeout: float = 5.0) -> str:
    """The configured follower bus, or the first bus found. Never returns a
    bus that wasn't asked for silently — log a warning so a leader/follower
    mix-up is visible."""
    buses = await list_buses(cli, timeout)
    if not buses:
        raise RuntimeError("No ST3215 bus in st3215/inference (servo bus connected?).")
    if config.BUS_SERIAL:
        for b in buses:
            if b["serial"] == config.BUS_SERIAL:
                return config.BUS_SERIAL
        raise RuntimeError(
            f"follower bus {config.BUS_SERIAL!r} not found; available: {[b['serial'] for b in buses]}")
    _log.warning("ROBOT_BUS_SERIAL unset — using first bus %s; set it to the follower!",
                 buses[0]["serial"])
    return buses[0]["serial"]


async def current_st3215(cli, bus_serial: str, timeout: float = 5.0, retries: int = 5):
    """Live joint norms + ranges straight from the motor bus.

    Retries the WHOLE read on transient glitches (stall, empty frame, missing
    bus, partial motor set) so a one-off gap cannot abort a motion mid-way.
    Returns ({jname: norm}, [(rmin, rmax) per joint in j1..j8 order],
             {jname: raw_ticks})."""
    _, st3215, _, _, _ = _norma()
    for attempt in range(retries):
        try:
            q: asyncio.Queue = asyncio.Queue()
            cli.follow(ST3215_QUEUE, q)
            e = await asyncio.wait_for(q.get(), timeout)
            if e is None:
                raise RuntimeError("st3215/inference closed")
            st = st3215.InferenceStateReader(memoryview(bytes(e.Data)))
            for bs in (st.get_buses() or []):
                info = bs.get_bus()
                if not info or _as_str(info.get_serial_number()) != bus_serial:
                    continue
                by_id = {}
                for m in (bs.get_motors() or []):
                    s = bytes(m.get_state() or b"")
                    raw = _norm_raw(struct.unpack("<H", s[ST3215_POS_REGISTER:ST3215_POS_REGISTER + 2])[0]) \
                        if len(s) >= ST3215_POS_REGISTER + 2 else 0
                    by_id[m.get_id()] = (raw, int(m.get_range_min()), int(m.get_range_max()))
                joints, ranges, raws = {}, [], {}
                for i, mid in enumerate(MOTOR_IDS):
                    raw, rmin, rmax = by_id[mid]      # KeyError on partial frame -> retry
                    name = JOINT_NAMES[i]
                    joints[name] = round((raw - rmin) / (rmax - rmin), 4) if rmax > rmin else 0.0
                    raws[name] = raw
                    ranges.append((rmin, rmax))
                return joints, ranges, raws
            raise RuntimeError(f"follower bus {bus_serial} not in this frame")
        except (asyncio.TimeoutError, TimeoutError, RuntimeError, KeyError) as ex:
            if attempt == retries - 1:
                raise RuntimeError(f"st3215 read failed after {retries} tries: {ex}") from ex
            _log.warning("st3215 read glitch (%d/%d): %s — retrying", attempt + 1, retries, ex)
            await asyncio.sleep(0.3)


# ---------------------------------------------------------------------------
# Camera frames

async def camera_jpegs(cli, timeout: float = 3.0) -> tuple[list[bytes], str]:
    """Latest camera JPEGs. Tries the combined inference/normvla frame first,
    then any usbvideo queues named in ROBOT_CAM_QUEUES. Returns (jpegs, source);
    source is "normvla" | "usbvideo" | "none"."""
    _, _, _, _, normvla = _norma()
    if normvla is not None:
        try:
            qr = cli.read_from_tail(NORMVLA_QUEUE, offset=b"\x00", limit=1, step=1, buf_size=1)
            entry = await asyncio.wait_for(qr.data.get(), timeout=timeout)
            if entry is not None:
                frame = normvla.FrameReader(memoryview(bytes(entry.Data)))
                imgs = [bytes(im.get_jpeg()) for im in (frame.get_images() or []) if im.get_jpeg()]
                if imgs:
                    return imgs, "normvla"
        except Exception as ex:
            _log.debug("normvla frame unavailable: %s", ex)
    if config.CAM_QUEUES:
        try:
            from target.gen_python.protobuf.drivers.usbvideo import usbvideo, frame as fr  # type: ignore
        except ImportError:
            usbvideo = None
        if usbvideo is not None:
            for qid in config.CAM_QUEUES:
                try:
                    q: asyncio.Queue = asyncio.Queue()
                    cli.follow(f"usbvideo/{qid}", q)
                    for _ in range(50):
                        e = await asyncio.wait_for(q.get(), timeout)
                        rx = usbvideo.RxEnvelopeReader(memoryview(bytes(e.Data)))
                        pack = rx.get_frames()
                        if rx.get_type() != usbvideo.RxEnvelopeType.ET_FRAMES or pack is None:
                            continue
                        frames = pack.get_frames_data() or []
                        fmt = pack.get_format()
                        if fmt and fmt.get_kind() == fr.FrameFormatKind.FF_JPEG and frames:
                            return [bytes(frames[-1])], "usbvideo"
                        break
                except Exception as ex:
                    _log.debug("usbvideo queue %s unavailable: %s", qid, ex)
    return [], "none"


# ---------------------------------------------------------------------------
# Motion

def norm_to_raw(g_norm: float, range_min: int, range_max: int) -> int:
    g = max(0.0, min(1.0, float(g_norm)))
    raw = int(round(range_min + g * (range_max - range_min)))
    return max(range_min, min(range_max, raw))


async def move_norm(cli, bus_serial: str, goals_norm: list[float],
                    ranges: list[tuple[int, int]]) -> list[int]:
    """One atomic sync-write of all joints to normalized goals. Returns the raw
    tick targets actually sent."""
    station_py, st3215, commands, drivers, _ = _norma()
    if not (len(goals_norm) == len(ranges) == len(MOTOR_IDS)):
        raise ValueError("goals_norm and ranges must cover all motors")
    motors = [
        st3215.ST3215SyncWriteCommand_MotorWrite(
            motor_id=mid, value=int(raw).to_bytes(2, byteorder="little"))
        for mid, raw in zip(MOTOR_IDS, [
            norm_to_raw(g, rmin, rmax) for g, (rmin, rmax) in zip(goals_norm, ranges)])
    ]
    sync = st3215.ST3215SyncWriteCommand(address=ST3215_GOAL_REGISTER, motors=motors)
    cmd = st3215.Command(target_bus_serial=bus_serial, sync_write=sync)
    await station_py.send_commands(cli, [commands.DriverCommand(
        type=drivers.StationCommandType.STC_ST3215_COMMAND, body=cmd.encode())])
    return [norm_to_raw(g, rmin, rmax) for g, (rmin, rmax) in zip(goals_norm, ranges)]
