"""Shared Station access for arm_mcp: norma-core imports, TCP + WebSocket NormFS client, motor-bus & camera reads."""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import socket
import struct
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

_HERE = Path(__file__).resolve().parent
_CANDIDATES = [
    os.environ.get("NORMA_CORE_REPO"),
    _HERE.parent / "norma-core",
    _HERE.parent.parent / "norma-core",
    _HERE.parent.parent / "normacorehack/norma-core",
]
NORMA_CORE = next((Path(p) for p in _CANDIDATES if p and (Path(p) / "software/station/shared/station_py").exists()), None)
if NORMA_CORE is None:
    raise SystemExit("norma-core checkout not found: set NORMA_CORE_REPO or place it next to this repo")
for p in (NORMA_CORE, NORMA_CORE / "software/station/shared"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from station_py import StreamEntry, StreamEntryId, new_station_client, send_commands  # noqa: E402
from target.gen_python.protobuf.drivers.st3215 import st3215  # noqa: E402
from target.gen_python.protobuf.drivers.usbvideo import frame as fr, usbvideo  # noqa: E402
from target.gen_python.protobuf.normfs import normfs as normfs_pb2  # noqa: E402
from target.gen_python.protobuf.station import commands, drivers, inference  # noqa: E402

ROBOT_MODEL = os.environ.get("ROBOT_MODEL", "elrobot").lower()
DEFAULT_BUS = "auto" if ROBOT_MODEL == "so101" else "5B61033301"
BUS = os.environ.get("ROBOT_BUS_SERIAL", DEFAULT_BUS)
STATION_HOST = os.environ.get("STATION_HOST", "localhost")
STATION_WS_URL = os.environ.get("STATION_WS_URL", "")
ST3215_QUEUE = "st3215/inference"
MOTOR_COUNT = 6 if ROBOT_MODEL == "so101" else 8
NAMES = [f"j{i}" for i in range(1, MOTOR_COUNT + 1)]
IDS = list(range(1, MOTOR_COUNT + 1))
_log = logging.getLogger("arm_mcp")


def as_str(v) -> str:
    return bytes(v).decode("utf-8", "replace") if isinstance(v, (bytes, bytearray, memoryview)) else str(v)


def u16(s: bytes, a: int) -> int:
    return struct.unpack("<H", s[a:a + 2])[0]


def norm_raw(raw: int) -> int:
    return (4096 - (raw & 4095)) & 4095 if raw & 0x8000 else raw & 4095


class WsStationClient:
    """Pure-stdlib asyncio NormFS client over Station's WebSocket endpoint (e.g. ws://127.0.0.1:8000/api)."""

    def __init__(self, ws_url: str):
        u = urlparse(ws_url)
        self.host = u.hostname or "127.0.0.1"
        self.port = u.port or 8000
        self.path = u.path or "/api"
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None
        self.connected = False
        self._send_lock = asyncio.Lock()
        self._next_read = 1
        self._next_write = 1
        self._next_ping = 1
        self._reads: dict[int, asyncio.Queue] = {}
        self._writes: dict[int, asyncio.Future] = {}
        self._pings: dict[int, asyncio.Future] = {}
        self._rx_task: asyncio.Task | None = None

    async def start(self, timeout: float = 5.0):
        r, w = await asyncio.wait_for(asyncio.open_connection(self.host, self.port), timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (f"GET {self.path} HTTP/1.1\r\nHost: {self.host}:{self.port}\r\n"
               f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
        w.write(req.encode())
        await w.drain()
        hdr = await asyncio.wait_for(r.readuntil(b"\r\n\r\n"), timeout)
        if b"101" not in hdr.splitlines()[0]:
            w.close()
            raise ConnectionError(f"WebSocket upgrade rejected: {hdr.splitlines()[0]!r}")
        self.reader, self.writer = r, w
        await self._ws_send(normfs_pb2.ClientRequest(setup=normfs_pb2.SetupRequest(version=1)).encode())
        raw = await asyncio.wait_for(self._ws_recv(), timeout)
        resp = normfs_pb2.ServerResponseReader(raw)
        if not resp.get_setup() or resp.get_setup().get_version() != 1:
            w.close()
            raise ConnectionError("NormFS WebSocket setup handshake failed")
        self.connected = True
        self._rx_task = asyncio.create_task(self._rx_loop())
        return self

    async def _ws_send(self, payload: bytes, opcode: int = 0x02):
        assert self.writer is not None
        mask = os.urandom(4)
        n = len(payload)
        hdr = bytearray([0x80 | opcode])
        if n < 126:
            hdr.append(0x80 | n)
        elif n < 65536:
            hdr.append(0x80 | 126)
            hdr += struct.pack(">H", n)
        else:
            hdr.append(0x80 | 127)
            hdr += struct.pack(">Q", n)
        masked = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
        async with self._send_lock:
            self.writer.write(bytes(hdr) + mask + masked)
            await self.writer.drain()

    async def _ws_recv(self) -> bytes:
        assert self.reader is not None
        chunks = bytearray()
        while True:
            b1, b2 = await self.reader.readexactly(2)
            fin, op, is_masked, n = bool(b1 & 0x80), b1 & 0x0F, bool(b2 & 0x80), b2 & 0x7F
            if n == 126:
                n = struct.unpack(">H", await self.reader.readexactly(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", await self.reader.readexactly(8))[0]
            mk = await self.reader.readexactly(4) if is_masked else b""
            data = await self.reader.readexactly(n)
            if is_masked:
                data = bytes(b ^ mk[i & 3] for i, b in enumerate(data))
            if op == 0x08:
                raise ConnectionResetError("WebSocket closed by server")
            if op == 0x09:
                await self._ws_send(data, opcode=0x0A)
                continue
            if op == 0x0A:
                continue
            chunks.extend(data)
            if fin:
                return bytes(chunks)

    async def _rx_loop(self):
        try:
            while self.connected:
                raw = await self._ws_recv()
                resp = normfs_pb2.ServerResponseReader(raw)
                if resp._read_buf is not None:
                    rid = resp.get_read().get_read_id()
                    q = self._reads.get(rid)
                    if q is not None:
                        await q.put(resp)
                if resp._write_buf is not None:
                    wid = resp.get_write().get_write_id()
                    fut = self._writes.pop(wid, None)
                    if fut is not None and not fut.done():
                        fut.set_result(resp)
                if resp._ping_buf is not None:
                    preq = resp.get_ping().get_request()
                    seq = preq.get_sequence() if preq else 0
                    fut = self._pings.pop(seq, None) or (next(iter(self._pings.values()), None) if self._pings else None)
                    if fut is not None and not fut.done():
                        fut.set_result(resp.get_ping())
        except Exception:
            self.connected = False
            for q in list(self._reads.values()):
                q.put_nowait(None)
            for fut in (*self._writes.values(), *self._pings.values()):
                if not fut.done():
                    fut.set_exception(ConnectionResetError("Station WebSocket disconnected"))

    async def ping_mono_ns(self, timeout: float = 3.0) -> int:
        seq = self._next_ping
        self._next_ping += 1
        fut = asyncio.get_running_loop().create_future()
        self._pings[seq] = fut
        req = normfs_pb2.ClientRequest(
            ping=normfs_pb2.PingRequest(sequence=seq, client_timestamp_ns=int(time.time() * 1e9))
        )
        await self._ws_send(req.encode())
        pong = await asyncio.wait_for(fut, timeout)
        return int(pong.get_monotonic_stamp_ns())

    async def read_one(self, queue_id: str, raw_offset: bytes = b"\x01\x00\x00\x00\x00\x00\x00\x00",
                       offset_type=None, timeout: float = 5.0) -> bytes | None:
        if offset_type is None:
            offset_type = normfs_pb2.OffsetType.OT_SHIFT_FROM_TAIL
        rid = self._next_read
        self._next_read += 1
        q: asyncio.Queue = asyncio.Queue()
        self._reads[rid] = q
        try:
            req = normfs_pb2.ClientRequest(read=normfs_pb2.ReadRequest(
                read_id=rid, queue_id=queue_id,
                offset=normfs_pb2.Offset(id=normfs_pb2.Id(raw=raw_offset), type=offset_type),
                limit=1, step=1,
            ))
            await self._ws_send(req.encode())
            out = None
            while True:
                resp = await asyncio.wait_for(q.get(), timeout)
                if resp is None:
                    raise ConnectionResetError("Station WebSocket disconnected during read")
                rr = resp.get_read()
                res = rr.get_result()
                if res == normfs_pb2.ReadResponse_Result.RR_ENTRY:
                    out = bytes(rr.get_data())
                elif res in (
                    normfs_pb2.ReadResponse_Result.RR_END,
                    normfs_pb2.ReadResponse_Result.RR_NOT_FOUND,
                    normfs_pb2.ReadResponse_Result.RR_QUEUE_NOT_FOUND,
                ):
                    return out
                elif res == normfs_pb2.ReadResponse_Result.RR_SERVER_ERROR:
                    raise RuntimeError(f"Station read error on {queue_id}")
        finally:
            self._reads.pop(rid, None)

    def follow(self, queue_id: str, target: asyncio.Queue) -> asyncio.Queue:
        err_q: asyncio.Queue = asyncio.Queue(maxsize=1)

        async def _run():
            rid = self._next_read
            self._next_read += 1
            q: asyncio.Queue = asyncio.Queue()
            self._reads[rid] = q
            try:
                req = normfs_pb2.ClientRequest(read=normfs_pb2.ReadRequest(
                    read_id=rid, queue_id=queue_id,
                    offset=normfs_pb2.Offset(id=normfs_pb2.Id(raw=b"\x00"), type=normfs_pb2.OffsetType.OT_SHIFT_FROM_TAIL),
                    limit=0, step=1,
                ))
                await self._ws_send(req.encode())
                while self.connected:
                    resp = await q.get()
                    if resp is None:
                        break
                    rr = resp.get_read()
                    if rr.get_result() == normfs_pb2.ReadResponse_Result.RR_ENTRY:
                        await target.put(StreamEntry(
                            ID=StreamEntryId(ID=bytes(rr.get_id().get_raw())),
                            Data=memoryview(bytes(rr.get_data())),
                            DataSource=rr.get_data_source(),
                        ))
                    elif rr.get_result() not in (normfs_pb2.ReadResponse_Result.RR_START,):
                        break
            except Exception as e:
                if not err_q.full():
                    err_q.put_nowait(e)
            finally:
                self._reads.pop(rid, None)

        asyncio.create_task(_run())
        return err_q

    async def enqueue(self, queue_id: str, data: bytes, timeout: float = 10.0) -> bytes:
        wid = self._next_write
        self._next_write += 1
        fut = asyncio.get_running_loop().create_future()
        self._writes[wid] = fut
        req = normfs_pb2.ClientRequest(write=normfs_pb2.WriteRequest(
            write_id=wid, queue_id=queue_id, packets=[data]
        ))
        await self._ws_send(req.encode())
        resp = await asyncio.wait_for(fut, timeout)
        wr = resp.get_write()
        if wr.get_result() != normfs_pb2.WriteResponse_Result.WR_DONE:
            raise RuntimeError(f"Station write failed on {queue_id}: {wr.get_result()}")
        ids = wr.get_ids()
        return bytes(ids[0].get_raw()) if ids else b""


def _tcp_open(host: str, port: int, timeout: float = 0.6) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


async def connect():
    if STATION_WS_URL:
        return await WsStationClient(STATION_WS_URL).start()
    host, _, port_s = STATION_HOST.partition(":")
    tcp_port = int(port_s) if port_s else 8888
    if _tcp_open(host, tcp_port):
        return await new_station_client(STATION_HOST, _log)
    for ws_port in (8000, 8889):
        if _tcp_open(host, ws_port):
            return await WsStationClient(f"ws://{host}:{ws_port}/api").start()
    return await new_station_client(STATION_HOST, _log)


def _select_bus(st_reader):
    global BUS
    buses = st_reader.get_buses() or []
    if BUS and BUS != "auto":
        for bs in buses:
            if as_str(bs.get_bus().get_serial_number()) == BUS:
                return bs
        return None
    for bs in buses:
        motors = bs.get_motors() or []
        if len(motors) == MOTOR_COUNT:
            BUS = as_str(bs.get_bus().get_serial_number())
            return bs
    if len(buses) == 1:
        BUS = as_str(buses[0].get_bus().get_serial_number())
        return buses[0]
    return None


async def bus_frame(cli):
    """One raw st3215/inference frame for BUS: list of motor readers, or None if the bus is absent."""
    if hasattr(cli, "read_one"):
        raw = await cli.read_one(ST3215_QUEUE)
        if not raw:
            return None
        bs = _select_bus(st3215.InferenceStateReader(memoryview(raw)))
        return (bs.get_motors() or []) if bs is not None else None
    q: asyncio.Queue = asyncio.Queue()
    cli.follow(ST3215_QUEUE, q)
    e = await asyncio.wait_for(q.get(), 5)
    bs = _select_bus(st3215.InferenceStateReader(memoryview(bytes(e.Data))))
    return (bs.get_motors() or []) if bs is not None else None


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
        await asyncio.sleep(0.25)
    raise RuntimeError(f"motor bus {BUS}: no complete error-free read (bus desync? restart Station)")


_known_cam_queues: set[str] = set()


async def read_live_cameras(cli) -> list[dict]:
    """Read latest USB camera frames directly from usbvideo/<hash> NormFS queues (works even when torque is off)."""
    if not hasattr(cli, "read_one"):
        return []
    mono_ns = await cli.ping_mono_ns()
    if not _known_cam_queues:
        for shift in range(1, 8):
            inf_raw = await cli.read_one("inference-states", shift.to_bytes(8, "little"), normfs_pb2.OffsetType.OT_SHIFT_FROM_TAIL)
            if not inf_raw:
                continue
            irx = inference.InferenceRxReader(memoryview(inf_raw))
            for entry in irx.get_entries() or []:
                q_str = as_str(entry.get_queue())
                if "usbvideo/" in q_str:
                    _known_cam_queues.add(q_str.split("usbvideo/")[-1].strip("/"))
            if _known_cam_queues:
                break
    cams = []
    for qhash in sorted(_known_cam_queues):
        vraw = await cli.read_one(f"usbvideo/{qhash}")
        if not vraw:
            continue
        env = usbvideo.RxEnvelopeReader(memoryview(vraw))
        cam = env.get_camera()
        pack = env.get_frames()
        if pack is None:
            continue
        fmt = pack.get_format()
        w, h, kind = fmt.get_width(), fmt.get_height(), fmt.get_kind()
        frames = pack.get_frames_data() or []
        if kind != fr.FrameFormatKind.FF_JPEG or not frames:
            continue
        stamps = pack.get_stamps() or []
        stamp = stamps[-1] if stamps else env.get_stamp()
        f_mono = int(stamp.get_monotonic_stamp_ns()) if stamp else 0
        age = max(0.0, (mono_ns - f_mono) / 1e9) if (mono_ns and f_mono) else 0.0
        uid = as_str(cam.get_unique_id()) if cam else ""
        prod = as_str(cam.get_product()) if cam else ""
        cams.append({
            "hash": qhash,
            "label": prod or uid or qhash[:8],
            "unique_id": uid,
            "w": w,
            "h": h,
            "age": round(age, 3),
            "live": age < 5.0,
            "jpeg": bytes(frames[-1]),
        })
    return cams


_cmd_seq = 1


async def sync_write(cli, address: int, values: dict[int, int], width: int) -> None:
    global _cmd_seq
    if BUS == "auto":
        await read_state(cli)
    cid = _cmd_seq
    _cmd_seq += 1
    motors = [st3215.ST3215SyncWriteCommand_MotorWrite(motor_id=mid, value=int(v).to_bytes(width, "little"))
              for mid, v in values.items()]
    cmd = st3215.Command(target_bus_serial=BUS,
                         sync_write=st3215.ST3215SyncWriteCommand(address=address, motors=motors))
    drv = commands.DriverCommand(
        command_id=cid.to_bytes(4, "big"),
        type=drivers.StationCommandType.STC_ST3215_COMMAND,
        body=cmd.encode(),
    )
    if hasattr(cli, "read_one"):
        pack = commands.StationCommandsPack(commands=[drv])
        await cli.enqueue("commands", pack.encode())
    else:
        await send_commands(cli, [drv])
