"""Live, read-only capture from Station: motor state, camera JPEGs, motor commands, hot-plug events.

Only follows/reads queues. It never calls send_commands / enqueue: nothing here writes to the arm.
Station access (norma-core path setup, string helper) comes from arm_mcp/station.py.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
from collections import deque
from pathlib import Path

from arm_mcp import station as stn  # norma-core sys.path setup + helpers; we only use its read side
from station_py import Client  # noqa: E402  (importable once arm_mcp.station ran)
from target.gen_python.protobuf.drivers.st3215 import st3215  # noqa: E402
from target.gen_python.protobuf.drivers.usbvideo import frame as fr, usbvideo  # noqa: E402
from target.gen_python.protobuf.station import commands, drivers, inference  # noqa: E402

from .fmt import bus_entry, bus_role, cam_role, decode_motor, sim_compat
from .store import SessionWriter

_log = logging.getLogger("recorder.live")
STATE_QUEUE = stn.ST3215_QUEUE
COMMANDS_QUEUE = "commands"      # StationCommandsPack: everything clients send (send_commands)
TX_QUEUE = "st3215/tx"           # TxEnvelope: commands as the st3215 driver dispatched them
INDEX_QUEUE = "inference-states"  # InferenceRx: one entry per live queue, used for camera discovery
CAM_RE = re.compile(r"usbvideo/([0-9a-f]{32})")
STALL_S = 3.0
ST3215_KINDS = ("write", "reg_write", "sync_write", "action", "reset", "reset_calibration",
                "freeze_calibration", "auto_calibrate", "stop_auto_calibrate")


def station_logs() -> list[Path]:
    if os.environ.get("STATION_LOG"):
        return [Path(os.environ["STATION_LOG"])]
    run = stn.NORMA_CORE.parent / "station-run"
    return [run / "run-station.out", run / "station.log"]


def cams_from_logs() -> set[str]:
    found = set()
    for p in station_logs():
        try:
            found |= set(CAM_RE.findall(p.read_text(errors="replace")))
        except OSError:
            pass
    return found


def decode_st3215(r) -> dict:
    """Readable form of an st3215.Command / TxEnvelope reader."""
    out = {"bus": r.get_target_bus_serial(), "kind": None}
    for kind in ST3215_KINDS:
        if getattr(r, f"_{kind}_buf", None) is None:
            continue
        out["kind"] = kind
        sub = getattr(r, f"get_{kind}")()
        if kind in ("write", "reg_write"):
            v = bytes(sub.get_value())
            out.update(motor_id=sub.get_motor_id(), address=sub.get_address(), value=int.from_bytes(v, "little"),
                       value_hex=v.hex())
        elif kind == "sync_write":
            out.update(address=sub.get_address(),
                       motors={int(m.get_motor_id()): int.from_bytes(bytes(m.get_value()), "little")
                               for m in sub.get_motors() or []})
        elif kind in ("action", "reset"):
            out["motor_id"] = sub.get_motor_id()
        break
    return out


def decode_commands_pack(data: bytes) -> list[dict]:
    pack = commands.StationCommandsPackReader(memoryview(data))
    tags = [str(t) for t in pack.get_tags() or []]
    out = []
    for dc in pack.get_commands() or []:
        typ = int(dc.get_type())
        known = typ in drivers.StationCommandType._value2member_map_
        rec = {"src": COMMANDS_QUEUE, "type": drivers.StationCommandType(typ).name if known else str(typ), "tags": tags}
        body = bytes(dc.get_body())
        if typ == drivers.StationCommandType.STC_ST3215_COMMAND:
            rec.update(decode_st3215(st3215.CommandReader(memoryview(body))))
        else:
            rec["body_hex"] = body.hex()
        out.append(rec)
    return out


def decode_tx(data: bytes) -> dict:
    r = st3215.TxEnvelopeReader(memoryview(data))
    return {"src": TX_QUEUE, "station_ns": r.get_local_stamp_ns(), "station_mono_ns": r.get_monotonic_stamp_ns(),
            "command_id": bytes(r.get_command_id()).hex(), **decode_st3215(r)}


class LiveCapture:
    """Runs its own asyncio loop in a daemon thread; everything public is safe to call from other threads."""

    def __init__(self, host: str | None = None, state_hz: float = 50.0):
        self.host = host or stn.STATION_HOST
        self.state_hz = state_hz
        self.lock = threading.Lock()
        self.writer: SessionWriter | None = None
        self.t0 = time.monotonic()
        self.connected = False
        self.motors: dict[str, dict[int, dict]] = {}
        self.bus_stamps: dict[str, tuple[int, int]] = {}
        self.bus_changed: dict[str, float] = {}
        self.buses_seen: set[str] = set()
        self.stale_buses: set[str] = set()
        self.latest_row: dict | None = None
        self.last_state_mono: float | None = None
        self.state_frames = 0
        self._since_emit = 0
        self._last_emit = 0.0
        self.cams: dict[str, dict] = {}
        self.events: deque = deque(maxlen=200)
        self.commands: deque = deque(maxlen=200)
        self.errors: dict[str, str] = {}
        self._followed: set[str] = set()

    # --- public ------------------------------------------------------------
    def start(self) -> "LiveCapture":
        threading.Thread(target=lambda: asyncio.run(self._main()), daemon=True, name="live-capture").start()
        return self

    def attach(self, writer: SessionWriter):
        with self.lock:
            self.writer = writer
            buses = {s: {"role": bus_role(s), "motor_ids": sorted(m)} for s, m in self.motors.items()}
            cams = {h[:8]: {"hash": h, "role": c["role"], "live": not c["stalled"]} for h, c in self.cams.items()}
        writer.event("start", name=writer.meta["name"], note=writer.meta["note"], buses=buses, cameras=cams,
                     station_connected=self.connected, state_hz=self.state_hz)

    def detach(self) -> SessionWriter | None:
        with self.lock:
            w, self.writer = self.writer, None
        return w

    def snapshot(self) -> dict:
        now = time.monotonic()
        with self.lock:
            return {"station": self.host, "connected": self.connected,
                    "state_age": None if self.last_state_mono is None else round(now - self.last_state_mono, 3),
                    "state_frames": self.state_frames,
                    "buses": {s: {"role": bus_role(s), "motor_ids": sorted(m), "stale": s in self.stale_buses}
                              for s, m in self.motors.items()},
                    "cameras": [self._cam_info(h, c, now) for h, c in sorted(self.cams.items())],
                    "errors": dict(self.errors)}

    def latest(self) -> dict | None:
        with self.lock:
            return self.latest_row

    def latest_jpeg(self, cam: str) -> bytes | None:
        with self.lock:
            for h, c in self.cams.items():
                if cam and (h.startswith(cam) or c["role"] == cam):
                    return c["jpeg"]
            return None

    # --- internals ---------------------------------------------------------
    @staticmethod
    def _cam_info(h, c, now):
        times = c["times"]
        fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else 0.0
        return {"hash": h, "hash8": h[:8], "role": c["role"], "fps": round(fps, 1), "frames": c["frames"],
                "age": None if c["mono"] is None else round(now - c["mono"], 3), "live": not c["stalled"],
                "w": c["w"], "h": c["h"]}

    def _event(self, type_: str, **data):
        _log.info("event %s %s", type_, data)
        with self.lock:
            w = self.writer
            self.events.append({"wall": round(time.time(), 3), "type": type_, **data})
        if w:
            w.event(type_, **data)

    def _error(self, where: str, ex: BaseException):
        msg = f"{type(ex).__name__}: {ex}"
        if self.errors.get(where) != msg:
            _log.warning("%s: %s", where, msg)
            self.errors[where] = msg

    async def _main(self):
        logging.getLogger("recorder.station").setLevel(logging.CRITICAL)
        # The client reconnects by itself every 5 s; one instance for the whole process.
        self.cli = Client(self.host if ":" in self.host else f"{self.host}:8888", logging.getLogger("recorder.station"))
        while not self.cli.setup_done:
            self.errors["connect"] = f"Station {self.host} not reachable yet"
            await asyncio.sleep(0.2)
        self.connected = True
        self.errors.pop("connect", None)
        self._event("station_connected", host=self.host)
        self._follow_all()
        while True:
            await asyncio.sleep(1.0)
            self.connected = bool(self.cli.connected and self.cli.setup_done)
            try:
                await self._discover()
            except Exception as ex:
                self._error("discover", ex)
            self._watchdog()

    def _cam_handler(self, h: str):
        return lambda e, mono, wall: self._on_cam(h, e, mono, wall)

    def _follow_all(self):
        self._followed.clear()
        self._follow(STATE_QUEUE, self._on_state)
        self._follow(COMMANDS_QUEUE, self._on_commands)
        self._follow(TX_QUEUE, self._on_tx)
        for h in list(self.cams):
            self._follow(f"usbvideo/{h}", self._cam_handler(h))

    def _follow(self, queue: str, handler):
        if queue in self._followed:
            return
        self._followed.add(queue)
        asyncio.get_running_loop().create_task(self._consume(queue, handler))

    async def _consume(self, queue: str, handler):
        q: asyncio.Queue = asyncio.Queue()
        err_q = self.cli.follow(queue, q)
        err = asyncio.ensure_future(err_q.get())
        while True:
            get = asyncio.ensure_future(q.get())
            done, _ = await asyncio.wait({get, err}, return_when=asyncio.FIRST_COMPLETED)
            if err in done:
                get.cancel()
                self._error(queue, err.result() if isinstance(err.result(), BaseException) else RuntimeError(str(err.result())))
                self._followed.discard(queue)
                return
            try:
                handler(get.result(), time.monotonic(), time.time())
            except Exception as ex:
                self._error(queue, ex)

    async def _discover(self):
        found: set[str] = set()
        try:
            qr = self.cli.read_from_tail(INDEX_QUEUE, offset=b"\x00", limit=1, step=1, buf_size=1)
            e = await asyncio.wait_for(qr.data.get(), 2)
            if e is not None:
                rx = inference.InferenceRxReader(memoryview(bytes(e.Data)))
                found = {m.group(1) for en in rx.get_entries() or [] if (m := CAM_RE.search(str(en.get_queue())))}
        except (asyncio.TimeoutError, TimeoutError):
            pass
        if not found and not self.cams:
            found = cams_from_logs()
        for h in found - set(self.cams):
            with self.lock:
                self.cams[h] = {"jpeg": None, "mono": None, "wall": None, "info": None, "times": deque(maxlen=60),
                                "stalled": True, "role": cam_role(h), "frames": 0, "w": None, "h": None}
            self._event("camera_discovered", cam=h[:8], hash=h, role=cam_role(h))
            self._follow(f"usbvideo/{h}", self._cam_handler(h))

    def _watchdog(self):
        now = time.monotonic()
        for h, c in list(self.cams.items()):
            if not c["stalled"] and (c["mono"] is None or now - c["mono"] > STALL_S):
                c["stalled"] = True
                self._event("camera_lost", cam=h[:8], role=c["role"],
                            last_frame_age=None if c["mono"] is None else round(now - c["mono"], 2))
        for s, changed in list(self.bus_changed.items()):
            if now - changed > STALL_S and s not in self.stale_buses and s in self.buses_seen:
                self.stale_buses.add(s)
                self._event("bus_stale", bus=s, role=bus_role(s))
        if self.last_state_mono is not None and now - self.last_state_mono > 2 * STALL_S:
            # Station restarted or the TCP link dropped: the old follows are dead, open new ones.
            self._event("state_stalled", age=round(now - self.last_state_mono, 1))
            self.last_state_mono = now
            self._follow_all()

    def _on_state(self, e, mono: float, wall: float):
        st = st3215.InferenceStateReader(memoryview(bytes(e.Data)))
        seen = set()
        for bs in st.get_buses() or []:
            info = bs.get_bus()
            serial = stn.as_str(info.get_serial_number()) if info else "?"
            seen.add(serial)
            motors = dict(self.motors.get(serial, {}))
            ids = set()
            for m in bs.get_motors() or []:
                mid, sns = int(m.get_id()), int(m.get_system_stamp_ns())
                lo, hi, frozen = int(m.get_range_min()), int(m.get_range_max()), bool(m.get_range_freezed())
                ids.add(mid)
                prev = motors.get(mid)
                if prev and prev["stamp_ns"] == sns and (prev["range_min"], prev["range_max"], prev["frozen"]) == (lo, hi, frozen):
                    continue
                err = m.get_error()
                text = stn.as_str(err.get_description()) if err else ""
                servo = [int(x) for x in (err.get_servo() or [])] if err else []
                if servo:
                    text = (text + f" servo={servo}").strip()
                motors[mid] = decode_motor(mid, bytes(m.get_state() or b""), lo, hi, frozen, text or None, sns)
            for gone in set(motors) - ids:
                del motors[gone]
            stamp = (int(bs.get_system_stamp_ns()), int(bs.get_monotonic_stamp_ns()))
            if self.bus_stamps.get(serial) != stamp:
                self.bus_changed[serial] = mono
                if serial in self.stale_buses:
                    self.stale_buses.discard(serial)
                    self._event("bus_resumed", bus=serial, role=bus_role(serial))
            self.bus_stamps[serial] = stamp
            with self.lock:
                self.motors[serial] = motors
        for s in seen - self.buses_seen:
            self._event("bus_connected", bus=s, role=bus_role(s), motor_ids=sorted(self.motors.get(s, {})))
        for s in self.buses_seen - seen:
            self._event("bus_disconnected", bus=s, role=bus_role(s))
            with self.lock:
                self.motors.pop(s, None)
            self.bus_changed.pop(s, None)
            self.stale_buses.discard(s)
        self.buses_seen = seen
        self.state_frames += 1
        self._since_emit += 1
        self.last_state_mono = mono
        if self.state_hz > 0 and mono - self._last_emit < 1.0 / self.state_hz:
            return
        self._last_emit = mono
        buses = {s: bus_entry(s, m) for s, m in self.motors.items()}
        stamps = [self.bus_stamps[s] for s in buses if s in self.bus_stamps]
        row = {"wall": round(wall, 4), "station_ns": max((s[0] for s in stamps), default=None),
               "station_mono_ns": max((s[1] for s in stamps), default=None), "buses": buses, **sim_compat(buses)}
        n, self._since_emit = self._since_emit, 0
        with self.lock:
            self.latest_row = {"t": round(mono - self.t0, 4), **row}
            w = self.writer
        if w:
            w.state(row, mono, n)

    def _on_cam(self, h: str, e, mono: float, wall: float):
        env = usbvideo.RxEnvelopeReader(memoryview(bytes(e.Data)))
        typ = env.get_type()
        if typ != usbvideo.RxEnvelopeType.ET_FRAMES:
            name = usbvideo.RxEnvelopeType(typ).name.lower().removeprefix("et_")
            self._event(f"camera_{name}", cam=h[:8], role=cam_role(h), error=str(env.get_error() or "") or None)
            return
        pack, c = env.get_frames(), self.cams.get(h)
        if pack is None or c is None:
            return
        fmt = pack.get_format()
        frames, stamps = pack.get_frames_data() or [], pack.get_stamps() or []
        if fmt is None or fmt.get_kind() != fr.FrameFormatKind.FF_JPEG:
            self._error(f"usbvideo/{h}", RuntimeError(f"non-JPEG frame kind {fmt and fmt.get_kind()} not recorded"))
            return
        for i, data in enumerate(frames):
            s = stamps[i] if i < len(stamps) else None
            info = {"station_ns": s.get_local_stamp_ns() if s else None,
                    "station_mono_ns": s.get_monotonic_stamp_ns() if s else None,
                    "index": s.get_index() if s else None, "w": fmt.get_width(), "h": fmt.get_height()}
            jpeg = bytes(data)
            with self.lock:
                c.update(jpeg=jpeg, mono=mono, wall=wall, info=info, w=info["w"], h=info["h"])
                c["frames"] += 1
                c["times"].append(mono)
                w = self.writer
            if w:
                w.frame(h, jpeg, mono, wall, info)
        if c["stalled"]:
            c["stalled"] = False
            self._event("camera_live", cam=h[:8], role=c["role"], w=c["w"], h=c["h"])

    def _record_command(self, rec: dict, mono: float, wall: float):
        rec = {"wall": round(wall, 4), **rec}
        with self.lock:
            self.commands.append(rec)
            w = self.writer
        if w:
            w.command(rec, mono)

    def _on_commands(self, e, mono: float, wall: float):
        for rec in decode_commands_pack(bytes(e.Data)):
            self._record_command(rec, mono, wall)

    def _on_tx(self, e, mono: float, wall: float):
        self._record_command(decode_tx(bytes(e.Data)), mono, wall)
