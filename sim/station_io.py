"""Read-only NormaCore Station access for the twin: camera JPEGs and follower raw joint ticks.

Everything goes through Station queues (never the camera or serial device directly):
  usbvideo/<hash>    -> JPEG frames (299x224 with Station's resize_target 224)
  st3215/inference   -> ST3215 register dumps (present position at 0x38, calibrated range)
Nothing here writes to Station. One subscription per (client, queue) for the whole run (perception.latest.Latest):
station_py has no unfollow, so following per call would pile up streams on the connection.

  python3 sim/station_io.py [outdir]     # save one frame per camera + print raw ticks
"""
from __future__ import annotations

import asyncio
import logging
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.append(str(HERE.parent))
from perception.latest import Latest  # noqa: E402

_CANDIDATES = [os.environ.get("NORMA_CORE_REPO"), HERE.parent / "norma-core", HERE.parent.parent / "norma-core",
               HERE.parents[2] / "repo/norma-core" if len(HERE.parents) > 2 else None]
NORMA_CORE = next((Path(p) for p in _CANDIDATES if p and (Path(p) / "software/station/shared/station_py").exists()), None)

QUEUES = {
    "overview": os.environ.get("TWIN_OVERVIEW_QUEUE", "usbvideo/9ea4cacd5be94751e1eea7be90bcf628"),
    "wrist": os.environ.get("TWIN_WRIST_QUEUE", "usbvideo/28ff0ca87978e8f993c2ac64a84d188a"),
}
ST3215_QUEUE = "st3215/inference"
BUS = os.environ.get("ROBOT_BUS_SERIAL", "5B61033301")
STATION_HOST = os.environ.get("STATION_HOST", "localhost")
_log = logging.getLogger("twin.station")


def _mods():
    if NORMA_CORE is None:
        raise RuntimeError("norma-core checkout not found: set NORMA_CORE_REPO")
    for p in (NORMA_CORE, NORMA_CORE / "software/station/shared"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from station_py import new_station_client
    from target.gen_python.protobuf.drivers.st3215 import st3215
    from target.gen_python.protobuf.drivers.usbvideo import frame as fr, usbvideo
    return new_station_client, usbvideo, fr, st3215


def _str(v) -> str:
    return bytes(v).decode("utf-8", "replace") if isinstance(v, (bytes, bytearray, memoryview)) else str(v)


def _pos(raw: int) -> int:
    return (4096 - (raw & 4095)) & 4095 if raw & 0x8000 else raw & 4095


async def connect():
    return await _mods()[0](STATION_HOST, _log)


def jpeg_of(e) -> bytes | None:
    """The newest JPEG in a usbvideo queue entry, None if it holds no JPEG frame."""
    _, usbvideo, fr, _ = _mods()
    env = usbvideo.RxEnvelopeReader(memoryview(bytes(e.Data)))
    pack = env.get_frames()
    if env.get_type() != usbvideo.RxEnvelopeType.ET_FRAMES or pack is None:
        return None
    frames = pack.get_frames_data() or []
    return bytes(frames[-1]) if pack.get_format().get_kind() == fr.FrameFormatKind.FF_JPEG and frames else None


async def grab_jpegs(cli, cam: str, n: int = 1, timeout: float = 5.0) -> list[bytes]:
    """The next n JPEG frames of a camera ("overview" | "wrist" | full queue name), each newer than the call."""
    sub = Latest.of(cli, QUEUES.get(cam, cam))
    out: list[bytes] = []
    while len(out) < n:
        j = jpeg_of(await sub.get(fresh=True, timeout=timeout))
        if j is not None:
            out.append(j)
    return out


class JointFollower:
    """Follows st3215/inference once per client; read() returns the newest {motor_id: raw_tick} of BUS (waiting for
    a new entry if the newest one was already returned) plus ranges and timing. A frame may lack motors (partial
    bus read): callers that need all of them check."""

    def __init__(self, cli):
        self.sub = Latest.of(cli, ST3215_QUEUE)
        self.seen = -1
        self.st3215 = _mods()[3]

    async def read(self, timeout: float = 5.0) -> tuple[dict, dict]:
        e = await self.sub.get(fresh=self.sub.seq == self.seen, timeout=timeout)
        self.seen = self.sub.seq
        t_recv = time.time()
        st = self.st3215.InferenceStateReader(memoryview(bytes(e.Data)))
        for bs in st.get_buses() or []:
            if _str(bs.get_bus().get_serial_number()) != BUS:
                continue
            raw, rng = {}, {}
            for m in bs.get_motors() or []:
                s = bytes(m.get_state() or b"")
                if len(s) >= 0x3A:
                    raw[int(m.get_id())] = _pos(struct.unpack_from("<H", s, 0x38)[0])
                    rng[int(m.get_id())] = (int(m.get_range_min()), int(m.get_range_max()))
            stamp = getattr(bs, "get_system_stamp_ns", lambda: 0)() or 0      # bus read time (wall clock, ns)
            return raw, {"ranges": rng, "t_recv": t_recv, "stamp_s": stamp / 1e9 if stamp > 1e17 else None}
        raise RuntimeError(f"bus {BUS} not in st3215/inference")


async def snapshot(cams=("overview", "wrist")) -> dict:
    """{'overview': BGR, 'wrist': BGR, 'raw': {id: tick}, 'ranges': {...}} in one Station connection."""
    import cv2
    cli = await connect()
    jf = JointFollower(cli)
    jpegs = await asyncio.gather(*(grab_jpegs(cli, c) for c in cams))
    raw, meta = await jf.read()
    out = {c: cv2.imdecode(np.frombuffer(j[0], np.uint8), cv2.IMREAD_COLOR) for c, j in zip(cams, jpegs)}
    out.update(raw=raw, ranges=meta["ranges"])
    return out


if __name__ == "__main__":
    import cv2
    logging.basicConfig(level=logging.ERROR)
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "real_compare" / "latest_frames"
    outdir.mkdir(parents=True, exist_ok=True)
    snap = asyncio.run(snapshot())
    for c in ("overview", "wrist"):
        cv2.imwrite(str(outdir / f"{c}.jpg"), snap[c])
        print(c, snap[c].shape, "->", outdir / f"{c}.jpg")
    print("raw ticks:", snap["raw"])
    print("ranges:", snap["ranges"])
