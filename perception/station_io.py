"""Read-only Station access for perception: camera frames and follower joint ticks.

Everything goes through NormaCore Station queues (never the camera or serial device directly):
  usbvideo/<hash>      -> JPEG frames (299x224 with resize_target 224)
  st3215/inference     -> motor register dumps (raw ticks, calibrated range)
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_CANDIDATES = [os.environ.get("NORMA_CORE_REPO"), ROOT / "norma-core", ROOT.parent / "norma-core",
               ROOT.parent.parent / "repo/norma-core"]
NORMA_CORE = next((Path(p) for p in _CANDIDATES if p and (Path(p) / "software/station/shared/station_py").exists()), None)

QUEUES = {
    "overview": os.environ.get("PERCEPTION_OVERVIEW_QUEUE", "usbvideo/9ea4cacd5be94751e1eea7be90bcf628"),
    "wrist": os.environ.get("PERCEPTION_WRIST_QUEUE", "usbvideo/28ff0ca87978e8f993c2ac64a84d188a"),
}
STATION_HOST = os.environ.get("STATION_HOST", "localhost")
_log = logging.getLogger("perception")


def _import_station():
    if NORMA_CORE is None:
        raise RuntimeError("norma-core checkout not found: set NORMA_CORE_REPO")
    for p in (NORMA_CORE, NORMA_CORE / "software/station/shared"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from station_py import new_station_client
    from target.gen_python.protobuf.drivers.usbvideo import frame as fr, usbvideo
    return new_station_client, usbvideo, fr


async def connect():
    new_station_client, _, _ = _import_station()
    return await new_station_client(STATION_HOST, _log)


def decode_jpeg(data: bytes) -> np.ndarray:
    import cv2
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


async def grab_jpegs(cli, cam: str, n: int = 1, timeout: float = 5.0) -> list[bytes]:
    """The next n JPEG frames from a camera queue ("overview" | "wrist" | full queue name)."""
    _, usbvideo, fr = _import_station()
    q: asyncio.Queue = asyncio.Queue()
    cli.follow(QUEUES.get(cam, cam), q)
    out: list[bytes] = []
    while len(out) < n:
        e = await asyncio.wait_for(q.get(), timeout)
        env = usbvideo.RxEnvelopeReader(memoryview(bytes(e.Data)))
        pack = env.get_frames()
        if env.get_type() != usbvideo.RxEnvelopeType.ET_FRAMES or pack is None:
            continue
        frames = pack.get_frames_data() or []
        if pack.get_format().get_kind() == fr.FrameFormatKind.FF_JPEG and frames:
            out.append(bytes(frames[-1]))
    return out


async def grab_frames(cli, cam: str, n: int = 1) -> list[np.ndarray]:
    return [decode_jpeg(b) for b in await grab_jpegs(cli, cam, n)]


async def read_joints(cli) -> dict:
    """{"j1": {"raw", "lo", "hi"}, ...} for the follower bus (read-only, via arm_mcp.station)."""
    if str(ROOT / "arm_mcp") not in sys.path:
        sys.path.insert(0, str(ROOT / "arm_mcp"))
    if NORMA_CORE and "NORMA_CORE_REPO" not in os.environ:
        os.environ["NORMA_CORE_REPO"] = str(NORMA_CORE)
    import station as stn
    st = await stn.read_state(cli)
    return {n: {"raw": st[i]["raw"], "lo": st[i]["lo"], "hi": st[i]["hi"]} for n, i in zip(stn.NAMES, stn.IDS)}


def grab(cam: str, n: int = 1) -> list[np.ndarray]:
    """Sync helper: n BGR frames from a camera."""
    async def go():
        return await grab_frames(await connect(), cam, n)
    return asyncio.run(go())


if __name__ == "__main__":
    import cv2
    out = ROOT / "frames"
    out.mkdir(exist_ok=True)
    for cam in QUEUES:
        img = grab(cam)[0]
        p = out / f"perception_{cam}.jpg"
        cv2.imwrite(str(p), img)
        print(cam, img.shape, "->", p)
