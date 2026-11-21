"""Grab the latest frame from every Station camera queue -> frames/<queue>.jpg|png
usage: python3 scripts/camera.py [queue_hash ...]   (default: all queues seen in station.log)"""
import os
import asyncio, logging, re, struct, sys, zlib
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
ROOT = BASE / "norma-core"
sys.path.insert(0, str(ROOT / "software/station/shared")); sys.path.insert(0, str(ROOT))
from station_py import new_station_client
from target.gen_python.protobuf.drivers.usbvideo import usbvideo, frame as fr

def png(w, h, rgb):
    def ch(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    raw = b"".join(b"\x00" + rgb[y * w * 3:(y + 1) * w * 3] for y in range(h))
    return b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + ch(b"IDAT", zlib.compress(raw)) + ch(b"IEND", b"")

def queues():
    log = (BASE / "station-run/station.log").read_text(errors="replace")
    return sorted(set(re.findall(r"usbvideo/([0-9a-f]{32})", log)))

async def grab(client, q, out):
    ch = asyncio.Queue(); client.follow(f"usbvideo/{q}", ch)
    for _ in range(200):
        try: e = await asyncio.wait_for(ch.get(), 3)
        except asyncio.TimeoutError: print(f"{q}: no frames (camera offline?)"); return
        env = usbvideo.RxEnvelopeReader(memoryview(bytes(e.Data)))
        pack = env.get_frames()
        if env.get_type() != usbvideo.RxEnvelopeType.ET_FRAMES or pack is None: continue
        fmt = pack.get_format(); w, h, kind = fmt.get_width(), fmt.get_height(), fmt.get_kind()
        frames = pack.get_frames_data() or []
        if kind == fr.FrameFormatKind.FF_JPEG and frames:
            p = out / f"{q[:8]}.jpg"; p.write_bytes(bytes(frames[-1]))
        else:
            data = bytes(frames[-1]) if frames else bytes(pack.get_linear_data() or b"")
            n = w * h
            if len(data) < 3 * n: print(f"{q}: unexpected frame size {len(data)} for {w}x{h} kind={kind}"); return
            data = data[-3 * n:]
            if len(data) == 3 * n and kind == fr.FrameFormatKind.FF_NCHW:   # planar CHW -> interleaved
                rgb = bytearray(3 * n); rgb[0::3] = data[:n]; rgb[1::3] = data[n:2 * n]; rgb[2::3] = data[2 * n:]
                data = bytes(rgb)
            p = out / f"{q[:8]}.png"; p.write_bytes(png(w, h, data))
        print(f"{q}: {w}x{h} kind={kind} frames_in_pack={len(frames)} -> {p}")
        return

async def main(qs):
    logging.basicConfig(level=logging.ERROR)
    client = await new_station_client(os.environ.get("STATION_HOST", "localhost"), logging.getLogger("camera"))
    await asyncio.gather(*(grab(client, q, BASE / "frames") for q in qs))

asyncio.run(main(sys.argv[1:] or queues()))
