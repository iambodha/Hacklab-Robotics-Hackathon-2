#!/usr/bin/env python3
"""Live web view of every Station usb-video camera.

Relays frames out of the running Station's usbvideo/<hash> queues into a small
MJPEG web server, so the cameras can be watched in a browser without opening
the video devices a second time (which would break the Station driver).

usage:
    python3 scripts/cam_stream.py [camera ...] [options]

    camera: usbvideo queue hash, usbvideo/<hash>, or camera unique id
            (e.g. 0x11200002c7f4a60). With no cameras given, every queue
            found under --data-dir is followed, and new ones are picked up
            automatically.

Endpoints (default http://127.0.0.1:8890):
    /                 live grid of all cameras
    /stream/<hash>    MJPEG stream for one camera
    /snapshot/<hash>  latest frame as a single image
    /api/cameras      JSON list of cameras (label, size, fps, staleness)

Options:
    --port N      listen port (default 8890)
    --host ADDR   bind address (default 127.0.0.1; use 0.0.0.0 for LAN)
    --station H   Station TCP host (default localhost:8888)
    --data-dir D  folder searched for Station normfs usbvideo/<hash> queues and
                  station.log (default station-run)
    --list        print discovered camera queues and exit
"""
import argparse
import asyncio
import hashlib
import json
import logging
import re
import struct
import sys
import threading
import time
import zlib
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import os

BASE = Path(__file__).resolve().parent.parent
NORMA = Path(os.environ.get("NORMA_CORE_REPO", BASE / "norma-core"))
for p in (NORMA / "software/station/shared", NORMA):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from station_py import new_station_client  # noqa: E402
from target.gen_python.protobuf.drivers.usbvideo import usbvideo, frame as fr  # noqa: E402

QUEUE_RE = re.compile(r"^[0-9a-f]{32}$")
CAMS = {}  # hash -> Camera


def png_encode(w, h, rgb):
    def ch(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))

    raw = b"".join(b"\x00" + rgb[y * w * 3 : (y + 1) * w * 3] for y in range(h))
    return (
        b"\x89PNG\r\n\x1a\n"
        + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + ch(b"IDAT", zlib.compress(raw))
        + ch(b"IEND", b"")
    )


PLACEHOLDER = png_encode(320, 240, b"\x18" * 320 * 240 * 3)


class Camera:
    def __init__(self, qhash):
        self.hash = qhash
        self.cond = threading.Condition()
        self.frame = PLACEHOLDER
        self.mime = "image/png"
        self.seq = 1  # placeholder is pushed first so <img> never looks broken
        self.label = qhash[:8]
        self.unique_id = ""
        self.width = self.height = 0
        self.last_frame_at = 0.0
        self.times = deque(maxlen=90)

    def fps(self):
        if len(self.times) < 2:
            return 0.0
        span = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / span if span > 0 else 0.0

    def age(self):
        return (time.time() - self.last_frame_at) if self.last_frame_at else None

    def push(self, data, mime, w, h):
        with self.cond:
            self.frame, self.mime = data, mime
            self.seq += 1
            self.width, self.height = w, h
            self.last_frame_at = time.time()
            self.times.append(self.last_frame_at)
            self.cond.notify_all()


def normalize_queue(spec):
    """Accept a queue hash, 'usbvideo/<hash>', or a camera unique id."""
    s = spec.strip()
    if s.lower().startswith("usbvideo/"):
        s = s[9:]
    if QUEUE_RE.match(s.lower()):
        return s.lower()
    return hashlib.md5(spec.encode()).hexdigest()


def scan_queues(data_dir):
    found = set()
    for pattern in ("*/usbvideo/*", "*/*/usbvideo/*"):
        for d in data_dir.glob(pattern):
            if d.is_dir() and QUEUE_RE.match(d.name):
                found.add(d.name)
    log = data_dir / "station.log"
    if log.exists():
        found |= set(re.findall(r"usbvideo/([0-9a-f]{32})", log.read_text(errors="replace")))
    return found


def pack_to_image(pack):
    """Return (image_bytes, mime, w, h) from a FramesPack, or None."""
    fmt = pack.get_format()
    w, h, kind = fmt.get_width(), fmt.get_height(), fmt.get_kind()
    frames = pack.get_frames_data() or []
    if kind == fr.FrameFormatKind.FF_JPEG and frames:
        return bytes(frames[-1]), "image/jpeg", w, h
    data = bytes(frames[-1]) if frames else bytes(pack.get_linear_data() or b"")
    n = w * h
    if len(data) < 3 * n or w <= 0 or h <= 0:
        return None
    data = data[-3 * n :]
    if kind == fr.FrameFormatKind.FF_NCHW:  # planar CHW -> interleaved RGB
        rgb = bytearray(3 * n)
        rgb[0::3], rgb[1::3], rgb[2::3] = data[:n], data[n : 2 * n], data[2 * n :]
        data = bytes(rgb)
    return png_encode(w, h, data), "image/png", w, h


async def follow_camera(cam, station, log):
    while True:
        try:
            client = await new_station_client(station, log)
            ch = asyncio.Queue()
            client.follow(f"usbvideo/{cam.hash}", ch)
            while True:
                e = await asyncio.wait_for(ch.get(), 15)
                env = usbvideo.RxEnvelopeReader(memoryview(bytes(e.Data)))
                if env.get_type() != usbvideo.RxEnvelopeType.ET_FRAMES:
                    continue
                meta = env.get_camera()
                if meta is not None:
                    uid = meta.get_unique_id()
                    product = meta.get_product()
                    if uid:
                        cam.unique_id = uid
                    if product or uid:
                        cam.label = product or uid
                pack = env.get_frames()
                if pack is None:
                    continue
                img = pack_to_image(pack)
                if img:
                    cam.push(*img)
        except Exception as ex:  # Station down or camera unplugged: retry
            log.warning("usbvideo/%s: stream lost (%s), retrying", cam.hash[:8], ex)
            await asyncio.sleep(2)


async def discovery(data_dir, station, extra, log):
    known = set()
    while True:
        for h in sorted((scan_queues(data_dir) | set(extra)) - known):
            if h in CAMS:
                known.add(h)
                continue
            cam = Camera(h)
            CAMS[h] = cam
            known.add(h)
            asyncio.create_task(follow_camera(cam, station, log))
            log.info("following usbvideo/%s", h)
        await asyncio.sleep(3)


async def relay(data_dir, station, extra, log):
    await asyncio.gather(discovery(data_dir, station, extra, log))


PAGE = br"""<!doctype html><meta charset=utf-8><title>ElRobot cameras</title>
<style>
body{margin:0;background:#111;color:#ccc;font:14px system-ui;padding:16px}
h1{font-size:16px;font-weight:600;margin:0 0 12px}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}
.card{background:#1c1c1c;border:1px solid #333;border-radius:8px;overflow:hidden}
.card img{display:block;width:100%;background:#000;aspect-ratio:4/3;object-fit:contain}
.meta{margin:0;padding:8px 10px;font-size:12px;color:#9aa;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.stale img{opacity:.35}
.empty{color:#777;padding:24px 0}
a{color:#7ab}</style>
<h1>ElRobot cameras &middot; <a href="/api/cameras">api</a></h1>
<div id=grid></div>
<script>
const grid = document.getElementById('grid');
const cards = new Map();
async function tick(){
  let list;
  try { list = await (await fetch('/api/cameras')).json(); }
  catch(e){ return; }
  const seen = new Set();
  for (const c of list){
    seen.add(c.hash);
    let card = cards.get(c.hash);
    if (!card){
      card = document.createElement('div');
      card.className = 'card';
      card.innerHTML = `<a href="/stream/${c.hash}" target="_blank"><img src="/stream/${c.hash}"></a><p class="meta"></p>`;
      grid.appendChild(card);
      cards.set(c.hash, card);
    }
    const age = c.age == null ? 'no frames yet' : c.age.toFixed(1) + 's ago';
    card.querySelector('.meta').textContent =
      c.label + ' \u00b7 ' + c.hash.slice(0, 6) + ' \u00b7 ' + (c.w || '?') + '\u00d7' + (c.h || '?') + ' \u00b7 ' + c.fps.toFixed(1) + ' fps \u00b7 ' + (c.live ? 'live' : age);
    card.classList.toggle('stale', !c.live);
  }
  for (const [h, card] of cards) if (!seen.has(h)){ card.remove(); cards.delete(h); }
  let empty = grid.querySelector('.empty');
  if (!list.length && !empty){
    empty = document.createElement('p');
    empty.className = 'empty';
    empty.textContent = 'No camera queues found yet. Waiting for Station usb-video...';
    grid.appendChild(empty);
  } else if (list.length && empty){ empty.remove(); }
}
setInterval(tick, 2000); tick();
</script>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _cam(self, prefix):
        h = self.path[len(prefix) :].split("?")[0].split("/")[0].lower()
        return CAMS.get(h)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)
        elif path == "/api/cameras":
            now = time.time()
            body = json.dumps(
                [
                    {
                        "hash": c.hash,
                        "label": c.label,
                        "unique_id": c.unique_id,
                        "w": c.width,
                        "h": c.height,
                        "fps": round(c.fps(), 1),
                        "age": None if not c.last_frame_at else round(now - c.last_frame_at, 1),
                        "live": bool(c.last_frame_at) and (now - c.last_frame_at) < 5,
                    }
                    for c in CAMS.values()
                ]
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path.startswith("/snapshot/"):
            cam = self._cam("/snapshot/")
            if cam is None:
                return self.send_error(404)
            with cam.cond:
                data, mime = cam.frame, cam.mime
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif path.startswith("/stream/"):
            cam = self._cam("/stream/")
            if cam is None:
                return self.send_error(404)
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = 0
            try:
                while True:
                    with cam.cond:
                        cam.cond.wait_for(lambda: cam.seq != last, timeout=10)
                        data, mime, last = cam.frame, cam.mime, cam.seq
                    self.wfile.write(
                        b"--f\r\nContent-Type: %s\r\nContent-Length: %d\r\n\r\n" % (mime.encode(), len(data))
                    )
                    self.wfile.write(data + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404)


def main():
    ap = argparse.ArgumentParser(description="Live web view of Station usb-video cameras")
    ap.add_argument("cameras", nargs="*", help="queue hash, usbvideo/<hash>, or camera unique id")
    ap.add_argument("--port", type=int, default=8890)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--station", default="localhost:8888")
    ap.add_argument("--data-dir", default=str(BASE / "station-run"))
    ap.add_argument("--list", action="store_true", help="print discovered queues and exit")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("cam_stream")

    data_dir = Path(args.data_dir)
    extra = {normalize_queue(c) for c in args.cameras}
    found = scan_queues(data_dir)

    if args.list:
        for h in sorted(found | extra):
            print(f"usbvideo/{h}")
        if not (found or extra):
            print("no camera queues found under", data_dir, file=sys.stderr)
        return

    if not found and not extra:
        log.warning("no camera queues under %s yet; will keep watching for new ones", data_dir)
    if args.host != "127.0.0.1":
        log.warning("binding to %s; stream is reachable from the network", args.host)

    threading.Thread(
        target=lambda: asyncio.run(relay(data_dir, args.station, extra, log)), daemon=True
    ).start()

    print(f"{len(found) + len(extra)} camera(s) -> http://{args.host}:{args.port}", flush=True)
    try:
        ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
