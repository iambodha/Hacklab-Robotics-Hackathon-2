"""Live web view of the scene camera, relayed from the running Station.
usage: python3 scripts/scene_cam.py [camera_unique_id] [port]      -> http://127.0.0.1:8890
Reads Station's own video queue (keyed by the camera's unique ID), so the camera is not opened a
second time and the right camera is picked regardless of device order. Loopback only."""
import asyncio, hashlib, logging, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent / "norma-core"
sys.path.insert(0, str(ROOT / "software/station/shared")); sys.path.insert(0, str(ROOT))
from station_py import new_station_client
from target.gen_python.protobuf.drivers.usbvideo import usbvideo

SCENE_CAM = "0x11200002c7f4a60"   # gripper camera on the follower arm is 0x11110002c7f4a60
UID = sys.argv[1] if len(sys.argv) > 1 else SCENE_CAM
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8890
QUEUE = "usbvideo/" + hashlib.md5(UID.encode()).hexdigest()
frame, seq, cond = None, 0, threading.Condition()

async def relay():
    global frame, seq
    log = logging.getLogger("scene_cam")
    while True:
        try:
            client = await new_station_client(os.environ.get("STATION_HOST", "localhost"), log)
            q = asyncio.Queue(); client.follow(QUEUE, q)
            while True:
                e = await asyncio.wait_for(q.get(), 10)
                pack = usbvideo.RxEnvelopeReader(memoryview(bytes(e.Data))).get_frames()
                frames = (pack.get_frames_data() or []) if pack else []
                if frames:
                    with cond:
                        frame, seq = bytes(frames[-1]), seq + 1
                        cond.notify_all()
        except Exception as ex:  # Station down or camera unplugged: retry
            log.warning("stream lost (%s), retrying", ex)
            await asyncio.sleep(2)

PAGE = b"""<!doctype html><meta charset=utf-8><title>Scene camera</title>
<style>body{margin:0;background:#111;color:#ccc;font:14px system-ui;display:grid;place-items:center;height:100vh}
img{width:min(96vw,128vh);background:#000}p{margin:8px}</style>
<div><img src=/stream alt="scene camera"><p>Scene camera &middot; live from Station</p></div>"""

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == "/":
            self.send_response(200); self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(PAGE))); self.end_headers(); self.wfile.write(PAGE)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.send_header("Cache-Control", "no-store"); self.end_headers()
            last = 0
            try:
                while True:
                    with cond:
                        cond.wait_for(lambda: seq != last, timeout=5)
                        data, last = frame, seq
                    if data:
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(data))
                        self.wfile.write(data + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

logging.basicConfig(level=logging.WARNING)
threading.Thread(target=lambda: asyncio.run(relay()), daemon=True).start()
print(f"camera {UID} -> http://127.0.0.1:{PORT}", flush=True)
ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
