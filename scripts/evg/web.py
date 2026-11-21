"""Browser UI for the same flow as the console (stdlib only). Loopback only; every POST needs the per-launch token."""
import asyncio, json, logging, secrets, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import common as C
from agent import mcp_client, providers
from agent import robot as robot_mod      # also sets up sys.path for station_py

PAGE = (Path(__file__).parent / "web.html").read_text()


class LiveCams:
    """Live camera feed for the page, independent of the agent. Connects to the arm only while the page is asking for frames
    (first request starts it; 15 s without a request stops it), so it adds no standing load to a shared Station."""
    IDLE = 15

    def __init__(self):
        self.frames, self.t_req, self.lock, self.thread = {}, 0.0, threading.Lock(), None

    def get(self, cam):
        self.t_req = time.time()
        with self.lock:
            if not (self.thread and self.thread.is_alive()):
                self.thread = threading.Thread(target=self._run, daemon=True); self.thread.start()
        f = self.frames.get(cam)
        return f[1] if f and time.time() - f[0] < 5 else None

    def _run(self):
        try:
            if mcp_client.backend() == "mcp": self._run_mcp()
            else: asyncio.run(self._run_station())
        except Exception as e:
            print(f"live camera feed stopped: {type(e).__name__}: {e}", flush=True)

    def _run_mcp(self):
        mcp = mcp_client.connect()
        while time.time() - self.t_req < self.IDLE:
            _, images = mcp.call("look")
            for label, jpeg in images:
                for cid, name in mcp_client.CAMERAS.items():
                    if cid in label and "STALE" not in label.upper(): self.frames[name] = (time.time(), jpeg)
            time.sleep(1.0)

    async def _run_station(self):
        from station_py import new_station_client
        from target.gen_python.protobuf.drivers.usbvideo import usbvideo
        client = await asyncio.wait_for(new_station_client(C.station_host(), logging.getLogger("evg-web")), 10)

        async def pump(name, queue_id):
            q = asyncio.Queue(); client.follow(queue_id, q)
            while True:
                e = await q.get()
                try:
                    pack = usbvideo.RxEnvelopeReader(memoryview(bytes(e.Data))).get_frames()
                    fr = (pack.get_frames_data() or []) if pack else []
                    if fr: self.frames[name] = (time.time(), bytes(fr[-1]))
                except Exception:
                    pass
        tasks = [asyncio.create_task(pump("over", robot_mod.OVERVIEW_CAM)), asyncio.create_task(pump("grip", robot_mod.GRIPPER_CAM))]
        try:
            while time.time() - self.t_req < self.IDLE: await asyncio.sleep(1)
        finally:
            for t in tasks: t.cancel()
            client.close_connection()


def latest_frame(s, cam):
    """Newest JPEG for 'over'|'grip': the take being recorded, else the agent's latest tick, else the last camera.py snapshot."""
    tag = "over" if cam == "over" else "grip"
    if s.rec:
        files = sorted((C.DEMOS / s.rec["name"]).glob(f"{tag}_*.jpg"))
        if files: return files[-1]
    for p in (C.RUN / f"latest_{tag}.jpg", C.ROOT / "frames" / ("5cd55831.jpg" if tag == "over" else "e091eeda.jpg")):
        if p.exists(): return p
    return None


def serve(session, port):
    token = secrets.token_urlsafe(16)
    live = LiveCams()
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def _send(self, code, body, ctype="application/json"):
            b = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body, default=str)).encode()
            self.send_response(code); self.send_header("content-type", ctype); self.send_header("content-length", str(len(b)))
            self.send_header("cache-control", "no-store"); self.end_headers(); self.wfile.write(b)

        def _ok(self): return self.headers.get("Host") in hosts

        def do_GET(self):
            if not self._ok(): return self._send(403, {"error": "bad host"})
            path = self.path.split("?")[0]
            if path == "/": return self._send(200, PAGE.replace("__TOKEN__", token).replace("__PRESETS__", json.dumps(list(providers.PRESETS))), "text/html; charset=utf-8")
            if path == "/api/status":
                return self._send(200, dict(session.status(), demos=session.demos(), log=session.log_tail(30)))
            if path in ("/frame/over.jpg", "/frame/grip.jpg"):
                cam = "over" if "over" in path else "grip"
                data = live.get(cam)                       # live feed first; saved frames only if the arm can't be reached
                if data: return self._send(200, data, "image/jpeg")
                f = latest_frame(session, cam)
                return self._send(200, f.read_bytes(), "image/jpeg") if f else self._send(404, b"", "image/jpeg")
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._ok() or self.headers.get("X-Token") != token: return self._send(403, {"error": "forbidden"})
            n = int(self.headers.get("content-length") or 0)
            try: d = json.loads(self.rfile.read(n) or b"{}")
            except ValueError: return self._send(400, {"error": "bad json"})
            a, s = self.path.removeprefix("/api/"), session
            try:
                if a == "task": r = {"task": s.set_task(d.get("text", ""))}
                elif a == "select": s.select_demos(d.get("names", [])); r = {}
                elif a == "cfg": s.set_cfg(**{k: d.get(k) for k in ("preset", "model", "base_url", "api_key_env", "step_mode", "max_ticks", "chunk", "effort")}); r = {}
                elif a == "station_start": r = {"up": s.start_station()}
                elif a == "snapshot": r = {"out": s._script("camera.py", timeout=40)[-300:]}
                elif a == "rec_start": r = dict(zip(("name", "teleop"), s.rec_start()))
                elif a == "rec_stop": r = dict(zip(("name", "samples"), s.rec_stop()))
                elif a == "rec_keep": r = dict(zip(("name", "samples", "kept"), s.rec_keep(d.get("description", ""))))
                elif a == "rec_discard": s.rec_discard(); r = {}
                elif a == "run": r = {"pid": s.run_start(dry_run=d.get("mode") == "dry", replay=(s.cfg["demos"] or [None])[0] if d.get("mode") == "replay" else None)}
                elif a == "stop": r = dict(zip(("how", "hold"), s.run_stop()))
                elif a == "pause": s.pause(bool(d.get("on", True))); r = {}
                elif a == "approve": s.approve(bool(d.get("yes", True))); r = {}
                else: return self._send(404, {"error": "unknown action"})
                self._send(200, r)
            except (RuntimeError, ValueError, FileNotFoundError, subprocess.TimeoutExpired) as e:
                self._send(400, {"error": str(e)})

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"evg web UI: http://127.0.0.1:{port}/   (loopback only; Ctrl-C to quit, a running agent keeps running)", flush=True)
    srv.serve_forever()
