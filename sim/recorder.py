"""Session recorder + live dashboard for sim_arm_mcp (for the human watching, never the agent).

Writes <dir>/<timestamp>/:
  frames/000000.jpg ...  side-camera video frames at `fps`
  state.jsonl            one row per frame: time, joint norms, goal norms, load, status
  events.jsonl           every MCP tool call: time, args, result, the images the agent saw
  looks/                 the camera JPEGs returned to the agent by look()
and serves http://localhost:<port>/ : live side-cam MJPEG, joint bars and the command log.
`python3 sim/make_replay.py <session>` (also run on shutdown) builds side.mp4 + replay.html.
"""
from __future__ import annotations

import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image as PILImage

NAMES = [f'j{i}' for i in range(1, 9)]


class Recorder:
    def __init__(self, arm, root: Path, fps: float = 10, port: int = 8891, cam: str = 'side_cam', size=(640, 480)):
        self.arm, self.fps, self.cam, self.size = arm, fps, cam, size
        joint_names = [f'j{i}' for i in range(1, arm.model.nu + 1)]
        self.dir = Path(root) / time.strftime('%Y%m%d-%H%M%S')
        (self.dir / 'frames').mkdir(parents=True)
        (self.dir / 'looks').mkdir()
        self.t0 = time.monotonic()
        self.frame = 0
        self.latest_jpeg = b''
        self.latest_state: dict = {}
        self.events: list[dict] = []
        self.cond = threading.Condition()
        self.running = True
        (self.dir / 'session.json').write_text(json.dumps(
            {'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'fps': fps, 'camera': cam, 'joints': joint_names}, indent=2))
        self._state_f = open(self.dir / 'state.jsonl', 'a', buffering=1)
        self._events_f = open(self.dir / 'events.jsonl', 'a', buffering=1)
        threading.Thread(target=self._capture, daemon=True).start()
        self.url = self._serve(port) if port else None

    def now(self):
        return round(time.monotonic() - self.t0, 3)

    # --- capture loop --------------------------------------------------------
    def _capture(self):
        arm, m = self.arm, self.arm.model
        renderer = mujoco.Renderer(m, self.size[1], self.size[0])
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False  # 150 -> 30 ms/frame on software GL
        next_t = time.monotonic()
        while self.running:
            with arm.lock:
                renderer.update_scene(arm.data, camera=self.cam, scene_option=arm.vis_option)
                q, ctrl = arm.data.qpos[arm.qids].copy(), arm.data.ctrl.copy()
                force, sim_t = np.abs(arm.data.actuator_force).copy(), float(arm.data.time)
                status, torque = arm.status.copy(), arm.torque.copy()
            buf = io.BytesIO()
            PILImage.fromarray(renderer.render()).save(buf, format='JPEG', quality=82)
            jpeg = buf.getvalue()
            row = {'t': self.now(), 'sim_t': round(sim_t, 3),
                   'norm': np.round(arm.norm(q), 4).tolist(), 'goal': np.round(arm.norm(ctrl), 4).tolist(),
                   'load': np.round(force / arm.fmax, 3).tolist(), 'status': status.tolist(), 'torque': torque.astype(int).tolist()}
            # Frame index stays locked to wall time (frame = t * fps): if capture lagged, repeat this frame.
            due = max(self.frame + 1, int((time.monotonic() - self.t0) * self.fps) + 1)
            while self.frame < due:
                (self.dir / 'frames' / f'{self.frame:06d}.jpg').write_bytes(jpeg)
                self._state_f.write(json.dumps({'frame': self.frame, **row}) + '\n')
                self.frame += 1
            with self.cond:
                self.latest_jpeg, self.latest_state = jpeg, {'frame': self.frame - 1, **row}
                self.cond.notify_all()
            next_t = self.t0 + self.frame / self.fps
            time.sleep(max(0.0, next_t - time.monotonic()))
        renderer.close()

    # --- tool events -----------------------------------------------------------
    def event(self, tool: str, args: dict, result: str = '', images: list[tuple[str, bytes]] = (),
              truth: dict | None = None, start: tuple[float, int] | None = None):
        """Record a finished tool call; `start` = (t, frame) when the call began."""
        n = len(self.events)
        looks = []
        for cam, jpeg in images:
            name = f'{n:04d}_{cam}.jpg'
            (self.dir / 'looks' / name).write_bytes(jpeg)
            looks.append(name)
        t0, f0 = start or (self.now(), self.frame)
        ev = {'n': n, 't': t0, 'frame': f0, 't_end': self.now(), 'frame_end': self.frame, 'tool': tool, 'args': args,
              'result': result, 'looks': looks, 'truth': truth}
        self._events_f.write(json.dumps(ev) + '\n')
        with self.cond:
            self.events.append(ev)
        return ev

    def close(self):
        self.running = False
        time.sleep(2 / self.fps)
        self._state_f.close(); self._events_f.close()

    # --- live dashboard ------------------------------------------------------
    def _serve(self, port):
        rec = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # stdout belongs to MCP; keep stderr quiet too
                pass

            def _send(self, body: bytes, ctype: str, code=200):
                self.send_response(code)
                self.send_header('Content-Type', ctype)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path, _, query = self.path.partition('?')
                if path == '/':
                    return self._send(LIVE_HTML.encode(), 'text/html; charset=utf-8')
                if path == '/api/live':
                    since = int(dict(p.split('=') for p in query.split('&') if '=' in p).get('since', 0))
                    with rec.cond:
                        body = {'state': rec.latest_state, 'events': rec.events[since:], 'session': str(rec.dir)}
                    return self._send(json.dumps(body).encode(), 'application/json')
                if path.startswith('/looks/'):
                    f = rec.dir / 'looks' / Path(path).name
                    return self._send(f.read_bytes(), 'image/jpeg') if f.exists() else self._send(b'', 'text/plain', 404)
                if path == '/replay':
                    f = rec.dir / 'replay.html'
                    return self._send(f.read_bytes(), 'text/html; charset=utf-8') if f.exists() else \
                        self._send(b'replay is built when the server stops (or run sim/make_replay.py)', 'text/plain', 404)
                if path == '/side.mp4':
                    f = rec.dir / 'side.mp4'
                    return self._send(f.read_bytes(), 'video/mp4') if f.exists() else self._send(b'', 'text/plain', 404)
                if path == '/stream.mjpg':
                    self.send_response(200)
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                    self.end_headers()
                    try:
                        while rec.running:
                            with rec.cond:
                                rec.cond.wait(timeout=1)
                                jpeg = rec.latest_jpeg
                            self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n' % len(jpeg) + jpeg + b'\r\n')
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                self._send(b'not found', 'text/plain', 404)

        for p in range(port, port + 10):
            try:
                httpd = ThreadingHTTPServer(('0.0.0.0', p), Handler)
            except OSError:
                continue
            httpd.daemon_threads = True
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            (self.dir / 'session.json').write_text(json.dumps(
                {**json.loads((self.dir / 'session.json').read_text()), 'dashboard': f'http://localhost:{p}/'}, indent=2))
            return f'http://localhost:{p}/'
        return None


LIVE_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>ElRobot sim — live</title>
<style>
body{margin:0;font:13px/1.4 system-ui,Segoe UI,sans-serif;background:#111418;color:#e6e6e6}
header{padding:8px 14px;background:#1a1f26;display:flex;gap:16px;align-items:center}
header b{font-size:15px} header span{color:#8a96a3}
main{display:grid;grid-template-columns:minmax(480px,1.2fr) 1fr;gap:12px;padding:12px}
img#cam{width:100%;border-radius:6px;background:#000}
.card{background:#1a1f26;border-radius:6px;padding:10px}
.joint{display:grid;grid-template-columns:28px 1fr 52px 46px 44px;gap:8px;align-items:center;margin:3px 0}
.bar{position:relative;height:14px;background:#2a313b;border-radius:3px}
.bar .v{position:absolute;top:0;bottom:0;left:0;background:#3d8bfd;border-radius:3px}
.bar .g{position:absolute;top:-3px;bottom:-3px;width:2px;background:#ffb020}
.num{font-variant-numeric:tabular-nums;text-align:right;color:#cfd6dd}
.load.hot{color:#ff6b6b}
#log{max-height:calc(100vh - 120px);overflow:auto}
.ev{border-bottom:1px solid #2a313b;padding:6px 2px}
.ev .h{display:flex;gap:8px;overflow-wrap:anywhere} .ev .t{color:#8a96a3;width:60px} .ev .tool{color:#7fd1ff;font-weight:600}
.ev pre{margin:4px 0 0;white-space:pre-wrap;color:#b8c2cc;font-size:11px}
.ev img{height:84px;margin:4px 4px 0 0;border-radius:3px}
.abort{color:#ff6b6b}
</style></head><body>
<header><b>ElRobot simulation — live</b><span id="sess"></span><span id="clock"></span>
<a href="/replay" style="color:#7fd1ff;margin-left:auto">replay (after stop)</a></header>
<main><div><img id="cam" src="/stream.mjpg"><div class="card" id="joints" style="margin-top:12px"></div>
<div style="color:#8a96a3;margin-top:6px">blue = measured position, orange = commanded goal (normalized 0..1; j8 1 = open). Side camera is for you only — the agent sees the wrist + overview images in the log.</div></div>
<div class="card" id="log"></div></main>
<script>
const fa=a=>JSON.stringify(a,(k,v)=>typeof v==='number'?Math.round(v*1000)/1000:v).replace(/"/g,'').replace(/,/g,', ');const fr=r=>{const f=(r||'').split('\n')[0];return f.startsWith('joint ')||f.startsWith('camera ')?'':f};
let since=0;const log=document.getElementById('log');
function joints(s){if(!s.norm)return;document.getElementById('clock').textContent=`t=${s.t.toFixed(1)}s  sim=${s.sim_t.toFixed(1)}s`;
 document.getElementById('joints').innerHTML=s.norm.map((v,i)=>`<div class="joint"><b>j${i+1}</b><div class="bar"><div class="v" style="width:${(v*100).toFixed(1)}%"></div><div class="g" style="left:${(s.goal[i]*100).toFixed(1)}%"></div></div><span class="num">${v.toFixed(3)}</span><span class="num load ${s.load[i]>0.9?'hot':''}">${Math.round(s.load[i]*100)}%</span><span class="num ${s.status[i]?'abort':''}">${s.status[i]?'0x'+s.status[i].toString(16):(s.torque[i]?'ok':'off')}</span></div>`).join('')}
function ev(e){const d=document.createElement('div');d.className='ev';const first=fr(e.result);
 d.innerHTML=`<div class="h"><span class="t">${e.t.toFixed(1)}s</span><span class="tool">${e.tool}</span><span>${fa(e.args)}</span></div>`+
 (first?`<pre class="${first.startsWith('ABORTED')||first.startsWith('ERROR')?'abort':''}">${first.replace(/</g,'&lt;')}</pre>`:'')+
 e.looks.map(l=>`<a href="/looks/${l}" target="_blank"><img src="/looks/${l}"></a>`).join('');log.prepend(d)}
async function poll(){try{const r=await fetch('/api/live?since='+since);const j=await r.json();document.getElementById('sess').textContent=j.session;
 joints(j.state);j.events.forEach(ev);since+=j.events.length}catch(e){}setTimeout(poll,250)}poll();
</script></body></html>"""
