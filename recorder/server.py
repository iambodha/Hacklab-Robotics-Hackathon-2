"""HTTP API on 127.0.0.1:8900 (stdlib only). See README.md for the endpoint list.

`Service` owns the optional live capture and the single active recording; `make_server` wraps it in
a ThreadingHTTPServer. Tests build a Service with capture=None (no Station, no norma-core).
"""
from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .fmt import strip_regs
from .replay import Recording
from .store import ROOT, SessionWriter, find, list_recordings
from .viewer import VIEWER_HTML

LOCAL_ORIGIN = re.compile(r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$")


class ApiError(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


class Service:
    def __init__(self, root: Path | None = None, capture=None):
        self.root = Path(root or ROOT)
        self.root.mkdir(parents=True, exist_ok=True)
        self.capture = capture
        self.lock = threading.Lock()
        self.active: SessionWriter | None = None
        self._cache: dict[str, tuple[tuple, Recording]] = {}

    # --- recording control -----------------------------------------------------
    def start(self, name: str, note: str = "") -> dict:
        with self.lock:
            if self.active:
                raise ApiError(409, f"recording {self.active.id} is already active")
            snap = self.capture.snapshot() if self.capture else {}
            w = SessionWriter(name or "rec", note, root=self.root, station=snap.get("station", ""),
                              extra={"state_hz": getattr(self.capture, "state_hz", None)})
            for c in snap.get("cameras", []):
                w.meta["cameras"].setdefault(c["hash8"], {"hash": c["hash"], "hash8": c["hash8"], "role": c["role"],
                                                          "frames": 0, "dropped": 0, "late": 0, "bytes": 0,
                                                          "w": c.get("w"), "h": c.get("h")})
            for s, b in snap.get("buses", {}).items():
                w.meta["buses"][s] = {"role": b["role"], "motor_ids": b["motor_ids"]}
            if self.capture:
                self.capture.attach(w)
            else:
                w.event("start", name=w.meta["name"], note=note, station_connected=False)
            self.active = w
            return w.meta

    def stop(self, rec_id: str) -> dict:
        with self.lock:
            if not self.active or rec_id not in (self.active.id, "active"):
                raise ApiError(404 if not self.active else 409, f"recording {rec_id} is not the active one")
            w, self.active = self.active, None
        if self.capture:
            self.capture.detach()
        return w.close()

    def marker(self, rec_id: str, label: str, data=None) -> dict:
        w = self.active
        if not w or rec_id not in (w.id, "active"):
            raise ApiError(409 if w else 404, f"recording {rec_id} is not active; markers go to the active recording")
        if not label:
            raise ApiError(400, "label is required")
        return w.event("marker", label=str(label), data=data)

    # --- reading -----------------------------------------------------------------
    def resolve(self, rec_id: str) -> Path:
        if rec_id == "active":
            if not self.active:
                raise ApiError(404, "no active recording")
            return self.active.dir
        p = find(rec_id, self.root)
        if p is None:
            raise ApiError(404, f"recording {rec_id} not found")
        return p

    def recording(self, rec_id: str) -> Recording:
        p = self.resolve(rec_id)
        files = [p / "session.json", p / "state.jsonl", p / "events.jsonl", *sorted(p.glob("cams/*/index.jsonl"))]
        key = tuple((f.stat().st_mtime_ns, f.stat().st_size) for f in files if f.exists())
        hit = self._cache.get(str(p))
        if hit and hit[0] == key:
            return hit[1]
        rec = Recording(p)
        if self.active and self.active.dir == p:
            rec.meta = dict(self.active.meta)
        self._cache[str(p)] = (key, rec)
        return rec

    def health(self) -> dict:
        snap = self.capture.snapshot() if self.capture else {"station": None, "connected": False, "buses": {},
                                                             "cameras": [], "errors": {"capture": "disabled"}}
        a = self.active
        return {"ok": True, "station_reachable": snap["connected"], **snap,
                "recording": {"id": a.id, "t": a.now(), "counts": a.meta["counts"]} if a else None}

    def listing(self) -> list[dict]:
        out = list_recordings(self.root)
        if self.active:
            out = [dict(self.active.meta, active=True) if m.get("id") == self.active.id else m for m in out]
        return out


def _num(q: dict, k: str, default=None):
    v = q.get(k)
    if v in (None, ""):
        return default
    try:
        return float(v)
    except ValueError:
        raise ApiError(400, f"{k} must be a number")


def make_handler(svc: Service):
    class Handler(BaseHTTPRequestHandler):
        server_version = "recorder/1"

        def log_message(self, fmt, *a):
            pass

        def _cors(self):
            origin = self.headers.get("Origin", "")
            if LOCAL_ORIGIN.match(origin):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _send(self, body: bytes, ctype: str, code: int = 200, headers: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", (headers or {}).pop("Cache-Control", "no-store"))
            for k, v in (headers or {}).items():
                self.send_header(k, str(v))
            self._cors()
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(json.dumps(obj, separators=(",", ":")).encode(), "application/json", code)

        def _jpeg(self, data: bytes | None, headers: dict | None = None, cache: bool = False):
            if data is None:
                raise ApiError(404, "no frame")
            h = dict(headers or {})
            if cache:
                h["Cache-Control"] = "max-age=86400, immutable"
            self._send(data, "image/jpeg", 200, h)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if not n:
                return {}
            try:
                body = json.loads(self.rfile.read(n))
            except json.JSONDecodeError:
                raise ApiError(400, "body must be JSON")
            if not isinstance(body, dict):
                raise ApiError(400, "body must be a JSON object")
            return body

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            self._dispatch("GET")

        def do_HEAD(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

        def _dispatch(self, method: str):
            u = urlparse(self.path)
            q = {k: v[-1] for k, v in parse_qs(u.query).items()}
            parts = [p for p in u.path.split("/") if p]
            try:
                self._route(method, parts, q)
            except ApiError as e:
                self._json({"error": str(e)}, e.code)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # keep the server up; report the failure to the caller
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def _route(self, method: str, parts: list[str], q: dict):
            if method == "GET" and parts in ([], ["index.html"]):
                return self._send(INDEX_HTML.encode(), "text/html; charset=utf-8")
            if method == "GET" and parts == ["health"]:
                return self._json(svc.health())
            if method == "GET" and parts == ["live"]:
                row = svc.capture.latest() if svc.capture else None
                if row is None:
                    raise ApiError(503, "no live motor state yet (Station unreachable or capture disabled)")
                return self._json(row if q.get("regs") == "1" else strip_regs(row))
            if method == "GET" and parts == ["live", "frame"]:
                cam = q.get("cam", "")
                return self._jpeg(svc.capture.latest_jpeg(cam) if svc.capture else None)
            if parts == ["recordings"]:
                if method == "GET":
                    return self._json({"recordings": svc.listing(),
                                       "active": svc.active.id if svc.active else None})
                b = self._body()
                meta = svc.start(str(b.get("name") or "rec"), str(b.get("note") or ""))
                return self._json({"id": meta["id"], "session": meta}, 201)
            if len(parts) < 2 or parts[0] != "recordings":
                raise ApiError(404, "not found")
            rid, rest = parts[1], parts[2:]
            if method == "POST":
                if rest == ["stop"]:
                    return self._json({"stopped": True, "session": svc.stop(rid)})
                if rest == ["markers"]:
                    b = self._body()
                    return self._json({"event": svc.marker(rid, b.get("label"), b.get("data"))}, 201)
                raise ApiError(404, "not found")
            if rest == ["viewer"]:
                svc.resolve(rid)
                return self._send(VIEWER_HTML.encode(), "text/html; charset=utf-8")
            rec = svc.recording(rid)
            if rest == []:
                return self._json({"session": rec.meta, "summary": rec.stats(),
                                   "markers": rec.markers(), "active": bool(svc.active and svc.active.dir == rec.path)})
            if rest == ["state"]:
                rows = rec.rows_between(_num(q, "from"), _num(q, "to"), _num(q, "every"))
                if q.get("regs") != "1":
                    rows = [strip_regs(r) for r in rows]
                return self._json({"id": rec.id, "count": len(rows), "rows": rows})
            if rest == ["at"]:
                t = _num(q, "t", 0.0)
                causal = q.get("causal") == "1"
                st = rec.state_at(t, causal)
                return self._json({"t": t, "state": st if (st is None or q.get("regs") == "1") else strip_regs(st),
                                   "frames": {h8: rec.frame_index(h8, t, causal) for h8 in rec.cams}})
            if rest == ["frame"]:
                hit = rec.frame_at(q.get("cam"), _num(q, "t", 0.0), q.get("causal") == "1")
                if hit is None:
                    raise ApiError(404, f"no frame for cam={q.get('cam')!r}")
                row, data = hit
                return self._jpeg(data, {"X-Frame-Cam": row["cam"], "X-Frame-N": row["n"], "X-Frame-T": row["t"]})
            if len(rest) == 3 and rest[0] == "frames" and rest[2].endswith(".jpg") and rest[2][:-4].isdigit():
                return self._jpeg(rec.frame_bytes(rest[1], int(rest[2][:-4])), cache=True)
            if rest == ["index"]:
                return self._json({"cameras": {h8: {"role": c["info"].get("role"), "hash": c["info"].get("hash"),
                                                    "t": c["ts"], "n": [r["n"] for r in c["rows"]],
                                                    "lat": [r.get("latency") for r in c["rows"]]}
                                               for h8, c in rec.cams.items()}})
            if rest == ["events"]:
                return self._json({"events": rec.events})
            if rest == ["commands"]:
                return self._json({"commands": rec.commands})
            raise ApiError(404, "not found")

    return Handler


def make_server(svc: Service, host: str = "127.0.0.1", port: int = 8900) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), make_handler(svc))
    httpd.daemon_threads = True
    return httpd


def serve(host: str = "127.0.0.1", port: int = 8900, state_hz: float = 50.0, station: str | None = None):
    from .live import LiveCapture  # needs norma-core; only for the live server

    cap = LiveCapture(host=station, state_hz=state_hz).start()
    svc = Service(capture=cap)
    httpd = make_server(svc, host, port)
    print(f"recorder API on http://{host}:{port}/  (recordings in {svc.root})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if svc.active:
            svc.stop(svc.active.id)
        httpd.server_close()


INDEX_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>recorder</title>
<style>body{margin:0;font:13px/1.5 system-ui,sans-serif;background:#111418;color:#e6e6e6;padding:14px 18px}
a{color:#7fd1ff} .dim{color:#8a96a3} table{border-collapse:collapse} td,th{padding:3px 10px;border-bottom:1px solid #232a33;text-align:left}
img{width:299px;border-radius:4px;background:#000;margin-right:8px}</style></head><body>
<h2 style="margin:0 0 6px">recorder <span class="dim" id="h"></span></h2>
<div id="cams"></div><h3>recordings</h3><table id="t"></table>
<script>
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function tick(){try{const h=await (await fetch('/health')).json();
 document.getElementById('h').textContent=`station ${h.station_reachable?'reachable':'UNREACHABLE'} · buses ${Object.keys(h.buses).join(', ')||'none'} · ${h.recording?'recording '+h.recording.id:'idle'}`;
 const c=document.getElementById('cams');if(!c.dataset.done){c.dataset.done=1;c.innerHTML=h.cameras.map(x=>`<img id="L${x.hash8}" alt="${esc(x.role||x.hash8)}" title="${esc(x.role||x.hash8)}">`).join('')}
 h.cameras.forEach(x=>{const i=document.getElementById('L'+x.hash8);if(i)i.src='/live/frame?cam='+x.hash8+'&_='+Date.now()})}catch(e){}
 setTimeout(tick,500)}
async function list(){const r=await (await fetch('/recordings')).json();
 document.getElementById('t').innerHTML='<tr><th>id</th><th>duration</th><th>rows</th><th>cams</th><th>note</th></tr>'+r.recordings.map(m=>`<tr><td><a href="/recordings/${encodeURIComponent(m.id)}/viewer">${esc(m.id)}</a>${m.active?' (active)':''}</td><td>${m.duration??'-'}</td><td>${m.counts?m.counts.state_rows:''}</td><td>${Object.keys(m.cameras||{}).length}</td><td class="dim">${esc(m.note)}</td></tr>`).join('')}
tick();list();
</script></body></html>"""
