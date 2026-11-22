"""Dashboard HTTP server on 127.0.0.1:8920 (stdlib only).

GET /                      the page (static/index.html, app.js, app.css)
GET /api/config            relay/recorder URLs and camera hashes for the page
GET /api/state             aggregated attempts from episodes.jsonl (+ ?v=<version> -> {"unchanged": true})
GET /api/agent             tail of actions.jsonl + latest OPERATOR_LOG assistant text (same ?v= protocol)
GET /api/live              proxy of recorder GET /live (503 when the recorder is down)
GET /api/trace/<N>         attempt_<N>/twist_trace.json normalised to columns
GET /files/<rel>           any file inside LEARN_DIR (images, report.md, SUMMARY.md); ?download=1 forces download
"""
from __future__ import annotations

import json
import mimetypes
import os
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from sites import get_site

from .data import AgentStore, LearnStore, trace_columns

STATIC = Path(__file__).parent / "static"
RECORDER_URL = os.environ.get("RECORDER_URL", "http://127.0.0.1:8900")
RELAY_URL = os.environ.get("RELAY_URL", "http://127.0.0.1:8890")
CAMERAS = [{"role": role, "hash": get_site().camera_hash(role)} for role in ("wrist", "overview")]
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}


class App:
    def __init__(self, learn: LearnStore, agent: AgentStore, recorder_url: str = RECORDER_URL,
                 relay_url: str = RELAY_URL):
        self.learn, self.agent, self.recorder_url, self.relay_url = learn, agent, recorder_url, relay_url

    def config(self) -> dict:
        return {"relay": self.relay_url, "recorder": self.recorder_url, "cameras": CAMERAS,
                "learn_dir": str(self.learn.dir), "actions_path": str(self.agent.actions),
                "operator_path": str(self.agent.operator) if self.agent.operator else None}

    def live(self) -> tuple[int, bytes]:
        try:
            with urllib.request.urlopen(self.recorder_url + "/live", timeout=1.5) as r:
                return 200, r.read()
        except Exception as e:  # recorder down, capture off (503), timeout
            return 503, json.dumps({"error": f"recorder /live unavailable: {e}"}).encode()


def make_handler(app: App):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj, default=str).encode(), "application/json")

        def _versioned(self, q: dict, state: dict):
            if q.get("v", [None])[0] == state["version"]:
                self._json({"unchanged": True, "version": state["version"]})
            else:
                self._json(state)

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            u = urlparse(self.path)
            path, q = unquote(u.path), parse_qs(u.query)
            try:
                if path in ("/", "/index.html"):
                    return self._static("index.html")
                if path.startswith("/static/"):
                    return self._static(path[len("/static/"):])
                if path == "/api/config":
                    return self._json(app.config())
                if path == "/api/state":
                    return self._versioned(q, app.learn.state())
                if path == "/api/agent":
                    return self._versioned(q, app.agent.state())
                if path == "/api/live":
                    code, body = app.live()
                    return self._send(code, body, "application/json")
                if path.startswith("/api/trace/"):
                    n = path.rsplit("/", 1)[-1]
                    if not n.isdigit():
                        return self._json({"error": "attempt must be an integer"}, 400)
                    tr = app.learn.trace(int(n))
                    return self._json({"attempt": int(n), "columns": trace_columns(tr)} if tr is not None
                                      else {"error": "no twist_trace.json yet"}, 200 if tr is not None else 404)
                if path.startswith("/files/"):
                    return self._file(path[len("/files/"):], "download" in q)
                self._json({"error": "not found"}, 404)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _static(self, name: str):
            p = (STATIC / name).resolve()
            if STATIC.resolve() not in p.parents or not p.is_file():
                return self._json({"error": "not found"}, 404)
            self._send(200, p.read_bytes(), STATIC_TYPES.get(p.suffix, "application/octet-stream"))

        def _file(self, rel: str, download: bool):
            p = app.learn.resolve(rel)
            if p is None:
                return self._json({"error": f"no such file in learn dir: {rel}"}, 404)
            ctype = "text/markdown; charset=utf-8" if p.suffix == ".md" else (
                mimetypes.guess_type(p.name)[0] or "application/octet-stream")
            extra = {"Content-Disposition": f'attachment; filename="{p.name}"'} if download else None
            self._send(200, p.read_bytes(), ctype, extra)

    return H


def make_server(app: App, host: str = "127.0.0.1", port: int = 8920) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), make_handler(app))
    srv.daemon_threads = True
    return srv
