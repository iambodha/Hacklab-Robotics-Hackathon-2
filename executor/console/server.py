"""Operator console + JSON API of the executor: http://127.0.0.1:8910 (loopback only, stdlib HTTP server).

The process that runs this owns all arm motion (the executor). The page, Claude (through the MCP proxy), Gemini
and scripts all use the same API:

  GET  /api/state  /api/live  /api/skills  /api/scene  /api/advisor  /api/timeline?since=N  /api/executions
  GET  /api/status/<execution_id>?wait=S   /api/cam/<role>.jpg   /api/frame/<obs_id>/<role>.jpg   /api/recordings
  POST /api/observe  /api/submit  /api/cancel/<execution_id|current>  /api/evaluate/<execution_id>
  GET  /api/health (status pills)  /api/overlay (projected world grid, estimates, planned path)  /api/estimates
  POST /api/task/start {prompt, advisor, backend?, model}  /api/task/stop  /api/backend {name}  /api/arm {enabled}  /api/tool {tool}
  GET  /api/voice   POST /api/voice/token  /api/voice/start {prompt, advisor} (fake/sim only)  /api/voice/say {role, text}
  GET  /vendor/<file>  /robot/<file>   (3D view: pinned libraries, arm model from NORMA_CORE_REPO)

Advisors: claude | gemini | scripted (executor/console/scripted.py) | none (manual proposals from the console).
"""
from __future__ import annotations

import collections
import json
import os
import shutil
import socket
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..advisor import make_advisor
from ..backends import make_backend
from ..contracts import Status, proposal_from_dict, to_dict
from ..core import EventLog, Executor
from ..kinematics import Kinematics, scene_calib_path
from ..recording import LOCAL_ROOT, SERVICE_URL, RecorderLink, service_up
from . import assets3d, voice

HERE = Path(__file__).resolve().parent
STATE_DIR = HERE.parent / "logs"
STATION_ADDR = ("127.0.0.1", int(os.environ.get("STATION_TCP_PORT", 8888)))
TWIN_MIRROR_URL = os.environ.get("TWIN_MIRROR_URL", "http://127.0.0.1:8892")
ADVISORS = ("scripted", "claude", "gemini", "none")
BACKENDS = ("fake", "sim", "real")


def _get_json(url: str, timeout: float = 0.6):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def _port_open(addr, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection(addr, timeout=timeout):
            return True
    except OSError:
        return False


class ApiError(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


class Console:
    def __init__(self, backend: str = "fake", port: int = 8910, local_rec_port: int = 8911):
        self.port, self.api_url = port, f"http://127.0.0.1:{port}"
        self.events = EventLog()
        self.kin = Kinematics()
        self.backends: dict[str, object] = {}
        self.ex: Executor | None = None
        self.advisor = None
        self.task = {"state": "idle"}
        self.lock = threading.RLock()
        self._cam_cache: dict[str, tuple[float, bytes]] = {}
        self._frames: dict = {}                                   # newest FrameSample per role (from cam())
        self._cam_seq: dict[str, collections.deque] = {}          # role -> (t, seq) for the fps pill
        self._est: tuple[float, list] = (0.0, [])
        self._probes: dict = {}
        self.local_rec_port = local_rec_port
        self._start_local_recorder()
        self.switch(backend)
        threading.Thread(target=self._sampler, daemon=True, name="sampler").start()
        threading.Thread(target=self._prober, daemon=True, name="prober").start()

    # --- backends -------------------------------------------------------------------------
    def _start_local_recorder(self):
        """A recorder API + viewer over the executor's own local recordings (fake / sim / replay runs)."""
        try:
            from recorder.server import Service, make_server
            LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
            srv = make_server(Service(root=LOCAL_ROOT, capture=None), port=self.local_rec_port)
            threading.Thread(target=srv.serve_forever, daemon=True, name="local-recorder").start()
        except OSError as e:
            self.events.add("error", text=f"local recorder viewer not started on :{self.local_rec_port}: {e}")

    def switch(self, name: str) -> None:
        with self.lock:
            if self.ex and self.ex.busy():
                raise ApiError(409, "an execution is running: cancel it first")
            if self.advisor and self.advisor.running:
                raise ApiError(409, "a task is running: stop it first")
            if self.ex:
                self.ex.end_task("backend switched")
                if getattr(self.ex.backend, "enable_motion", None):
                    self.ex.backend.enable_motion(False)
            b = self.backends.get(name)
            if b is None:
                # fake: real-time clock and jaws that stop on an object part-way, so a pick can succeed
                b = make_backend(name, kin=self.kin, **({"realtime": True, "object_norm": 0.3} if name == "fake" else {}))
                if name in ("sim", "real"):     # one twin / one Station capture per process
                    self.backends[name] = b
            mode = "service" if name == "real" and service_up() else "local"
            layout = getattr(b, "layout_name", "default")
            locator = None
            if name == "fake":
                from .demo_scene import DemoScene
                locator = DemoScene()
            self.ex = Executor(b, self.kin, recorder=RecorderLink(mode), events=self.events, layout=layout, locator=locator,
                               table_memory_path=STATE_DIR / "table_memory.json")
            self._frames, self._cam_cache, self._cam_seq, self._est = {}, {}, {}, (0.0, [])
            self.events.add("backend", name=name, recorder=mode, setup_id=self.ex.setup_id(), describe=b.describe())

    def _sampler(self):
        """Low-rate state + frames into local recordings (the recorder service records the real arm itself)."""
        last_frames = 0.0
        while True:
            time.sleep(0.1)
            ex = self.ex
            try:
                if ex and hasattr(ex.locator, "tick"):
                    m = ex.backend.read_state().motors
                    ex.locator.tick({i: m[i]["norm"] for i in ex.kin.ids}, ex.kin)
                if ex and ex.rec.active and ex.rec.mode == "local":
                    frames = None
                    if time.monotonic() - last_frames > 0.5:
                        frames, last_frames = ex.backend.frames(), time.monotonic()
                    ex.rec.sample(ex.backend.read_state().motors, frames)
            except Exception:  # noqa: BLE001
                pass

    def _prober(self):
        """Slow checks for the header pills, off the request path: Station, recorder service, twin mirror."""
        while True:
            p = {"station": _port_open(STATION_ADDR)}
            try:
                p["recorder"] = _get_json(SERVICE_URL + "/health")
            except (OSError, ValueError):
                p["recorder"] = None
            try:
                _get_json(TWIN_MIRROR_URL + "/state.json", 0.4)
                p["mirror"] = True
            except (OSError, ValueError):
                p["mirror"] = False
            self._probes = p
            time.sleep(2.0)

    # --- task / advisor -------------------------------------------------------------------------
    def start_task(self, prompt: str, advisor: str, model: str | None, backend: str | None = None) -> dict:
        with self.lock:
            if self.advisor and self.advisor.running:
                raise ApiError(409, "a task is already running")
            if advisor not in ADVISORS:
                raise ApiError(400, f"advisor must be one of {', '.join(ADVISORS)}")
            if backend and backend != self.ex.backend.name:
                self.switch(backend)
            if advisor == "gemini":
                from ..advisor.gemini_er import enabled
                if not enabled():
                    raise ApiError(400, "Gemini ER needs GEMINI_API_KEY in the executor's environment")
            if advisor == "claude" and not shutil.which("claude"):
                raise ApiError(400, "the Claude advisor needs the `claude` CLI on PATH")
            rid = self.ex.start_task(advisor, prompt=prompt, advisor=advisor)
            kw = {"model": model or "sonnet"} if advisor == "claude" else {}
            skills = {k: s.describe() for k, s in self.ex.skills.items()}
            if advisor == "scripted":
                from .scripted import ScriptedAdvisor
                self.advisor = ScriptedAdvisor(self.api_url, self.events, skills, self.scene(), executor=self.ex, on_done=self._advisor_done)
            else:
                self.advisor = make_advisor(advisor, self.api_url, self.events, skills, self.scene(), executor=self.ex,
                                            on_done=self._advisor_done, **kw)
            self.task = {"state": "running", "prompt": prompt, "advisor": advisor, "model": kw.get("model"), "recording_id": rid,
                         "viewer": self.ex.rec.viewer_url(rid), "started": time.time(), "backend": self.ex.backend.name}
            self.advisor.start(prompt)
            return self.task

    def _advisor_done(self, why: str):
        if self.advisor and self.advisor.name == "none":
            return                                   # manual task: stays open until the operator stops it
        self.task.update(state="ended", why=why, ended=time.time())
        self.ex.end_task(why)

    def stop_task(self) -> dict:
        if self.advisor:
            self.advisor.stop()
        self.ex.cancel()
        self.ex.end_task("stopped by the operator")
        self.task.update(state="ended", why="stopped by the operator", ended=time.time())
        return self.task

    # --- views ----------------------------------------------------------------------------------
    def scene(self) -> dict:
        P = self.kin.P
        return {"frame": "world: metres, origin = arm base on the table top, +z up, +x forward (scene-contract.md)",
                "T_world_base": self.kin.T.tolist(), "scene_calib": str(scene_calib_path()) if scene_calib_path() else None,
                "scene_version": self.ex.scene_version, "min_tcp_z": P.workspace.min_tcp_z, "max_tcp_z": P.workspace.max_tcp_z,
                "max_radius": P.workspace.max_radius, "table_tcp_z": P.touch.nominal_table_tcp_z,
                "park_world": self.kin.to_world(P.poses.park).round(3).tolist(), "profile": P.name, "tool": self.ex.tool,
                "layout": self.ex.layout, "setup_id": self.ex.setup_id(), "locator": self.ex.locator.name,
                "cameras": list(self.ex.backend.cameras)}

    def state(self) -> dict:
        ex = self.ex
        return {"backend": ex.backend.describe(), "busy": ex.busy(), "current": ex.current, "setup_id": ex.setup_id(),
                "tool": ex.tool, "layout": ex.layout, "task": self.task, "recording": {"id": ex.rec.id, "mode": ex.rec.mode,
                                                                                         "viewer": ex.rec.viewer_url(), "errors": ex.rec.errors[-3:]},
                "advisor": self.advisor.describe() | {"system_prompt": None} if self.advisor else None,
                "locator": ex.locator.name, "recorder_service": SERVICE_URL, "local_viewer": f"http://127.0.0.1:{self.local_rec_port}"}

    def live(self) -> dict:
        ex = self.ex
        st = ex.backend.read_state()
        m = st.motors
        q = ex.motion.q_now(m)
        tcp = ex.kin.fk(q)[0]
        # joint angles (radians, URDF joint order rev_motor_01..08) for the 3D view: measured and commanded
        q_goal = ex.kin.q_from_norm({i: (m[i]["goal"] - m[i]["lo"]) / max(1, m[i]["hi"] - m[i]["lo"]) for i in ex.kin.ids},
                                    {i: m[i]["goal"] for i in ex.kin.ids})
        return {"t": ex.now(), "q": [round(float(v), 4) for v in q], "q_goal": [round(float(v), 4) for v in q_goal], "age_s": round(ex.now() - st.t_mono, 3), "busy": ex.busy(), "current": ex.current,
                "tcp_world": tcp.round(4).tolist(), "tilt_deg": round(ex.kin.tilt_deg(q), 1),
                "joints": [{"name": ex.kin.name_of[i], "norm": m[i]["norm"], "raw": m[i]["raw"], "goal": m[i]["goal"],
                            "goal_norm": round((m[i]["goal"] - m[i]["lo"]) / max(1, m[i]["hi"] - m[i]["lo"]), 4),
                            "load": m[i]["load"], "temp": m[i]["temp"], "torque": m[i]["torque"], "status": m[i]["status"],
                            "in_range": m[i]["in_range"]} for i in ex.kin.ids]}

    def cam(self, role: str) -> bytes | None:
        hit = self._cam_cache.get(role)
        if hit and time.monotonic() - hit[0] < 0.15:
            return hit[1]
        frames = self.ex.backend.frames()
        now = time.monotonic()
        for r, f in frames.items():
            if f.jpeg:
                self._cam_cache[r] = (now, f.jpeg)
            self._frames[r] = f
            dq = self._cam_seq.setdefault(r, collections.deque(maxlen=60))
            if not dq or dq[-1][1] != f.seq:
                dq.append((now, f.seq))
            while len(dq) > 2 and now - dq[0][0] > 4.0:
                dq.popleft()
        return frames[role].jpeg if role in frames else None

    def cam_stats(self) -> dict:
        """Per camera: frames/s as they arrive here (source sequence numbers) and age of the newest frame."""
        out, now = {}, time.monotonic()
        for r, dq in self._cam_seq.items():
            fps = None
            if len(dq) >= 2 and dq[-1][0] > dq[0][0]:
                fps = round((dq[-1][1] - dq[0][1]) / (dq[-1][0] - dq[0][0]), 1)
            f = self._frames.get(r)
            out[r] = {"fps": fps, "age_s": round(self.ex.now() - f.t_capture, 2) if f else None,
                      "seen_s": round(now - dq[-1][0], 2) if dq else None}
        return out

    def estimates(self) -> list:
        """Perception's estimates on the newest frames, at most once a second (for the overlay; not an observation)."""
        t, objs = self._est
        ex = self.ex
        if time.monotonic() - t < 1.0:
            return objs
        try:
            frames = {r: f for r, f in (self._frames or ex.backend.frames()).items() if f.jpeg}
            m = ex.backend.read_state().motors
            objs = ex.locator.locate(frames, {i: m[i]["norm"] for i in ex.kin.ids}, ex.kin, ex.camera_info())
        except Exception as e:  # noqa: BLE001  (keep the last estimates; report the first failure only)
            if not t:
                self.events.add("error", text=f"overlay perception failed: {type(e).__name__}: {e}")
        if ex is self.ex:                      # the backend may have been switched meanwhile
            self._est = (time.monotonic(), objs)
        return objs

    def health(self) -> dict:
        ex, p = self.ex, self._probes
        try:
            m = ex.backend.read_state().motors
            arm = {"torque_on": sum(1 for s in m.values() if s["torque"]), "motors": len(m),
                   "calibrated": all(s["hi"] > s["lo"] for s in m.values()), "in_range": all(s["in_range"] for s in m.values()),
                   "faults": [ex.kin.name_of.get(i, i) for i, s in m.items() if s["status"]]}
        except Exception as e:  # noqa: BLE001
            arm = {"error": str(e)}
        rec = p.get("recorder")
        adv = self.advisor
        return {"station": {"ok": p.get("station"), "addr": "%s:%d" % STATION_ADDR,
                            "via_recorder": rec.get("station_reachable") if rec else None},
                "recorder": {"ok": bool(rec and rec.get("ok")), "url": SERVICE_URL, "recording": (rec or {}).get("recording"),
                             "cameras": {c.get("role"): c.get("fps") for c in (rec or {}).get("cameras", [])},
                             "local_viewer": f"http://127.0.0.1:{self.local_rec_port}", "mode": ex.rec.mode, "active": ex.rec.id},
                "cameras": self.cam_stats(), "arm": arm,
                "backend": {"name": ex.backend.name, "can_move": ex.backend.can_move,
                            "needs_enable": hasattr(ex.backend, "enable_motion")},
                "advisor": {"name": adv.name if adv else None, "running": bool(adv and adv.running),
                            "available": {"scripted": True, "none": True, "claude": bool(shutil.which("claude")),
                                          "gemini": bool(os.environ.get("GEMINI_API_KEY"))}},
                "setup_id": ex.setup_id(), "locator": ex.locator.name,
                "mirror": {"ok": p.get("mirror"), "url": TWIN_MIRROR_URL}}

    def overlay(self) -> dict:
        from .overlay import build
        return build(self)

    def executions(self) -> list[dict]:
        ex = self.ex
        out = []
        for eid, r in list(ex.results.items())[-40:]:
            p = ex.proposals.get(eid)
            out.append({**to_dict(r), "proposal": to_dict(p) if p else None, "viewer": ex.rec.viewer_url(r.recording_id)})
        return out[::-1]

    def recordings(self) -> dict:
        from recorder.store import list_recordings
        local = [{"id": s["id"], "name": s.get("name"), "started": s.get("started_iso"), "duration": s.get("duration"),
                  "markers": s.get("counts", {}).get("markers"), "viewer": f"http://127.0.0.1:{self.local_rec_port}/recordings/{s['id']}/viewer"}
                 for s in list_recordings(LOCAL_ROOT)[:30]]
        service = []
        try:
            with urllib.request.urlopen(SERVICE_URL + "/recordings", timeout=1) as r:
                service = [{"id": s["id"], "name": s.get("name"), "started": s.get("started_iso"), "duration": s.get("duration"),
                            "markers": s.get("counts", {}).get("markers"), "viewer": f"{SERVICE_URL}/recordings/{s['id']}/viewer"}
                           for s in json.loads(r.read()).get("recordings", [])[:30]]
        except (OSError, ValueError):
            pass
        return {"local": local, "service": service}

    def wait_status(self, eid: str, wait: float) -> dict:
        r = self.ex.status(eid)
        if r is None:
            raise ApiError(404, f"no execution {eid}")
        t_end = time.monotonic() + wait
        while not r.status.final and time.monotonic() < t_end:
            time.sleep(0.1)
            r = self.ex.status(eid)
        return to_dict(r)


def make_handler(c: Console):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="application/json", cache="no-store"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", cache)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, default=str).encode())

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            try:
                return json.loads(self.rfile.read(n) or b"{}") if n else {}
            except ValueError:
                raise ApiError(400, "body must be JSON")

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

        def _dispatch(self, method):
            u = urlparse(self.path)
            parts = [p for p in u.path.split("/") if p]
            q = {k: v[-1] for k, v in parse_qs(u.query).items()}
            try:
                self._route(method, parts, q)
            except ApiError as e:
                self._json({"error": str(e)}, e.code)
            except (KeyError, ValueError, TypeError) as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 400)
            except Exception as e:  # noqa: BLE001
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def _route(self, method, parts, q):
            ex = c.ex
            if method == "GET" and not parts:
                return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            if method == "GET" and len(parts) >= 2 and parts[0] in ("vendor", "robot"):   # 3D view: libraries, arm model
                try:
                    data, ctype = assets3d.vendor(parts[1]) if parts[0] == "vendor" else assets3d.robot(parts[1:])
                except assets3d.NotAvailable as e:
                    raise ApiError(404, str(e))
                return self._send(200, data, ctype, cache="max-age=3600")
            if not parts or parts[0] != "api":
                raise ApiError(404, "not found")
            r = parts[1:]
            if method == "GET":
                if r == ["state"]:
                    return self._json(c.state())
                if r == ["live"]:
                    return self._json(c.live())
                if r == ["health"]:
                    return self._json(c.health())
                if r == ["overlay"]:
                    return self._json(c.overlay())
                if r == ["estimates"]:
                    return self._json({"objects": to_dict(c.estimates()), "locator": ex.locator.name})
                if r == ["skills"]:
                    return self._json({k: s.describe() for k, s in ex.skills.items()})
                if r == ["scene"]:
                    return self._json(c.scene())
                if r == ["advisor"]:
                    return self._json(c.advisor.describe() if c.advisor else {"name": None})
                if r == ["timeline"]:
                    return self._json({"events": c.events.since(int(q.get("since", 0)), int(q.get("limit", 400))), "seq": c.events.seq})
                if r == ["executions"]:
                    return self._json({"executions": c.executions()})
                if r == ["recordings"]:
                    return self._json(c.recordings())
                if len(r) == 2 and r[0] == "status":
                    return self._json(c.wait_status(r[1], float(q.get("wait", 0))))
                if len(r) == 2 and r[0] == "cam":
                    data = c.cam(r[1].removesuffix(".jpg"))
                    return self._send(200, data, "image/jpeg") if data else self._json({"error": "no frame"}, 404)
                if len(r) == 3 and r[0] == "frame":
                    data = ex.frame_jpeg(r[1], r[2].removesuffix(".jpg"))
                    return self._send(200, data, "image/jpeg") if data else self._json({"error": "no such frame"}, 404)
                if r == ["observe"]:
                    return self._json(to_dict(ex.observe()))
                if r == ["skillcfg"]:
                    return self._json(ex.skill_config())
                if r == ["voice"]:
                    return self._json(voice.describe())
            if method == "POST":
                b = self._body()
                if r == ["observe"]:
                    return self._json(to_dict(ex.observe()))
                if r == ["submit"]:
                    if "skill" not in b:
                        raise ApiError(400, "skill is required")
                    if b.get("based_on_obs") in (None, "", "latest"):     # console form: observe now
                        o = ex.observe()
                        b = {**b, "based_on_obs": o.obs_id, "setup_id": b.get("setup_id") or o.setup_id}
                    p = proposal_from_dict({"source": self.headers.get("X-Advisor") or "operator", **b}, ex.now())
                    eid = ex.submit(p)
                    res = ex.status(eid)
                    return self._json({"execution_id": eid, "status": res.status.value, "accepted": res.status != Status.REJECTED,
                                       "reasons": res.validation.reasons if res.validation else [],
                                       "checks": res.validation.checks if res.validation else {}})
                if len(r) == 2 and r[0] == "cancel":
                    return self._json({"cancelled": ex.cancel(None if r[1] == "current" else r[1])})
                if len(r) == 2 and r[0] == "evaluate":
                    return self._json(to_dict(ex.evaluate(r[1])))
                if r == ["task", "start"]:
                    prompt = (b.get("prompt") or "").strip()
                    if not prompt:
                        raise ApiError(400, "type a task first")
                    return self._json(c.start_task(prompt, b.get("advisor", "scripted"), b.get("model"), b.get("backend")))
                if r == ["task", "stop"]:
                    return self._json(c.stop_task())
                if len(r) == 2 and r[0] == "voice":
                    try:
                        if r[1] == "token":
                            out = voice.token()
                            c.events.add("voice", role="console", text=f"voice session opened ({out['model']})")
                            return self._json(out)
                        if r[1] == "start":
                            return self._json(voice.start_by_voice(c, b))
                    except voice.VoiceError as e:
                        raise ApiError(e.code, str(e))
                    if r[1] == "say" and b.get("role") in ("you", "robot") and str(b.get("text", "")).strip():
                        c.events.add("voice", role=b["role"], text=str(b["text"]).strip()[:500])
                        return self._json({"ok": True})
                if r == ["backend"]:
                    c.switch(b["name"])
                    return self._json(c.state())
                if r == ["arm"]:
                    fn = getattr(ex.backend, "enable_motion", None)
                    if fn is None:
                        raise ApiError(400, f"backend {ex.backend.name} has no arm-enable switch (it always may move)")
                    msg = fn(bool(b.get("enabled")))
                    c.events.add("arm", text=msg, enabled=ex.backend.can_move)
                    return self._json({"message": msg, "can_move": ex.backend.can_move})
                if r == ["skillcfg", "propose"]:      # no approve route: the real config changes only by the CLI
                    return self._json(ex.propose_skill_change(b.get("changes") or {}, b.get("reason", ""),
                                                              b.get("evidence") or [], self.headers.get("X-Advisor") or "operator",
                                                              bool(b.get("twin_check"))))
                if r == ["tool"]:
                    sid = ex.set_tool(b.get("tool", "none"))
                    return self._json({"tool": ex.tool, "setup_id": sid})
            raise ApiError(404, f"no route {method} /{'/'.join(parts)}")
    return Handler


def serve(backend: str = "fake", port: int = 8910, host: str = "127.0.0.1"):
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("the console has no login: keep it on loopback")
    c = Console(backend, port)
    srv = ThreadingHTTPServer((host, port), make_handler(c))
    srv.daemon_threads = True
    print(f"executor console -> http://{host}:{port}  (backend {backend}; local recordings viewer :{c.local_rec_port})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        c.ex.cancel()
        c.ex.end_task("console stopped")
    return c


if __name__ == "__main__":
    serve(os.environ.get("EXECUTOR_BACKEND", "fake"))
