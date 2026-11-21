"""Every task runs inside a recorder session; proposals, validations, phase boundaries, results and outcomes
are markers in it.

mode "service": the recorder HTTP API (http://127.0.0.1:8900, records the real Station: state, both cameras,
                commands). Used for the real backend when it is up.
mode "local":   recorder.store.SessionWriter in <repo>/recordings (same format, same viewer). Used for fake / sim /
                replay runs (and as the fallback); it also stores the backend's state and frames at a low rate so
                the recorder viewer can replay a sim run. Served by a second recorder API on :8911.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SERVICE_URL = os.environ.get("RECORDER_URL", "http://127.0.0.1:8900")
LOCAL_ROOT = Path(os.environ.get("EXECUTOR_RECORDINGS", REPO / "recordings"))
LOCAL_VIEWER = os.environ.get("EXECUTOR_LOCAL_RECORDER_URL", "http://127.0.0.1:8911")


def _http(method: str, url: str, body: dict | None = None, timeout: float = 2.0):
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def service_up(url: str = SERVICE_URL) -> bool:
    try:
        return bool(_http("GET", url + "/health", timeout=1.0).get("ok"))
    except (OSError, ValueError):
        return False


class RecorderLink:
    def __init__(self, mode: str = "local", url: str = SERVICE_URL, root: Path = LOCAL_ROOT, frame_hz: float = 2.0,
                 state_hz: float = 10.0):
        self.mode, self.url, self.root = mode, url, Path(root)
        self.frame_hz, self.state_hz = frame_hz, state_hz
        self.id: str | None = None
        self.writer = None
        self.owned = False
        self.lock = threading.Lock()
        self._last_frame: dict[str, float] = {}
        self._last_state = 0.0
        self.errors: list[str] = []

    @property
    def active(self) -> bool:
        return self.id is not None

    def viewer_url(self, rec_id: str | None = None) -> str | None:
        rid = rec_id or self.id
        if not rid:
            return None
        base = self.url if self.mode == "service" else LOCAL_VIEWER
        return f"{base}/recordings/{rid}/viewer"

    def start(self, name: str, note: str = "") -> str:
        with self.lock:
            if self.id:
                return self.id
            if self.mode == "service":
                try:
                    self.id = _http("POST", self.url + "/recordings", {"name": name, "note": note})["id"]
                    self.owned = True
                    return self.id
                except urllib.error.HTTPError as e:
                    if e.code == 409:      # someone else is recording: add our markers to theirs
                        self.id = _http("GET", self.url + "/recordings").get("active")
                        self.owned = False
                        if self.id:
                            return self.id
                    self.errors.append(f"recorder service: {e}")
                except (OSError, ValueError, KeyError) as e:
                    self.errors.append(f"recorder service unreachable ({e}); recording locally")
                self.mode = "local"
            from recorder.store import SessionWriter
            self.root.mkdir(parents=True, exist_ok=True)
            self.writer = SessionWriter(name, note, root=self.root, station="executor",
                                        extra={"source": "executor", "state_hz": self.state_hz})
            self.writer.event("start", name=name, note=note)
            self.id, self.owned = self.writer.id, True
            return self.id

    def marker(self, label: str, data: dict | None = None) -> str:
        if not self.id:
            return ""
        data = json.loads(json.dumps(data or {}, default=str))
        try:
            if self.mode == "service":
                ev = _http("POST", f"{self.url}/recordings/{self.id}/markers", {"label": label, "data": data})["event"]
            else:
                ev = self.writer.event("marker", label=label, data=data)
            return f"{self.id}@{ev['t']:.3f}:{label}"
        except (OSError, ValueError, KeyError) as e:
            self.errors.append(f"marker {label}: {e}")
            return ""

    def sample(self, state: dict | None = None, frames: dict | None = None, now: float | None = None) -> None:
        """Local mode only: keep a low-rate copy of the backend's state and frames in the session."""
        if self.mode != "local" or self.writer is None:
            return
        now = time.monotonic() if now is None else now
        if state and now - self._last_state >= 1.0 / self.state_hz:
            self._last_state = now
            ids = sorted(state)
            row = {"wall": time.time(), "buses": {}, "norm": [state[m]["norm"] for m in ids],
                   "goal": [round((state[m]["goal"] - state[m]["lo"]) / max(1, state[m]["hi"] - state[m]["lo"]), 4) for m in ids],
                   "load": [state[m]["load"] for m in ids], "status": [state[m]["status"] for m in ids],
                   "torque": [state[m]["torque"] for m in ids]}
            self.writer.state(row, time.monotonic())
        for role, f in (frames or {}).items():
            if f.jpeg and now - self._last_frame.get(role, 0.0) >= 1.0 / self.frame_hz:
                self._last_frame[role] = now
                h = (role + "0" * 32)[:32] if not f.queue.startswith("usbvideo/") else f.queue.split("/", 1)[1]
                self.writer.frame(h, f.jpeg, time.monotonic(), time.time(), {"index": f.seq, "w": 299, "h": 224})
                self.writer.meta["cameras"][h[:8]]["role"] = role

    def stop(self) -> None:
        with self.lock:
            if not self.id:
                return
            try:
                if self.mode == "service" and self.owned:
                    _http("POST", f"{self.url}/recordings/{self.id}/stop", {})
                elif self.writer is not None:
                    self.writer.close()
            except (OSError, ValueError) as e:
                self.errors.append(f"stop: {e}")
            self.id, self.writer = None, None


class TableMemory:
    """Gripper heights at which the fingertips touched the table, KEYED BY setup_id. A changed setup (new
    calibration, ranges, tool, layout) has no memory: its picks use the nominal table until felt again."""

    def __init__(self, path: Path | None, touch, keep_setups: int = 6):
        self.path, self.TS, self.keep_setups = path, touch, keep_setups
        try:
            self.data: dict[str, list] = json.loads(path.read_text()) if path and path.exists() else {}
        except (OSError, ValueError):
            self.data = {}
        if not isinstance(self.data, dict):     # old un-keyed list (sim/real_sessions/table_touch.json): discard
            self.data = {}

    def lookup(self, setup_id: str, x: float, y: float) -> float | None:
        TS = self.TS
        near = [(float(((t[0] - x) ** 2 + (t[1] - y) ** 2) ** 0.5), t[2]) for t in self.data.get(setup_id, [])]
        near = [n for n in near if n[0] <= TS.reuse_radius and abs(n[1] - TS.nominal_table_tcp_z) <= TS.plausible]
        return min(near)[1] if near else None

    def remember(self, setup_id: str, x: float, y: float, z: float) -> bool:
        if abs(z - self.TS.nominal_table_tcp_z) > self.TS.plausible:
            return False
        lst = self.data.setdefault(setup_id, [])
        lst.append([round(x, 4), round(y, 4), round(z, 4)])
        del lst[:-self.TS.keep]
        for k in list(self.data)[:-self.keep_setups]:
            self.data.pop(k)
        if self.path:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(self.data))
            except OSError:
                pass
        return True
