"""Writes one recording directory (see README.md for the format) and lists recordings."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from .fmt import FORMAT_VERSION, NAMES, cam_role, slug

ROOT = Path(os.environ.get("RECORDER_DIR", Path(__file__).resolve().parent.parent / "recordings"))
LATE_S = 0.25  # camera frame counted as late when it arrives this long after Station stamped it


def _dumps(o) -> str:
    return json.dumps(o, separators=(",", ":"))


class SessionWriter:
    """Thread-safe writer: the capture thread writes rows/frames, the HTTP thread adds markers/stops."""

    def __init__(self, name: str, note: str = "", root: Path | None = None, station: str = "",
                 extra: dict | None = None):
        root = Path(root or ROOT)
        self.t0, self.wall0 = time.monotonic(), time.time()
        self.id = f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(self.wall0))}_{slug(name)}"
        self.dir = root / self.id
        n = 1
        while self.dir.exists():
            n += 1
            self.dir = root / f"{self.id}-{n}"
        self.id = self.dir.name
        self.dir.mkdir(parents=True)
        self.lock = threading.Lock()
        self.closed = False
        self.meta = {"id": self.id, "name": name, "note": note, "source": "real", "format_version": FORMAT_VERSION,
                     "started": self.wall0, "started_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.wall0)),
                     "stopped": None, "duration": None, "station": station, "joints": NAMES,
                     "buses": {}, "cameras": {}, "counts": {"state_rows": 0, "state_frames_seen": 0, "events": 0,
                                                            "markers": 0, "commands": 0}, **(extra or {})}
        self._state = open(self.dir / "state.jsonl", "a", buffering=1 << 16)
        self._events = open(self.dir / "events.jsonl", "a", buffering=1)
        self._commands = open(self.dir / "commands.jsonl", "a", buffering=1)
        self._cams: dict[str, dict] = {}
        self._last_meta = 0.0
        self.write_meta()

    def now(self, mono: float | None = None) -> float:
        return round((time.monotonic() if mono is None else mono) - self.t0, 4)

    def write_meta(self):
        tmp = self.dir / "session.json.tmp"
        tmp.write_text(json.dumps(self.meta, indent=2))
        tmp.replace(self.dir / "session.json")
        self._last_meta = time.monotonic()

    def _maybe_meta(self):
        if time.monotonic() - self._last_meta > 2:
            for f in (self._state, *(c["f"] for c in self._cams.values())):
                f.flush()
            self.write_meta()

    def state(self, row: dict, mono: float, frames_seen: int = 1):
        with self.lock:
            if self.closed:
                return
            c = self.meta["counts"]
            c["state_rows"] += 1
            c["state_frames_seen"] += frames_seen
            for serial, b in row.get("buses", {}).items():
                info = self.meta["buses"].setdefault(serial, {"role": b.get("role"), "motor_ids": []})
                ids = sorted(m["id"] for k, m in b.items() if isinstance(m, dict) and "id" in m)
                info["motor_ids"] = sorted(set(info["motor_ids"]) | set(ids))
            self._state.write(_dumps({"t": self.now(mono), **row}) + "\n")
            self._maybe_meta()

    def frame(self, qhash: str, jpeg: bytes, mono: float, wall: float, info: dict):
        """info: station_ns, station_mono_ns, index, w, h."""
        with self.lock:
            if self.closed:
                return
            h8 = qhash[:8]
            cam = self._cams.get(h8)
            if cam is None:
                d = self.dir / "cams" / h8
                d.mkdir(parents=True, exist_ok=True)
                cam = self._cams[h8] = {"dir": d, "f": open(d / "index.jsonl", "a", buffering=1 << 14), "n": 0,
                                        "last_index": None}
                self.meta["cameras"][h8] = {"hash": qhash, "hash8": h8, "role": cam_role(qhash), "frames": 0,
                                            "dropped": 0, "late": 0, "bytes": 0, "w": info.get("w"), "h": info.get("h")}
            m = self.meta["cameras"][h8]
            idx = info.get("index")
            if idx is not None and cam["last_index"] is not None and idx > cam["last_index"] + 1:
                m["dropped"] += idx - cam["last_index"] - 1
            cam["last_index"] = idx if idx is not None else cam["last_index"]
            sns = info.get("station_ns")
            latency = round(wall - sns / 1e9, 4) if sns else None
            if latency is not None and latency > LATE_S:
                m["late"] += 1
            n = cam["n"]
            (cam["dir"] / f"{n}.jpg").write_bytes(jpeg)
            cam["f"].write(_dumps({"n": n, "t": self.now(mono), "wall": round(wall, 4), "latency": latency,
                                   "bytes": len(jpeg), **info}) + "\n")
            cam["n"] += 1
            m["frames"] += 1
            m["bytes"] += len(jpeg)
            self._maybe_meta()

    def event(self, type_: str, mono: float | None = None, **data) -> dict:
        ev = {"t": self.now(mono), "wall": round(time.time(), 4), "type": type_, **data}
        with self.lock:
            if self.closed:
                return ev
            self.meta["counts"]["events"] += 1
            if type_ == "marker":
                self.meta["counts"]["markers"] += 1
            self._events.write(_dumps(ev) + "\n")
        return ev

    def command(self, rec: dict, mono: float):
        with self.lock:
            if self.closed:
                return
            self.meta["counts"]["commands"] += 1
            self._commands.write(_dumps({"t": self.now(mono), **rec}) + "\n")

    def close(self) -> dict:
        self.event("stop")
        with self.lock:
            if self.closed:
                return self.meta
            self.closed = True
            stopped = time.time()
            self.meta.update(stopped=stopped, duration=round(time.monotonic() - self.t0, 3))
            for f in (self._state, self._events, self._commands, *(c["f"] for c in self._cams.values())):
                f.close()
            self.meta["size_bytes"] = dir_size(self.dir)
            self.write_meta()
            return self.meta


def dir_size(d: Path) -> int:
    return sum(p.stat().st_size for p in Path(d).rglob("*") if p.is_file())


def list_recordings(root: Path | None = None) -> list[dict]:
    root = Path(root or ROOT)
    out = []
    for f in sorted(root.glob("*/session.json"), reverse=True):
        try:
            out.append(json.loads(f.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def find(rec_id: str, root: Path | None = None) -> Path | None:
    root = Path(root or ROOT)
    if (root / rec_id / "session.json").exists():
        return root / rec_id
    hits = sorted(p.parent for p in root.glob(f"*{rec_id}*/session.json"))
    return hits[-1] if len(hits) == 1 else None
