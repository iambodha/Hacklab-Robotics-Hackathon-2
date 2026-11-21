"""Read recordings back: nearest/causal lookups, timed iteration, and a drop-in state/camera source.

    rec = Recording("recordings/20261121-235900_test")
    rec.state_at(3.2); rec.frame_at("wrist", 3.2); rec.markers()
    for t, kind, payload in rec.iter(speed=2.0): ...

    src = ReplaySource.from_spec("replay:recordings/20261121-235900_test?speed=1&loop=1")
    st = await src.read_state()      # same dict shape as arm_mcp.station.read_state
    jpeg = src.frame("wrist")        # latest camera JPEG at the replay clock

No norma-core import: this runs on any machine with the recording folder.
"""
from __future__ import annotations

import bisect
import heapq
import json
import os
import time
from pathlib import Path
from urllib.parse import parse_qs

from .fmt import FOLLOWER_BUS, IDS, STATE_LEN, read_state_entry
from .store import ROOT, dir_size, find


def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:  # a recording still being written can end mid-line
                pass
    return out


def _pick(ts: list[float], t: float, causal: bool) -> int | None:
    """Index into sorted `ts` for time t: last <= t when causal (first item if t precedes all), else nearest."""
    if not ts:
        return None
    i = bisect.bisect_right(ts, t)
    if causal:
        return max(i - 1, 0)
    if i == 0:
        return 0
    if i == len(ts):
        return len(ts) - 1
    return i if ts[i] - t < t - ts[i - 1] else i - 1


class Recording:
    def __init__(self, path: str | Path):
        p = Path(path)
        if not (p / "session.json").exists():
            p = find(str(path)) or p
        if not (p / "session.json").exists():
            raise FileNotFoundError(f"no recording at {path}")
        self.path = p
        self.meta = json.loads((p / "session.json").read_text())
        self.rows = _jsonl(p / "state.jsonl")
        self.ts = [r["t"] for r in self.rows]
        self.events = _jsonl(p / "events.jsonl")
        self.commands = _jsonl(p / "commands.jsonl")
        self.cams: dict[str, dict] = {}
        for d in sorted((p / "cams").glob("*")) if (p / "cams").exists() else []:
            rows = _jsonl(d / "index.jsonl")
            info = self.meta.get("cameras", {}).get(d.name, {"hash8": d.name, "hash": d.name, "role": None})
            self.cams[d.name] = {"rows": rows, "ts": [r["t"] for r in rows], "info": info, "dir": d}

    @property
    def id(self) -> str:
        return self.meta.get("id", self.path.name)

    @property
    def duration(self) -> float:
        ends = [self.ts[-1] if self.ts else 0.0] + [c["ts"][-1] for c in self.cams.values() if c["ts"]]
        return max([float(self.meta.get("duration") or 0.0)] + ends)

    # --- lookups -------------------------------------------------------------
    def state_index(self, t: float, causal: bool = False) -> int | None:
        return _pick(self.ts, t, causal)

    def state_at(self, t: float, causal: bool = False) -> dict | None:
        """Nearest state row to t (or the last one at/before t when causal=True)."""
        i = self.state_index(t, causal)
        return None if i is None else self.rows[i]

    def cam(self, cam: str | None) -> str | None:
        """Resolve hash8, full hash or role ('wrist'/'overview') to the hash8 folder name. None = first camera."""
        if not cam:
            return next(iter(self.cams), None)
        for h8, c in self.cams.items():
            if cam.startswith(h8) or h8.startswith(cam) or c["info"].get("role") == cam:
                return h8
        return None

    def frame_index(self, cam: str | None, t: float, causal: bool = False) -> dict | None:
        h8 = self.cam(cam)
        if h8 is None:
            return None
        c = self.cams[h8]
        i = _pick(c["ts"], t, causal)
        return None if i is None else {**c["rows"][i], "cam": h8}

    def frame_bytes(self, cam: str | None, n: int) -> bytes | None:
        h8 = self.cam(cam)
        f = self.path / "cams" / h8 / f"{int(n)}.jpg" if h8 else None
        return f.read_bytes() if f and f.exists() else None

    def frame_at(self, cam: str | None, t: float, causal: bool = False) -> tuple[dict, bytes] | None:
        """(index row, JPEG bytes) of the nearest frame (or last at/before t when causal=True)."""
        row = self.frame_index(cam, t, causal)
        if row is None:
            return None
        data = self.frame_bytes(row["cam"], row["n"])
        return (row, data) if data is not None else None

    def markers(self) -> list[dict]:
        return [e for e in self.events if e.get("type") == "marker"]

    def rows_between(self, frm: float | None = None, to: float | None = None, every: float | None = None) -> list[dict]:
        lo = bisect.bisect_left(self.ts, frm) if frm is not None else 0
        hi = bisect.bisect_right(self.ts, to) if to is not None else len(self.ts)
        out, last = [], None
        for r in self.rows[lo:hi]:
            if every and last is not None and r["t"] - last < every - 1e-9:
                continue
            out.append(r)
            last = r["t"]
        return out

    # --- iteration -------------------------------------------------------------
    def iter(self, speed: float = 1.0, realtime: bool = True, start: float = 0.0, end: float | None = None):
        """Yield (t, kind, payload) in time order. kind: 'state' (row), 'frame' (index row with 'cam'),
        'event' (event), 'command' (command). realtime=True sleeps so items come out at t/speed."""
        def src(kind, rows, extra=None):
            for i, r in enumerate(rows):
                if r["t"] >= start and (end is None or r["t"] <= end):
                    yield r["t"], kind, i, ({**r, **extra} if extra else r)
        by_t = lambda rows: sorted(rows, key=lambda r: r["t"])  # noqa: E731  (markers may be posted out of order)
        streams = [src("state", self.rows), src("event", by_t(self.events)), src("command", by_t(self.commands))]
        streams += [src("frame", c["rows"], {"cam": h8}) for h8, c in self.cams.items()]
        c0 = time.monotonic()
        for t, kind, _, payload in heapq.merge(*streams, key=lambda x: (x[0], x[1], x[2])):
            if realtime:
                delay = c0 + (t - start) / speed - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            yield t, kind, payload

    # --- summary -------------------------------------------------------------
    def stats(self) -> dict:
        dur = self.duration or 0.0
        span = (self.ts[-1] - self.ts[0]) if len(self.ts) > 1 else 0.0
        cams = {}
        for h8, c in self.cams.items():
            rows, cspan = c["rows"], (c["ts"][-1] - c["ts"][0]) if len(c["ts"]) > 1 else 0.0
            gaps = [b - a for a, b in zip(c["ts"], c["ts"][1:])]
            lat = sorted(r["latency"] for r in rows if r.get("latency") is not None)
            info = c["info"]
            cams[h8] = {"role": info.get("role"), "frames": len(rows),
                        "fps": round((len(rows) - 1) / cspan, 2) if cspan else 0.0,
                        "dropped": info.get("dropped", 0), "late": info.get("late", 0),
                        "max_gap_s": round(max(gaps), 3) if gaps else None,
                        "latency_p50_s": lat[len(lat) // 2] if lat else None,
                        "latency_max_s": lat[-1] if lat else None,
                        "bytes": sum(r.get("bytes", 0) for r in rows)}
        counts = self.meta.get("counts", {})
        return {"duration": round(dur, 3), "state_rows": len(self.rows),
                "state_hz": round((len(self.rows) - 1) / span, 2) if span else 0.0,
                "station_state_hz": round(counts.get("state_frames_seen", 0) / dur, 1) if dur else None,
                "events": len(self.events), "markers": len(self.markers()), "commands": len(self.commands),
                "cameras": cams, "size_bytes": dir_size(self.path)}


class ReplaySource:
    """Replay clock over a Recording that answers like the live Station helpers.

    read_state() returns {motor_id: {raw, goal, lo, hi, norm, in_range, torque, status, temp, volt, load}}
    exactly as arm_mcp.station.read_state does, from the recorded register dumps. Lookups are causal:
    the source never shows data from after the replay clock.
    """

    def __init__(self, rec: Recording | str | Path, speed: float = 1.0, start: float = 0.0, loop: bool = False,
                 bus: str | None = None, clock=time.monotonic):
        self.rec = rec if isinstance(rec, Recording) else Recording(rec)
        self.speed, self.loop, self.bus, self._clock = float(speed), bool(loop), bus or FOLLOWER_BUS, clock
        self._base, self._c0, self._paused = float(start), clock(), False

    @classmethod
    def from_spec(cls, spec: str, **kw) -> "ReplaySource":
        """'replay:<path or recording id>[?speed=2&start=5&loop=1&bus=SERIAL]'"""
        body = spec.removeprefix("replay:")
        path, _, query = body.partition("?")
        q = {k: v[-1] for k, v in parse_qs(query).items()}
        opts = {"speed": float(q.get("speed", 1.0)), "start": float(q.get("start", 0.0)),
                "loop": q.get("loop", "0") in ("1", "true", "yes"), "bus": q.get("bus")}
        p = Path(path)
        if not (p / "session.json").exists():
            p = find(path) or ROOT / path
        return cls(p, **{**opts, **kw})

    @classmethod
    def from_env(cls, var: str = "ROBOT_SOURCE") -> "ReplaySource | None":
        spec = os.environ.get(var, "")
        return cls.from_spec(spec) if spec.startswith("replay:") else None

    # --- clock -----------------------------------------------------------------
    def now(self) -> float:
        t = self._base if self._paused else self._base + (self._clock() - self._c0) * self.speed
        dur = self.rec.duration
        if self.loop and dur > 0:
            return t % dur
        return max(0.0, min(t, dur))

    @property
    def done(self) -> bool:
        return not self.loop and not self._paused and self._base + (self._clock() - self._c0) * self.speed >= self.rec.duration

    def seek(self, t: float):
        self._base, self._c0 = float(t), self._clock()

    def pause(self):
        if not self._paused:
            self._base, self._paused = self.now(), True

    def resume(self):
        if self._paused:
            self._c0, self._paused = self._clock(), False

    def set_speed(self, speed: float):
        self._base, self._c0, self.speed = self.now(), self._clock(), float(speed)

    # --- data --------------------------------------------------------------------
    def row(self) -> dict | None:
        return self.rec.state_at(self.now(), causal=True)

    def read_state_sync(self) -> dict:
        row = self.row()
        bus = (row or {}).get("buses", {}).get(self.bus) or {}
        out = {}
        for k, m in bus.items():
            if not isinstance(m, dict) or "regs" not in m:
                continue
            s = bytes.fromhex(m["regs"])
            if len(s) < STATE_LEN or m.get("error"):
                out = None
                break
            out[int(m["id"])] = read_state_entry(s, int(m["range_min"]), int(m["range_max"]))
        if out and len(out) == len(IDS):
            return dict(sorted(out.items()))
        raise RuntimeError(f"motor bus {self.bus}: no complete error-free read at replay t={self.now():.3f}")

    async def read_state(self, cli=None) -> dict:
        """Drop-in for arm_mcp.station.read_state(cli); `cli` is ignored."""
        return self.read_state_sync()

    def frame(self, cam: str | None = None) -> bytes | None:
        """Latest JPEG at or before the replay clock for a camera (hash8, hash or role; default first)."""
        hit = self.rec.frame_at(cam, self.now(), causal=True)
        return hit[1] if hit else None

    def frames(self) -> dict[str, bytes]:
        return {h8: f for h8 in self.rec.cams if (f := self.frame(h8)) is not None}

    def cameras(self) -> list[dict]:
        """Same fields as scripts/cam_stream.py /api/cameras."""
        t, out = self.now(), []
        for h8, c in self.rec.cams.items():
            row = self.rec.frame_index(h8, t, causal=True)
            info = c["info"]
            age = round((t - row["t"]) / max(self.speed, 1e-9), 3) if row else None
            out.append({"hash": info.get("hash", h8), "label": info.get("role") or h8, "unique_id": "",
                        "w": info.get("w"), "h": info.get("h"),
                        "fps": round((len(c["ts"]) - 1) / (c["ts"][-1] - c["ts"][0]), 1) if len(c["ts"]) > 1 and c["ts"][-1] > c["ts"][0] else 0.0,
                        "age": age, "live": age is not None and age < 5})
        return out
