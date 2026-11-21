"""python -m recorder serve | record --name X --seconds N | ls | show <id>"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request

from .replay import Recording
from .store import list_recordings

API = os.environ.get("RECORDER_URL", "http://127.0.0.1:8900")


def _api(method: str, path: str, body: dict | None = None, timeout: float = 3):
    req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _mb(n: int) -> str:
    return f"{n / 1e6:.1f} MB"


def cmd_record(a):
    try:
        if not a.local:
            _api("GET", "/health", timeout=1)
            rid = _api("POST", "/recordings", {"name": a.name, "note": a.note})["id"]
            print(f"recording {rid} via {API} for {a.seconds:g} s")
            try:
                time.sleep(a.seconds)
            finally:
                _api("POST", f"/recordings/{rid}/stop")
            return show(rid)
    except urllib.error.HTTPError as e:
        sys.exit(f"recorder API refused: {e.code} {e.read().decode(errors='replace')}")
    except (urllib.error.URLError, OSError):
        print(f"no recorder server at {API}; capturing directly from Station")
    from .live import LiveCapture
    from .server import Service

    cap = LiveCapture(host=a.station, state_hz=a.state_hz).start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and (cap.latest() is None or any(c["frames"] == 0 for c in cap.snapshot()["cameras"])
                                           or not cap.snapshot()["cameras"]):
        time.sleep(0.2)
    svc = Service(capture=cap)
    meta = svc.start(a.name, a.note)
    print(f"recording {meta['id']} for {a.seconds:g} s")
    try:
        time.sleep(a.seconds)
    finally:
        svc.stop(meta["id"])
    show(meta["id"])


def show(rec_id: str):
    rec = Recording(rec_id)
    s = rec.stats()
    m = rec.meta
    print(f"{rec.id}  ({rec.path})")
    print(f"  name={m.get('name')!r} note={m.get('note')!r} started={m.get('started_iso')} duration={s['duration']} s")
    buses = ", ".join(f"{k} ({v.get('role')})" for k, v in m.get("buses", {}).items()) or "none"
    print(f"  buses: {buses}")
    print(f"  state: {s['state_rows']} rows @ {s['state_hz']} Hz (Station published {s['station_state_hz']} frames/s)")
    for h8, c in s["cameras"].items():
        print(f"  cam {h8} ({c['role']}): {c['frames']} frames @ {c['fps']} fps, dropped {c['dropped']}, late {c['late']}, "
              f"max gap {c['max_gap_s']} s, latency p50 {c['latency_p50_s']} s max {c['latency_max_s']} s, {_mb(c['bytes'])}")
    print(f"  events {s['events']} (markers {s['markers']}), commands {s['commands']}, size on disk {_mb(s['size_bytes'])}")
    for e in rec.events:
        if e["type"] not in ("start", "stop"):
            print(f"    {e['t']:8.3f}s {e['type']} {e.get('label', '')}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m recorder")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="HTTP API + live capture")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8900)
    s.add_argument("--station", default=None, help="Station host[:port] (default STATION_HOST or localhost)")
    s.add_argument("--state-hz", type=float, default=50.0, help="max state rows/s written (0 = every Station frame)")
    r = sub.add_parser("record", help="record N seconds (through a running server if there is one)")
    r.add_argument("--name", default="rec")
    r.add_argument("--note", default="")
    r.add_argument("--seconds", type=float, default=10.0)
    r.add_argument("--local", action="store_true", help="capture in this process even if a server is running")
    r.add_argument("--station", default=None)
    r.add_argument("--state-hz", type=float, default=50.0)
    sub.add_parser("ls", help="list recordings")
    sh = sub.add_parser("show", help="summary of one recording")
    sh.add_argument("id")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("recorder.station").setLevel(logging.CRITICAL)
    if a.cmd == "serve":
        from .server import serve
        serve(a.host, a.port, a.state_hz, a.station)
    elif a.cmd == "record":
        cmd_record(a)
    elif a.cmd == "ls":
        for m in list_recordings():
            c = m.get("counts", {})
            print(f"{m['id']:40} {m.get('duration') or '-':>8} s  rows {c.get('state_rows', 0):>6}  "
                  f"cams {len(m.get('cameras', {}))}  markers {c.get('markers', 0)}  {m.get('note', '')}")
    elif a.cmd == "show":
        show(a.id)


if __name__ == "__main__":
    main()
