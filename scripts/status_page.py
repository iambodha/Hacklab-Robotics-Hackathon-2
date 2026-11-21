"""Live status page for the whole robot setup: http://127.0.0.1:8891

Shows, refreshing every 2 s: this laptop's network address, every service (Station, camera relay,
robot_mcp, arm_mcp, the Station forwarder) up or down, both cameras, every follower and leader
joint (position, range, torque, current, load, temperature), teleop state, MCP call count, who has
connected to the exposed Station ports, disk space and Station's storage.

Read-only: it listens to Station's state stream and reads log files. It never sends commands.

  python3 scripts/status_page.py            # this laptop only
  python3 scripts/status_page.py --lan      # also reachable from other laptops (no password; read-only info)
"""
import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import socket
import struct
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RUN = BASE / "station-run"
NORMA = Path(os.environ.get("NORMA_CORE_REPO", BASE / "norma-core"))
for p in (NORMA / "software/station/shared", NORMA):
    sys.path.insert(0, str(p))
from station_py import new_station_client  # noqa: E402
from target.gen_python.protobuf.drivers.st3215 import st3215  # noqa: E402

FOLLOWER, LEADER = "5B61033301", "5B61035034"
STATE = {"buses": {}, "stamp": 0.0, "error": None}
SLOW = {"stamp": 0.0}


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def port_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=0.4):
            return True
    except OSError:
        return False


def http_json(url, timeout=1.5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def tail(path, nbytes=3_000_000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - nbytes))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def u16(b, a):
    return struct.unpack_from("<H", b, a)[0]


async def follow_station():
    log = logging.getLogger("status")
    while True:
        try:
            cli = await new_station_client("localhost", log)
            q = asyncio.Queue(maxsize=2)
            cli.follow("st3215/inference", q)
            while True:
                e = await asyncio.wait_for(q.get(), 5)
                buses = {}
                for bus in st3215.InferenceStateReader(memoryview(bytes(e.Data))).get_buses() or []:
                    serial = str(bus.get_bus().get_serial_number())
                    rows = []
                    for m in bus.get_motors() or []:
                        s = bytes(m.get_state() or b"")
                        if len(s) < 0x47:
                            continue
                        raw = u16(s, 0x38)
                        raw = (4096 - (raw & 4095)) & 4095 if raw & 0x8000 else raw & 4095
                        lo, hi = int(m.get_range_min()), int(m.get_range_max())
                        rows.append({"id": int(m.get_id()), "raw": raw, "goal": u16(s, 0x2A) & 4095, "lo": lo, "hi": hi,
                                     "norm": round((raw - lo) / (hi - lo), 3) if hi > lo else None,
                                     "in_range": lo <= raw <= hi, "frozen": bool(m.get_range_freezed()),
                                     "torque": s[0x28], "mA": u16(s, 0x45) & 0x7FFF, "load": u16(s, 0x3C) & 0x3FF,
                                     "temp": s[0x3F], "volt": s[0x3E] / 10, "status": s[0x41]})
                    buses[serial] = sorted(rows, key=lambda r: r["id"])
                STATE.update(buses=buses, stamp=time.time(), error=None)
        except Exception as ex:
            STATE["error"] = f"{type(ex).__name__}: {ex}"
            await asyncio.sleep(2)


def slow_facts():
    """Things that are expensive to compute: refreshed at most every 20 s."""
    if time.time() - SLOW["stamp"] < 20:
        return SLOW
    data = RUN / "station_data"
    total = sum(f.stat().st_size for f in data.rglob("*") if f.is_file()) if data.exists() else 0
    slog = tail(RUN / "station.log")
    mirror = re.findall(r"^\[(\S+)Z INFO  motors_mirroring\] (Starting|Stopping) mirroring", slog, re.M)
    calib = re.findall(r"^\[(\S+)Z INFO .*(Calibration completed successfully|Freeze calibration command completed|"
                       r"Starting auto-calibration sequence)", slog, re.M)
    SLOW.update(stamp=time.time(),
                disk_free_gb=round(shutil.disk_usage(BASE).free / 1e9, 1),
                station_data_gb=round(total / 1e9, 2),
                footage_deletions=len(re.findall(r"disk_monitor\] Deleted store file", slog)),
                teleop=("on" if mirror[-1][1] == "Starting" else "off") if mirror else "unknown",
                teleop_since=mirror[-1][0] if mirror else None,
                last_calibration=(calib[-1][0] + " " + calib[-1][1]) if calib else None)
    return SLOW


def status():
    ip = lan_ip()
    fwd = {}
    for line in tail(RUN / "lan_forward.log", 2_000_000).splitlines():
        m = re.match(r"^(\d+\.\d+\.\d+\.\d+) -> :(\d+)", line)
        if m and m.group(1) != ip:
            fwd[f"{m.group(1)} :{m.group(2)}"] = fwd.get(f"{m.group(1)} :{m.group(2)}", 0) + 1
    cams = http_json("http://127.0.0.1:8890/api/cameras") or []
    services = [
        ("Station API", "127.0.0.1:8888", port_open("127.0.0.1", 8888), "this laptop"),
        ("Station web UI", "127.0.0.1:8889", port_open("127.0.0.1", 8889), "this laptop"),
        ("Station forwarder (API)", f"{ip}:8888", port_open(ip, 8888), "network, no password"),
        ("Station forwarder (web UI)", f"{ip}:8889", port_open(ip, 8889), "network, no password"),
        ("Camera relay", "127.0.0.1:8890", bool(cams), "this laptop"),
        ("robot_mcp", "127.0.0.1:8765", bool(http_json("http://127.0.0.1:8765/health")), "this laptop"),
        ("arm_mcp", f"{ip}:8766/mcp", bool(http_json(f"http://{ip}:8766/health")), "network, token"),
    ]
    amlog = tail(RUN / "arm_mcp.log", 1_000_000)
    return {"now": time.strftime("%H:%M:%S"), "ip": ip,
            "services": [{"name": n, "addr": a, "up": u, "scope": s} for n, a, u, s in services],
            "cameras": [{"hash": c["hash"][:8], "fps": c["fps"], "age": c["age"], "live": c["live"]} for c in cams],
            "state_age": round(time.time() - STATE["stamp"], 1) if STATE["stamp"] else None,
            "state_error": STATE["error"], "buses": STATE["buses"],
            "mcp_calls": amlog.count("CallToolRequest"), "forward_connections": fwd, **{k: v for k, v in slow_facts().items() if k != "stamp"}}


PAGE = rb"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Robot status</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1d2430;--mut:#667085;--ok:#12805c;--bad:#c0362c;--warn:#b25e09;--line:#e4e7ec}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--card:#1a1f26;--fg:#e6e9ee;--mut:#98a2b3;--ok:#3ccf91;--bad:#ff6b5e;--warn:#f0a43a;--line:#2a313b}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif;padding:16px}
h1{font-size:18px;margin:0 0 4px}h2{font-size:14px;margin:0 0 8px;color:var(--mut);font-weight:600}
.sub{color:var(--mut);margin-bottom:14px}.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:3px 6px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--mut);font-weight:600}.ok{color:var(--ok)}.bad{color:var(--bad);font-weight:600}.warn{color:var(--warn);font-weight:600}
.wide{grid-column:1/-1}</style>
<h1>Robot status</h1><div class=sub id=sub>loading...</div>
<div class=grid><div class=card><h2>Services</h2><table id=svc></table></div>
<div class=card><h2>Arm, cameras, storage</h2><table id=facts></table></div>
<div class="card wide"><h2>Follower 5B61033301</h2><table id=fol></table></div>
<div class="card wide"><h2>Leader 5B61035034</h2><table id=lea></table></div>
<div class=card><h2>Connections to the exposed Station ports</h2><table id=fwd></table></div></div>
<script>
const $=id=>document.getElementById(id), esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function joints(rows){ if(!rows||!rows.length) return '<tr><td class=bad>no motors reported</td></tr>';
 let h='<tr><th>joint</th><th>pos</th><th>norm</th><th>range</th><th>torque</th><th>mA</th><th>load</th><th>temp &deg;C</th><th>V</th></tr>';
 for(const r of rows){ const t=r.temp>=55?'bad':r.temp>=45?'warn':''; const rg=r.in_range?'':'bad';
  h+=`<tr><td>j${r.id}</td><td>${r.raw}</td><td>${r.norm??'-'}</td><td class=${rg}>${r.lo}-${r.hi}${r.frozen?'':' (not frozen)'}</td><td>${r.torque?'on':'off'}</td><td>${r.mA}</td><td>${r.load}</td><td class=${t}>${r.temp}</td><td>${r.volt}</td></tr>`;}
 return h;}
async function tick(){ let s; try{ s=await (await fetch('/api/status')).json(); }catch(e){ $('sub').textContent='status page lost contact'; return; }
 $('sub').innerHTML=`Laptop address <b>${s.ip}</b> &middot; updated ${s.now}`+(s.state_error?` &middot; <span class=bad>Station: ${esc(s.state_error)}</span>`:'');
 $('svc').innerHTML='<tr><th>service</th><th>address</th><th>reach</th><th></th></tr>'+s.services.map(x=>`<tr><td>${x.name}</td><td>${x.addr}</td><td>${x.scope}</td><td class=${x.up?'ok':'bad'}>${x.up?'up':'DOWN'}</td></tr>`).join('');
 const cams=s.cameras.map(c=>`${c.hash}: <span class=${c.live?'ok':'bad'}>${c.live?c.fps+' fps':'stale '+c.age+' s'}</span>`).join('<br>')||'<span class=bad>none</span>';
 const f=[['Arm state age', s.state_age==null?'<span class=bad>none</span>':(s.state_age>3?`<span class=bad>${s.state_age} s</span>`:`${s.state_age} s`)],
  ['Teleop', `${s.teleop}${s.teleop_since?' (since '+s.teleop_since.slice(11,19)+' UTC)':''}`],['Last calibration event', s.last_calibration?esc(s.last_calibration.replace('T',' ')):'-'],
  ['Cameras', cams],['MCP tool calls (arm_mcp log)', s.mcp_calls],['Disk free', s.disk_free_gb<2?`<span class=bad>${s.disk_free_gb} GB</span>`:s.disk_free_gb<5?`<span class=warn>${s.disk_free_gb} GB</span>`:s.disk_free_gb+' GB'],
  ['Station data', s.station_data_gb+' GB'],['Old footage chunks deleted', s.footage_deletions]];
 $('facts').innerHTML=f.map(([k,v])=>`<tr><th>${k}</th><td>${v}</td></tr>`).join('');
 $('fol').innerHTML=joints(s.buses['5B61033301']); $('lea').innerHTML=joints(s.buses['5B61035034']);
 const fw=Object.entries(s.forward_connections); $('fwd').innerHTML=fw.length?'<tr><th>from</th><th>connections</th></tr>'+fw.map(([k,v])=>`<tr><td>${k}</td><td>${v}</td></tr>`).join(''):'<tr><td>none from other machines</td></tr>';}
setInterval(tick,2000); tick();
</script>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/api/status":
            body, ctype = json.dumps(status()).encode(), "application/json"
        elif self.path in ("/", "/index.html"):
            body, ctype = PAGE, "text/html; charset=utf-8"
        else:
            return self.send_error(404)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser(description="Read-only live status page for the robot setup")
    ap.add_argument("--lan", action="store_true", help="listen on all interfaces")
    ap.add_argument("--port", type=int, default=8891)
    a = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)
    threading.Thread(target=lambda: asyncio.run(follow_station()), daemon=True).start()
    host = "0.0.0.0" if a.lan else "127.0.0.1"
    print(f"status page -> http://{lan_ip() if a.lan else '127.0.0.1'}:{a.port}", flush=True)
    ThreadingHTTPServer((host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
