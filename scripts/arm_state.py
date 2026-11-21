"""Print a snapshot of every ST3215 bus/motor Station sees.  usage: python3 scripts/arm_state.py [seconds]"""
import asyncio, logging, struct, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent / "norma-core"
sys.path.insert(0, str(ROOT / "software/station/shared")); sys.path.insert(0, str(ROOT))
from station_py import new_station_client
from target.gen_python.protobuf.drivers.st3215 import st3215

def u16(b, a): return struct.unpack_from("<H", b, a)[0] if len(b) >= a + 2 else None
def pos(v):
    if v is None: return None
    return (4096 - (v & 4095)) & 4095 if v & 0x8000 else v & 4095

def snapshot(data):
    out = {}
    for bus in st3215.InferenceStateReader(memoryview(data)).get_buses() or []:
        info = bus.get_bus()
        serial = str(info.get_serial_number()) if info else "?"
        rows = []
        for m in bus.get_motors() or []:
            s = bytes(m.get_state() or b"")
            rows.append(dict(id=m.get_id(), pos=pos(u16(s, 0x38)), goal=pos(u16(s, 0x2A)),
                             torque=s[0x28] if len(s) > 0x28 else None, cur=u16(s, 0x45),
                             volt=s[0x3E] / 10 if len(s) > 0x3E else None, temp=s[0x3F] if len(s) > 0x3F else None,
                             rmin=m.get_range_min(), rmax=m.get_range_max(), frozen=m.get_range_freezed()))
        out[serial] = rows
    return out

async def main(secs):
    log = logging.getLogger("arm_state"); logging.basicConfig(level=logging.WARNING)
    client = await new_station_client("localhost", log)
    q = asyncio.Queue(); client.follow("st3215/inference", q)
    first = last = None; t0 = time.time()
    while time.time() - t0 < secs or first is None:
        e = await asyncio.wait_for(q.get(), 10)
        last = snapshot(bytes(e.Data)); first = first or last
    for serial, rows in last.items():
        print(f"\nBUS {serial}  ({len(rows)} motors)")
        print(" id   pos  goal torque  cur(mA)  volt temp   range_min range_max frozen  moved")
        f = {r["id"]: r for r in first.get(serial, [])}
        for r in rows:
            d = (r["pos"] or 0) - (f.get(r["id"], r)["pos"] or 0)
            print(f"{r['id']:3} {r['pos']!s:>5} {r['goal']!s:>5} {r['torque']!s:>6} {r['cur']!s:>8} {r['volt']!s:>5} {r['temp']!s:>4}   {r['rmin']:9} {r['rmax']:9} {r['frozen']!s:>6}  {d:+d}")

asyncio.run(main(float(sys.argv[1]) if len(sys.argv) > 1 else 1.0))
