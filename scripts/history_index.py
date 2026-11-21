"""Map Station history frame pointers to clock times, for dataset-generator -from/-to.
usage: python3 scripts/history_index.py [HH:MM:SS ...]     (no args: print every 3000th pointer)
  e.g. python3 scripts/history_index.py 12:31:50 12:48:00  -> pointer nearest each time"""
import asyncio, datetime, logging, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent / "norma-core"
sys.path.insert(0, str(ROOT / "software/station/shared")); sys.path.insert(0, str(ROOT))
from station_py import new_station_client
from target.gen_python.protobuf.station import inference

async def index(step=3000):
    c = await new_station_client("localhost", logging.getLogger("history"))
    qr = c.read_from_offset("inference-states", b"\x00", 0, step, 64); out = []
    while True:
        try: e = await asyncio.wait_for(qr.data.get(), 5)
        except asyncio.TimeoutError: break
        if e is None: break
        ts = inference.InferenceRxReader(memoryview(bytes(e.Data))).get_local_stamp_ns()
        out.append((int.from_bytes(bytes(e.ID.ID), "little"), datetime.datetime.fromtimestamp(ts / 1e9)))
    return out

async def main(times):
    idx = await index()
    if not times:
        for p, t in idx: print(p, t.strftime("%H:%M:%S"))
        return
    for s in times:
        h, m, sec = map(int, s.split(":"))
        p, t = min(idx, key=lambda x: abs((x[1].hour * 3600 + x[1].minute * 60 + x[1].second) - (h * 3600 + m * 60 + sec)))
        print(f"{s} -> pointer {p} (at {t:%H:%M:%S})")

asyncio.run(main(sys.argv[1:]))
