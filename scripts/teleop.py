"""Start/stop Station motor mirroring (teleop).
usage: python3 scripts/teleop.py start [LEADER_SERIAL FOLLOWER_SERIAL] | stop | status"""
import asyncio, logging, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent / "norma-core"
sys.path.insert(0, str(ROOT / "software/station/shared")); sys.path.insert(0, str(ROOT))
from station_py import new_station_client, send_commands
from target.gen_python.protobuf.station import commands, drivers
from target.gen_python.protobuf.drivers.motors_mirroring import mirroring as mm

LEADER, FOLLOWER = "5B61035034", "5B61033301"   # 7.6 V arm -> 12 V arm
def bus(s): return mm.MirroringBus(type=mm.BusType.MBT_ST3215, unique_id=s)

async def status(client):
    q = asyncio.Queue(); client.follow("inference/mirroring", q)
    try: e = await asyncio.wait_for(q.get(), 5)
    except asyncio.TimeoutError: print("no mirroring state published"); return
    st = mm.RxEnvelopeReader(memoryview(bytes(e.Data))).get_state()
    pairs = [(m.get_source().get_id().get_unique_id(), [t.get_id().get_unique_id() for t in m.get_targets() or []])
             for m in (st.get_mirroring() or [])] if st else []
    print("active mirrors:", pairs or "none")

async def main(argv):
    logging.basicConfig(level=logging.WARNING)
    client = await new_station_client(os.environ.get("STATION_HOST", "localhost"), logging.getLogger("teleop"))
    op = argv[0] if argv else "status"
    if op in ("start", "stop"):
        src, dst = (argv[1], argv[2]) if len(argv) >= 3 else (LEADER, FOLLOWER)
        ct = mm.CommandType.CT_START_MIRROR if op == "start" else mm.CommandType.CT_STOP_MIRROR
        cmd = mm.Command(type=ct, source=bus(src), targets=[bus(dst)])
        await send_commands(client, [commands.DriverCommand(
            type=drivers.StationCommandType.STC_MOTOR_MIRRORING_COMMAND, body=cmd.encode())])
        print(f"{op}: leader {src} -> follower {dst}")
        await asyncio.sleep(1.0)
    await status(client)

asyncio.run(main(sys.argv[1:]))
