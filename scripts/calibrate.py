"""Start/stop Station auto-calibration on one ST3215 bus.
usage: python3 scripts/calibrate.py start|stop|reset|freeze BUS_SERIAL
  start/stop = auto-calibration; reset = clear ranges (then sweep joints by hand); freeze = lock ranges"""
import os
import asyncio, logging, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent / "norma-core"
sys.path.insert(0, str(ROOT / "software/station/shared")); sys.path.insert(0, str(ROOT))
from station_py import new_station_client, send_commands
from target.gen_python.protobuf.station import commands, drivers
from target.gen_python.protobuf.drivers.st3215 import st3215

async def main(op, serial):
    logging.basicConfig(level=logging.WARNING)
    client = await new_station_client(os.environ.get("STATION_HOST", "localhost"), logging.getLogger("calibrate"))
    C = st3215.Command
    cmd = {"start": lambda: C(target_bus_serial=serial, auto_calibrate=st3215.AutoCalibrateCommand(calibrate=True)),
           "stop": lambda: C(target_bus_serial=serial, stop_auto_calibrate=st3215.StopAutoCalibrateCommand(stop=True)),
           "reset": lambda: C(target_bus_serial=serial, reset_calibration=st3215.ResetCalibrationCommand(reset=True)),
           "freeze": lambda: C(target_bus_serial=serial, freeze_calibration=st3215.FreezeCalibrationCommand(freeze=True))}[op]()
    await send_commands(client, [commands.DriverCommand(
        type=drivers.StationCommandType.STC_ST3215_COMMAND, body=cmd.encode())])
    print(f"calibration {op} sent to bus {serial}")
    await asyncio.sleep(0.5)

asyncio.run(main(sys.argv[1], sys.argv[2]))
