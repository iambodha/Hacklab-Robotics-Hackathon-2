"""Read-only health check for the real arm: motor-bus reads, torque, calibration ranges. No motion."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import station as stn  # noqa: E402


async def main():
    cli = await stn.connect()
    motors = await stn.bus_frame(cli)
    errors = {int(m.get_id()): stn.as_str(m.get_error().get_description()) for m in motors or []
              if m.get_error() and stn.as_str(m.get_error().get_description())}
    report = {"bus": stn.BUS, "bus_found": motors is not None, "read_errors": errors}
    if motors and not errors:
        st = await stn.read_state(cli)
        report["joints"] = {stn.NAMES[stn.IDS.index(mid)]: {k: m[k] for k in ("raw", "lo", "hi", "norm", "in_range", "torque", "status", "volt")}
                            for mid, m in sorted(st.items())}
        report["all_in_range"] = all(m["in_range"] for m in st.values())
        report["torque_on"] = all(m["torque"] for m in st.values())
    print(json.dumps(report, indent=1))


asyncio.run(main())
