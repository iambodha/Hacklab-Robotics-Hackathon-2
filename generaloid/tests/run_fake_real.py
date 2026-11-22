"""Subprocess body of test_real_mode.py: mount a connector in REAL mode on a FakeStation and call its tools.

    python -m generaloid.tests.run_fake_real elrobot      -> one JSON line

Runs in its own process because real_arm_mcp puts sim/ in front of sys.path and registers its tools at import.
"""
from __future__ import annotations

import asyncio
import json
import sys


def _text(res) -> str:
    content = res[0] if isinstance(res, tuple) else res
    return "\n".join(getattr(c, "text", "") for c in content)


async def main(name: str) -> dict:
    from generaloid import connectors
    from generaloid.core import station as S
    from generaloid.core.server import build_server, tool_names
    from generaloid.testing import GOAL_REG, TORQUE_REG, FakeStation

    conn = connectors.get(name)
    st = FakeStation([j.motor_id for j in conn.joints])
    server, mount = build_server(conn, "real", st, addons=False)
    out = {"tools": sorted(tool_names(server)), "server_name": mount.tools.name}
    out["joint_state"] = _text(await server.call_tool("joint_state", {}))
    await server.call_tool("set_torque", {"enabled": True})
    out["torque_writes"] = [w[0] for w in st.bus_fake.writes]
    first = conn.arm_joints[0]
    before = {m: v["raw"] for m, v in st.bus_fake.motors.items()}
    st.bus_fake.writes.clear()
    out["move_to"] = _text(await server.call_tool("move_to", {"targets": {first.name: 0.55}}))[:300]
    out["moved"] = {m: v["raw"] - before[m] for m, v in st.bus_fake.motors.items() if v["raw"] != before[m]}
    out["other_writes"] = sorted({a for a, _, _ in st.bus_fake.writes if a not in (GOAL_REG, TORQUE_REG)})
    out["fake_connects"] = st.connects
    try:                                             # the real Station is never reachable from here
        await S.Station().connect()
        out["real_connect"] = "connected"
    except S.StationRefused as e:
        out["real_connect"] = f"refused: {e}"
    return out


if __name__ == "__main__":
    print(json.dumps(asyncio.run(main(sys.argv[1]))))
