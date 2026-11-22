"""Test helpers for connectors: a fake NormaCore Station with an ST3215-like motor bus.

    st = FakeStation(motor_ids=range(1, 9))
    mount = CONNECTOR.mount("real", st)          # the real-mode server, wired to the fake bus
    await mount.tools.call_tool("joint_state", {})
    st.bus_fake.writes                           # [(register, {motor: value}, width), ...]

The real-arm server reads and writes the bus only through the Station I/O layer (arm_mcp/station.py read_state /
sync_write / connect, routed by Station.install()), so swapping those three is enough to run its real-mode tools
with no Station and no arm. Goal writes move a motor instantly (no dynamics); use the twin for motion behaviour.
"""
from __future__ import annotations

import copy

from generaloid.core.station import Station

GOAL_REG, TORQUE_REG = 0x2A, 0x28          # ST3215 goal position / torque enable (arm_mcp/server.py)
FAKE_RANGE = (1000, 3000)                  # calibrated range of every fake motor (raw ticks)
FAKE_CENTRE = 2048


class FakeClient:
    connected = True

    def follow(self, *_a, **_k):
        raise AssertionError("fake Station: nothing may subscribe to a raw queue (bus reads go through read_state)")


class FakeBus:
    def __init__(self, motor_ids):
        lo, hi = FAKE_RANGE
        self.motors = {int(i): {"raw": FAKE_CENTRE, "goal": FAKE_CENTRE, "lo": lo, "hi": hi, "torque": 0, "status": 0,
                                "temp": 30, "volt": 12.0, "load": 0} for i in motor_ids}
        self.writes: list[tuple[int, dict, int]] = []
        self.reads = 0

    async def read_state(self, _cli) -> dict:
        self.reads += 1
        out = {}
        for mid, m in self.motors.items():
            m = copy.deepcopy(m)
            m.update(norm=round((m["raw"] - m["lo"]) / (m["hi"] - m["lo"]), 4), in_range=m["lo"] <= m["raw"] <= m["hi"])
            out[mid] = m
        return out

    async def sync_write(self, _cli, address: int, values: dict, width: int) -> None:
        self.writes.append((address, {int(k): int(v) for k, v in values.items()}, width))
        for mid, v in values.items():
            m = self.motors[int(mid)]
            if address == GOAL_REG:
                m["goal"] = int(v)
                if m["torque"]:
                    m["raw"] = int(v)
            elif address == TORQUE_REG:
                m["torque"] = int(v)


class FakeStation(Station):
    def __init__(self, motor_ids):
        super().__init__(host="fake-station.invalid")
        self.bus_fake = FakeBus(motor_ids)
        self.connects = 0

    async def connect(self, log=None):
        self.connects += 1
        return FakeClient()

    def install(self) -> "FakeStation":
        bus = self.bus()

        async def _connect():
            return await self.connect()
        bus.connect, bus.read_state, bus.sync_write = _connect, self.bus_fake.read_state, self.bus_fake.sync_write
        return self
