"""Checks against the real arm_mcp.station / norma-core code. Skipped when norma-core is not available."""
import asyncio
import json

import pytest

from recorder import fmt
from recorder.replay import ReplaySource

from .conftest import BUS, build

try:
    from arm_mcp import station as stn
except BaseException:  # station.py raises SystemExit without a norma-core checkout
    stn = None

pytestmark = pytest.mark.skipif(stn is None, reason="norma-core checkout not found (set NORMA_CORE_REPO)")


class FakeErr:
    def get_description(self):
        return ""


class FakeMotor:
    def __init__(self, m):
        self.m = m

    def get_state(self):
        return bytes.fromhex(self.m["regs"])

    def get_error(self):
        return FakeErr()

    def get_range_min(self):
        return self.m["range_min"]

    def get_range_max(self):
        return self.m["range_max"]

    def get_id(self):
        return self.m["id"]


def test_helpers_match_station():
    for raw in (0, 5, 2048, 4095, 0x8000 | 7, 0x8FFF, 0xFFFF):
        assert fmt.norm_raw(raw) == stn.norm_raw(raw)
    b = bytes(range(80))
    assert all(fmt.u16(b, a) == stn.u16(b, a) for a in (0x2A, 0x38, 0x3C))
    assert fmt.NAMES == stn.NAMES and fmt.IDS == stn.IDS


def test_replay_read_state_equals_station_read_state(tmp_path, monkeypatch):
    d = build(tmp_path / "r", seconds=0.5)
    row = json.loads((d / "state.jsonl").read_text().splitlines()[3])
    motors = [FakeMotor(m) for k, m in row["buses"][BUS].items() if isinstance(m, dict) and "regs" in m]

    async def fake_bus_frame(cli):
        return motors

    monkeypatch.setattr(stn, "bus_frame", fake_bus_frame)
    live = asyncio.run(stn.read_state(None))
    src = ReplaySource(d, clock=lambda: 0.0)
    src.seek(row["t"])
    assert src.read_state_sync() == live


def test_command_decoders_round_trip():
    from target.gen_python.protobuf.drivers.st3215 import st3215
    from target.gen_python.protobuf.station import commands, drivers

    from recorder.live import decode_commands_pack, decode_tx

    sync = st3215.ST3215SyncWriteCommand(address=0x2A, motors=[
        st3215.ST3215SyncWriteCommand_MotorWrite(motor_id=1, value=(2100).to_bytes(2, "little")),
        st3215.ST3215SyncWriteCommand_MotorWrite(motor_id=8, value=(1500).to_bytes(2, "little"))])
    cmd = st3215.Command(target_bus_serial=BUS, sync_write=sync)
    write = st3215.Command(target_bus_serial=BUS, write=st3215.ST3215WriteCommand(motor_id=3, address=0x28, value=b"\x00"))
    pack = commands.StationCommandsPack(commands=[
        commands.DriverCommand(type=drivers.StationCommandType.STC_ST3215_COMMAND, body=cmd.encode()),
        commands.DriverCommand(type=drivers.StationCommandType.STC_ST3215_COMMAND, body=write.encode())], tags=["t1"])
    recs = decode_commands_pack(pack.encode())
    assert recs[0] == {"src": "commands", "type": "STC_ST3215_COMMAND", "tags": ["t1"], "bus": BUS,
                       "kind": "sync_write", "address": 0x2A, "motors": {1: 2100, 8: 1500}}
    assert recs[1]["kind"] == "write" and recs[1]["motor_id"] == 3 and recs[1]["address"] == 0x28
    tx = st3215.TxEnvelope(local_stamp_ns=5, target_bus_serial=BUS, sync_write=sync)
    t = decode_tx(tx.encode())
    assert t["kind"] == "sync_write" and t["station_ns"] == 5 and t["motors"] == {1: 2100, 8: 1500}
