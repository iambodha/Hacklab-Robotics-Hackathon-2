"""On-disk format constants and pure ST3215 register decoding (no norma-core import).

`u16`, `norm_raw` and `read_state_entry` mirror arm_mcp/station.py exactly so a recording can be
turned back into the dict `arm_mcp.station.read_state` returns; tests/test_read_state.py checks the
two stay identical.
"""
from __future__ import annotations

import os
import re
import struct

FORMAT_VERSION = 1
NAMES = [f"j{i}" for i in range(1, 9)]
IDS = list(range(1, 9))
GRIPPER = "j8"

FOLLOWER_BUS = os.environ.get("ROBOT_BUS_SERIAL", "5B61033301")
LEADER_BUS = os.environ.get("ROBOT_LEADER_SERIAL", "5B61035034")
_DEFAULT_ROLES = "28ff0ca8=wrist,9ea4cacd=overview"

# ST3215 / STS3215 register addresses inside the state dump Station sends
REG_TORQUE, REG_GOAL, REG_POS, REG_SPEED, REG_LOAD = 0x28, 0x2A, 0x38, 0x3A, 0x3C
REG_VOLT, REG_TEMP, REG_STATUS, REG_MOVING, REG_CURRENT = 0x3E, 0x3F, 0x41, 0x42, 0x45
STATE_LEN = 0x46  # read_state rejects shorter dumps


def u16(s: bytes, a: int) -> int:
    return struct.unpack("<H", s[a:a + 2])[0]


def norm_raw(raw: int) -> int:
    return (4096 - (raw & 4095)) & 4095 if raw & 0x8000 else raw & 4095


def signed(v: int, bit: int) -> int:
    """Feetech sign-magnitude: `bit` is the direction flag."""
    return -(v & ((1 << bit) - 1)) if v & (1 << bit) else v & ((1 << bit) - 1)


def bus_role(serial: str) -> str | None:
    return {FOLLOWER_BUS: "follower", LEADER_BUS: "leader"}.get(serial)


def cam_roles() -> dict[str, str]:
    spec = os.environ.get("RECORDER_CAM_ROLES", _DEFAULT_ROLES)
    return dict(p.split("=", 1) for p in spec.split(",") if "=" in p)


def cam_role(qhash: str) -> str | None:
    return cam_roles().get(qhash[:8])


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", name.strip()).strip("-")[:40] or "rec"


def decode_motor(mid: int, regs: bytes, lo: int, hi: int, frozen: bool, error: str | None,
                 stamp_ns: int | None = None) -> dict:
    """One motor's recorded entry. Short dumps keep the raw bytes and leave the decoded fields null."""
    m = {"id": mid, "range_min": lo, "range_max": hi, "frozen": bool(frozen), "error": error or None,
         "stamp_ns": stamp_ns, "regs": regs.hex()}
    if len(regs) < STATE_LEN + 1:
        return {**m, "pos": None, "goal": None, "speed": None, "load": None, "current": None, "volt": None,
                "temp": None, "torque": None, "moving": None, "status": None, "norm": None}
    pos = norm_raw(u16(regs, REG_POS))
    return {**m, "pos": pos, "goal": u16(regs, REG_GOAL), "speed": signed(u16(regs, REG_SPEED), 15),
            "load": signed(u16(regs, REG_LOAD), 10), "current": signed(u16(regs, REG_CURRENT), 15),
            "volt": regs[REG_VOLT] / 10, "temp": regs[REG_TEMP], "torque": regs[REG_TORQUE],
            "moving": regs[REG_MOVING], "status": regs[REG_STATUS],
            "norm": round((pos - lo) / (hi - lo), 4) if hi > lo else None}


def gripper_opening(m: dict | None) -> float | None:
    """j8 opening 0..1 from its calibrated range (higher = more open); None when uncalibrated."""
    if not m or m.get("norm") is None:
        return None
    return round(min(1.0, max(0.0, m["norm"])), 4)


def read_state_entry(regs: bytes, lo: int, hi: int) -> dict:
    """Exactly the per-motor dict arm_mcp.station.read_state builds from the same register dump."""
    s = regs
    raw = norm_raw(u16(s, 0x38))
    return {"raw": raw, "goal": u16(s, 0x2A), "lo": lo, "hi": hi,
            "norm": round((raw - lo) / (hi - lo), 4) if hi > lo else None,
            "in_range": lo <= raw <= hi, "torque": s[0x28], "status": s[0x41],
            "temp": s[0x3F], "volt": s[0x3E] / 10, "load": u16(s, 0x3C) & 0x3FF}


def bus_entry(serial: str, motors: dict[int, dict]) -> dict:
    """Per-bus block of a state row: joints j1..j8 keyed by name, plus gripper opening."""
    out = {"role": bus_role(serial), "n_motors": len(motors)}
    for mid, m in sorted(motors.items()):
        out[NAMES[mid - 1] if 1 <= mid <= len(NAMES) else f"m{mid}"] = m
    out["gripper"] = gripper_opening(out.get(GRIPPER))
    return out


def primary_bus(buses: dict) -> str | None:
    if FOLLOWER_BUS in buses:
        return FOLLOWER_BUS
    return next(iter(sorted(buses)), None)


def sim_compat(buses: dict) -> dict:
    """Top-level `norm`/`goal`/`load` arrays (j1..j8 of the follower) like sim state rows."""
    b = buses.get(primary_bus(buses) or "", {})
    js = [b.get(n) or {} for n in NAMES]
    goal = [(round((j["goal"] - j["range_min"]) / (j["range_max"] - j["range_min"]), 4)
             if j.get("goal") is not None and j["range_max"] > j["range_min"] else None) for j in js]
    return {"norm": [j.get("norm") for j in js], "goal": goal, "load": [j.get("load") for j in js],
            "status": [j.get("status") for j in js], "torque": [j.get("torque") for j in js]}


def strip_regs(row: dict) -> dict:
    """Copy of a state row without the raw register hex (the bulky part)."""
    buses = {s: {k: ({kk: vv for kk, vv in v.items() if kk != "regs"} if isinstance(v, dict) else v)
                 for k, v in b.items()} for s, b in row.get("buses", {}).items()}
    return {**row, "buses": buses}
