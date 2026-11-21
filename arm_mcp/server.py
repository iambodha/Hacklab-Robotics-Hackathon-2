"""MCP server for the real ElRobot arm with freshness-checked cameras and jump-free motion.

State comes from the motor bus (st3215/inference), never from the inference/normvla
frame, which freezes whenever torque is off. Camera frames come from the cam_stream
relay and are rejected if stale or byte-identical to the previous capture.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image as PILImage  # noqa: E402
from mcp.server.fastmcp import FastMCP, Image  # noqa: E402

import station as stn  # noqa: E402

CAM_URL = os.environ.get("ROBOT_CAM_URL", "http://localhost:8890")
MAX_CAM_AGE = float(os.environ.get("ROBOT_MAX_CAM_AGE", "1.0"))
MAX_DELTA = float(os.environ.get("ROBOT_MAX_DELTA", "0.35"))
TRACK_TOL = int(os.environ.get("ROBOT_TRACK_TOL", "250"))
GRIPPER = os.environ.get("ROBOT_GRIPPER_JOINT", "j8")
NAMES, IDS = stn.NAMES, stn.IDS

server = FastMCP("elrobot-hardware")
_cli = None
_last_hash: dict[str, str] = {}


async def _client():
    global _cli
    if _cli is None or not getattr(_cli, "connected", True):
        _cli = await stn.connect()
    return _cli


async def _read() -> dict:
    return await stn.read_state(await _client())


async def _write(address: int, values: dict[int, int], width: int) -> None:
    await stn.sync_write(await _client(), address, values, width)


def _table(st: dict) -> str:
    rows = [f"{'joint':>5} {'norm':>7} {'raw':>5} {'goal':>5} {'range':>12} in_rng torque status load temp"]
    for i, mid in enumerate(IDS):
        m = st[mid]
        rows.append(f"{NAMES[i]:>5} {m['norm']:>7} {m['raw']:>5} {m['goal']:>5} [{m['lo']:>4},{m['hi']:>4}] "
                    f"{'y' if m['in_range'] else 'NO':>6} {m['torque']:>6} {hex(m['status']):>6} {m['load']:>4} {m['temp']:>4}")
    return "\n".join(rows)


def _cameras() -> list[dict]:
    with urllib.request.urlopen(f"{CAM_URL}/api/cameras", timeout=3) as r:
        return [c for c in json.load(r) if c.get("live")]


def _snapshot(h: str) -> bytes:
    with urllib.request.urlopen(f"{CAM_URL}/snapshot/{h}", timeout=3) as r:
        return r.read()


@server.tool(annotations={"readOnlyHint": True})
async def look() -> list:
    """FRESH images from every live camera (cam_stream relay) plus the live motor-bus joint table.
    Each image is checked: relay age <= ROBOT_MAX_CAM_AGE seconds and bytes differ from the previous
    capture of that camera; a stale camera is retried and then reported as STALE instead of shown."""
    out: list = []
    for cam in _cameras():
        h, note = cam["hash"], ""
        for _ in range(4):
            data = _snapshot(h)
            digest = hashlib.sha1(data).hexdigest()
            age = next((c["age"] for c in _cameras() if c["hash"] == h), None)
            if age is not None and age <= MAX_CAM_AGE and digest != _last_hash.get(h):
                break
            note = f"age={age}"
            await asyncio.sleep(0.3)
        else:
            out.append(f"camera {h[:8]} ({cam['unique_id']}): STALE ({note}) — not shown")
            continue
        _last_hash[h] = digest
        with PILImage.open(io.BytesIO(data)) as im:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="JPEG", quality=85)
        out.append(f"camera {h[:8]} ({cam['unique_id']}) fresh, age={age:.2f}s, captured {time.strftime('%H:%M:%S')}")
        out.append(Image(data=buf.getvalue(), format="jpeg"))
    out.append(_table(await _read()))
    return out


@server.tool(annotations={"readOnlyHint": True})
async def joint_state() -> str:
    """Live joint table from the motor bus: raw ticks, calibrated range, in-range flag, torque, status, load."""
    return _table(await _read())


@server.tool(annotations={"destructiveHint": True})
async def set_torque(enabled: bool) -> str:
    """Enable/disable torque on all motors. Enabling first writes goal = present position for every
    motor so nothing jumps. Disabling makes the arm limp: support it before calling with False."""
    st = await _read()
    if enabled:
        await _write(0x2A, {mid: st[mid]["raw"] for mid in IDS}, 2)
        await asyncio.sleep(0.1)
    await _write(0x28, {mid: int(enabled) for mid in IDS}, 1)
    await asyncio.sleep(0.4)
    return _table(await _read())


@server.tool(annotations={"destructiveHint": True})
async def move_to(targets: dict[str, float], step: float = 0.02, dt: float = 0.12) -> str:
    """Creep joints to ABSOLUTE normalized targets in [0,1] (calibrated range), e.g. {"j1":0.5}.
    Joints: j1 base yaw, j2 shoulder, j3 elbow, j4/j5/j6 wrist, j7 gripper rotation, j8 gripper jaws
    (higher = more open). Unlisted joints hold their present raw position. Moves `step` per `dt`.
    Refuses: torque off, a commanded joint currently outside its calibrated range (it would snap),
    or a request > ROBOT_MAX_DELTA from present. Aborts and holds if a joint stops tracking
    (|goal-present| > ROBOT_TRACK_TOL ticks) or a motor reports a fault status."""
    return await _move(targets, step, dt, TRACK_TOL)


async def _move(targets: dict[str, float], step: float, dt: float, track_tol: int) -> str:
    unknown = [k for k in targets if k not in NAMES]
    if unknown:
        raise ValueError(f"unknown joints {unknown}; valid {NAMES}")
    st = await _read()
    if any(st[mid]["torque"] == 0 for mid in IDS):
        raise RuntimeError("torque is off on some motors — call set_torque(true) first")
    plan = {}
    for name, want in targets.items():
        mid = IDS[NAMES.index(name)]
        m = st[mid]
        if not m["in_range"]:
            raise RuntimeError(f"{name} raw {m['raw']} is outside calibrated range [{m['lo']},{m['hi']}] — "
                               "commanding it would snap; recalibrate or move it into range by hand")
        want = min(1.0, max(0.0, float(want)))
        if abs(want - m["norm"]) > MAX_DELTA:
            raise ValueError(f"{name}: {m['norm']:.3f}->{want:.3f} exceeds max delta {MAX_DELTA}; use waypoints")
        plan[mid] = (m["norm"], want)
    hold = {mid: st[mid]["raw"] for mid in IDS}
    n = max([1] + [int(abs(b - a) / step + 0.999) for a, b in plan.values()])
    for k in range(1, n + 1):
        goals = dict(hold)
        for mid, (a, b) in plan.items():
            m = st[mid]
            goals[mid] = int(round(m["lo"] + (a + (b - a) * k / n) * (m["hi"] - m["lo"])))
        await _write(0x2A, goals, 2)
        await asyncio.sleep(dt)
        now = await _read()
        bad = [(NAMES[IDS.index(mid)], goals[mid], now[mid]["raw"], now[mid]["status"]) for mid in IDS
               if now[mid]["status"] or abs(goals[mid] - now[mid]["raw"]) > track_tol]
        if bad:
            await _write(0x2A, {mid: now[mid]["raw"] for mid in IDS}, 2)
            return f"ABORTED at step {k}/{n}: (joint, goal, present, status) {bad}; holding present pose.\n" + _table(now)
    await asyncio.sleep(0.3)
    return f"reached in {n} steps.\n" + _table(await _read())


@server.tool(annotations={"destructiveHint": True})
async def gripper(open_fraction: float) -> str:
    """Set the gripper jaws (j8) to a normalized target: higher = more open. Closing on an object
    stalls the jaw short of the target, which is a grip; it does not abort for the gripper."""
    return await _move({GRIPPER: open_fraction}, 0.05, 0.1, 4096)


if __name__ == "__main__":
    server.run()
