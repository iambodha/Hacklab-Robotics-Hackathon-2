"""The real follower arm through NormaCore Station.

Reads: one read-only recorder.live.LiveCapture (single subscription per queue; Station frame indices as seq;
each camera frame paired with the follower's motor norms at arrival). Writes: arm_mcp.station.sync_write of the
goal register on the capture's own Station client.

Motion is OFF unless BOTH hold:
  * the process was started with EXECUTOR_REAL_MOTION=1 (the integrator's gate), and
  * the operator switched "arm enabled" on (enable_motion(True)); it starts off.
Otherwise write_goals raises MotionDisabled and validation rejects every motion proposal ("arm disabled").
"""
from __future__ import annotations

import asyncio
import os
import threading
import time

from .base import Backend, FrameSample, MotionDisabled, MotorRead

GOAL_REG, TORQUE_LIMIT_REG = 42, 48


def real_motion_allowed() -> bool:
    return os.environ.get("EXECUTOR_REAL_MOTION") == "1"


def make_capture():
    """A LiveCapture that also remembers the follower's joints at each camera frame and its event loop."""
    from recorder.fmt import FOLLOWER_BUS
    from recorder.live import LiveCapture

    class Capture(LiveCapture):
        loop = None

        async def _main(self):
            self.loop = asyncio.get_running_loop()
            await super()._main()

        def _on_cam(self, h, e, mono, wall):
            super()._on_cam(h, e, mono, wall)
            with self.lock:
                ms = self.motors.get(FOLLOWER_BUS, {})
                c = self.cams.get(h)
                if c is not None:
                    c["joints"] = [ms[m]["norm"] if m in ms else None for m in range(1, 9)]
    return Capture().start()


class StationBackend(Backend):
    name = "real"

    def __init__(self, capture=None, writer=None, bus: str | None = None, max_state_age: float = 0.5):
        from recorder.fmt import FOLLOWER_BUS
        self.bus = bus or FOLLOWER_BUS
        self.cap = capture if capture is not None else make_capture()
        self._writer = writer          # (address, {mid: value}, width) -> None; default: Station sync_write
        self.max_state_age = max_state_age
        self._enabled = False
        self.lock = threading.Lock()

    # --- gate --------------------------------------------------------------------------
    @property
    def can_move(self) -> bool:
        return self._enabled and real_motion_allowed()

    def enable_motion(self, on: bool) -> str:
        if on and not real_motion_allowed():
            self._enabled = False
            return "refused: start the executor with EXECUTOR_REAL_MOTION=1 to allow real motion"
        self._enabled = bool(on)
        return "arm enabled" if self._enabled else "arm disabled"

    # --- reads --------------------------------------------------------------------------
    def read_state(self) -> MotorRead:
        from recorder.fmt import read_state_entry
        deadline = time.monotonic() + 2.0
        while True:
            with self.cap.lock:
                motors = dict(self.cap.motors.get(self.bus, {}))
                age = None if self.cap.last_state_mono is None else time.monotonic() - self.cap.last_state_mono
            out = {}
            for mid, m in motors.items():
                regs = bytes.fromhex(m.get("regs") or "")
                if m.get("error") or len(regs) < 0x46:
                    out = None
                    break
                out[mid] = read_state_entry(regs, int(m["range_min"]), int(m["range_max"]))
            if out and len(out) == len(self.motor_ids) and age is not None and age <= self.max_state_age:
                return MotorRead(time.monotonic() - age, dict(sorted(out.items())))
            if time.monotonic() > deadline:
                raise RuntimeError(f"motor bus {self.bus}: no complete, fresh, error-free state (age {age})")
            time.sleep(0.02)

    def frames(self) -> dict[str, FrameSample]:
        out = {}
        with self.cap.lock:
            for h, c in self.cap.cams.items():
                role = c.get("role") or h[:8]
                if c.get("jpeg") is None:
                    continue
                info = c.get("info") or {}
                seq = info.get("index") if info.get("index") is not None else c["frames"]
                out[role] = FrameSample(role, f"usbvideo/{h}", int(seq), float(c["mono"]), c["jpeg"], c.get("joints"))
        return out

    # --- writes (gated) --------------------------------------------------------------------
    def _write(self, address: int, values: dict[int, int], width: int) -> None:
        if not self.can_move:
            raise MotionDisabled("real arm motion is disabled (arm-enabled toggle off or EXECUTOR_REAL_MOTION unset)")
        if self._writer is not None:
            return self._writer(address, values, width)
        from arm_mcp import station as stn
        if self.cap.loop is None:
            raise RuntimeError("Station client not connected yet")
        asyncio.run_coroutine_threadsafe(stn.sync_write(self.cap.cli, address, values, width), self.cap.loop).result(2.0)

    def _holds(self, st: dict, gap: int = 300) -> dict[int, int]:
        """Each motor's current goal register, or its present position if the goal is far off (real_arm_mcp._holds)."""
        return {m: (s["goal"] if abs(s["goal"] - s["raw"]) < gap else s["raw"]) for m, s in st.items()}

    def write_goals(self, goals: dict[int, float]) -> None:
        if not self.can_move:
            raise MotionDisabled("real arm motion is disabled")
        st = self.read_state().motors
        ticks = {}
        for m, n in goals.items():
            s = st[m]
            if not s["in_range"]:
                raise RuntimeError(f"motor {m} raw {s['raw']} is outside its calibrated range; commanding it would snap")
            ticks[m] = int(round(s["lo"] + min(1.0, max(0.0, n)) * (s["hi"] - s["lo"])))
        self._write(GOAL_REG, {**self._holds(st), **ticks}, 2)

    def hold(self) -> None:
        st = self.read_state().motors
        self._write(GOAL_REG, {**self._holds(st), **{m: st[m]["raw"] for m in self.motor_ids[:-1]}}, 2)

    def set_torque_limit(self, mid: int, fraction: float) -> None:
        self._write(TORQUE_LIMIT_REG, {mid: max(0, min(1000, int(round(fraction * 1000))))}, 2)

    def describe(self) -> dict:
        snap = self.cap.snapshot() if hasattr(self.cap, "snapshot") else {}
        return {**super().describe(), "arm_enabled": self._enabled, "real_motion_allowed": real_motion_allowed(),
                "station_connected": snap.get("connected"), "bus": self.bus}
