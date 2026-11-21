"""One motion/feedback interface for every backend (fake, sim, real Station, replay).

State uses the shape of arm_mcp.station.read_state (and recorder.ReplaySource.read_state):
{motor_id: {raw, goal, lo, hi, norm, in_range, torque, status, temp, volt, load}}. Goals are written as
normalized positions in each motor's calibrated range (0..1; the gripper: higher = more open).
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass


class MotionDisabled(RuntimeError):
    """The backend refuses motor commands (real arm not enabled, replay, read-only)."""


@dataclass
class MotorRead:
    t_mono: float
    motors: dict[int, dict]


@dataclass
class FrameSample:
    role: str
    queue: str
    seq: int                       # source frame counter (Station frame index); never a byte hash
    t_capture: float               # executor-clock monotonic seconds
    jpeg: bytes | None
    joints_norm: list[float] | None = None   # motor norms (ids order) at capture time


class Backend:
    name = "base"
    cameras: tuple[str, ...] = ("wrist", "overview")
    motor_ids: list[int] = list(range(1, 9))

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, s: float) -> None:
        time.sleep(max(0.0, s))

    @property
    def can_move(self) -> bool:
        return True

    def read_state(self) -> MotorRead:
        raise NotImplementedError

    def frames(self) -> dict[str, FrameSample]:
        raise NotImplementedError

    def write_goals(self, goals: dict[int, float]) -> None:
        """Set goal positions (norm) for the listed motors; every other motor keeps its current goal."""
        raise NotImplementedError

    def set_torque_limit(self, mid: int, fraction: float) -> None:
        """Cap a motor's torque (gripper force) as a fraction of full; backends without it ignore the call."""

    def hold(self) -> None:
        """Stop pushing: goal := present position for every arm motor (gripper keeps its goal)."""
        st = self.read_state().motors
        self.write_goals({m: st[m]["norm"] for m in self.motor_ids[:-1] if st[m]["norm"] is not None})

    def describe(self) -> dict:
        return {"name": self.name, "can_move": self.can_move, "cameras": list(self.cameras)}

    def close(self) -> None:
        pass


def placeholder_jpeg(text: str = "", size=(299, 224), color=(200, 200, 196)) -> bytes:
    """A small valid JPEG (PIL if present), for backends without real cameras."""
    try:
        from PIL import Image, ImageDraw
        im = Image.new("RGB", size, color)
        if text:
            ImageDraw.Draw(im).text((8, 8), text, fill=(60, 60, 60))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=70)
        return buf.getvalue()
    except ImportError:
        return b"\xff\xd8\xff\xd9"
