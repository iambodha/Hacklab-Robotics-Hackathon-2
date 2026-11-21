import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recorder.fmt import bus_entry, decode_motor, sim_compat  # noqa: E402
from recorder.store import SessionWriter  # noqa: E402

BUS = "5B61033301"
CAM = "28ff0ca87978e8f993c2ac64a84d188a"   # role "wrist" by default
CAM2 = "9ea4cacd5be94751e1eea7be90bcf628"  # role "overview"
LO, HI = 1000, 3000


def regs(pos, goal=2000, torque=1, load=0, volt=124, temp=31, cur=0, speed=0, status=0, moving=0) -> bytes:
    b = bytearray(71)
    struct.pack_into("<H", b, 0x38, pos)
    struct.pack_into("<H", b, 0x2A, goal)
    struct.pack_into("<H", b, 0x3A, speed)
    struct.pack_into("<H", b, 0x3C, load)
    struct.pack_into("<H", b, 0x45, cur)
    b[0x28], b[0x3E], b[0x3F], b[0x41], b[0x42] = torque, volt, temp, status, moving
    return bytes(b)


def jpeg(tag: str) -> bytes:
    return b"\xff\xd8" + tag.encode() + b"\xff\xd9"


def pos_at(mid: int, t: float) -> int:
    return 1000 + mid * 100 + int(t * 100)


def state_row(t: float, wall0: float) -> dict:
    motors = {mid: decode_motor(mid, regs(pos_at(mid, t), load=(1 << 10) | 50 if mid == 2 else 30), LO, HI, True, None,
                                int((wall0 + t) * 1e9)) for mid in range(1, 9)}
    buses = {BUS: bus_entry(BUS, motors)}
    return {"wall": wall0 + t, "station_ns": int((wall0 + t) * 1e9), "station_mono_ns": int(t * 1e9),
            "buses": buses, **sim_compat(buses)}


def build(root: Path, seconds: float = 2.0, state_hz: int = 50, fps: int = 10) -> Path:
    w = SessionWriter("synthetic", "unit test", root=root, station="test")
    w.event("start", mono=w.t0)
    items = [(i / state_hz, "s") for i in range(int(seconds * state_hz))]
    items += [(i / fps + 0.003, "f") for i in range(int(seconds * fps))]
    items += [(i / fps + 0.051, "g") for i in range(int(seconds * fps))]
    for t, kind in sorted(items):
        if kind == "s":
            w.state(state_row(t, w.wall0), w.t0 + t, frames_seen=8)
        else:
            cam = CAM if kind == "f" else CAM2
            n = round((t - 0.003) * fps) if kind == "f" else round((t - 0.051) * fps)
            w.frame(cam, jpeg(f"{cam[:8]}-{n}"), w.t0 + t, w.wall0 + t,
                    {"station_ns": int((w.wall0 + t - 0.02) * 1e9), "station_mono_ns": int(t * 1e9),
                     "index": n + (5 if n >= 10 and kind == "f" else 0), "w": 299, "h": 224})
        if kind == "s" and abs(t - 0.5) < 1e-9:
            w.event("marker", mono=w.t0 + t, label="grasp attempt", data={"try": 1})
    w.close()
    return w.dir


@pytest.fixture
def rec_dir(tmp_path) -> Path:
    return build(tmp_path / "recordings")
