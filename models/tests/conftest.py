import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAKE_KEY = "AIzaSyFAKEfakeFAKEfake0123456789abcdefg"


def tiny_jpeg(w: int = 299, h: int = 224) -> bytes:
    """Smallest byte string image_size() accepts as a JPEG of w x h (SOI, SOF0, EOI); not decodable."""
    sof = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, h, w, 1) + b"\x01\x11\x00"
    return b"\xff\xd8" + b"\xff\xe0" + struct.pack(">H", 4) + b"JF" + sof + b"\xff\xd9"


@pytest.fixture
def log_dir(tmp_path):
    return tmp_path / "logs"


@pytest.fixture
def jpeg():
    return tiny_jpeg()
