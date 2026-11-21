"""One Gemini Robotics ER pointing call on a benchmark frame: checks key, model id and parsing. Prints no key."""
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.gemini_er import GeminiER, Image  # noqa: E402

frame = Path(__file__).resolve().parent / "data" / "frames" / "markers_20261122-000849_live-20s_overview.jpg"
er = GeminiER(source_tag="smoke")
pts = er.point(Image.from_path(frame, role="overview"), ["the centre of each black-and-white square ArUco fiducial marker"])
print(json.dumps([dataclasses.asdict(p) for p in pts]))
print(json.dumps(dataclasses.asdict(er.last)))
