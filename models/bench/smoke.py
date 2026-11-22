"""Gemini Robotics ER smoke test: key, model id, access, parsing. Prints no key (every message is redacted).

    uv run --no-project --with google-genai==3.3.0 python models/bench/smoke.py [--list] [--fallbacks]

1. --list: models.list() filtered to robotics / ER ids (free, no generation).
2. One pointing call on a marker frame with DEFAULT_MODEL (logged to models/logs/ like every GeminiER call).
3. --fallbacks (or the primary failing): the same call on the documented fallback ids, so the report can
   say exactly which ids this key can use.
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models._secrets import gemini_key, redact  # noqa: E402
from models.gemini_er import API_VERSION, DEFAULT_MODEL, GeminiER, GeminiError, Image, ParseError  # noqa: E402

FALLBACKS = ["gemini-robotics-er-1.6-preview", "gemini-robotics-er-1.5-preview", "gemini-3.8-flash"]
FRAME = Path(__file__).resolve().parent / "data" / "frames" / "markers_20261122-000849_live-20s_overview.jpg"


def list_models():
    from google import genai
    try:
        client = genai.Client(api_key=gemini_key(), http_options={"api_version": API_VERSION})
        names = [m.name for m in client.models.list()]
    except Exception as e:
        print("LIST ERROR", redact(f"{type(e).__name__}: {e}"))
        return
    print(f"models visible to this key: {len(names)}")
    for n in names:
        if "robotics" in n or "er-" in n or n.endswith("flash") or "3-" in n:
            print("  ", n)


def try_model(model: str) -> bool:
    er = GeminiER(model, source_tag="smoke")
    try:
        pts = er.point(Image.from_path(FRAME, role="overview"),
                       ["the centre of each black-and-white square ArUco fiducial marker"])
        print(f"OK {model}: {json.dumps([dataclasses.asdict(p) for p in pts])}")
        ok = True
    except (GeminiError, ParseError) as e:
        print(f"FAIL {model}: {redact(e)}")
        ok = False
    if er.last:
        print("   call:", json.dumps(dataclasses.asdict(er.last)))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--fallbacks", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()
    print("key present:", bool(gemini_key()))
    if a.list:
        list_models()
    ok = try_model(a.model)
    if a.fallbacks or not ok:
        for m in FALLBACKS:
            try_model(m)


if __name__ == "__main__":
    main()
