"""Optional Gemini ER judgement for check_cap: "is the cap separated from the bottle?".

Uses models/gemini_er.py (key from $GEMINI_API_KEY or secrets.env, never logged). The real-arm server's Python has
no google-genai, so when the SDK is missing the call runs in a subprocess: `uv run --with google-genai==3.3.0
python -m learning.judge <wrist.jpg> <overview.jpg>`. Any failure returns None: the signal-based judgement stands.

    python -m learning.judge wrist.jpg overview.jpg     -> one JSON line {"verdict", "done", "confidence", ...}
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TASK = "unscrew the cap of the black bottle the gripper is holding"
CRITERIA = ("the cap is separated from the bottle: it is held in the gripper jaws with a visible gap between the cap "
            "and the bottle's neck or thread, or it is clearly off the bottle")


def _judge_here(images: dict[str, bytes]) -> dict:
    from models.gemini_er import GeminiER, Image
    imgs = [Image(data, role=role, ref=f"check_{role}") for role, data in images.items() if data]
    j = GeminiER(source_tag="bottle-check").judge(imgs, TASK, CRITERIA)
    word = {True: "cap separated", False: "cap NOT separated", None: "cannot tell"}[j.done]
    return {"verdict": f"{word} (confidence {j.confidence}): {j.evidence}" + (f"; problem: {j.failure}" if j.failure else ""),
            "done": j.done, "confidence": j.confidence}


def gemini_cap_judge(images: dict[str, bytes], timeout: float = 60.0) -> dict | None:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    try:
        from models._secrets import has_gemini_key
        if not has_gemini_key():
            return None
    except Exception:  # noqa: BLE001
        return None
    try:
        import google.genai  # noqa: F401
        return _judge_here(images)
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001
        return {"verdict": f"Gemini ER failed: {type(e).__name__}", "done": None, "error": True}
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for role, data in images.items():
            if data:
                p = Path(d) / f"{role}.jpg"
                p.write_bytes(data)
                paths.append(str(p))
        try:
            out = subprocess.run(["uv", "run", "--no-project", "-q", "--with", "google-genai==3.3.0", "python", "-m",
                                  "learning.judge", *paths], cwd=REPO, capture_output=True, text=True, timeout=timeout,
                                 env={**os.environ})
            line = [l for l in out.stdout.splitlines() if l.startswith("{")]
            return json.loads(line[-1]) if line else None
        except (OSError, subprocess.TimeoutExpired, ValueError):
            return None


if __name__ == "__main__":
    sys.path.insert(0, str(REPO))
    imgs = {Path(p).stem: Path(p).read_bytes() for p in sys.argv[1:]}
    try:
        print(json.dumps(_judge_here(imgs)))
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"verdict": f"Gemini ER failed: {type(e).__name__}", "done": None, "error": True}))
