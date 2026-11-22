"""Load the Gemini API key at runtime and keep it out of everything we write.

Order: $GEMINI_API_KEY, then $GOOGLE_API_KEY, then GEMINI_API_KEY=... in the secrets file
($MODELS_SECRETS_FILE, default <hackathon root>/secrets.env). The key is never printed, logged or
put in a URL; `redact()` scrubs it (and anything shaped like a Google API key) from text we persist.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

DEFAULT_SECRETS = Path(__file__).resolve().parents[3] / "secrets.env"
_GOOGLE_KEY = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")
_KEY_PARAM = re.compile(r"(?i)((?:api[_-]?key|x-goog-api-key|[?&]key)[\"']?\s*[=:]\s*[\"']?)([^\s\"'&,}]+)")
_seen: set[str] = set()


def _from_file(path: Path, name: str) -> str | None:
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:]
        k, sep, v = line.partition("=")
        if sep and k.strip() == name:
            return v.strip().strip("'\"") or None
    return None


def gemini_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or \
        _from_file(Path(os.environ.get("MODELS_SECRETS_FILE", DEFAULT_SECRETS)), "GEMINI_API_KEY")
    if key:
        _seen.add(key)
    return key or None


def has_gemini_key() -> bool:
    return gemini_key() is not None


def redact(text, extra: tuple[str, ...] = ()) -> str:
    """Scrub every key we have loaded (plus `extra`), Google-key-shaped tokens and key=... parameters."""
    s = str(text)
    for k in sorted(_seen | {e for e in extra if e}, key=len, reverse=True):
        if len(k) >= 6:
            s = s.replace(k, "[REDACTED]")
    s = _GOOGLE_KEY.sub("[REDACTED]", s)
    return _KEY_PARAM.sub(lambda m: m.group(1) + "[REDACTED]", s)


def redact_obj(obj, extra: tuple[str, ...] = ()):
    """redact() applied to every string inside a JSON-like structure."""
    if isinstance(obj, str):
        return redact(obj, extra)
    if isinstance(obj, dict):
        return {redact(k, extra) if isinstance(k, str) else k: redact_obj(v, extra) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact_obj(v, extra) for v in obj]
    return obj
