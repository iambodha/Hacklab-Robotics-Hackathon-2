"""Read-only access to the perception workstream's localisation API (pixels -> world metres).

perception lives in another worktree (ws/perception, $PERCEPTION_ROOT). It is imported by path, never
copied or modified. If it can't be imported (no opencv/numpy in this environment, no scene_calib.json),
`locator()` returns None and callers fall back to pixel-only proposals.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PERCEPTION_ROOT = Path(os.environ.get("PERCEPTION_ROOT", Path(__file__).resolve().parents[2] / "perception"))
_LOC = None
_ERR: str | None = None


def locator():
    """perception.locate.Locator or None (reason in `last_error()`)."""
    global _LOC, _ERR
    if _LOC is not None:
        return _LOC
    try:
        if str(PERCEPTION_ROOT) not in sys.path:
            sys.path.append(str(PERCEPTION_ROOT))
        from perception import locate  # noqa: PLC0415
        if not locate.scene_file_exists():
            raise FileNotFoundError("perception/calibration/scene_calib.json is missing")
        _LOC = locate.Locator()
        _ERR = None
    except Exception as e:  # ImportError (cv2), missing calibration, bad JSON
        _ERR = f"{type(e).__name__}: {e}"
        _LOC = None
    return _LOC


def last_error() -> str | None:
    return _ERR


def joints_for_locate(raw: list[int], ranges: dict | list) -> dict:
    """perception's joint dict {"j1": {"raw", "lo", "hi"}, ...} from raw ticks and calibrated ranges
    (dict {"j1": [lo, hi]} or list of [lo, hi] in joint order)."""
    rng = [ranges[f"j{i + 1}"] for i in range(len(raw))] if isinstance(ranges, dict) else list(ranges)
    return {f"j{i + 1}": {"raw": int(r), "lo": int(lo), "hi": int(hi)} for i, (r, (lo, hi)) in enumerate(zip(raw, rng))}


def pixel_to_world(cam: str, u: float, v: float, z: float = 0.0, joints: dict | None = None) -> dict:
    """{"ok", "xyz", "sigma_mm", ...} as perception.locate.pixel_to_table, or {"ok": False, "reason"}."""
    loc = locator()
    if loc is None:
        return {"ok": False, "reason": f"perception unavailable ({_ERR})"}
    if cam == "wrist" and joints is None:
        return {"ok": False, "reason": "wrist pixels need the joint readings at capture time"}
    try:
        return loc.pixel_to_table(cam, u, v, z, joints)
    except Exception as e:
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}


def world_to_pixel(cam: str, xyz, joints: dict | None = None) -> dict | None:
    loc = locator()
    return None if loc is None else loc.world_to_pixel(cam, xyz, joints)
