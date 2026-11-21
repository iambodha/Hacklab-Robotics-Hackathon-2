"""Real follower raw ticks -> twin joint angles (URDF radians), configurable.

Modes (layout["arm"]["joint_mapping"], or a JSON file named by env TWIN_JOINT_MAP):
  range_fraction  (default, as real_arm_mcp.py with the elrobot profile)
                  q_i = lower_i + norm_i * (upper_i - lower_i), norm = (raw - range_min) / (range_max - range_min);
                  gripper j8: q = upper_8 * (1 - norm_8)  (norm 1 = open = 0 rad).
                  Needs Station's calibrated range; a joint with range_min == range_max falls back to "ticks".
  ticks           q_i = anchor_rad_i + sign_i * (raw - raw_anchor_i) * 2 pi / 4096  (camera_pose/HANDOFF.md);
                  defaults: anchor 0, raw_anchor 2048, sign +1 (j8: -1, like sim_arm_mcp's raw ticks).
Example override:
  {"mode": "ticks", "raw_anchor": [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048],
   "anchor_rad": [0, 0, 0, 0, 0, 0, 0, 0], "sign": [1, 1, 1, 1, 1, 1, 1, -1]}
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

TICK_RAD = 2 * np.pi / 4096
DEFAULT = {"mode": "range_fraction"}


def load_mapping(layout: dict | None = None) -> dict:
    path = os.environ.get("TWIN_JOINT_MAP")
    if path:
        return {**DEFAULT, **json.loads(Path(path).read_text())}
    return {**DEFAULT, **((layout or {}).get("arm", {}).get("joint_mapping") or {})}


def to_q(raw: dict, ranges: dict, lo_rad, hi_rad, mapping: dict | None = None) -> np.ndarray:
    """raw {motor_id: tick}, ranges {motor_id: (min, max)}, URDF limits per joint (8,) -> q (8,) radians."""
    m = {**DEFAULT, **(mapping or {})}
    sign = m.get("sign", [1, 1, 1, 1, 1, 1, 1, -1])
    raw_anchor = m.get("raw_anchor", [2048] * 8)
    anchor = m.get("anchor_rad", [0.0] * 8)
    q = np.zeros(8)
    for i in range(8):
        r = raw[i + 1]
        lo_t, hi_t = ranges.get(i + 1, (0, 0))
        if m["mode"] == "range_fraction" and hi_t > lo_t:
            n = (r - lo_t) / (hi_t - lo_t)
            q[i] = hi_rad[i] * (1 - n) if i == 7 else lo_rad[i] + n * (hi_rad[i] - lo_rad[i])
        else:
            q[i] = anchor[i] + sign[i] * (r - raw_anchor[i]) * TICK_RAD
    return np.clip(q, lo_rad, hi_rad)
