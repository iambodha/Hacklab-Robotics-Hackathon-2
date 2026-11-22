"""Real follower raw ticks -> twin joint angles (URDF radians), configurable.

Modes (layout["arm"]["joint_mapping"], or a JSON file named by env TWIN_JOINT_MAP):
  range_fraction  (default, as real_arm_mcp.py with the elrobot profile)
                  q_i = lower_i + norm_i * (upper_i - lower_i), norm = (raw - range_min) / (range_max - range_min);
                  gripper j8: q = upper_8 * (1 - norm_8)  (norm 1 = open = 0 rad).
                  Needs Station's calibrated range; a joint with range_min == range_max falls back to "ticks".
  ticks           q_i = anchor_rad_i + sign_i * (raw - raw_anchor_i) * 2 pi / 4096  (camera_pose/HANDOFF.md);
                  defaults: anchor 0, raw_anchor 2048, sign +1 (j8: -1, like sim_arm_mcp's raw ticks).
Joint zero offsets: the robot profile's kinematics.joint_offsets_deg {"j1": deg, ...} are added to the
range_fraction angle (before the clip), here, in the executor (Kinematics.q_from_norm) and in perception's FK
(camera_pose/elrobot_camera.py); a mapping may override them with "offsets_deg".
Example override:
  {"mode": "ticks", "raw_anchor": [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048],
   "anchor_rad": [0, 0, 0, 0, 0, 0, 0, 0], "sign": [1, 1, 1, 1, 1, 1, 1, -1]}
"""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import numpy as np

TICK_RAD = 2 * np.pi / 4096
DEFAULT = {"mode": "range_fraction"}
HERE = Path(__file__).resolve().parent


def profile_offsets_deg(path: str | None = None) -> dict:
    """kinematics.joint_offsets_deg of the robot profile (path, else env ROBOT_PROFILE as a path or a name in
    sim/robots/, else elrobot; resolved like robot_profile.load, which needs mujoco). Missing = no offsets."""
    choice = path or os.environ.get("ROBOT_PROFILE") or "elrobot"
    p = Path(choice)
    if not p.exists():
        p = HERE / "robots" / (choice if choice.endswith(".json") else choice + ".json")
    return dict(_offsets_of(str(p), p.stat().st_mtime_ns))


@functools.lru_cache(maxsize=8)
def _offsets_of(path: str, mtime_ns: int) -> tuple:
    return tuple((json.loads(Path(path).read_text())["kinematics"].get("joint_offsets_deg") or {}).items())


def offsets_rad(offsets_deg: dict | None = None, n: int = 8) -> np.ndarray:
    """Per-joint offsets (radians) for j1..jn; offsets_deg None = the robot profile's."""
    d = profile_offsets_deg() if offsets_deg is None else offsets_deg
    return np.radians([float(d.get(f"j{i + 1}", 0.0)) for i in range(n)])


def load_mapping(layout: dict | None = None) -> dict:
    path = os.environ.get("TWIN_JOINT_MAP")
    if path:
        return {**DEFAULT, **json.loads(Path(path).read_text())}
    return {**DEFAULT, **((layout or {}).get("arm", {}).get("joint_mapping") or {})}


def set_arm_q(model, data, q) -> None:
    """Put the twin at joint angles q (8,) for rendering/kinematics, including equality-coupled joints (the jaw
    slides follow rev_motor_08 only through the solver, so a kinematic pose would leave them where they were)."""
    import mujoco
    data.qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]] = q
    for e in range(model.neq):
        if model.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT and model.eq_obj2id[e] >= 0:
            a1, a2 = model.jnt_qposadr[model.eq_obj1id[e]], model.jnt_qposadr[model.eq_obj2id[e]]
            x = data.qpos[a2] - model.qpos0[a2]
            data.qpos[a1] = model.qpos0[a1] + np.polyval(model.eq_data[e, 4::-1], x)
    mujoco.mj_forward(model, data)


def to_q(raw: dict, ranges: dict, lo_rad, hi_rad, mapping: dict | None = None) -> np.ndarray:
    """raw {motor_id: tick}, ranges {motor_id: (min, max)}, URDF limits per joint (8,) -> q (8,) radians."""
    m = {**DEFAULT, **(mapping or {})}
    sign = m.get("sign", [1, 1, 1, 1, 1, 1, 1, -1])
    raw_anchor = m.get("raw_anchor", [2048] * 8)
    anchor = m.get("anchor_rad", [0.0] * 8)
    off = offsets_rad(m.get("offsets_deg"))
    q = np.zeros(8)
    for i in range(8):
        r = raw[i + 1]
        lo_t, hi_t = ranges.get(i + 1, (0, 0))
        if m["mode"] == "range_fraction" and hi_t > lo_t:
            n = (r - lo_t) / (hi_t - lo_t)
            q[i] = (hi_rad[i] * (1 - n) if i == 7 else lo_rad[i] + n * (hi_rad[i] - lo_rad[i])) + off[i]
        else:
            q[i] = anchor[i] + sign[i] * (r - raw_anchor[i]) * TICK_RAD
    return np.clip(q, lo_rad, hi_rad)
