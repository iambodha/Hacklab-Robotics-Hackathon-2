"""Wrist camera mount: the CAD pose vs the measured correction, with apply / revert (camera_pose/wrist_camera.json
stays the single source; the twin reads it through sim/layouts/*.json pose_file, perception through WristCamera).

  python camera_pose/wrist_mount.py            # status
  python camera_pose/wrist_mount.py apply      # T_link_camera = CAD @ Rx(pitch)   (rerun calib_scene + sync_from_calib after)
  python camera_pose/wrist_mount.py revert     # T_link_camera = CAD
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

CAMERA_JSON = Path(__file__).resolve().parent / "wrist_camera.json"


def rx(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])


def corrected(cfg: dict) -> np.ndarray:
    """CAD mount rotated about the camera's own x axis by the measured pitch (right-multiplied: camera frame)."""
    return np.array(cfg["T_link_camera_cad"], float) @ rx(cfg["mount_correction"]["pitch_deg"])


def set_applied(applied: bool, path: Path = CAMERA_JSON) -> dict:
    cfg = json.loads(path.read_text())
    T = corrected(cfg) if applied else np.array(cfg["T_link_camera_cad"], float)
    cfg["T_link_camera"] = T.round(6).tolist()
    cfg["optical_axis"] = T[:3, 2].round(5).tolist()
    cfg["tilt_below_gripper_axis_deg"] = round(math.degrees(math.atan2(-T[2, 2], T[1, 2])), 2)
    cfg["mount_correction"]["applied"] = applied
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    return cfg


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd in ("apply", "revert"):
        set_applied(cmd == "apply")
    cfg = json.loads(CAMERA_JSON.read_text())
    mc = cfg["mount_correction"]
    print(f"wrist mount: {'CORRECTED (CAD @ Rx(%+.2f deg))' % mc['pitch_deg'] if mc['applied'] else 'CAD'}; "
          f"measured correction {mc['pitch_deg']:+.2f} +- {mc['sigma_deg']} deg ({mc['source']})")
