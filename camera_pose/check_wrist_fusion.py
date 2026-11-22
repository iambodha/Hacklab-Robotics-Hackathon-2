"""Offline check of the wrist camera chain against the arm foot: rerun perception's arm-base fusion
(perception.calib_scene.calibrate, nothing written) with a given wrist mount pitch and joint zero offsets, and
print how far the wrist-only arm base lands from the foot-based one.

  python camera_pose/check_wrist_fusion.py --frames-dir frames/calib-20261122-013512 \
      --extra-wrist-dir calibration/wrist_pose_0122 --pitch 0 --pitch 9.31 --offsets j1=-2,j2=5.3
Each --pitch x each --offsets (plus "none") is one row. Pitch is about the camera's x axis on top of the CAD mount.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from camera_pose.wrist_mount import CAMERA_JSON, rx  # noqa: E402


def fusion(data: dict, cfg: dict, pitch_deg: float, offsets_deg: dict) -> dict:
    from camera_pose.elrobot_camera import JOINT_OF
    from perception import calib_scene, wrist
    cam = wrist.wrist_camera()
    cls, saved = type(cam), (cam.T_link_cam, type(cam).joints_to_q)
    base = saved[1]

    def j2q(self, state, mapping=None):
        q = base(self, state, mapping)
        for j, d in offsets_deg.items():
            if JOINT_OF.get(j) in q:
                q[JOINT_OF[j]] += math.radians(d)
        return q
    cam.T_link_cam = np.array(json.loads(CAMERA_JSON.read_text())["T_link_camera_cad"]) @ rx(pitch_deg)
    cls.joints_to_q = j2q
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return calib_scene.calibrate(data, cfg)["calib"]["arm"]["base_on_table"]
    finally:
        cam.T_link_cam, cls.joints_to_q = saved


def main():
    from perception import calib_scene
    from perception.scene import CONFIG_PATH
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--extra-wrist-dir", type=Path, action="append", default=[])
    ap.add_argument("--pitch", type=float, action="append")
    ap.add_argument("--offsets", action="append", default=[], help="j1=-2,j2=5.3 (degrees, added after range_fraction)")
    a = ap.parse_args()
    data = calib_scene.load_dir(a.frames_dir)
    for d in a.extra_wrist_dir:
        old = calib_scene.load_dir(d)
        data["wrist_poses"] = data.get("wrist_poses", []) + [{"joints": old["joints"], "frames": old["wrist"], "source": str(d)}]
    cfg = json.loads(CONFIG_PATH.read_text())
    sets = [{}] + [{k: float(v) for k, v in (kv.split("=") for kv in s.split(","))} for s in a.offsets]
    for p in a.pitch or [0.0]:
        for o in sets:
            b = fusion(data, cfg, p, o)
            wo = b.get("wrist_only", {})
            print(f"pitch {p:+6.2f}  offsets {o or 'none'}: wrist-vs-foot {wo.get('mm_off', float('nan')):6.1f} mm, "
                  f"yaw {wo.get('yaw_off_deg', float('nan')):+.2f} deg, wrist-only rms {wo.get('rms_px', float('nan')):.2f} px; "
                  f"fused sigma {b['sigma_mm']:.1f} mm / {b['sigma_yaw_deg']:.2f} deg, fused wrist rms {b.get('wrist_rms_px', float('nan')):.2f} px")


if __name__ == "__main__":
    main()
