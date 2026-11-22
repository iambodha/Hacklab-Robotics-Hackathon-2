"""Wrist camera mount: CAD vs measured correction in camera_pose/wrist_camera.json (single source), apply / revert,
and the home twin reading it through pose_file."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))
from camera_pose import wrist_mount  # noqa: E402

TWINFIX = [[0.999967, -0.002271, 0.007853, 0.001517], [-0.008175, -0.277793, 0.960606, 0.022752],
           [0.0, -0.960639, -0.277802, 0.06075], [0, 0, 0, 1]]


def test_active_mount_is_consistent_with_the_recorded_cad_and_correction():
    cfg = json.loads(wrist_mount.CAMERA_JSON.read_text())
    mc = cfg["mount_correction"]
    want = wrist_mount.corrected(cfg) if mc["applied"] else np.array(cfg["T_link_camera_cad"])
    assert np.allclose(cfg["T_link_camera"], want, atol=1e-6)
    assert np.allclose(wrist_mount.corrected(cfg), TWINFIX, atol=1e-5)          # the r2-twinfix matrix


def test_apply_and_revert_round_trip(tmp_path):
    p = tmp_path / "wrist_camera.json"
    p.write_text(wrist_mount.CAMERA_JSON.read_text())
    cad = json.loads(p.read_text())["T_link_camera_cad"]
    cfg = wrist_mount.set_applied(True, p)
    assert cfg["mount_correction"]["applied"] and np.allclose(cfg["T_link_camera"], TWINFIX, atol=1e-5)
    assert abs(cfg["tilt_below_gripper_axis_deg"] - (25.44 - 9.31)) < 0.05
    cfg = wrist_mount.set_applied(False, p)
    assert not cfg["mount_correction"]["applied"] and cfg["T_link_camera"] == cad and cfg["tilt_below_gripper_axis_deg"] == 25.44


def test_sync_from_calib_points_the_twin_at_the_pose_file():
    import sync_from_calib
    layout = json.loads((ROOT / "sim/layouts/home.json").read_text())
    layout["wrist_camera"]["T_link_cam"] = np.eye(4).tolist()                 # a stale inline copy
    calib = {"cameras": {"wrist": {"T_link_cam": np.eye(4).tolist(), "link": "Gripper_Base_v1_1", "fovy_deg": 75.0,
                                   "source": "CAD (camera_pose/wrist_camera.json)"}}}
    out, _ = sync_from_calib.apply_calib(calib, layout, "scene_calib", True)
    assert "T_link_cam" not in out["wrist_camera"] and out["wrist_camera"]["pose_file"] == "camera_pose/wrist_camera.json"
