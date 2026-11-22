"""kinematics.joint_offsets_deg of the robot profile: the twin (sim/real_joints.to_q), the executor
(Kinematics.q_from_norm / norm_from_q) and perception's FK (camera_pose/elrobot_camera.joints_to_q) all add the
same offsets. Needs the norma-core URDF (NORMA_CORE_REPO); skipped without it."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from perception import wrist

try:
    wrist.wrist_camera()
except (SystemExit, Exception):  # noqa: BLE001
    pytest.skip("norma-core URDF not available", allow_module_level=True)

ROOT = Path(wrist.ROOT)
sys.path.append(str(ROOT / "sim"))
import real_joints  # noqa: E402

from tests.test_wrist import POSE  # noqa: E402

OFFSETS = {"j1": -1.0, "j2": 3.0, "j5": -4.0, "j6": 1.5, "j8": 2.0}
NAMES = [f"rev_motor_0{i}" for i in range(1, 9)]


@pytest.fixture
def offset_profile(tmp_path, monkeypatch):
    prof = json.loads((ROOT / "sim/robots/elrobot.json").read_text())
    prof["kinematics"]["joint_offsets_deg"] = OFFSETS
    p = tmp_path / "elrobot_offsets.json"
    p.write_text(json.dumps(prof))
    monkeypatch.setenv("ROBOT_PROFILE", str(p))
    return p


def _limits():
    lim = wrist.wrist_camera().urdf.limits
    return np.array([lim[n][0] for n in NAMES]), np.array([lim[n][1] for n in NAMES])


def _norm(j):
    v = POSE[j]
    return (v["raw"] - v["lo"]) / (v["hi"] - v["lo"])


def test_shipped_profile_applies_no_offsets():
    assert all(v == 0 for v in real_joints.profile_offsets_deg().values())


def test_twin_and_perception_add_the_profile_offsets(offset_profile):
    lo, hi = _limits()
    raw = {i: POSE[f"j{i}"]["raw"] for i in range(1, 8)} | {8: 2000}
    ranges = {i: (POSE[f"j{i}"]["lo"], POSE[f"j{i}"]["hi"]) for i in range(1, 8)} | {8: (1228, 2868)}
    q_twin = real_joints.to_q(raw, ranges, lo, hi)
    q_plain = real_joints.to_q(raw, ranges, lo, hi, {"offsets_deg": {}})
    q_fk = wrist.joints_to_q(POSE)
    for i in range(7):
        d = math.radians(OFFSETS.get(f"j{i + 1}", 0.0))
        assert q_twin[i] == pytest.approx(q_plain[i] + d, abs=1e-12)
        assert q_fk[NAMES[i]] == pytest.approx(q_twin[i], abs=1e-9)
    assert q_twin[7] == pytest.approx(q_plain[7] + math.radians(OFFSETS["j8"]), abs=1e-12)
    assert wrist.joints_to_q(POSE, {"offsets_deg": {}})[NAMES[1]] == pytest.approx(q_plain[1], abs=1e-12)


def test_executor_adds_the_same_offsets_and_inverts_them(offset_profile):
    try:
        import robot_profile
        from executor.kinematics import Kinematics
        kin = Kinematics(profile=robot_profile.load(str(offset_profile)))
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"executor kinematics unavailable: {e}")
    norm = {i: _norm(f"j{i}") for i in range(1, 8)} | {8: 0.5}
    q_exec = kin.q_from_norm(norm)
    q_fk = wrist.joints_to_q(POSE)
    assert np.allclose([q_fk[n] for n in NAMES[:7]], q_exec[:7], atol=1e-4)
    back = kin.norm_from_q(q_exec)
    assert all(back[i] == pytest.approx(norm[i], abs=1e-9) for i in range(1, 8))
    plain = Kinematics(profile=robot_profile.load(str(ROOT / "sim/robots/elrobot.json"))).q_from_norm(norm)
    assert q_exec[7] - plain[7] == pytest.approx(math.radians(OFFSETS["j8"]), abs=1e-12)
