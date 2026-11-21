"""Wrist FK + base solve. Needs the norma-core URDF (NORMA_CORE_REPO); skipped without it."""
import math

import numpy as np
import pytest

from perception import geometry as g, wrist
from perception.aruco import marker_object_points
from tests import synth

try:
    wrist.wrist_camera()
except (SystemExit, Exception):  # noqa: BLE001
    pytest.skip("norma-core URDF not available", allow_module_level=True)

POSE = {"j1": {"raw": 2006, "lo": 1056, "hi": 3040}, "j2": {"raw": 1015, "lo": 975, "hi": 3121},
        "j3": {"raw": 1907, "lo": 879, "hi": 3217}, "j4": {"raw": 3196, "lo": 880, "hi": 3216},
        "j5": {"raw": 1854, "lo": 126, "hi": 3970}, "j6": {"raw": 1944, "lo": 1005, "hi": 3091},
        "j7": {"raw": 2204, "lo": 109, "hi": 3987}}


def test_uncalibrated_ranges_fall_back_to_servo_centre():
    flat = {j: {**v, "lo": 2048, "hi": 2048} for j, v in POSE.items()}
    assert np.allclose(wrist.T_base_wristcam(flat), wrist.T_base_wristcam(POSE), atol=1e-9)


def test_reach_direction_is_base_plus_y_and_world_plus_x():
    T = wrist.T_base_wristcam(POSE)
    assert T[1, 2] > 0.8                                       # optical axis along base +y
    Tw = wrist.T_world_base() @ T
    assert Tw[0, 2] > 0.8                                      # ... which is world +x


@pytest.mark.parametrize("known_f", [True, False])
def test_solve_base_from_wrist_recovers_planted_base(known_f):
    T_table_base = g.T_from(g.rot_z(math.radians(-87)), [-0.30, 0.05, 0.0])
    T_base_cam = wrist.T_base_wristcam(POSE)
    T_table_cam = T_table_base @ T_base_cam
    # markers in front of the wrist camera on the table
    fwd = T_table_cam[:3, 2] * np.array([1, 1, 0])
    fwd /= np.linalg.norm(fwd)
    side = np.array([-fwd[1], fwd[0], 0])
    p0 = T_table_cam[:3, 3] * np.array([1, 1, 0])
    T_tm = {20: g.planar_T(*(p0 + 0.45 * fwd - 0.15 * side)[:2], 0.3),
            21: g.planar_T(*(p0 + 0.45 * fwd + 0.15 * side)[:2], -0.2)}
    obs = {i: g.project(T_table_cam, synth.TRUE_INTR, synth.W, synth.H, g.apply(T, marker_object_points(0.1)))[0]
           for i, T in T_tm.items()}
    r = wrist.solve_base_from_wrist(obs, T_tm, T_base_cam, 0.1, synth.W, synth.H,
                                    intr=synth.TRUE_INTR if known_f else None)
    assert np.linalg.norm(r["T_table_base"][:2, 3] - T_table_base[:2, 3]) < 1e-4
    assert abs(math.remainder(wrist.planar_of(r["T_table_base"])[2] - math.radians(-87), 2 * math.pi)) < 1e-4
    if not known_f:
        assert abs(r["intr"]["fx"] / synth.TRUE_INTR["fx"] - 1) < 1e-3


def test_solve_base_from_several_hand_placed_poses():
    T_table_base = g.T_from(g.rot_z(math.radians(-92)), [-0.30, 0.04, 0.0])
    T_tm = {20: g.planar_T(0.10, -0.30, 0.3), 21: g.planar_T(0.12, 0.25, -0.2)}
    obs, Ts = [], []
    for j1 in (1850, 2250):                                     # base turned right / left by hand
        pose = {**POSE, "j1": {**POSE["j1"], "raw": j1}}
        Tc = wrist.T_base_wristcam(pose)
        T_table_cam = T_table_base @ Tc
        o = {}
        for i, T in T_tm.items():
            uv, d = g.project(T_table_cam, synth.TRUE_INTR, synth.W, synth.H, g.apply(T, marker_object_points(0.1)))
            if g.in_image(uv, d, synth.W, synth.H).all():
                o[i] = uv + np.random.default_rng(j1).normal(0, 0.2, uv.shape)
        assert o
        obs.append(o)
        Ts.append(Tc)
    r = wrist.solve_base_from_wrist(obs, T_tm, Ts, 0.1, synth.W, synth.H, intr=None)
    assert r["n_poses"] == 2
    assert np.linalg.norm(r["T_table_base"][:2, 3] - T_table_base[:2, 3]) < 0.01
