"""Geometry, grasp choice and the success predicate (no simulator, no camera)."""
import math

import numpy as np
import pytest

from executor.contracts import ObjectEstimate
from tasks.blue_to_bin.task import Spec, judge, lowered_release, plan_grasp, release_point, truth_in_bin


@pytest.fixture
def spec():
    return Spec.load()


def est(xyz, sigma=8.0, visible=True):
    return ObjectEstimate("blue_cap", list(xyz), uncertainty_mm=sigma, visible=visible, source="test")


def test_measured_values_are_loaded(spec):
    assert spec.source.get("created")
    assert 0.05 < spec.cap_diameter < 0.075 and 0.015 < spec.cap_height < 0.04
    assert 0.09 < spec.bin_rim_diameter < 0.13 and 0.06 < spec.bin_rim_height < 0.1


def test_wide_lid_uses_rim_pinch_and_narrow_object_goes_across(spec):
    assert spec.cap_diameter > spec.max_grip_width and spec.grasp_mode() == "rim"
    assert Spec(cap_diameter=0.03).grasp_mode() == "across"


def test_rim_grasp_puts_the_near_wall_in_the_middle_of_the_jaws(spec):
    c = np.array([0.26, -0.13])
    g = plan_grasp(spec, [*c, 0.013])
    u = c / np.linalg.norm(c)
    assert np.linalg.norm(c - g.tcp_xy) == pytest.approx(spec.cap_r - spec.cap_wall / 2, abs=1e-6)
    assert np.dot(c - g.tcp_xy, u) > 0                                   # the TCP is on the arm's side of the centre
    assert g.yaw_deg == pytest.approx(math.degrees(math.atan2(u[1], u[0])))   # jaws close radially
    assert g.z_hover - spec.cap_height >= 0.05 - 1e-9 and spec.cap_height / 2 < g.z_grasp < spec.cap_height


def test_release_clears_the_rim_and_open_jaws_fit_inside_it(spec):
    g = plan_grasp(spec, [0.26, -0.13, 0.013])
    above = release_point(spec, g)
    assert above[2] - g.z_grasp >= spec.bin_rim_height + spec.rim_margin - 1e-9
    low, fit = lowered_release(spec, g)
    assert fit["fits"] and low[2] > spec.bin_rim_height       # the TCP stays above the rim; the pads reach ~1 cm below it
    tight = Spec.load(bin_rim_diameter=0.06)
    assert not lowered_release(tight, plan_grasp(tight, [0.26, -0.13, 0.013]))[1]["fits"]


def test_tcp_offset_shifts_every_commanded_point_but_not_the_physics(spec):
    lid = [0.26, -0.13, 0.013]
    g0, a0 = plan_grasp(spec, lid), release_point(spec, plan_grasp(spec, lid))
    s = Spec.load(tcp_offset=(0.01, -0.005, 0.004))
    g1 = plan_grasp(s, lid)
    assert np.allclose(g1.tcp_xy - g0.tcp_xy, [0.01, -0.005]) and g1.z_grasp - g0.z_grasp == pytest.approx(0.004)
    assert np.allclose(g1.offset_xy, g0.offset_xy)                         # the lid hangs the same way
    assert np.allclose(release_point(s, g1) - a0, [0.01, -0.005, 0.004])
    assert lowered_release(s, g1)[1] == lowered_release(spec, g0)[1]       # the jaw fit is judged where the jaws really are


def test_judge_needs_camera_and_gripper_evidence(spec):
    ok_grip = {"held_after_lift": True, "open_after_release": True}
    inside = est([*spec.bin_xy, spec.cap_height / 2])
    assert judge(spec, inside, ok_grip)["success"] is True
    assert judge(spec, inside, {**ok_grip, "held_after_lift": False})["success"] is False
    assert judge(spec, inside, {**ok_grip, "open_after_release": None})["success"] is None
    outside = est([spec.bin_xy[0], spec.bin_xy[1] - 0.15, spec.cap_height / 2])
    assert judge(spec, outside, ok_grip)["success"] is False
    on_rim = est([*spec.bin_xy, spec.bin_rim_height + 0.01])          # resting on top of the rim, not inside
    assert judge(spec, on_rim, ok_grip)["success"] is False
    assert judge(spec, None, ok_grip)["success"] is None              # not seen = unknown, never success
    assert judge(spec, est([*spec.bin_xy, 0.013], visible=False), ok_grip)["success"] is None


def test_truth_in_bin(spec):
    b = [*spec.bin_xy, 0.039]
    assert truth_in_bin(spec, [*spec.bin_xy, 0.014], b)
    assert not truth_in_bin(spec, [spec.bin_xy[0] + 0.1, spec.bin_xy[1], 0.013], b)
