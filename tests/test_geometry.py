import math

import numpy as np
import pytest

from perception import aruco, geometry as g, scene_fit
from tests import synth


def test_ray_plane_roundtrip_with_distortion():
    T = synth.overview_cam()
    intr = {**synth.TRUE_INTR, "dist": [-0.25, 0.08, 0.001, -0.001, 0.0]}
    rng = np.random.default_rng(0)
    P = np.c_[rng.uniform(0.0, 0.8, 50), rng.uniform(-0.4, 0.4, 50), np.zeros(50)]
    uv, depth = g.project(T, intr, synth.W, synth.H, P)
    assert (depth > 0).all()
    back = g.pixel_to_plane(T, intr, synth.W, synth.H, uv, 0.0)
    assert np.abs(back - P).max() < 1e-7


def test_pixel_to_plane_known_height():
    T = synth.overview_cam()
    P = np.array([[0.4, 0.1, 0.07]])
    uv, _ = g.project(T, synth.TRUE_INTR, synth.W, synth.H, P)
    assert np.allclose(g.pixel_to_plane(T, synth.TRUE_INTR, synth.W, synth.H, uv, 0.07), P, atol=1e-9)


def test_ray_parallel_and_behind():
    P = g.ray_plane(np.array([0, 0, 1.0]), np.array([[1, 0, 0], [0, 0, 1.0]]))
    assert np.isnan(P).all()


def test_triangulate_rays():
    X = np.array([0.3, -0.1, 0.05])
    o1, o2 = np.array([1.0, 0, 0.7]), np.array([0.0, 0.2, 0.3])
    p, gap, _, _ = g.triangulate_rays(o1, (X - o1) / np.linalg.norm(X - o1), o2, (X - o2) / np.linalg.norm(X - o2))
    assert gap < 1e-12 and np.allclose(p, X)


def test_inverse_and_planar():
    T = g.T_from_rvec([0.1, -0.2, 0.3], [0.5, 0.1, -0.2])
    assert np.allclose(g.inv(T) @ T, np.eye(4))
    assert math.isclose(synth.yaw_of(g.planar_T(0, 0, 0.7)), 0.7)


def test_scene_fit_exact_corners_recovers_pose_and_focal():
    T_wc = synth.overview_cam()
    markers = synth.scene_markers()
    obs = synth.true_corners(T_wc, synth.TRUE_INTR, markers)
    fit = scene_fit.fit(obs, 0.10, synth.W, synth.H, intr=None, ref_id=22)
    assert abs(fit.intr["fx"] - synth.TRUE_INTR["fx"]) / synth.TRUE_INTR["fx"] < 1e-4
    T_table_world = g.inv(markers[22])                       # table frame = marker 22 frame
    for i, T in markers.items():
        x, y, yaw = fit.markers[i]
        Tt = T_table_world @ T
        assert np.hypot(x - Tt[0, 3], y - Tt[1, 3]) < 1e-5
        assert abs(math.remainder(yaw - synth.yaw_of(Tt), 2 * math.pi)) < 1e-4
    assert np.allclose(fit.T_table_cam, T_table_world @ T_wc, atol=1e-5)
    assert fit.rms_px() < 1e-4


def test_render_detect_fit_millimetres():
    """Render real ArUco images, detect them, fit. At 299x224 one pixel is 3-10 mm on the far table,
    so marker layout within 1 cm over a 0.8 m table and focal within 2 % is the honest target."""
    T_wc = synth.overview_cam()
    markers = synth.scene_markers()
    img = synth.render(T_wc, synth.TRUE_INTR, markers)
    name, _ = aruco.autodetect_dictionary([img])
    assert name == "DICT_4X4_50"
    obs = aruco.detect(img, name)
    assert set(obs) == set(markers)
    truth = synth.true_corners(T_wc, synth.TRUE_INTR, markers)
    err = np.concatenate([obs[i] - truth[i] for i in obs])
    assert np.sqrt((err ** 2).sum(1).mean()) < 0.3          # sub-pixel corners after edge refinement
    raw = aruco.detect(img, name, edge_refine=False)
    raw_err = np.concatenate([raw[i] - truth[i] for i in raw])
    assert np.sqrt((err ** 2).sum(1).mean()) < np.sqrt((raw_err ** 2).sum(1).mean())
    fit = scene_fit.fit(obs, 0.10, synth.W, synth.H, intr=None, ref_id=22)
    assert abs(fit.intr["fx"] / synth.TRUE_INTR["fx"] - 1) < 0.02
    T_table_world = g.inv(markers[22])
    for i, T in markers.items():
        Tt = T_table_world @ T
        assert np.hypot(fit.markers[i][0] - Tt[0, 3], fit.markers[i][1] - Tt[1, 3]) < 0.010


@pytest.mark.parametrize("known", [True, False])
def test_held_out_marker_side_lengths(known):
    """0.15 px corner noise -> a few mm on the held-out marker's sides (resolution-limited)."""
    from perception.accuracy import held_out
    T_wc = synth.overview_cam()
    markers = synth.scene_markers()
    obs = synth.true_corners(T_wc, synth.TRUE_INTR, markers)
    rng = np.random.default_rng(1)
    obs = {i: c + rng.normal(0, 0.15, c.shape) for i, c in obs.items()}
    rows = held_out(obs, 0.10, synth.W, synth.H, synth.TRUE_INTR if known else None)
    assert len(rows) == 4
    assert max(r["rms_mm"] for r in rows) < 5.0
    assert max(r["side_err_mm_max"] for r in rows) < 8.0
