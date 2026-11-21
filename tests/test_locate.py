import math

import cv2
import numpy as np
import pytest

from perception import geometry as g
from perception.locate import Locator, footprint, segment
from perception.scene import Scene
from tests import synth


def synth_scene(T_world_cam=None, intr=None) -> Scene:
    T = synth.overview_cam() if T_world_cam is None else T_world_cam
    it = synth.TRUE_INTR if intr is None else intr
    d = {"version": 1, "created": "test", "world_frame": "test",
         "table": {"plane_world": [0, 0, 1, 0], "extent_xy": [[-0.05, -0.45], [0.9, 0.45]], "height_m": 0.0},
         "markers": {"dictionary": "DICT_4X4_50", "size_m": 0.1,
                     "items": [{"id": i, "role": "base", "T_world_marker": M.tolist(), "rms_mm": 0}
                               for i, M in synth.scene_markers().items()]},
         "cameras": {"overview": {"image_size": [synth.W, synth.H], "K_normalized": {k: it[k] for k in ("fx", "fy", "cx", "cy")},
                                  "dist": it.get("dist", [0] * 5), "T_world_cam": T.tolist()},
                     "wrist": {"image_size": [synth.W, synth.H], "K_normalized": None, "T_link_cam": np.eye(4).tolist()}},
         "arm": {"T_world_base": np.eye(4).tolist(), "base_on_table": {"sigma_mm": 5, "sigma_yaw_deg": 0.5}},
         "quality": {"held_out": [{"rms_mm": 1.0}]}}
    return Scene(d)


def render_prism(T, intr, centre, size, yaw, height, color=(0, 0, 220)):
    """Grey table, a coloured box (centre xy, size (l, w), yaw rad, height) drawn as its silhouette."""
    img = np.full((synth.H, synth.W, 3), 170, np.uint8)
    l, w = size
    base = np.array([[l / 2, w / 2], [-l / 2, w / 2], [-l / 2, -w / 2], [l / 2, -w / 2]]) @ g.rot_z(yaw)[:2, :2].T + centre
    V = np.r_[np.c_[base, np.zeros(4)], np.c_[base, np.full(4, height)]]
    ss = 8
    uv, _ = g.project(T, intr, synth.W, synth.H, V)
    big = cv2.resize(img, (synth.W * ss, synth.H * ss), interpolation=cv2.INTER_NEAREST)
    hull = cv2.convexHull(np.float32((uv + 0.5) * ss)).reshape(-1, 2)
    cv2.fillConvexPoly(big, np.int32(np.round(hull)), color, cv2.LINE_AA)
    return cv2.resize(big, (synth.W, synth.H), interpolation=cv2.INTER_AREA)


def test_pixel_world_roundtrip():
    loc = Locator(synth_scene())
    P = [0.4, -0.12, 0.0]
    px = loc.world_to_pixel("overview", P)
    assert px["in_view"]
    r = loc.pixel_to_table("overview", *px["pixel"])
    assert r["ok"] and np.allclose(r["xyz"], P, atol=2e-4)
    assert r["mm_per_px"][0] >= r["mm_per_px"][1] > 0
    assert r["sigma_mm"] >= r["sigma_parts_mm"]["calibration"]


def test_pixel_above_horizon_is_refused():
    loc = Locator(synth_scene())
    assert loc.pixel_to_table("overview", 150, -400)["ok"] is False


REAL_LIKE_T = synth.look_at([0.65, -0.02, 0.38], [0.2, 0.0, 0.0])      # like the calibrated overview camera
REAL_LIKE_INTR = g.intr_from_K(np.array([[145.0, 0, 149], [0, 145.0, 111.5], [0, 0, 1]]), [0] * 5, synth.W, synth.H)


@pytest.mark.parametrize("height", [0.0, 0.06, 0.10])
@pytest.mark.parametrize("centre,yaw_deg", [((0.25, 0.1), 30), ((0.1, -0.2), -50), ((0.4, 0.25), 80)])
def test_locate_prism_footprint(height, centre, yaw_deg):
    T, intr = REAL_LIKE_T, REAL_LIKE_INTR
    centre, size, yaw = np.array(centre), (0.08, 0.05), math.radians(yaw_deg)
    img = render_prism(T, intr, centre, size, yaw, height)
    r = Locator(synth_scene(T, intr)).locate_object("overview", img, color="red", height_m=height)
    assert r["ok"], r
    err = np.linalg.norm(np.array(r["centre_xy"]) - centre) * 1000
    mmpp = r["mm_per_px"]
    assert err < 1.0 + 0.3 * mmpp, (err, mmpp)                    # sub-millimetre to ~2 mm
    assert -0.5 * mmpp < (r["size_m"][0] - size[0]) * 1000 < 1.5 * mmpp     # boundary adds up to ~1 px
    assert -0.5 * mmpp < (r["size_m"][1] - size[1]) * 1000 < 1.5 * mmpp
    assert abs(math.remainder(math.radians(r["yaw_deg"]) - yaw, math.pi)) < math.radians(4)


def test_tall_object_without_height_is_biased_and_reported():
    T, intr = REAL_LIKE_T, REAL_LIKE_INTR
    centre = np.array([0.3, 0.1])
    img = render_prism(T, intr, centre, (0.05, 0.05), 0.0, 0.08)
    loc = Locator(synth_scene(T, intr))
    flat = loc.locate_object("overview", img, color="red", height_m=0.0)
    good = loc.locate_object("overview", img, color="red", height_m=0.08)
    e_flat = np.linalg.norm(np.array(flat["centre_xy"]) - centre)
    e_good = np.linalg.norm(np.array(good["centre_xy"]) - centre)
    assert e_good < e_flat and flat["centre_shift_mm_per_cm_height"] > 1.0


def test_grabcut_bbox_segmentation():
    T, intr = synth.overview_cam(), synth.TRUE_INTR
    img = render_prism(T, intr, np.array([0.45, 0.0]), (0.06, 0.06), 0.0, 0.0, color=(40, 160, 40))
    m, how = segment(img, color="green")
    ys, xs = np.nonzero(m)
    bbox = (xs.min() - 3, ys.min() - 3, xs.max() + 3, ys.max() + 3)
    gm, how = segment(img, bbox=bbox)
    assert how == "grabcut in bbox"
    iou = (gm & m).sum() / (gm | m).sum()
    assert iou > 0.8


def test_triangulate_two_synthetic_cameras(monkeypatch):
    sc = synth_scene()
    loc = Locator(sc)
    T2 = synth.look_at([0.0, 0.0, 0.35], [0.45, 0.0, 0.0])
    cams = {"overview": (synth.overview_cam(), synth.TRUE_INTR, synth.W, synth.H),
            "wrist": (T2, synth.TRUE_INTR, synth.W, synth.H)}
    monkeypatch.setattr(loc, "camera", lambda cam, joints=None: cams[cam])
    X = np.array([0.42, 0.05, 0.03])
    uv1 = g.project(*cams["overview"][:2], synth.W, synth.H, X)[0][0]
    uv2 = g.project(*cams["wrist"][:2], synth.W, synth.H, X)[0][0]
    r = loc.triangulate(uv1, uv2, joints={})
    assert r["ok"] and np.allclose(r["xyz"], X, atol=2e-4) and r["ray_gap_mm"] < 0.5


def test_scene_map_shape():
    m = Locator(synth_scene()).scene_map()
    assert {"table", "markers", "arm_base", "cameras", "objects"} <= set(m)
    assert len(m["markers"]) == 4
    assert all(c is not None for c in m["cameras"]["overview"]["table_footprint_corners_xy"][2:])
