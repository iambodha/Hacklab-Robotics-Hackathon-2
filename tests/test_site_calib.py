import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from perception import aruco, geometry as g
from tests import synth

ROOT = Path(__file__).resolve().parent.parent
W, H = synth.W, synth.H


def _banded_closeup():
    """A marker filling a third of the frame (a wrist close-up) whose black is crossed by thin bright horizontal
    streaks, like the office prints; -> (image, true corners)."""
    T_cam = synth.look_at([0.0, -0.02, 0.22], [0.0, 0.0, 0.0])
    markers = {23: g.planar_T(0.0, 0.0, 0.2)}
    img = synth.render(T_cam, synth.TRUE_INTR, markers)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    rows = np.arange(H) % 3 == 0
    band = gray[rows].astype(int)
    gray[rows] = np.where(band < 100, np.minimum(band + 90, 255), band).astype(np.uint8)
    uv, _ = g.project(T_cam, synth.TRUE_INTR, W, H, g.apply(markers[23], aruco.marker_object_points(0.10)))
    return gray, uv


def test_destripe_finds_banded_marker_accurately():
    img, truth = _banded_closeup()
    found = aruco.detect(img, "DICT_4X4_50", destripe=aruco.DESTRIPE_WRIST_PX)
    assert 23 in found
    assert np.abs(found[23] - truth).max() < 0.6


def test_destripe_decodes_the_real_banded_office_print():
    """Office capture 22 Nov, wrist pose 3: marker 23 fills the view, its black banded by the print; the plain
    detector misses it, the de-striped pass finds a full-size quad."""
    img = cv2.imread(str(ROOT / "tests/data/office_wrist_banded_marker23.jpg"))
    assert 23 not in aruco.detect(img, "DICT_4X4_50")
    found = aruco.detect(img, "DICT_4X4_50", destripe=aruco.DESTRIPE_WRIST_PX)
    assert 23 in found
    sides = np.linalg.norm(found[23] - np.roll(found[23], 1, 0), axis=1)
    assert 60 < sides.min() and sides.max() < 1.3 * sides.min()


def test_destripe_off_by_default_and_plain_detection_wins():
    img = synth.render(synth.overview_cam(), synth.TRUE_INTR, synth.scene_markers())
    plain = aruco.detect(img, "DICT_4X4_50")
    both = aruco.detect(img, "DICT_4X4_50", destripe=aruco.DESTRIPE_OVERVIEW_PX)
    assert set(plain) <= set(both)
    for i in plain:
        np.testing.assert_allclose(both[i], plain[i])


@pytest.fixture(scope="module")
def build_scene():
    pytest.importorskip("mujoco")
    sys.path.insert(0, str(ROOT / "sim"))
    import build_scene as bs
    return bs


K_N = {"fx": 0.479236, "fy": 0.641638, "cx": 0.513426, "cy": 0.526548}   # the overview camera's calibration


def _model(bs, cam):
    import mujoco
    attrs = " ".join(f'{k}="{v}"' for k, v in bs.camera_lens(cam, [W, H]).items())
    xml = f"""<mujoco><visual><global offwidth="{2 * W}" offheight="{2 * H}"/></visual><worldbody>
      <light pos="0 0 3" dir="0 0 -1"/><geom type="plane" size="2 2 .1" rgba="0 0 0 1"/>
      <site name="p" pos="0.1 -0.05 0" size="0.02" rgba="1 1 1 1"/>
      <camera name="c" pos="0 0 1" xyaxes="1 0 0 0 1 0" {attrs}/></worldbody></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def test_camera_lens_compiles_to_the_calibrated_pinhole(build_scene):
    m = _model(build_scene, {"fovy": 75.86, "K_normalized": K_N})
    K = build_scene.camera_K(m, 0, W, H)
    np.testing.assert_allclose(K, [[K_N["fx"] * W, 0, K_N["cx"] * W], [0, K_N["fy"] * H, K_N["cy"] * H], [0, 0, 1]], atol=1e-3)
    K2 = build_scene.camera_K(m, 0, 2 * W, 2 * H)                 # pixel-centre coordinates scale about -0.5
    np.testing.assert_allclose(K2[:2, 2], (K[:2, 2] + 0.5) * 2 - 0.5, atol=1e-3)
    plain = _model(build_scene, {"fovy": 75.86})
    np.testing.assert_allclose(build_scene.camera_K(plain, 0, W, H)[:2, 2], [(W - 1) / 2, (H - 1) / 2])


def test_render_puts_a_point_where_camera_K_projects_it(build_scene):
    import mujoco
    m = _model(build_scene, {"fovy": 75.86, "K_normalized": K_N})
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    try:
        r = mujoco.Renderer(m, H, W)
    except Exception as ex:                        # no GL context (headless CI / sandbox)
        pytest.skip(f"no renderer: {ex}")
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 1
    r.update_scene(d, camera="c", scene_option=opt)
    ys, xs = np.nonzero(r.render()[..., 0] > 128)
    r.close()
    p_cam = np.array([0.1, 0.05, 1.0])             # site in the OpenCV camera frame (camera looks down, image y = -world y)
    uv = build_scene.camera_K(m, 0, W, H) @ (p_cam / p_cam[2])
    assert abs(xs.mean() - uv[0]) < 0.3 and abs(ys.mean() - uv[1]) < 0.3


def test_load_capture_reads_a_calib_scene_dir(tmp_path, build_scene):
    import compare_real
    img = np.full((H, W, 3), 128, np.uint8)
    for name in ("overview_00", "overview_01", "wrist_00", "pose01_wrist_00", "pose01_wrist_01"):
        cv2.imwrite(str(tmp_path / f"{name}.jpg"), img)
    joint = lambda raw: {f"j{i}": {"raw": raw + i, "lo": 100, "hi": 4000} for i in range(1, 9)}
    (tmp_path / "joints.json").write_text(json.dumps({"joints": joint(2000), "joints_after": joint(2000),
                                                      "poses": [joint(1500), joint(1700)]}))
    frames, raw, ranges, poses = compare_real.load_capture(tmp_path)
    assert frames["overview"].shape == (H, W, 3) and len(frames["overview_all"]) == 2
    assert raw[3] == 2003 and ranges[8] == (100, 4000)
    assert [p["name"] for p in poses] == ["pose01"]                # pose 2 has no frames
    assert poses[0]["raw"][1] == 1501 and len(poses[0]["wrist_all"]) == 2
