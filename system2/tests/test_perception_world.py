import numpy as np

from conftest import make_world
from s2.arm import parse_joints, parse_tcp
from s2.perception import detect_overview

SIM_TABLE = """joint    norm   raw  goal        range in_rng torque status load temp
   j1   0.512  2100  2100 [ 612,3484]      y      1    0x0   12   31
   j2   0.418  1900  1902 [1030,3060]      y      1    0x0  210   31
   j3   0.3    1700  1700 [ 900,3000]      y      1    0x0    0   31
   j4   0.5    2048  2048 [1024,3072]      y      1    0x0    0   31
   j5   0.5    2048  2048 [1024,3072]      y      1    0x0    0   31
   j6   0.5    2048  2048 [1024,3072]      y      1    0x0    0   31
   j7   0.5    2048  2048 [1024,3072]      y      0   0x20    0   31
   j8   0.031   655   612 [ 612,2048]      y      1    0x0 1000   31"""


def test_parse_sim_joint_table_and_tcp():
    j = parse_joints(SIM_TABLE)
    assert len(j) == 8 and j['j8'].norm == 0.031 and j['j8'].load == 1000
    assert j['j7'].status == 0x20 and not j['j7'].torque
    tcp, tilt, yaw, level = parse_tcp('tcp xyz = [0.024, 0.155, 0.018] m; jaws point 2 deg from straight down; '
                                      'jaw closing axis yaw = -15 deg (40 deg from level).\nBase frame ...')
    assert np.allclose(tcp, [0.024, 0.155, 0.018]) and tilt == 2 and yaw == -15 and level == 40
    # Older arm servers do not report the jaw axis level: still parses.
    assert parse_tcp('tcp xyz = [0, 0, 0.1] m; jaws point 2 deg from straight down; jaw closing axis yaw = 3 deg.')[3] is None


def test_camera_round_trip(overview):
    p = np.array([0.05, 0.2, 0.01])
    (uv,), front = overview.project(p)
    assert front[0]
    assert np.allclose(overview.to_plane(*uv, 0.01), p, atol=1e-6)


def _render_square(cam, centre, half=0.01, z=0.0, size=(224, 299)):
    """White table with one black square drawn by projecting its outline (no simulator needed)."""
    import cv2
    img = np.full((*size, 3), 235, np.uint8)
    corners = [[centre[0] + sx * half, centre[1] + sy * half, z] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    uv, _ = cam.project(corners)
    # OpenCV puts pixel centres at integers; the camera model at i + 0.5. Rasterize with 1/16 px precision.
    cv2.fillPoly(img, [((uv - 0.5) * 16).round().astype(np.int32)], (15, 15, 18), shift=4)
    return img


def test_detects_a_dark_object_and_locates_it_on_the_table(overview):
    img = _render_square(overview, (0.05, 0.2), z=0.02)  # top face of a 2 cm cube
    dets = detect_overview(img, overview, arm_points=[[0, 0, 0.05]], workspace=(-0.4, 0.4, -0.1, 0.5))
    assert len(dets) == 1
    p = dets[0].at_height(overview, 0.02)
    assert np.hypot(p[0] - 0.05, p[1] - 0.2) < 0.004


def test_arm_blobs_are_excluded(overview):
    img = _render_square(overview, (0.0, 0.0), half=0.04)  # dark blob over the arm base
    dets = detect_overview(img, overview, arm_points=[[0, 0, 0.0]], workspace=(-0.4, 0.4, -0.1, 0.5))
    assert dets == []


def test_grip_detection_and_hysteresis():
    objs = {'cube': ((0.02, 0.16, 0.01), 'object', 0.02, (0.02, 0.02))}
    w = make_world(tcp=(0.02, 0.16, 0.02), jaw=0.03, jaw_cmd=0.0, load=1000, objects=objs)
    w._update_grasp()
    assert w.entities['cube'].attached
    w.gripper.jaw_load = 50          # load dips while carrying: still held
    w._update_grasp()
    assert w.entities['cube'].attached
    w.gripper.jaw, w.gripper.jaw_cmd = 1.0, 1.0   # opened: released where the gripper is
    w._update_grasp()
    assert not w.entities['cube'].attached and 'released' in w.entities['cube'].note


def test_closing_on_nothing_is_not_a_grip():
    objs = {'cube': ((0.02, 0.16, 0.01), 'object', 0.02, (0.02, 0.02))}
    w = make_world(tcp=(0.02, 0.16, 0.02), jaw=0.0, jaw_cmd=0.0, load=0, objects=objs)
    w._update_grasp()
    assert not w.entities['cube'].attached


def test_container_belief_is_not_dragged_by_a_merged_blob(overview):
    from s2.perception import Detection
    w = make_world(objects={'box': ((0.24, 0.06, 0.045), 'container', 0.09, (0.18, 0.13))})
    w.frame.cameras['overview'] = overview
    contour = np.array([[60.0, 100], [120, 100], [120, 140], [60, 140]])
    # A much bigger blob (arm + box merged) whose bottom row lands 9 cm away: must be ignored.
    big = Detection('o1', 'overview', 90, 120, (40, 60, 140, 150), 9000, 'black', (10, 10, 10), contour, base_u=(40, 140, 150))
    w._associate([big])
    assert np.allclose(w.entities['box'].xyz[:2], [0.24, 0.06]) and not w.entities['box'].visible


def test_camera_distortion_roundtrip_matches_opencv_model():
    from conftest import OVERVIEW
    from s2.geometry import Camera
    cam = Camera.from_info({**OVERVIEW, 'dist': [0.0688, -0.0506, 0.001, -0.002, 0.0]})
    pin = Camera.from_info(OVERVIEW)
    pts = np.array([[x, y, 0.0] for x in (-0.15, 0.0, 0.15) for y in (0.0, 0.1, 0.25)])
    uv, _ = cam.project(pts)
    assert np.abs(uv - pin.project(pts)[0]).max() > 0.5                   # the distortion is really applied
    k1, k2, p1, p2, k3 = cam.dist                                          # independent OpenCV formula on one point
    pc = (pts[0] - cam.t) @ cam.R
    x, y = pc[:2] / pc[2]
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd, yd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    assert np.allclose(uv[0], [cam.fx * xd + cam.cx, cam.fy * yd + cam.cy])
    back = np.array([cam.to_plane(u, v, 0.0) for u, v in uv])
    assert np.abs(back - pts).max() < 1e-6
