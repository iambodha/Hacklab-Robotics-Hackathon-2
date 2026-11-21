import cv2
import numpy as np

from perception import calib_intrinsics as ci, geometry as g
from tests import synth


def _render_board(b, T_cam_board, K, W, H, ss=4):
    """Render the board (z=0 plane, origin at its top-left corner) through a pinhole camera."""
    px_per_m = 4000
    bw, bh = int(ci.SQUARES[0] * b.getSquareLength() * px_per_m), int(ci.SQUARES[1] * b.getSquareLength() * px_per_m)
    tile = b.generateImage((bw, bh), marginSize=0, borderBits=1)
    tile = cv2.copyMakeBorder(tile, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    m = 40 / px_per_m
    src = np.float32([[0, 0], [tile.shape[1], 0], [tile.shape[1], tile.shape[0]], [0, tile.shape[0]]]) - 0.5
    corners = np.array([[-m, -m, 0], [bw / px_per_m + m, -m, 0], [bw / px_per_m + m, bh / px_per_m + m, 0],
                        [-m, bh / px_per_m + m, 0]])
    Pc = g.apply(T_cam_board, corners)
    uv = (Pc[:, :2] / Pc[:, 2:]) @ K[:2, :2].T + K[:2, 2]
    dst = (uv * ss + (ss - 1) / 2).astype(np.float32)
    Hm = cv2.getPerspectiveTransform(src, dst)
    big = cv2.warpPerspective(tile, Hm, (W * ss, H * ss), flags=cv2.INTER_LINEAR, borderValue=140)
    return cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)


def test_charuco_calibration_recovers_focal():
    W, H = synth.W, synth.H
    K = g.K_matrix(synth.TRUE_INTR, W, H)
    b = ci.board()
    rng = np.random.default_rng(3)
    views = []
    centre = np.array([ci.SQUARES[0] * b.getSquareLength() / 2, ci.SQUARES[1] * b.getSquareLength() / 2, 0])
    for _ in range(40):
        dist = rng.uniform(0.25, 0.4)
        R = cv2.Rodrigues(np.r_[rng.uniform(-0.5, 0.5, 2), rng.uniform(-0.3, 0.3)])[0]
        t = np.r_[rng.uniform(-0.05, 0.05, 2), dist] - R @ centre
        img = _render_board(b, g.T_from(R, t), K, W, H)
        det = ci.detect_board(img, b)
        if det is not None and len(det[0]) >= 12:
            views.append(det)
    assert len(views) >= 10
    res = ci.calibrate(views, W, H)
    assert res["rms_px"] < 0.5
    assert res["sigma_f_rel"] < 0.03 and max(res["max_tilt_deg"]) > 20
    assert abs(res["K"][0, 0] / K[0, 0] - 1) < 0.02
    assert abs(res["K"][1, 1] / K[1, 1] - 1) < 0.02


def test_board_files(tmp_path):
    paths = ci.make_board(tmp_path)
    img = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)
    assert img.shape == (2480, 3508)                    # A4 landscape at 300 dpi
    det = ci.detect_board(cv2.resize(img, (1200, 848), interpolation=cv2.INTER_AREA), ci.board())
    assert det is not None and len(det[0]) == (ci.SQUARES[0] - 1) * (ci.SQUARES[1] - 1)
