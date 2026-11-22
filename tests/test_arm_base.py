"""Arm base from the overview foot edge and from tape distances (synthetic)."""
import math

import cv2
import numpy as np

from perception import arm_base, geometry as g, scene_fit
from tests import synth

T_WORLD_BASE_TRUE = g.T_from(g.rot_z(math.radians(-90 + 2.0)), [0.004, 0.012, 0.0])   # 2 deg off square


def _fit():
    T_wc, markers = synth.overview_cam(), synth.scene_markers()
    fit = scene_fit.fit(synth.true_corners(T_wc, synth.TRUE_INTR, markers), 0.10, synth.W, synth.H,
                        intr=synth.TRUE_INTR, ref_id=22)
    T_table_world = g.inv(markers[22])                       # exact corners: the table frame is marker 22's frame
    return fit, T_table_world, T_wc, markers


def _render_foot(T_wc, ss=4, foot=True):
    """Bright table up to a far edge at world x = -0.03, dark background beyond, black CAD foot box on the table."""
    W, H = synth.W, synth.H
    K = g.K_matrix(synth.TRUE_INTR, W, H)
    Kb = K.copy()
    Kb[:2] *= ss
    Kb[:2, 2] += (ss - 1) / 2

    def px(P):
        Pc = g.apply(g.inv(T_wc), P)
        uv = (Kb @ (Pc / Pc[:, 2:3]).T).T[:, :2]
        return np.round(uv).astype(np.int32)
    img = np.full((H * ss, W * ss), 30, np.uint8)
    table = np.array([[-0.03, -1.0, 0], [2.0, -1.0, 0], [2.0, 1.0, 0], [-0.03, 1.0, 0]])
    cv2.fillPoly(img, [px(table)], 140)
    box = np.array([[x, y, z] for x in (-0.05, 0.05) for y in (-0.0034, arm_base.FOOT_FRONT_Y) for z in (0, 0.062)])
    if foot:
        cv2.fillPoly(img, [cv2.convexHull(px(g.apply(T_WORLD_BASE_TRUE, box)))], 20)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    return img + np.random.default_rng(0).normal(0, 2.0, img.shape).astype(np.float32)


def test_foot_edge_recovers_base_position():
    fit, T_table_world, T_wc, _ = _fit()
    det = arm_base.detect_foot(_render_foot(T_wc), fit, [22, 23])
    assert det is not None
    F, cov, width = arm_base.foot_front_point(det, fit)
    assert abs(width - 0.100) < 0.008
    F_true = g.apply(T_table_world @ T_WORLD_BASE_TRUE, [[*arm_base.FOOT_FRONT_CENTRE, 0]])[0, :2]
    assert np.linalg.norm(F - F_true) < 0.004
    assert np.sqrt(np.linalg.eigvalsh(cov)).max() < 0.008
    yaw = math.atan2(*(T_table_world @ T_WORLD_BASE_TRUE)[1::-1, 0])
    p = arm_base.base_from_foot(F, yaw)
    assert np.linalg.norm(p - (T_table_world @ T_WORLD_BASE_TRUE)[:2, 3]) < 0.004


def test_no_foot_found_on_bare_table():
    fit, _, T_wc, _ = _fit()
    assert arm_base.detect_foot(_render_foot(T_wc, foot=False), fit, [22, 23]) is None


def test_tape_distances_recover_base():
    _, T_table_world, _, markers = _fit()
    T_tm = {i: T_table_world @ T for i, T in markers.items()}
    T_true = T_table_world @ T_WORLD_BASE_TRUE
    T_guess = T_true @ g.T_from(g.rot_z(math.radians(3)), [0.04, -0.03, 0])
    segs = arm_base.tape_segments(T_guess, T_tm, 0.10, [22, 23])
    assert [s["label"] for s in segs] == list("ABCD")
    assert {s["marker"] for s in segs} == {20, 21, 22, 23}
    tape = {s["label"]: 1000 * float(np.linalg.norm(
        g.apply(T_true, [[*arm_base.FOOT_CORNERS[s["foot_corner"]], 0]])[0, :2] - s["marker_corner_xy"])) for s in segs}
    r = arm_base.solve_tape(segs, tape, T_guess)
    assert np.linalg.norm(r["T_table_base"][:2, 3] - T_true[:2, 3]) < 1e-4
    assert abs(math.remainder(math.atan2(r["T_table_base"][1, 0], r["T_table_base"][0, 0])
                              - math.atan2(T_true[1, 0], T_true[0, 0]), 2 * math.pi)) < 1e-4
    assert r["sigma_mm"] < 3.0 and r["sigma_yaw_deg"] < 1.0           # 1 mm tape -> well-conditioned geometry
