"""Evaluator on synthetic board images: drawn stains, strokes of known width, shadows, occluders."""
import numpy as np
import pytest

from skills.eval import Board, Camera, evaluate, homography_from_points, synth
from skills.eval.geometry import apply_H

CAM = synth.default_camera()
H = CAM.H_img_world()
T = synth.TARGET


def lanes(n, w_hat, region=T):
    x0, y0, x1, y1 = region
    ys = [y0 + w_hat / 2 + i * (y1 - y0 - w_hat) / (n - 1) for i in range(n)] if n > 1 else [(y0 + y1) / 2]
    return [((x0 - 0.01, y), (x1 + 0.01, y)) if i % 2 == 0 else ((x1 + 0.01, y), (x0 - 0.01, y)) for i, y in enumerate(ys)]


def run(w_true, strokes, seed=0, offset=(0.0, 0.0), push=0.03, after_kw=None, before_kw=None, cam=CAM, **kw):
    w = synth.SynthWorld().add_stain_rect(T, seed=seed)
    b = w.render(cam, seed=seed, **(before_kw or {}))
    w.wipe(strokes, w_true, push_frac=push, offset=offset)
    a = w.render(cam, seed=seed + 1, **(after_kw or {}))
    return evaluate(b, a, H=cam.H_img_world(), board=synth.BOARD, execution_id="t", strokes=strokes, target=T, **kw), w


def test_homography_matches_projection_and_dlt():
    pts = np.array([[0.2, -0.1], [0.4, -0.1], [0.4, 0.1], [0.2, 0.1], [0.3, 0.0]])
    uv = CAM.project(np.c_[pts, np.zeros(len(pts))])
    assert np.allclose(apply_H(H, pts), uv, atol=1e-6)
    H2 = homography_from_points(uv[:4], pts[:4])
    assert np.allclose(apply_H(H2, pts), uv, atol=1e-6)


def test_scene_calib_loader(tmp_path):
    import json
    p = tmp_path / "scene_calib.json"
    p.write_text(json.dumps(synth.OVERVIEW_CALIB))
    c = Camera.from_scene_calib(p)
    assert np.allclose(c.K, CAM.K) and c.size == (299, 224)


@pytest.mark.parametrize("w_true", [0.02, 0.03, 0.045, 0.06])
def test_width_recovered_single_stroke(w_true):
    ev, _ = run(w_true, lanes(1, 0.03))                     # one stroke through the middle: both edges visible
    assert ev.width.valid, ev.width.why
    assert abs(ev.width.w_observed - w_true) <= 0.005, (ev.width.w_observed, w_true)


@pytest.mark.parametrize("w_true", [0.02, 0.03])
def test_width_recovered_separate_lanes(w_true):
    ev, _ = run(w_true, lanes(2, 0.04))                     # two lanes 60 mm apart
    assert ev.width.valid, ev.width.why
    assert abs(ev.width.w_observed - w_true) <= 0.005, (ev.width.w_observed, w_true)


def test_width_recovered_with_overlapping_lanes():
    ev, _ = run(0.03, lanes(3, 0.05))                       # w_hat overestimated: lanes overlap, outer edges show
    assert ev.width.valid
    assert abs(ev.width.w_observed - 0.03) <= 0.005
    assert ev.outcome.success is False and ev.outcome.value < 0.9


def test_width_finer_camera_is_more_accurate():
    ev, _ = run(0.03, lanes(2, 0.04), cam=synth.default_camera(scale=2))
    assert abs(ev.width.w_observed - 0.03) <= 0.003


def test_lateral_offset_recovered():
    ev, _ = run(0.03, lanes(2, 0.04), offset=(0.0, 0.006))
    assert ev.width.valid and abs(ev.width.offset - 0.006) <= 0.003 and abs(ev.width.w_observed - 0.03) <= 0.005


def test_full_coverage_width_not_identifiable():
    ev, _ = run(0.03, lanes(5, 0.03))
    assert ev.outcome.success is True and ev.outcome.value >= 0.9
    assert not ev.width.valid                               # no dirt beyond the band edges: only a lower bound


def test_shadow_is_not_dirt():
    s = {"shadow": (0.3, 0.0, 0.06, 0.45)}
    ev_shadow, _ = run(0.03, lanes(5, 0.03), after_kw=s)
    ev_plain, _ = run(0.03, lanes(5, 0.03))
    assert ev_shadow.outcome.success is True
    assert abs(ev_shadow.outcome.value - ev_plain.outcome.value) < 0.02


def test_arm_in_image_gives_unknown():
    arm = [[(150, 100), (175, 100), (180, 224), (140, 224)]]       # a dark arm across the board
    ev, _ = run(0.03, lanes(5, 0.03), after_kw={"occluder": arm})
    assert ev.outcome.success is None and "appeared" in ev.outcome.evidence[-1]
    ev2, _ = run(0.03, lanes(5, 0.03), arm_in_view=True)
    assert ev2.outcome.success is None
    ev3, _ = run(0.03, lanes(5, 0.03), stale=True)
    assert ev3.outcome.success is None


def test_board_out_of_view_gives_unknown():
    big = Board(-0.2, -0.6, 0.8, 0.6)
    w = synth.SynthWorld().add_stain_rect(T)
    img = w.render()
    ev = evaluate(img, img, H=H, board=big, execution_id="t")
    assert ev.outcome.success is None


def test_dirt_pushed_off_target_still_counts():
    ev_push, w = run(0.03, lanes(5, 0.03), push=0.6)
    ev_clean, _ = run(0.03, lanes(5, 0.03), push=0.0)
    assert ev_push.stats["dirt_after_outside_px"] > ev_clean.stats["dirt_after_outside_px"] + 50
    assert ev_push.outcome.value < ev_clean.outcome.value - 0.05
    assert ev_push.stats["dirt_after_inside_px"] < 30          # target itself looks clean, the metric still drops


def test_deterministic_on_saved_images(tmp_path):
    w = synth.SynthWorld().add_stain_rect(T, seed=3)
    b = synth.to_jpeg(w.render(seed=3))
    w.wipe(lanes(3, 0.05), 0.03)
    a = synth.to_jpeg(w.render(seed=4))
    (tmp_path / "b.jpg").write_bytes(b)
    (tmp_path / "a.jpg").write_bytes(a)
    res = [evaluate(synth.from_jpeg((tmp_path / "b.jpg").read_bytes()), synth.from_jpeg((tmp_path / "a.jpg").read_bytes()),
                    H=H, board=synth.BOARD, execution_id="d", strokes=lanes(3, 0.05), out_dir=tmp_path) for _ in range(2)]
    assert res[0].outcome.value == res[1].outcome.value and res[0].width.w_observed == res[1].width.w_observed
    assert any(e.endswith(".png") for e in res[0].outcome.evidence)


def test_no_stain_gives_unknown():
    w = synth.SynthWorld()
    img = w.render()
    assert evaluate(img, img, H=H, board=synth.BOARD, execution_id="t").outcome.success is None
