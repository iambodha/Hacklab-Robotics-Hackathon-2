"""sim/fit_appearance.py helpers (frames, table edge scans, hole fill, texture paths) and build_scene's texture
resolution; no rendering, no Station."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("mujoco")
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "sim"))
import build_scene  # noqa: E402
import fit_appearance as fa  # noqa: E402

CAPTURE = {"joints": {f"j{i}": {"raw": 2000 + i, "lo": 1000, "hi": 3000} for i in range(1, 9)}}
ESTIMATE = {"raw": {str(i): 2000 + i for i in range(1, 9)}, "ranges": {str(i): [1000, 3000] for i in range(1, 9)}}


def test_parse_joints_both_formats():
    a, b = fa.parse_joints(CAPTURE), fa.parse_joints(ESTIMATE)
    assert a == b
    assert a[0][3] == 2003 and a[1][8] == (1000, 3000)


def test_frames_median_drops_a_passing_hand(tmp_path):
    table = np.full((24, 32, 3), 180, np.uint8)
    for k in range(5):
        img = table.copy()
        if k == 2:
            img[5:15, 5:15] = (40, 60, 120)                        # a hand in one frame
        cv2.imwrite(str(tmp_path / f"overview_{k:02d}.jpg"), img)
    cv2.imwrite(str(tmp_path / "pose01_wrist_00.jpg"), np.zeros((24, 32, 3), np.uint8))   # not a wrist_* frame
    (tmp_path / "joints.json").write_text(json.dumps(CAPTURE))
    ov, wrist, raw, ranges, src = fa.load_frames(tmp_path)
    assert np.abs(ov.astype(int) - 180).max() <= 3
    assert wrist is None and raw[1] == 2001 and "median of 5" in src
    cv2.imwrite(str(tmp_path / "overview.jpg"), np.zeros((24, 32, 3), np.uint8))           # a single frame wins
    assert fa.load_frames(tmp_path)[0].max() <= 3


def test_first_off_finds_the_edge_not_an_object():
    t = np.array([1] * 10 + [0] * 4 + [1] * 20 + [0] * 30, bool)   # a 16 mm object, then the real edge at 34
    assert fa.first_off(t, np.ones_like(t)) == 34
    assert fa.first_off(t[::-1], np.ones_like(t)) is None            # starts off the table
    vis = np.ones_like(t); vis[20:] = False                          # leaves the view before the edge
    assert fa.first_off(t, vis) is None


def test_robust_line_ignores_outliers():
    x = np.linspace(-0.6, 0.6, 100)
    y = math.tan(math.radians(2.0)) * x - 0.05
    y[::10] += 0.3                                                   # chair legs, a cup
    slope, off, keep = fa.robust_line(x, y)
    assert abs(math.degrees(math.atan(slope)) - 2.0) < 0.05 and abs(off + 0.05) < 1e-3
    assert keep.sum() == 90


def test_table_from_edges_rotates_the_centre():
    e = {"yaw_deg": 90.0, "x_min": -0.5, "x_max": 0.7, "y_far": -0.1, "y_near": 0.9}
    size, center, yaw = fa.table_from_edges(e)
    assert size == [1.2, 1.0] and yaw == 90.0
    assert np.allclose(center, [-0.4, 0.1], atol=1e-4)              # table-axes centre (0.1, 0.4) turned by 90 deg


def test_push_pull_fill_does_not_drag_a_border_shadow_across():
    img = np.full((80, 80, 3), 200, np.uint8)
    hole = np.zeros((80, 80), bool); hole[20:60, 30:50] = True
    img[19:61, 29] = 60                                              # a dark 1 px shadow along one side
    out = fa.push_pull_fill(img, hole)
    assert (out[~hole] == img[~hole]).all()
    assert out[hole].mean() > 185 and out[40, 45].mean() > 190


def test_texture_paths_keep_home_and_use_site_folders(tmp_path):
    home = json.loads((ROOT / "sites/home/layout_home.json").read_text())
    f, name = fa.texture_target(home, ROOT / "sites/home/layout_home.json")
    assert name == "assets/home_table_texture.png" and f == ROOT / "sim/assets/home_table_texture.png"
    assert build_scene.table_texture_file(home["table"]["texture"], ROOT / "sites/home") == f
    assert build_scene.table_texture_file(home["table"]["texture"]) == f
    office = {"table": {}}
    f, name = fa.texture_target(office, ROOT / "sites/office/layout.json")
    assert f == ROOT / "sites/office/table_texture.png" and name == "../sites/office/table_texture.png"
    assert (build_scene.HERE / name).resolve() == f                 # compare_real.load_model resolves sim/<texture>
    (tmp_path / "t.png").write_bytes(b"")
    assert build_scene.table_texture_file("t.png", tmp_path) == tmp_path / "t.png"
    assert build_scene.table_texture_file("t.png") == build_scene.HERE / "t.png"


def test_office_layout_texture_exists_where_every_consumer_looks():
    layout = json.loads((ROOT / "sites/office/layout.json").read_text())
    tex = layout["table"]["texture"]
    assert (build_scene.HERE / tex).is_file()
    assert build_scene.table_texture_file(tex, ROOT / "sites/office").is_file()
    assert (build_scene.HERE / tex).stat().st_size < 2_000_000


def test_dump_like_keeps_the_file_style():
    obj = {"a": [1, 2], "b": {"c": "d"}}
    one = json.dumps(obj, indent=1)
    assert fa.dump_like(one, {**obj, "e": 3}) == json.dumps({**obj, "e": 3}, indent=1)
    assert fa.dump_like(one + "\n", obj) == one + "\n"
    two = fa.sync_from_calib.dumps(obj) + "\n"
    assert fa.dump_like(two, obj) == two
