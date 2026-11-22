"""Measurement and the locator on two real overview frames (Station, 22 Nov): the lid on the table, then in the tub."""
import json

import cv2
import numpy as np
import pytest

pytest.importorskip("scipy")

from tasks.blue_to_bin.locate import BlueLocator  # noqa: E402
from tasks.blue_to_bin.measure import DEFAULT_BIN_PIXELS, Cam, measure_bin, measure_cap  # noqa: E402
from tasks.blue_to_bin.task import MEASURED, Spec, judge  # noqa: E402

from .conftest import DATA  # noqa: E402

ON_TABLE, IN_BIN = DATA / "real_overview_on_table.jpg", DATA / "real_overview_in_bin.jpg"


@pytest.fixture(scope="module")
def loc():
    from perception.locate import Locator
    return Locator()


def test_measure_cap_and_bin_on_the_real_frame(loc):
    cam = Cam(loc)
    cap = measure_cap(cv2.imread(str(ON_TABLE)), loc, cam)
    assert cap["ok"] and cap["model_iou"] > 0.7
    assert 0.052 < cap["diameter_m"] < 0.072 and 0.015 < cap["height_m"] < 0.04
    tub = measure_bin(loc, cam, DEFAULT_BIN_PIXELS)
    assert tub["fit_rms_px"] < 2.0
    assert 0.06 < tub["rim_height_m"] < 0.1 and tub["base_diameter_m"] < tub["rim_diameter_m"]
    saved = json.loads(MEASURED.read_text())
    assert np.allclose(cap["centre_xy"], saved["blue_cap"]["centre_xy"], atol=0.003)
    assert np.allclose(tub["centre_xy"], saved["bin"]["centre_xy"], atol=0.003)


def test_background_blue_is_ignored(loc):
    """The top-right of the overview has a blue patch (off the table) larger than the lid."""
    r = BlueLocator(Spec.load()).fit(cv2.imread(str(ON_TABLE)))
    assert r["ok"] and abs(r["z_base"]) < 0.005
    assert np.allclose(r["xyz"][:2], [0.2619, -0.1298], atol=0.01)


def test_predicate_on_real_frames():
    from executor.contracts import ObjectEstimate
    spec = Spec.load()
    L = BlueLocator(spec)
    grip = {"held_after_lift": True, "open_after_release": True}
    out = {}
    for name, f in (("table", ON_TABLE), ("bin", IN_BIN)):
        r = L.fit(cv2.imread(str(f)))
        assert r["ok"], r
        out[name] = judge(spec, ObjectEstimate("blue_cap", r["xyz"], uncertainty_mm=r["sigma_mm"]), grip)
    assert out["table"]["success"] is False
    assert out["bin"]["success"] is True and out["bin"]["overview"]["radial_mm"] < 30
