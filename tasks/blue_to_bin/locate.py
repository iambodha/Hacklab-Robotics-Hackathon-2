"""Thin perception adapter for the executor (executor.perception.Locator interface).

The executor's make_locator() looks for a module-level `perception.locate.locate(images, joints, scene)`, which
perception does not have (it has Locator.locate_object / segment). This adapter fills the gap for the demo task:
the blue blob in the overview frame (only pixels that look at the table), fitted with an upright cylinder of the
KNOWN lid size (measured.json), so the fit gives the centre AND the height of its base above the table (on the
table ~0, lifted in the jaws, or on the tub floor). Calibration: the selected site's scene_calib.json (sites.get_site();
the site twin's overview camera is synced from the same file, so sim and real use the same geometry).
"""
from __future__ import annotations

import cv2
import numpy as np

from executor.contracts import ObjectEstimate
from executor.perception import Locator as ExecutorLocator
from perception.locate import Locator as PerceptionLocator

from .measure import PIXEL_SIGMA, Cam, _fit, circle, table_blob, table_roi
from .task import Spec

NAME = "blue_cap"


class BlueLocator(ExecutorLocator):
    name = "tasks.blue_to_bin (blue blob -> known-size cylinder, scene_calib overview)"

    def __init__(self, spec: Spec | None = None, color: str = "blue", max_rms_px: float = 2.5, rest_below: float = 0.012):
        self.spec = spec or Spec.load()
        self.color, self.max_rms_px, self.rest_below = color, max_rms_px, rest_below
        self.loc = PerceptionLocator()
        self.cam = Cam(self.loc)
        self.roi = table_roi(self.loc, self.cam)
        self.s_cal = self.loc.calib_sigma_mm("overview")

    def locate(self, frames, joints_norm, kin, camera_info):
        f = frames.get("overview")
        if f is None or f.jpeg is None:
            return []
        img = cv2.imdecode(np.frombuffer(f.jpeg, np.uint8), cv2.IMREAD_COLOR)
        r = self.fit(img)
        if not r["ok"]:
            return []
        return [ObjectEstimate(name=NAME, xyz_world=r["xyz"], size=[self.spec.cap_diameter, self.spec.cap_diameter, self.spec.cap_height],
                               yaw=None, uncertainty_mm=r["sigma_mm"], visible=True, t_seen=f.t_capture,
                               source=f"overview seq {f.seq}: {r['how']}, fit {r['rms_px']} px, base z {1000 * r['z_base']:.0f} mm")]

    def fit(self, img: np.ndarray) -> dict:
        if img.shape[:2] != (self.cam.H, self.cam.W):
            img = cv2.resize(img, (self.cam.W, self.cam.H))
        mask, how = table_blob(img, self.roi, self.color)
        if mask is None:
            return {"ok": False, "why": how}
        ys, xs = np.nonzero(mask)
        ext = np.array([xs.min() - 0.5, xs.max() + 0.5, ys.min() - 0.5, ys.max() + 0.5])
        p0 = self.loc.pixel_to_table("overview", float(xs.mean()), float(ys.mean()), self.spec.cap_height / 2)["xyz"]
        R, Hc = self.spec.cap_r, self.spec.cap_height

        def res(x):
            return self.cam.extremes(np.r_[circle(x[:2], R, x[2], 48), circle(x[:2], R, x[2] + Hc, 48)]) - ext
        x, sd, rms = _fit(res, [p0[0], p0[1], 0.0], [p0[0] - 0.15, p0[1] - 0.15, -0.01], [p0[0] + 0.15, p0[1] + 0.15, 0.4])
        resting = x[2] < self.rest_below
        if resting:      # on the table (or the tub floor, 2 mm): the base height is known, so the centre is better fixed
            x2, sd2, rms2 = _fit(lambda y: res([y[0], y[1], 0.0]), x[:2], x[:2] - 0.05, x[:2] + 0.05)
            x, sd, rms = np.r_[x2, 0.0], sd2, rms2
        s_px = 1000 * float(np.hypot(*sd[:2])) * max(1.0, rms / PIXEL_SIGMA)
        sigma = float(np.hypot(s_px, self.s_cal))
        ok = rms <= self.max_rms_px
        return {"ok": ok, "why": None if ok else f"silhouette does not fit a {1000 * 2 * R:.0f} mm lid ({rms:.1f} px)",
                "xyz": [round(float(x[0]), 4), round(float(x[1]), 4), round(float(x[2] + Hc / 2), 4)],
                "z_base": float(x[2]), "sigma_mm": round(sigma, 1), "rms_px": round(rms, 2), "how": how, "px": int(mask.sum())}
