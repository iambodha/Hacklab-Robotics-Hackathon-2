"""Synthetic object estimates for the fake backend (its cameras are placeholders), so the console, the scripted
advisor and the overlay can be exercised end to end without hardware or MuJoCo. Every estimate says it is synthetic.

The blue object follows the jaws while they are closed on it (fake backend: object_norm stops them part-way) and
drops straight down when they open; it lands "in the bin" if that point is inside the bin footprint.
"""
from __future__ import annotations

import numpy as np

from ..contracts import ObjectEstimate
from ..perception import Locator

BLUE_START = (0.20, 0.06, 0.0)
BIN_XY, BIN_SIZE = (0.06, -0.24), (0.14, 0.14, 0.09)


class DemoScene(Locator):
    name = "demo scene (synthetic, fake backend)"

    def __init__(self):
        self.blue = np.array(BLUE_START, float)
        self.held = False

    def tick(self, joints_norm: dict, kin) -> None:
        """Called ~10 Hz by the console's sampler so grasp and release are seen as they happen."""
        g, G = kin.grip_id, kin.P.gripper
        tcp = kin.fk(kin.q_from_norm(joints_norm))[0]
        closed_on = G.closed_norm < joints_norm[g] < G.open_enough
        if closed_on and not self.held and np.linalg.norm(tcp[:2] - self.blue[:2]) < 0.03 and tcp[2] < 0.08:
            self.held = True
        if self.held and joints_norm[g] >= G.open_enough:
            self.held = False
            inside = all(abs(tcp[i] - BIN_XY[i]) < BIN_SIZE[i] / 2 for i in (0, 1))
            self.blue = np.array([tcp[0], tcp[1], 0.03 if inside else 0.0])
        elif self.held:
            self.blue = np.array([tcp[0], tcp[1], max(0.0, tcp[2] - 0.02)])

    def locate(self, frames, joints_norm, kin, camera_info):
        if joints_norm:
            self.tick(joints_norm, kin)
        t = frames["overview"].t_capture if "overview" in frames else None
        return [ObjectEstimate("blue object", np.round(self.blue, 4).tolist(), [0.03, 0.03, 0.02], None, 8.0,
                               "synthetic (fake backend)" + (", in the jaws" if self.held else ""), True, t),
                ObjectEstimate("bin", [*BIN_XY, 0.0], list(BIN_SIZE), 0.0, 10.0, "synthetic (fake backend)", True, t)]
