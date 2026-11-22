"""Task "put the blue object in the bin": goal, geometry, grasp choice and the success predicate.

World frame of prompts/scene-contract.md (metres; origin = follower base on the table, +x forward, z up).
Sizes come from measured.json (tasks/blue_to_bin/measure.py); positions are re-located on every attempt.

Success (`judge`) needs BOTH, and neither is the executor's own claim:
  1. overview camera, independent: the blue object is seen, its centre lies inside the bin's rim footprint
     and its top is below the rim (object fitted with its known size, so its height above the table is measured);
  2. gripper evidence: it was holding after the lift, and the jaws are open and empty after the release.
Unknown (not seen, stale) stays unknown: success None, never True.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MEASURED = HERE / "measured.json"

GOAL = "Take the blue object (a jar lid on the table) and put it in the bin (the transparent tub)."


@dataclass
class Spec:
    cap_diameter: float = 0.062
    cap_height: float = 0.026
    cap_wall: float = 0.002
    bin_xy: tuple = (0.2545, 0.1234)
    bin_rim_diameter: float = 0.1094
    bin_base_diameter: float = 0.0875
    bin_rim_height: float = 0.078
    bin_sigma_mm: float = 8.6
    max_grip_width: float = 0.04         # profile gripper.max_grip_width (jaw gap ~2 x 25.5 mm fully open)
    grasp: str = "auto"                  # auto | across | rim
    hover_above: float = 0.05            # TCP above the object's top before the descent
    grasp_above_mid: float = 0.008       # TCP above the object's mid-height when closing (profile pick.grasp_above_mid)
    test_lift: float = 0.04              # lift after closing, then look: is the object coming with the jaws?
    carry_z: float = 0.20
    rim_margin: float = 0.04             # object bottom above the bin rim when carried over it
    release_depth: float = 0.006         # TCP this far above the rim when the jaws open (pad bottoms ~1 cm below the rim)
    release_shift: float = 0.5           # fraction of the grasp offset the TCP is moved off the bin centre
    retreat_z: float = 0.20
    grip_force: float = 0.5
    max_attempts: int = 3
    moved_tol: float = 0.015             # re-plan the approach if the object moved more than this
    sigma_max_mm: float = 20.0           # do not act on a location worse than this
    tcp_offset: tuple = (0.0, 0.0, 0.0)  # world xyz added to every commanded TCP point near the objects (dry run / touch_table)
    source: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = MEASURED, **over) -> "Spec":
        s = cls()
        if path.exists():
            d = json.loads(path.read_text())
            c, b = d.get("blue_cap", {}), d.get("bin", {})
            if c.get("ok"):
                s.cap_diameter, s.cap_height = float(c["diameter_m"]), float(c["height_m"])
            if b.get("ok"):
                s.bin_xy = tuple(b["centre_xy"])
                s.bin_rim_diameter, s.bin_base_diameter = float(b["rim_diameter_m"]), float(b["base_diameter_m"])
                s.bin_rim_height, s.bin_sigma_mm = float(b["rim_height_m"]), float(b["sigma_mm"]["centre"])
            s.source = {"file": str(path), "created": d.get("created"), "frames": d.get("source")}
        for k, v in over.items():
            setattr(s, k, v)
        return s

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def cap_r(self) -> float:
        return self.cap_diameter / 2

    @property
    def offset_xy(self) -> np.ndarray:
        return np.asarray(self.tcp_offset, float)[:2]

    @property
    def offset_z(self) -> float:
        o = tuple(self.tcp_offset)
        return float(o[2]) if len(o) > 2 else 0.0

    def grasp_mode(self) -> str:
        """across = jaws straddle the whole object (needs it narrower than the jaws can grip);
        rim = pinch the lid's wall with one pad inside (lid lying open side up)."""
        if self.grasp != "auto":
            return self.grasp
        return "across" if self.cap_diameter <= self.max_grip_width else "rim"


@dataclass
class Grasp:
    mode: str
    tcp_xy: np.ndarray          # where the TCP goes down
    yaw_deg: float              # world yaw of the jaw closing axis
    z_grasp: float              # TCP z when closing
    z_hover: float
    offset_xy: np.ndarray       # object centre - TCP (world xy), constant while the yaw is held

    def describe(self) -> dict:
        return {"mode": self.mode, "tcp_xy": np.round(self.tcp_xy, 4).tolist(), "yaw_deg": round(self.yaw_deg, 1),
                "z_grasp": round(self.z_grasp, 4), "z_hover": round(self.z_hover, 4),
                "object_offset_mm": np.round(self.offset_xy * 1000, 1).tolist()}


def plan_grasp(spec: Spec, cap_xy) -> Grasp:
    c = np.asarray(cap_xy, float)[:2]
    u = c / max(np.linalg.norm(c), 1e-6)             # from the arm base towards the object
    yaw = math.degrees(math.atan2(u[1], u[0]))       # jaws close radially: one pad inside the near wall
    off, dz = spec.offset_xy, spec.offset_z
    z_grasp = spec.cap_height / 2 + spec.grasp_above_mid + dz
    z_hover = spec.cap_height + spec.hover_above + dz
    if spec.grasp_mode() == "across":
        return Grasp("across", c + off, yaw, z_grasp, z_hover, np.zeros(2))
    tcp = c - (spec.cap_r - spec.cap_wall / 2) * u   # the wall nearest the arm, in the middle of the jaw gap
    return Grasp("rim", tcp + off, yaw, z_grasp, z_hover, c - tcp)


JAW_REACH = 0.031    # TCP -> outer face of a pad along the closing axis, jaws open (24.5 mm half gap + 6 mm pad)


def release_point(spec: Spec, g: Grasp) -> np.ndarray:
    """TCP over the bin, object bottom clearing the rim + margin. The TCP sits `release_shift` of the grasp offset
    off the bin centre: the object hangs (tilted) between centred and offset, and both open pads stay inside the rim."""
    xy = np.asarray(spec.bin_xy, float) - spec.release_shift * g.offset_xy + spec.offset_xy
    return np.r_[xy, spec.bin_rim_height + spec.rim_margin + g.z_grasp]      # z_grasp already carries the z offset


def lowered_release(spec: Spec, g: Grasp) -> tuple[np.ndarray, dict]:
    """TCP point inside the rim (pads `release_depth` below it) where the jaws open, and the check that the open jaws
    fit inside the rim there. Opening above the rim leaves a pinched lid hanging on a pad (seen in the twin)."""
    above = release_point(spec, g)
    p = np.r_[above[:2], spec.bin_rim_height + spec.release_depth + spec.offset_z]
    u = np.array([math.cos(math.radians(g.yaw_deg)), math.sin(math.radians(g.yaw_deg))])
    c = np.asarray(spec.bin_xy, float)
    actual = p[:2] - spec.offset_xy                              # where the TCP really is once the offset is applied
    reach = max(np.linalg.norm(actual + s * JAW_REACH * u - c) for s in (-1, 1))
    inner = spec.bin_rim_diameter / 2 - 0.002
    return p, {"jaw_reach_from_bin_centre_mm": round(reach * 1000, 1), "rim_inner_radius_mm": round(inner * 1000, 1),
               "fits": bool(reach <= inner - 0.004)}


# ---- the success predicate -----------------------------------------------------------------------------------
def in_bin_from_overview(spec: Spec, est) -> dict:
    """est: the blue object's ObjectEstimate (or None). Inside = centre within the rim footprint, top below the rim."""
    if est is None or not est.visible:
        return {"result": None, "why": "blue object not seen by the overview camera (unknown, not absent)"}
    xyz = np.asarray(est.xyz_world, float)
    r = float(np.linalg.norm(xyz[:2] - np.asarray(spec.bin_xy)))
    top = float(xyz[2] + spec.cap_height / 2)
    tol = 0.001 * (est.uncertainty_mm or 10.0)
    inside = r <= spec.bin_rim_diameter / 2
    below = top <= spec.bin_rim_height
    return {"result": bool(inside and below), "radial_mm": round(r * 1000, 1), "rim_radius_mm": round(spec.bin_rim_diameter * 500, 1),
            "top_z_mm": round(top * 1000, 1), "rim_z_mm": round(spec.bin_rim_height * 1000, 1),
            "sigma_mm": est.uncertainty_mm, "margin_mm": round((spec.bin_rim_diameter / 2 - r) * 1000, 1),
            "confident": bool(abs(spec.bin_rim_diameter / 2 - r) > tol and abs(spec.bin_rim_height - top) > tol),
            "source": est.source}


def judge(spec: Spec, overview_est, gripper: dict) -> dict:
    """gripper: {"held_after_lift": bool|None, "open_after_release": bool|None, "empty_after_release": bool|None}"""
    cam = in_bin_from_overview(spec, overview_est)
    grip_ok = None if any(gripper.get(k) is None for k in ("held_after_lift", "open_after_release")) else \
        bool(gripper["held_after_lift"] and gripper["open_after_release"] and gripper.get("empty_after_release", True))
    if cam["result"] is False or grip_ok is False:
        success = False
    elif cam["result"] is None or grip_ok is None:
        success = None
    else:
        success = True
    return {"success": success, "overview": cam, "gripper": {**gripper, "result": grip_ok}}


def truth_in_bin(spec: Spec, cap_world, bin_world) -> bool:
    """Simulator ground truth (reporting only, never used by the predicate)."""
    cap, b = np.asarray(cap_world, float), np.asarray(bin_world, float)
    return bool(np.linalg.norm(cap[:2] - b[:2]) <= spec.bin_rim_diameter / 2 and cap[2] + spec.cap_height / 2 <= spec.bin_rim_height + 0.005)
