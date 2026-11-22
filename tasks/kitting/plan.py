"""The scripted pick-and-place plan for one kit object: the same function drives the twin (sim_env.py) and the real
arm (run_real.py), so the learned numbers mean the same thing on both.

All in the table (arm base) frame, metres and degrees. Yaw = direction of the jaw closing axis (build_scene.ArmIK).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .spec import Pose2D, axis_wrap_deg

# Motion constants of the plan (not learned): where they come from.
TRAVEL_Z = 0.08          # profile workspace.travel_height: sideways moves at this TCP height (pads 64 mm up)
HOVER_ABOVE = 0.05       # profile pick.hover_above: hover this far above the grasp height before the descent
HOVER_MIN = 0.07         # profile pick.hover_min
PAD_REACH = 0.010        # the fingertips reach 10 mm below the TCP: profile touch.nominal_table_tcp_z (real arm, 21 Nov);
                         # the kitting twin's pads are moved to match (sim/kitting_scene.PAD_TIP_SHIFT_M)


@dataclass
class ObjectPlan:
    name: str
    cls: str
    pick_xy: tuple[float, float]
    yaw_pick: float
    grasp_z: float
    hover_z: float
    place_xy: tuple[float, float]
    yaw_place: float
    place_z: float
    grip_force: float
    release_open: float
    travel_z: float = TRAVEL_Z

    def describe(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v] if isinstance(v, tuple) else v)
                for k, v in asdict(self).items()}


def plan_pick_place(target: dict, params: dict, round_obj: bool = False) -> ObjectPlan:
    """target: one entry of Spec.targets(sheet_pose) (start / goal Pose2D); params: that class's kit.* values."""
    s, g = target["start"], target["goal"]
    yaw = axis_wrap_deg(s.yaw_deg + (0.0 if round_obj else params["yaw_deg"]))
    grasp_z = params["grasp_z_mm"] / 1000
    return ObjectPlan(target["name"], target["class"], (s.x, s.y), yaw, grasp_z, max(grasp_z + HOVER_ABOVE, HOVER_MIN),
                      (g.x, g.y), axis_wrap_deg(yaw + g.yaw_deg - s.yaw_deg), params["place_z_mm"] / 1000,
                      params["grip_force"], params["release_open"])


def order(targets: list[dict]) -> list[dict]:
    """Pick order: the objects nearest the arm base first (the arm never reaches over an object it has not moved yet
    at low height), ties by x."""
    return sorted(targets, key=lambda t: (round(t["start"].y, 3), t["start"].x))


def as_pose(p) -> Pose2D:
    return p if isinstance(p, Pose2D) else Pose2D(*p)
