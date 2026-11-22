"""Boustrophedon coverage planner for a rectangular region of the table (world frame, metres).

Lanes are parallel to the stroke axis. With an effective cleaning width w_hat and spacing s = beta * w_hat, the
outermost lane centres sit s / 2 inside the region edges and the lanes in between are evenly spaced at most s
apart (so beta < 1 is a safety margin at the edges as well as an overlap between lanes). Each lane is one wipe stroke (one pass of the existing wipe tool), alternating direction. Strokes run a
little past the region (`overshoot`) so the ends are covered. A stroke budget caps the number of strokes; if the
lanes needed do not fit, the budget's lanes are spread evenly over the region and the plan says so (`capped`).
"""
from __future__ import annotations

from ..config import get as strategy

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Region:
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self):
        if not (self.x1 > self.x0 and self.y1 > self.y0):
            raise ValueError(f"region must have x1 > x0 and y1 > y0: {self}")

    @property
    def size(self):
        return self.x1 - self.x0, self.y1 - self.y0

    @property
    def centre(self):
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2

    def shifted(self, dx: float, dy: float) -> "Region":
        return Region(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)

    def as_list(self):
        return [self.x0, self.y0, self.x1, self.y1]


@dataclass
class Stroke:
    index: int
    start: tuple[float, float]
    end: tuple[float, float]
    lane: int
    segment: int

    @property
    def length(self) -> float:
        return float(math.dist(self.start, self.end))


@dataclass
class WipePlan:
    region: Region
    axis: str                     # "x" or "y": the direction strokes run in
    w_hat: float
    beta: float
    spacing_target: float         # beta * w_hat
    spacing: float                # actual lane-centre spacing (0 for a single lane)
    lanes_needed: int
    lanes: int
    segments: int
    budget: int
    capped: bool
    strokes: list[Stroke]
    passes: int = 1
    notes: list[str] = field(default_factory=list)

    @property
    def expected_gap(self) -> float:
        """Uncleaned strip between neighbouring lanes if w_hat were the true width (0 if they overlap)."""
        return max(0.0, self.spacing - self.w_hat) if self.lanes > 1 else 0.0

    def summary(self) -> dict:
        return {"region": self.region.as_list(), "axis": self.axis, "w_hat_m": round(self.w_hat, 4), "beta": self.beta,
                "spacing_m": round(self.spacing, 4), "lanes": self.lanes, "lanes_needed": self.lanes_needed,
                "segments": self.segments, "strokes": len(self.strokes), "budget": self.budget, "capped": self.capped,
                "passes": self.passes, "notes": self.notes}


def lane_offsets(cross: float, spacing: float, max_lanes: int | None = None) -> tuple[list[float], int]:
    """Lane-centre offsets from the region's low edge across a cross-width `cross`, and the lanes needed. The
    outer lanes sit spacing/2 inside the edges (the same safety factor beta as between lanes)."""
    inset = spacing / 2
    if cross <= 2 * inset:
        return [cross / 2], 1
    need = math.ceil((cross - 2 * inset) / spacing - 1e-9) + 1
    n = need if max_lanes is None else max(1, min(need, max_lanes))
    if n == 1:
        return [cross / 2], need
    return [inset + i * (cross - 2 * inset) / (n - 1) for i in range(n)], need


def plan_coverage(region: Region, w_hat: float, beta: float, budget: int, limits: dict, axis: str = "auto",
                  overshoot: float | None = None, passes: int = 1) -> WipePlan:
    overshoot = strategy("wipe.plan.overshoot_m") if overshoot is None else overshoot
    sx, sy = region.size
    max_len, min_len = limits["max_stroke_m"], limits["min_stroke_m"]
    if axis == "auto":                            # strokes along the longer side, unless only the shorter one fits
        axis = "x" if sx >= sy else "y"
        if (sx if axis == "x" else sy) + 2 * overshoot > max_len and (sy if axis == "x" else sx) + 2 * overshoot <= max_len:
            axis = "y" if axis == "x" else "x"
    along, cross = (sx, sy) if axis == "x" else (sy, sx)
    run = along + 2 * overshoot
    segments = max(1, math.ceil(run / max_len - 1e-9))
    seg_len = run / segments
    notes = []
    if seg_len < min_len:
        seg_len, notes = min_len, notes + [f"strokes lengthened to the tool's minimum {min_len} m"]
    spacing_target = beta * w_hat
    offs, need = lane_offsets(cross, spacing_target, max(1, budget // segments))
    capped = len(offs) < need
    if capped:
        notes.append(f"budget {budget} allows {len(offs)} of {need} lanes: lanes spread wider than beta*w_hat")
    spacing = (offs[1] - offs[0]) if len(offs) > 1 else 0.0
    a0 = (region.x0 if axis == "x" else region.y0) - overshoot + (run - seg_len * segments) / 2
    c0 = region.y0 if axis == "x" else region.x0
    strokes = []
    for seg in range(segments):
        lo, hi = a0 + seg * seg_len, a0 + (seg + 1) * seg_len
        for lane, off in enumerate(offs):
            fwd = (lane + seg) % 2 == 0                    # boustrophedon: alternate direction lane to lane
            s, e = (lo, hi) if fwd else (hi, lo)
            c = c0 + off
            p, q = ((s, c), (e, c)) if axis == "x" else ((c, s), (c, e))
            strokes.append(Stroke(len(strokes), tuple(map(float, p)), tuple(map(float, q)), lane, seg))
    return WipePlan(region, axis, w_hat, beta, spacing_target, spacing, need, len(offs), segments, budget, capped,
                    strokes, passes, notes)


def check_reach(plan: WipePlan, limits: dict, T_world_base=None) -> list[str]:
    """Stroke endpoints that are beyond the arm's horizontal reach (profile max_radius, measured in the base frame)."""
    Tinv = np.linalg.inv(np.asarray(T_world_base, float)) if T_world_base is not None else np.eye(4)
    bad = []
    for s in plan.strokes:
        for p in (s.start, s.end):
            b = Tinv @ np.array([p[0], p[1], 0.0, 1.0])
            if float(np.hypot(b[0], b[1])) > limits["max_radius_m"]:
                bad.append(f"stroke {s.index} point {np.round(p, 3).tolist()} is beyond reach ({limits['max_radius_m']} m)")
    return bad
