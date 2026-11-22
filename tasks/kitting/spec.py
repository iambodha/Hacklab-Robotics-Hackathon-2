"""The kit sheet spec (kit_a4.json) and its geometry: sheet frame <-> table (arm base) frame, marker corners, cells,
start spots, object classes. Pure numpy; every other kitting module goes through this one.

Frames
  sheet  mm, origin = page centre, +x = right edge of the printed page, +y = top edge, z out of the paper.
  table  metres, the arm base frame of the twin (= sim frame, the SITE's layout, sites.get_site().layout): +y = arm
         forward, +x = the arm's right, z up, z = 0 the table top. The executor's world frame is a rotation of it
         (kin.to_world).
A sheet pose is (x_m, y_m, yaw_deg) of the sheet origin in the table frame; the paper lies flat on z = 0. The spec's
placement is the home pose; a site may place the sheet elsewhere (sites/<site>/tasks.json "kitting", site_sheet_pose).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SITES = HERE.parents[1] / "sites"
DEFAULT_SPEC = HERE / "kit_a4.json"
TABLE_MARKER_IDS = (20, 21, 22, 23)      # home + office table markers; table_marker_ids() reads every site layout


def table_marker_ids() -> tuple[int, ...]:
    """Marker ids on any site's table (the same printed sheet travels between sites: it must reuse none of them)."""
    ids = set(TABLE_MARKER_IDS)
    for manifest in SITES.glob("*/site.json"):
        layout = SITES.parent / json.loads(manifest.read_text())["files"]["layout"]
        if layout.is_file():
            ids |= {int(m["id"]) for m in json.loads(layout.read_text()).get("markers", {}).get("items", [])}
    return tuple(sorted(ids))


def site_tasks(site: str) -> dict:
    """sites/<site>/tasks.json (task placements measured for that table), {} if the site has none or is a layout name."""
    path = SITES / site / "tasks.json"
    return json.loads(path.read_text()) if path.is_file() else {}


@dataclass(frozen=True)
class Pose2D:
    """Rigid 2D pose: x, y (metres in the table frame, or mm in the sheet frame) and yaw (deg)."""
    x: float
    y: float
    yaw_deg: float = 0.0

    def R(self) -> np.ndarray:
        a = math.radians(self.yaw_deg)
        return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])

    def apply(self, pts) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(pts, float))
        return pts @ self.R().T + [self.x, self.y]

    def compose(self, other: "Pose2D", scale: float = 1.0) -> "Pose2D":
        """self * other, other's translation multiplied by scale first (mm in the sheet -> m in the table)."""
        p = self.apply([other.x * scale, other.y * scale])[0]
        return Pose2D(float(p[0]), float(p[1]), wrap_deg(self.yaw_deg + other.yaw_deg))

    def as_list(self) -> list[float]:
        return [round(self.x, 5), round(self.y, 5), round(self.yaw_deg, 3)]


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def axis_wrap_deg(a: float) -> float:
    """An undirected axis angle in (-90, 90]: a jaw closing axis or a box's long side is the same after 180 deg."""
    a = wrap_deg(a)
    return a - 180.0 if a > 90.0 else a + 180.0 if a <= -90.0 else a


@dataclass
class Spec:
    raw: dict
    path: Path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Spec":
        p = Path(path or DEFAULT_SPEC)
        s = cls(json.loads(p.read_text()), p)
        errs = s.problems()
        if errs:
            raise ValueError(f"{p}: " + "; ".join(errs))
        return s

    # --- the parts -------------------------------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.raw["name"]

    @property
    def paper_mm(self) -> tuple[float, float]:
        w, h = self.raw["paper"]["size_mm"]
        return float(w), float(h)

    @property
    def marker_size_mm(self) -> float:
        return float(self.raw["markers"]["size_mm"])

    @property
    def dictionary(self) -> str:
        return self.raw["markers"]["dictionary"]

    def markers(self) -> dict[int, Pose2D]:
        return {int(m["id"]): Pose2D(*m["center"], m.get("yaw_deg", 0.0)) for m in self.raw["markers"]["items"]}

    def cells(self) -> dict[str, dict]:
        size = self.raw["grid"]["cell_size"]
        return {c["id"]: {"center": np.array(c["center"], float), "size": np.array(c.get("size", size), float),
                          "label": c.get("label", c["id"]), "yaw_deg": float(c.get("yaw_deg", 0.0))}
                for c in self.raw["cells"]}

    def spots(self) -> dict[str, Pose2D]:
        return {s["id"]: Pose2D(*s["center"], s.get("yaw_deg", 0.0)) for s in self.raw["start_spots"]}

    def classes(self) -> dict[str, dict]:
        return self.raw["classes"]

    def objects(self) -> list[dict]:
        return self.raw["objects"]

    def object(self, name: str) -> dict:
        return next(o for o in self.objects() if o["name"] == name)

    def nominal_sheet_pose(self) -> Pose2D:
        p = self.raw["placement"]
        return Pose2D(*p["sheet_xy_m"], p.get("sheet_yaw_deg", 0.0))

    def site_sheet_pose(self, site: str) -> Pose2D:
        """The sheet pose on a site's table: its tasks.json "kitting" placement, else the spec's nominal (home)."""
        p = site_tasks(site).get("kitting")
        return Pose2D(*p["sheet_xy_m"], p.get("sheet_yaw_deg", 0.0)) if p else self.nominal_sheet_pose()

    # --- geometry in the sheet frame (mm) ----------------------------------------------------------------------
    def marker_corners_mm(self, mid: int) -> np.ndarray:
        """(4, 2) corners of marker mid's black square in the sheet frame, OpenCV order TL, TR, BR, BL (marker x right,
        y up, as sim/build_scene.add_markers)."""
        h = self.marker_size_mm / 2
        return self.markers()[mid].apply([[-h, h], [h, h], [h, -h], [-h, -h]])

    def cell_corners_mm(self, cid: str) -> np.ndarray:
        c = self.cells()[cid]
        w, h = c["size"] / 2
        return Pose2D(*c["center"], c["yaw_deg"]).apply([[-w, h], [w, h], [w, -h], [-w, -h]])

    # --- sheet frame <-> table frame ---------------------------------------------------------------------------
    @staticmethod
    def to_table(sheet_pose: Pose2D, pts_mm) -> np.ndarray:
        """Sheet points (mm) -> table points (m) for a sheet lying at sheet_pose."""
        return sheet_pose.apply(np.atleast_2d(np.asarray(pts_mm, float)) / 1000.0)

    @staticmethod
    def to_sheet(sheet_pose: Pose2D, pts_m) -> np.ndarray:
        """Table points (m) -> sheet points (mm)."""
        d = np.atleast_2d(np.asarray(pts_m, float)) - [sheet_pose.x, sheet_pose.y]
        return 1000.0 * d @ sheet_pose.R()

    def cell_pose(self, sheet_pose: Pose2D, cid: str) -> Pose2D:
        c = self.cells()[cid]
        return sheet_pose.compose(Pose2D(*c["center"], c["yaw_deg"]), scale=1e-3)

    def spot_pose(self, sheet_pose: Pose2D, sid: str) -> Pose2D:
        return sheet_pose.compose(self.spots()[sid], scale=1e-3)

    def targets(self, sheet_pose: Pose2D) -> list[dict]:
        """Per object: its class, start pose and target cell pose in the table frame (what the planner needs)."""
        return [{"name": o["name"], "class": o["class"], "spot": o["spot"], "cell": o["cell"],
                 "start": self.spot_pose(sheet_pose, o["spot"]), "goal": self.cell_pose(sheet_pose, o["cell"])}
                for o in self.objects()]

    # --- object classes ----------------------------------------------------------------------------------------
    def class_dims(self, cls: str) -> dict:
        """Resting height (m), footprint half-extents (m) along the object's long axis and across it, mass (kg)."""
        c = self.classes()[cls]
        if c["shape"] == "screw":
            L = c["shank_length"] + c["head_height"]
            return {"height": c["head_diameter"] / 1000, "grip_width": c["shank_diameter"] / 1000,
                    "half_long": L / 2000, "half_across": c["head_diameter"] / 2000, "mass": c["mass_g"] / 1000,
                    "round": False}
        if c["shape"] == "cylinder":
            r = c["diameter"] / 2000
            return {"height": c["height"] / 1000, "grip_width": 2 * r, "half_long": r, "half_across": r,
                    "mass": c["mass_g"] / 1000, "round": True}
        if c["shape"] == "box":
            sx, sy, sz = c["size"]
            return {"height": sz / 1000, "grip_width": sy / 1000, "half_long": sx / 2000, "half_across": sy / 2000,
                    "mass": c["mass_g"] / 1000, "round": False}
        raise ValueError(f"unknown shape {c['shape']!r}")

    # --- validation --------------------------------------------------------------------------------------------
    def problems(self) -> list[str]:
        out = []
        for label, rows, field in (("marker", self.raw["markers"]["items"], "id"),
                                   ("cell", self.raw["cells"], "id"),
                                   ("spot", self.raw["start_spots"], "id"),
                                   ("object", self.objects(), "name")):
            ids = [row[field] for row in rows]
            if len(set(ids)) != len(ids):
                out.append(f"duplicate {label} identifier")
        if self.dictionary != "DICT_4X4_50":
            out.append("sheet dictionary must be DICT_4X4_50")
        if self.marker_size_mm <= 0:
            out.append("marker size must be positive")
        for mid in self.markers():
            if not 0 <= mid < 50:
                out.append(f"marker id {mid} outside DICT_4X4_50")
        for cid, cell in self.cells().items():
            if np.any(cell["size"] <= 0):
                out.append(f"cell {cid} must have positive size")
        W, H = self.paper_mm
        margin = 4.0                      # printers cannot print the outer ~4 mm of a page
        inside = lambda pts: bool(np.all(np.abs(pts[:, 0]) <= W / 2 - margin) and np.all(np.abs(pts[:, 1]) <= H / 2 - margin))
        boxes = {}
        table_ids = table_marker_ids()
        for mid in self.markers():
            if mid in table_ids:
                out.append(f"marker id {mid} is a table marker {list(table_ids)}")
            q = self.raw["markers"].get("quiet_zone_mm", 0.0)
            c = self.marker_corners_mm(mid)
            if not inside(c):
                out.append(f"marker {mid} is outside the printable page")
            boxes[f"marker {mid}"] = (c.min(0) - q, c.max(0) + q)
        for cid in self.cells():
            c = self.cell_corners_mm(cid)
            if not inside(c):
                out.append(f"cell {cid} is outside the printable page")
            boxes[f"cell {cid}"] = (c.min(0), c.max(0))
        names = list(boxes)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                (a0, a1), (b0, b1) = boxes[a], boxes[b]
                if np.all(a0 < b1) and np.all(b0 < a1):
                    out.append(f"{a} overlaps {b}")
        cells, spots = self.cells(), self.spots()
        used_c, used_s = [o["cell"] for o in self.objects()], [o["spot"] for o in self.objects()]
        for o in self.objects():
            if o["class"] not in self.classes():
                out.append(f"{o['name']}: unknown class {o['class']}")
            if o["cell"] not in cells:
                out.append(f"{o['name']}: unknown cell {o['cell']}")
            if o["spot"] not in spots:
                out.append(f"{o['name']}: unknown spot {o['spot']}")
        if len(set(used_c)) != len(used_c):
            out.append("two objects share a target cell")
        if len(set(used_s)) != len(used_s):
            out.append("two objects share a start spot")
        return out
