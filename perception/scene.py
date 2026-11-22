"""Load the selected site's scene_calib.json (sites.get_site(), SITE=home|office) into a Scene: cameras (intrinsics,
poses), markers, arm base."""
from __future__ import annotations

import json
from sites import get_site
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import geometry as g

ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = get_site().scene_calib
CALIB_DIR = SCENE_PATH.parent
CONFIG_PATH = Path(__file__).resolve().parent / "scene_config.json"


def intrinsics_file(cam: str) -> Path:
    return get_site().intrinsics(cam)


def load_intrinsics(cam: str) -> dict | None:
    """Board-calibrated intrinsics (perception/calib_intrinsics.py) in the internal dict form, or None."""
    p = intrinsics_file(cam)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {**d["K_normalized"], "dist": d["dist"], "source": f"charuco ({d.get('created', '?')}, rms {d.get('rms_px')} px)"}


def _intr(cam: dict) -> dict | None:
    if not cam.get("K_normalized"):
        return None
    return {**cam["K_normalized"], "dist": cam.get("dist") or [0] * 5, "source": cam.get("intrinsics_source")}


@dataclass
class Scene:
    d: dict

    @classmethod
    def load(cls, path: Path = SCENE_PATH) -> "Scene":
        return cls(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, d: dict) -> "Scene":
        return cls(json.loads(json.dumps(d, default=float)))       # normalise numpy / int keys

    # ---- cameras ----
    def size(self, cam: str) -> tuple[int, int]:
        return tuple(self.d["cameras"][cam]["image_size"])

    def intr(self, cam: str) -> dict | None:
        return _intr(self.d["cameras"][cam])

    @property
    def T_world_overview(self) -> np.ndarray:
        return np.array(self.d["cameras"]["overview"]["T_world_cam"])

    @property
    def T_world_base(self) -> np.ndarray:
        return np.array(self.d["arm"]["T_world_base"])

    @property
    def T_link_wrist(self) -> np.ndarray:
        return np.array(self.d["cameras"]["wrist"]["T_link_cam"])

    def T_world_cam(self, cam: str, joints: dict | None = None) -> np.ndarray:
        if cam == "overview":
            return self.T_world_overview
        if joints is None:
            raise ValueError("the wrist camera pose needs the live joint readings")
        from . import wrist
        return self.T_world_base @ wrist.T_base_wristcam(joints, self.T_link_wrist)

    # ---- markers / table ----
    @property
    def markers(self) -> dict[int, np.ndarray]:
        return {int(m["id"]): np.array(m["T_world_marker"]) for m in self.d["markers"]["items"]}

    @property
    def marker_size(self) -> float:
        return float(self.d["markers"]["size_m"])

    @property
    def table_extent(self) -> np.ndarray:
        return np.array(self.d["table"]["extent_xy"], float)

    def status(self) -> dict:
        ov, wr, q = self.d["cameras"]["overview"], self.d["cameras"]["wrist"], self.d.get("quality", {})
        ho = q.get("held_out") or []
        return {"created": self.d["created"],
                "markers": [m["id"] for m in self.d["markers"]["items"]], "dictionary": self.d["markers"]["dictionary"],
                "overview_intrinsics": ov.get("intrinsics_source"), "overview_reproj_px": ov.get("reproj_px"),
                "overview_fovy_deg": ov.get("fovy_deg"),
                "overview_position_m": np.round(self.T_world_overview[:3, 3], 4).tolist(),
                "wrist_intrinsics": wr.get("intrinsics_source"),
                "arm_base": self.d["arm"].get("base_on_table"),
                "held_out_rms_mm": [round(r["rms_mm"], 2) for r in ho],
                "pixel_footprint_mm": q.get("pixel_footprint_mm"),
                "wrist_check": (q.get("wrist_check") or {}).get("markers")}
