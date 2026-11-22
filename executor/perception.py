"""Object estimates for observations, behind one small interface.

Order of preference (make_locator):
  1. PerceptionLocator: `perception.locate` (this repo's perception package) with calibration/scene_calib.json (its
     camera intrinsics + pose). Every object hint in executor/object_hints.json (or EXECUTOR_OBJECT_HINTS) is
     segmented by colour on the overview image; its footprint on the table becomes world xyz + size + yaw with a 1-sigma
     `uncertainty_mm` (pixel + calibration + unknown object height). On the twin, the twin's own overview camera
     replaces the calibrated one (same code, the twin's pose), so sim observations measure what the twin renders.
  2. BlobLocator: the blob detector of system2/s2/perception.py on the overview image (no perception package).
Estimates from a stale frame are not produced: the last known ones are re-emitted with visible=False (core.observe).
"""
from __future__ import annotations

import copy
import io
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contracts import ObjectEstimate

REPO = Path(__file__).resolve().parent.parent
HINTS_PATH = Path(__file__).resolve().parent / "object_hints.json"


@dataclass
class Hint:
    """What to look for: a name and a colour (perception.locate.HSV_RANGES), with the object's height for the footprint."""
    name: str
    color: str
    height_m: float = 0.02
    height_sigma_m: float = 0.01       # how well the height is known (moves the footprint centre, see locate_object)
    min_px: int = 6                    # smaller silhouettes are noise
    role: str = "object"               # "object" | "bin" (a place target)

    @classmethod
    def of(cls, d: dict) -> "Hint":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def load_hints(path: str | Path | None = None) -> list[Hint]:
    p = Path(path or os.environ.get("EXECUTOR_OBJECT_HINTS") or HINTS_PATH)
    return [Hint.of(d) for d in json.loads(p.read_text())] if p.exists() else [Hint("blue object", "blue")]


def repo_perception():
    """(perception.locate, perception.scene) of THIS repo (perception/ at the repo root), even when another
    `perception` was imported first (an older worktree's package, or this file run with executor/ on sys.path)."""
    import importlib
    mod = sys.modules.get("perception")
    if mod is None or Path(getattr(mod, "__file__", None) or "").resolve().parent != (REPO / "perception").resolve():
        for k in [k for k in sys.modules if k == "perception" or k.startswith("perception.")]:
            del sys.modules[k]
        if str(REPO) in sys.path:
            sys.path.remove(str(REPO))
        sys.path.insert(0, str(REPO))
    return importlib.import_module("perception.locate"), importlib.import_module("perception.scene")


def colour_in(text: str | None) -> str | None:
    """The first perception colour name in free text ("the blue cap" -> "blue")."""
    try:
        HSV_RANGES = repo_perception()[0].HSV_RANGES
    except ImportError:
        return None
    words = str(text or "").lower().replace(",", " ").split()
    return next((w for w in words if w in HSV_RANGES), None)


class Locator:
    name = "none"

    def locate(self, frames: dict, joints_norm: dict, kin, camera_info: dict) -> list[ObjectEstimate]:
        return []

    def locate_hint(self, jpeg: bytes, hint: Hint, camera_info: dict, seq=None, t_seen=None) -> tuple[ObjectEstimate | None, str]:
        return None, f"locator {self.name} cannot localise {hint.color!r}"

    def pixel_to_world(self, cam: str, u: float, v: float, z: float, camera_info: dict) -> dict:
        return {"ok": False, "reason": f"locator {self.name} has no pixel-to-world map"}

    def describe(self) -> dict:
        return {"name": self.name, **({"fallback_reason": r} if (r := getattr(self, "fallback_reason", None)) else {})}


class PerceptionLocator(Locator):
    name = "perception.locate"

    def __init__(self, scene: dict | None = None, hints: list[Hint] | None = None):
        pl, ps = repo_perception()
        self.pl, self.Scene = pl, ps.Scene
        self.base = self.Scene.from_dict(scene) if scene else pl.scene()
        self.hints = load_hints() if hints is None else hints
        self._scenes: dict = {}

    def scene_for(self, camera_info: dict):
        """The calibrated scene, with the overview camera replaced by the backend's when that is not scene_calib.json."""
        info = (camera_info or {}).get("overview")
        if not info or info.get("source") == "scene_calib.json":
            return self.base
        key = json.dumps(info, sort_keys=True, default=float)
        if key not in self._scenes:
            d = copy.deepcopy(self.base.d)
            W, H = int(info["width"]), int(info["height"])
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = np.array(info["R_base_cam"], float), np.array(info["t_base_cam"], float)
            d["cameras"]["overview"].update(image_size=[W, H], dist=list(info.get("dist") or [0.0] * 5), T_world_cam=T.tolist(), intrinsics_source=info.get("source"),
                                            K_normalized={"fx": info["fx"] / W, "fy": info["fy"] / H, "cx": info["cx"] / W, "cy": info["cy"] / H})
            self._scenes = {key: self.Scene.from_dict(d)}
        return self._scenes[key]

    @staticmethod
    def decode(jpeg: bytes):
        import cv2
        return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)      # BGR, as perception.locate.segment expects

    def locate_hint(self, jpeg, hint, camera_info, seq=None, t_seen=None):
        if not jpeg:
            return None, "no overview image"
        img = self.decode(jpeg)
        if img is None:
            return None, "overview image does not decode"
        loc = self.pl.Locator(self.scene_for(camera_info))
        W, H = loc.sc.size("overview")
        if img.shape[1] != W or img.shape[0] != H:
            return None, f"overview image {img.shape[1]}x{img.shape[0]} != calibrated size {W}x{H}"
        r = loc.locate_object("overview", img, color=hint.color, height_m=hint.height_m)
        if not r.get("ok"):
            return None, f"{hint.name}: {r.get('reason')}"
        if "centre_xy" not in r or r.get("silhouette_px", 0) < hint.min_px:
            return None, f"{hint.name}: only {r.get('silhouette_px', 0)} {hint.color} pixels (< {hint.min_px})"
        s_h = float(r.get("centre_shift_mm_per_cm_height") or 0.0) * hint.height_sigma_m * 100
        sigma = math.hypot(float(r["sigma_centre_mm"]), s_h)
        (cx, cy), (l, w) = r["centre_xy"], r["size_m"]
        intr = str(loc.sc.d["cameras"]["overview"].get("intrinsics_source") or "scene_calib").split(" (")[0]
        return ObjectEstimate(
            name=hint.name, xyz_world=[round(cx, 4), round(cy, 4), round(hint.height_m / 2, 4)],
            size=[l, w, hint.height_m], yaw=round(math.radians(r["yaw_deg"]), 4), uncertainty_mm=round(sigma, 1),
            source=f"perception.locate overview colour {hint.color} ({r['silhouette_px']} px"
                   + (f", seq {seq}" if seq is not None else "") + f"; intrinsics: {intr})",
            visible=True, t_seen=t_seen), "ok"

    def locate(self, frames, joints_norm, kin, camera_info):
        f = frames.get("overview")
        if f is None or not f.jpeg:
            return []
        out = []
        for h in self.hints:
            est, _ = self.locate_hint(f.jpeg, h, camera_info, f.seq, f.t_capture)
            if est is not None:
                out.append(est)
        return out

    def pixel_to_world(self, cam, u, v, z, camera_info):
        if cam != "overview":
            return {"ok": False, "reason": "only overview pixels are localised by the executor (wrist needs per-frame joints)"}
        try:
            return self.pl.Locator(self.scene_for(camera_info)).pixel_to_table(cam, float(u), float(v), float(z))
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "reason": f"{type(e).__name__}: {e}"}

    def describe(self) -> dict:
        ov = self.base.d["cameras"]["overview"]
        board = REPO / "calibration" / "intrinsics_overview.json"
        return {"name": self.name, "scene_calib": self.base.d.get("created"), "intrinsics": ov.get("intrinsics_source"),
                "board_intrinsics_file": str(board) if board.exists() else None,
                "hints": [h.__dict__ for h in self.hints]}


class BlobLocator(Locator):
    """Dark or saturated blobs on a light table (system2 perception), height assumed 1 cm (unknown)."""
    name = "blob (system2)"

    def __init__(self, uncertainty_mm: float = 25.0, workspace=(-0.1, 0.6, -0.45, 0.45)):
        if str(REPO / "system2") not in sys.path:
            sys.path.insert(0, str(REPO / "system2"))
        from s2 import perception as s2p
        from s2.geometry import Camera
        self.s2p, self.Camera = s2p, Camera
        self.unc, self.workspace = uncertainty_mm, workspace

    def locate(self, frames, joints_norm, kin, camera_info):
        f = frames.get("overview")
        info = camera_info.get("overview")
        if f is None or f.jpeg is None or info is None:
            return []
        from PIL import Image
        rgb = np.asarray(Image.open(io.BytesIO(f.jpeg)).convert("RGB"))
        cam = self.Camera.from_info({**info, "width": rgb.shape[1], "height": rgb.shape[0]})
        q = kin.q_from_norm(joints_norm) if joints_norm else None
        arm = [kin.to_world([0, 0, 0.03])] + ([kin.fk(q)[0]] if q is not None else [])
        # workspace (x0, x1, y0, y1) in the WORLD frame for detect_overview's foot check
        dets = self.s2p.detect_overview(rgb, cam, arm, self.workspace)
        out = []
        for d in dets:
            p = cam.to_plane(d.u, d.v, 0.01)
            if p is None:
                continue
            fp = d.footprint(cam, 0.0)
            out.append(ObjectEstimate(name=f"{d.color} {d.id}", xyz_world=np.round(p, 4).tolist(),
                                      size=np.round(fp[1], 3).tolist() if fp else None,
                                      yaw=float(np.radians(fp[2])) if fp else None, uncertainty_mm=self.unc,
                                      source=f"blob overview seq {f.seq}", visible=True, t_seen=f.t_capture))
        return out


def make_locator(scene: dict | None) -> Locator:
    """PerceptionLocator, else BlobLocator, else none; a fallback carries why (describe()["fallback_reason"])."""
    try:
        return PerceptionLocator(scene)
    except Exception as e:  # noqa: BLE001  (no cv2 / no calibration: fall back to blobs; the console shows which locator is used)
        why = f"perception.locate unavailable: {type(e).__name__}: {e}"
    try:
        loc = BlobLocator()
    except ImportError:
        loc = Locator()
    loc.fallback_reason = why
    return loc


def camera_info_from_scene(scene: dict | None) -> dict:
    """Overview camera in s2.geometry.Camera.from_info form, WORLD frame, from scene_calib.json."""
    if not scene or "overview" not in scene.get("cameras", {}):
        return {}
    c = scene["cameras"]["overview"]
    T = np.array(c["T_world_cam"], float)
    W, H = c["image_size"]
    K = c["K_normalized"]
    return {"overview": {"width": W, "height": H, "fx": K["fx"] * W, "fy": K["fy"] * H, "cx": K["cx"] * W, "cy": K["cy"] * H,
                         "R_base_cam": T[:3, :3].tolist(), "t_base_cam": T[:3, 3].tolist(), "dist": [float(d) for d in c.get("dist", [0.0] * 5)],
                         "source": "scene_calib.json"}}
