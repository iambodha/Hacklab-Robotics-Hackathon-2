"""Object estimates for observations, behind one small interface.

Order of preference (make_locator):
  1. `perception.locate` (perception workstream, branch `perception`; EXECUTOR_PERCEPTION_PATH or ../perception)
     if it exists: called as locate(images={role: jpeg}, joints={motor_id: norm}, scene=<scene_calib dict>) and
     expected to return a list of dicts with ObjectEstimate fields.
  2. BlobLocator: the blob detector of system2/s2/perception.py on the overview image, rays onto the table through
     the overview calibration (scene_calib.json on the real arm; the twin's own camera in sim).
Estimates from a stale frame are not produced: the last known ones are re-emitted with visible=False.
"""
from __future__ import annotations

import importlib
import io
import os
import sys
from pathlib import Path

import numpy as np

from .contracts import ObjectEstimate

REPO = Path(__file__).resolve().parent.parent


class Locator:
    name = "none"

    def locate(self, frames: dict, joints_norm: dict, kin, camera_info: dict) -> list[ObjectEstimate]:
        return []


class ModuleLocator(Locator):
    name = "perception.locate"

    def __init__(self, fn, scene: dict | None):
        self.fn, self.scene = fn, scene

    def locate(self, frames, joints_norm, kin, camera_info):
        out = self.fn(images={r: f.jpeg for r, f in frames.items()}, joints=joints_norm, scene=self.scene)
        return [o if isinstance(o, ObjectEstimate) else ObjectEstimate(**{k: v for k, v in o.items() if k in ObjectEstimate.__dataclass_fields__})
                for o in out or []]


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
    path = os.environ.get("EXECUTOR_PERCEPTION_PATH") or str(REPO.parent / "perception")
    if Path(path, "perception", "locate.py").exists():
        if path not in sys.path:
            sys.path.insert(0, path)
        try:
            mod = importlib.import_module("perception.locate")
            return ModuleLocator(mod.locate, scene)
        except Exception:  # noqa: BLE001  (fall back to blobs; the console shows which locator is used)
            pass
    try:
        return BlobLocator()
    except ImportError:
        return Locator()


def camera_info_from_scene(scene: dict | None) -> dict:
    """Overview camera in s2.geometry.Camera.from_info form, WORLD frame, from scene_calib.json."""
    if not scene or "overview" not in scene.get("cameras", {}):
        return {}
    c = scene["cameras"]["overview"]
    T = np.array(c["T_world_cam"], float)
    W, H = c["image_size"]
    K = c["K_normalized"]
    return {"overview": {"width": W, "height": H, "fx": K["fx"] * W, "fy": K["fy"] * H, "cx": K["cx"] * W, "cy": K["cy"] * H,
                         "R_base_cam": T[:3, :3].tolist(), "t_base_cam": T[:3, 3].tolist(), "source": "scene_calib.json"}}
