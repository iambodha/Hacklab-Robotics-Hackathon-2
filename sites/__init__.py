"""Table setups for NormaCore Station consumers; stdlib only, no Station connections.

Manifests use repository-relative asset paths and data-root-relative log paths.
Call get_site() after changing SITE; long-running services select a site at startup.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Worktrees live in <hackathon>/ws/<branch>; preserve the original home records there.
DATA_ROOT = ROOT.parent.parent if ROOT.parent.name == "ws" else ROOT


@dataclass(frozen=True)
class Site:
    name: str
    manifest: dict

    @property
    def directory(self) -> Path:
        return ROOT / "sites" / self.name

    def asset(self, key: str) -> Path:
        return ROOT / self.manifest["files"][key]

    @property
    def scene_calib(self) -> Path:
        return self.asset("scene_calib")

    @property
    def layout(self) -> Path:
        return self.asset("layout")

    @property
    def overview_camera(self) -> Path:
        return Path(os.environ.get("ROBOT_OVERVIEW_CAL", self.asset("overview_camera")))

    @property
    def wrist_mount(self) -> Path:
        return self.asset("wrist_mount")

    def intrinsics(self, role: str) -> Path:
        return self.asset(f"intrinsics_{role}")

    def queue(self, role: str, consumer: str = "PERCEPTION") -> str:
        return os.environ.get(f"{consumer}_{role.upper()}_QUEUE", self.manifest["cameras"][role])

    def camera_hash(self, role: str, consumer: str = "PERCEPTION") -> str:
        return self.queue(role, consumer).removeprefix("usbvideo/")

    def robot_camera(self, role: str) -> str:
        return os.environ.get(f"ROBOT_{role.upper()}_CAM", self.manifest["cameras"][role].split("/")[-1][:8])

    def recorder_roles(self) -> str:
        default = ",".join(f"{self.manifest['cameras'][r].split('/')[-1][:8]}={r}" for r in ("wrist", "overview"))
        return os.environ.get("RECORDER_CAM_ROLES", default)

    def data_path(self, key: str) -> Path:
        return DATA_ROOT / self.manifest["directories"][key]

    @property
    def action_log_dir(self) -> Path:
        return Path(os.environ.get("ROBOT_REAL_LOG_DIR", self.data_path("actions")))

    @property
    def episode_dir(self) -> Path:
        return Path(os.environ.get("BOTTLE_LEARN_DIR", self.data_path("episodes")))

    @property
    def record_dir(self) -> Path:
        # Recorder historically used <checkout>/recordings; preserve home in place.
        return Path(os.environ.get("RECORDER_DIR", ROOT / self.manifest["directories"]["recordings"]))


def get_site(name: str | None = None) -> Site:
    name = name if name is not None else os.environ.get("SITE", "home")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(f"invalid SITE {name!r}: use a folder name (letters, digits, _, -)")
    path = ROOT / "sites" / name / "site.json"
    try:
        manifest = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"unknown SITE {name!r}: missing {path}") from exc
    return Site(name, manifest)
