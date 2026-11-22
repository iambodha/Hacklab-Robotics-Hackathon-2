"""Files for the console's 3D arm view: the arm model (URDF + STL meshes) and the pinned JavaScript libraries.

The arm model is read from the local NormaCore checkout (NORMA_CORE_REPO/hardware/elrobot/simulation), the same
URDF the twin is built from, so nothing is copied into this repo. The libraries (three.js, urdf-loader) are fetched
once from jsDelivr at a pinned version, checked against a SHA-256, and kept in executor/logs/vendor: after the
first page load the view works offline.
"""
from __future__ import annotations

import hashlib
import os
import threading
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
VENDOR_DIR = HERE.parent / "logs" / "vendor"
CDN = "https://cdn.jsdelivr.net/npm/"

# served name -> (pinned package path, sha256)
VENDOR = {
    "three.module.js": ("three@0.170.0/build/three.module.js", "ce1fa418de16a19495a9f72495580e3015d7745c296d3ce0485897f902ddedfb"),
    "OrbitControls.js": ("three@0.170.0/examples/jsm/controls/OrbitControls.js", "80efaadea4f8a636a65fb0bd08bfef62f3d93a0bb94e2e7500f23176c5c07f4e"),
    "STLLoader.js": ("three@0.170.0/examples/jsm/loaders/STLLoader.js", "a0a83c88b269c94e25b690fae770d350c4728c81853195186976be7af0f8a3b3"),
    "URDFLoader.js": ("urdf-loader@0.12.3/src/URDFLoader.js", "78e0ffe5edc0780a2885b1be9103fc40d8ca86c72946212db23c39b961e49bf2"),
    "URDFClasses.js": ("urdf-loader@0.12.3/src/URDFClasses.js", "d5112785611d360b501e824a4811252a57a0a44b761138daa6a35041b06bd569"),
}
# urdf-loader imports the Collada loader unconditionally; this arm only has STL meshes.
STUBS = {"ColladaLoader.js": b"export class ColladaLoader{load(u,ok,_p,err){err&&err(new Error('Collada meshes are not supported here'))}}\n"}
TYPES = {".js": "text/javascript; charset=utf-8", ".urdf": "application/xml; charset=utf-8", ".stl": "model/stl"}
_lock = threading.Lock()


class NotAvailable(Exception):
    pass


def vendor(name: str) -> tuple[bytes, str]:
    if name in STUBS:
        return STUBS[name], TYPES[".js"]
    if name not in VENDOR:
        raise NotAvailable(f"no vendored file {name}")
    path, sha = VENDOR[name]
    f = VENDOR_DIR / name
    with _lock:
        if f.exists() and hashlib.sha256(f.read_bytes()).hexdigest() == sha:
            return f.read_bytes(), TYPES[".js"]
        try:
            with urllib.request.urlopen(CDN + path, timeout=20) as r:
                data = r.read()
        except OSError as e:
            raise NotAvailable(f"could not download {path} (needed once, then cached): {e}") from None
        if hashlib.sha256(data).hexdigest() != sha:
            raise NotAvailable(f"{path}: checksum mismatch, refusing to serve it")
        VENDOR_DIR.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)
        return data, TYPES[".js"]


def model_dir() -> Path | None:
    for c in (os.environ.get("NORMA_CORE_REPO"), REPO / "norma-core", REPO.parent / "norma-core", REPO.parent / "repo" / "norma-core",
              REPO.parent.parent / "repo" / "norma-core"):
        d = Path(c) / "hardware" / "elrobot" / "simulation" if c else None
        if d and (d / "elrobot_follower.urdf").exists():
            return d.resolve()
    return None


def robot(rel: list[str]) -> tuple[bytes, str]:
    base = model_dir()
    if base is None:
        raise NotAvailable("arm model not found: set NORMA_CORE_REPO to a norma-core checkout")
    f = base.joinpath(*rel).resolve()
    if base not in f.parents or f.suffix.lower() not in (".urdf", ".stl") or not f.is_file():
        raise NotAvailable(f"no model file {'/'.join(rel)}")
    return f.read_bytes(), TYPES[f.suffix.lower()]
