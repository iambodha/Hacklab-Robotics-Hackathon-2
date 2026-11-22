"""Bootstrap pinned NormaCore model assets and run a simulation-only MCP demo."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

REVISION = "6b494f77fac366a878ae71222458b196f8400dee"
ARCHIVE_URL = f"https://codeload.github.com/norma-core/norma-core/tar.gz/{REVISION}"
ARCHIVE_SHA256 = "6737c213bf41f3ec4431538854a98663b7563d0497fbb54d7d60d37b0c221d77"


def assets() -> None:
    root = Path(os.environ["NORMA_CORE_REPO"]).resolve()
    required = root / "hardware/elrobot/simulation/elrobot_follower.urdf"
    if required.is_file():
        print(f"Using NormaCore assets: {root}", file=sys.stderr)
        return
    if root.exists():
        raise SystemExit(f"{root} exists but has no ElRobot URDF; set NORMA_CORE_REPO to a full checkout")
    root.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading separate NormaCore source archive at {REVISION}", file=sys.stderr)
    with tempfile.TemporaryDirectory(prefix="generaloid-assets-", dir=root.parent) as tmp:
        archive = Path(tmp) / "norma.tar.gz"
        urllib.request.urlretrieve(ARCHIVE_URL, archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != ARCHIVE_SHA256:
            raise SystemExit("NormaCore archive checksum mismatch; no files installed")
        with tarfile.open(archive) as tar:
            # The upstream device OS images include absolute system symlinks.
            # Only install the model/client assets this source checkout uses.
            prefix = f"norma-core-{REVISION}/"
            roots = ("hardware/", "software/station/shared/", "software/station/clients/station-viewer/public/devices/so101/")
            members = [m for m in tar.getmembers()
                       if m.name.startswith(prefix)
                       and (m.name[len(prefix):].startswith(roots)
                            or m.name[len(prefix):] in ("LICENSE", "README.md"))
                       and (m.isfile() or m.isdir())]
            tar.extractall(Path(tmp) / "source", members=members, filter="data")
        source = Path(tmp) / "source" / f"norma-core-{REVISION}"
        if not (source / "hardware/elrobot/simulation/elrobot_follower.urdf").is_file():
            raise SystemExit("Upstream archive does not contain the required robot assets")
        source.rename(root)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--connector", choices=("elrobot", "so101"), default="elrobot")
    ap.add_argument("--serve", action="store_true", help="serve the ordinary simulator MCP on stdio")
    ap.add_argument("--rendered", action="store_true", help="use real MuJoCo camera rendering in the pick demo")
    args = ap.parse_args()
    assets()
    # build_scene defaults to osmesa on Linux. Honor an explicit backend choice.
    os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "osmesa")
    if args.serve:
        cmd = [sys.executable, "-m", "generaloid", "serve", "--connector", args.connector, "--mode", "sim"]
    else:
        if not args.rendered:
            os.environ["GENERALOID_CHECK_HEADLESS"] = "1"
            print("Physics-only MCP pick demo; camera images are labeled diagnostics, not renders.", file=sys.stderr)
        else:
            os.environ.pop("GENERALOID_CHECK_HEADLESS", None)
        cmd = [sys.executable, "-m", "generaloid.checks", "pick", "--connector", args.connector]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
