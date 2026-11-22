"""Read-only camera relay discovery and shell defaults; never imports Station."""
from __future__ import annotations

import argparse
import io
import json
import shlex
import sys
import urllib.error
import urllib.request

from . import ROOT, get_site

# Coarse normalised view: compare lower side edges with the gap and upper table/background.
JAW_THUMBNAIL = (120, 90)
JAW_LEFT = (15, 62, 48, 87)
JAW_RIGHT = (72, 62, 105, 87)
JAW_GAP = (48, 62, 72, 87)
JAW_BACKGROUND = (15, 15, 105, 55)
JAW_SCORE_THRESHOLD = 0.0  # Both side edge energies must exceed the gap/background mean.


def jaw_score(jpeg: bytes) -> float:
    """Tentative symmetric bottom-jaw edge score, not camera identity recognition.

    Jaws normally occupy the bottom left/right of the wrist view with a central gap.
    Clutter can match this too; discovery always asks for visual confirmation.
    Pillow is optional for the CLI (metadata and snapshot URLs work without it).
    """
    from PIL import Image, ImageFilter, ImageStat
    im = Image.open(io.BytesIO(jpeg)).convert("L").resize(JAW_THUMBNAIL)
    edge = im.filter(ImageFilter.FIND_EDGES)
    def energy(box):
        return ImageStat.Stat(edge.crop(box)).mean[0]
    left, right = energy(JAW_LEFT), energy(JAW_RIGHT)
    centre, upper = energy(JAW_GAP), energy(JAW_BACKGROUND)
    return round(min(left, right) - (centre + upper) / 2, 2)


def discover(relay: str, *, opener=urllib.request.urlopen) -> list[dict]:
    relay = relay.rstrip("/")
    with opener(relay + "/api/cameras", timeout=3) as response:
        cameras = json.load(response)
    results = []
    known = {get_site(n).manifest["cameras"][r].split("/")[-1]: (n, r)
             for n in sorted(p.parent.name for p in (ROOT / "sites").glob("*/site.json")) for r in ("wrist", "overview")}
    for camera in cameras:
        if not camera.get("live"):
            continue
        h = camera["hash"]
        item = dict(camera, queue="usbvideo/" + h, snapshot=relay + "/snapshot/" + h)
        if h in known:
            name, role = known[h]
            item["registered_as"] = f"{name}/{role}"
            item["looks_like"] = role + " (registered queue; visually verify jaws at bottom for wrist)"
        else:
            try:
                with opener(item["snapshot"], timeout=3) as response:
                    score = jaw_score(response.read())
                item["bottom_jaw_score"] = score
                item["looks_like"] = ("wrist candidate (symmetric bottom edges)" if score > JAW_SCORE_THRESHOLD else
                                      "overview candidate (no strong bottom-jaw pattern)") + "; visually verify"
            except (ImportError, OSError, ValueError) as exc:
                item["looks_like"] = "unknown; open snapshot: wrist has jaws at bottom, overview sees the table"
                item["image_note"] = str(exc)
        results.append(item)
    return results


def shell_defaults() -> dict[str, str]:
    site = get_site()
    values = {"SITE": site.name, "ROBOT_OVERVIEW_CAL": str(site.overview_camera),
              "ROBOT_REAL_LOG_DIR": str(site.action_log_dir), "BOTTLE_LEARN_DIR": str(site.episode_dir)}
    for role in ("overview", "wrist"):
        values[f"ROBOT_{role.upper()}_CAM"] = site.robot_camera(role)
        for consumer in ("PERCEPTION", "TWIN"):
            values[f"{consumer}_{role.upper()}_QUEUE"] = site.queue(role, consumer)
    values["RECORDER_CAM_ROLES"] = site.recorder_roles()
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("discover", help="GET live camera metadata/snapshots from the existing relay")
    d.add_argument("--relay", default="http://127.0.0.1:8890")
    sub.add_parser("env", help="print shell-quoted exports, preserving explicit overrides")
    args = parser.parse_args()
    if args.command == "env":
        for key, value in shell_defaults().items():
            print(f"export {key}={shlex.quote(value)}")
        return
    try:
        print(json.dumps(discover(args.relay), indent=2))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"camera discovery failed: {exc}\n")


if __name__ == "__main__":
    main()
