"""Pipeline self-test input: a hand-held-like video of the twin's table, rendered by MuJoCo, plus the true cameras.

  python sim/splat/synth_video.py [--layout home] [--frames 90] [--out DATA_ROOT/splat/_synthetic]

Writes <out>/video.mp4 and <out>/truth.json (camera centres in the sim frame per frame). prepare.sh on that video must
recover the centres (after the marker alignment) to millimetres; check_synthetic() in this file compares them.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/Volumes/Dell HDD/Robotics-Hackathon"))


def render(layout_name: str, n: int, out: Path, W=640, H=480):
    import build_scene  # noqa: F401  (sets MUJOCO_GL)
    import compare_real
    import cv2
    import mujoco
    model, layout = compare_real.load_model(build_scene.layout_path(layout_name))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("rest").id)
    mujoco.mj_forward(model, data)
    centre = np.mean([np.asarray(it["T"])[:3, 3] for it in layout["markers"]["items"]], axis=0)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    truth = []
    with tempfile.TemporaryDirectory() as tmp, mujoco.Renderer(model, H, W) as r:
        for i in range(n):                                  # a 300-degree walk around the table, bobbing height
            t = i / (n - 1)
            cam.lookat[:] = centre + [0.05 * np.sin(5 * t), 0.05 * np.cos(3 * t), 0]
            cam.azimuth, cam.elevation, cam.distance = -60 + 300 * t, -35 - 12 * np.sin(2 * np.pi * t), 0.95
            r.update_scene(data, camera=cam)
            img = r.render()
            gl = r.scene.camera
            truth.append({"frame": f"f{i + 1:04d}.jpg", "centre": np.mean([np.array(c.pos) for c in gl], 0).tolist()})
            cv2.imwrite(f"{tmp}/{i:04d}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", "10", "-i", f"{tmp}/%04d.png",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(out / "video.mp4")], check=True)
    (out / "truth.json").write_text(json.dumps({"layout": layout_name, "fps": 10, "cameras": truth}, indent=1))
    return out / "video.mp4"


def check_synthetic(work: Path, truth_file: Path) -> dict:
    """Aligned camera centres (dataset/sparse/0) vs the rendered ones; frame names match when SPLAT_FPS = video fps."""
    import pycolmap
    truth = {c["frame"]: np.array(c["centre"]) for c in json.loads(truth_file.read_text())["cameras"]}
    rec = pycolmap.Reconstruction(work / "dataset" / "sparse" / "0")
    err = [np.linalg.norm(img.projection_center() - truth[img.name]) for img in rec.images.values()
           if img.has_pose and img.name in truth]
    return {"frames_compared": len(err), "centre_err_mm_mean": round(float(np.mean(err) * 1000), 2),
            "centre_err_mm_max": round(float(np.max(err) * 1000), 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", default="home")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "splat" / "_synthetic")
    ap.add_argument("--check", type=Path, help="compare DATA_ROOT/splat/<name> against --out/truth.json instead")
    a = ap.parse_args()
    if a.check:
        print(json.dumps(check_synthetic(a.check, a.out / "truth.json")))
    else:
        print(render(a.layout, a.frames, a.out))


if __name__ == "__main__":
    main()
