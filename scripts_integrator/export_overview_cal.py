"""Write the overview camera file sim/real_arm_mcp.py reads (pos, xmat, fovy in the arm-base frame, MuJoCo camera
convention) from a twin whose overview camera is synced from perception's scene_calib.json.

  python scripts_integrator/export_overview_cal.py                # home twin sim/generated/home -> sim/real_overview_camera.json
  SITE=office python scripts_integrator/export_overview_cal.py    # the site's twin -> its real_overview_camera.json
  python scripts_integrator/export_overview_cal.py --site office  # same

Build the twin first (SITE=<site> python sim/build_scene.py). Run from the repository root.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import mujoco

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))


def export(xml: Path, layout_file: Path | None, dst: Path, label: str, synced_from: str) -> dict:
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]
    cam = next(n for n in names if n and "overview" in n)
    i = names.index(cam)
    # the layout's fovy: a calibrated twin camera renders with its intrinsics and MuJoCo re-derives cam_fovy from fy
    entry = next((c for c in json.loads(layout_file.read_text())["cameras"] if c["name"] == cam), {}) if layout_file else {}
    out = {"source": f"{label} twin {xml.relative_to(ROOT)} camera '{cam}' (synced from {synced_from})",
           "pos": d.cam_xpos[i].round(6).tolist(), "xmat": d.cam_xmat[i].round(6).tolist(),
           "fovy": float(entry.get("fovy", m.cam_fovy[i]))}
    dst.write_text(json.dumps(out, indent=2))
    print("cameras:", names)
    print(json.dumps(out))
    print("wrote", dst)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default=os.environ.get("SITE"), help="site name (default: env SITE; unset = legacy home)")
    a = ap.parse_args()
    if a.site is None:
        layout = ROOT / "sim/layouts/home.json"
        export(ROOT / "sim/generated/home/workspace.xml", layout if layout.exists() else None,
               ROOT / "sim/real_overview_camera.json", "home", "perception scene_calib.json")
        return
    import build_scene
    from sites import get_site
    site = get_site(a.site)
    export(build_scene.generated_dir(site.layout) / "workspace.xml", site.layout, site.asset("overview_camera"),
           site.name, str(site.scene_calib.relative_to(ROOT)))


if __name__ == "__main__":
    main()
