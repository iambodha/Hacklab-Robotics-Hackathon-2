"""The stir twin: the selected site's table layout plus a mug and a spoon leaning in it (learning/stir/task.json).

    python -m learning.stir.twin [--site office] [--out sim/layouts/stir_office.json]

writes a full layout (select it with SIM_LAYOUT=stir_office). The site layout itself is never edited: the task objects
live in task.json, like the bottle twin (learning/bottle_twin.py) keeps the bottle out of the site layout. The spoon's
start pose is not hand-set: it is placed against the rim analytically, then MuJoCo lets it settle (task settle_s) and
the settled pose is written, so a reset starts from rest and nothing slides at t = 0.

The layout's "stir" block tells sim/sim_arm_mcp.py what to track for the judge (tool tip = spoon_tip, the cup's rim
height, inner radius and liquid level).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TASK = Path(__file__).resolve().parent / "task.json"
BOWL_PAST_AXIS = 0.4   # bowl centre this x inner radius past the cup axis (sim/stir_skill.handle_line's assumption)


def load_task(path: Path = TASK) -> dict:
    return json.loads(Path(path).read_text())


def default_out(site: str) -> Path:
    return REPO / "sim" / "layouts" / f"stir_{site}.json"


def spoon_length(spoon: dict) -> float:
    """Bowl centre (site spoon_tip) to the handle end (site spoon_top)."""
    return spoon["handle_length"] + spoon["bowl"][2] / 2 - 0.004


def _quat_from_axes(x, y, z):
    """[w, x, y, z] of the rotation whose columns are x, y, z."""
    m = [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]]
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return [0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s]
    i = max(range(3), key=lambda k: m[k][k])
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + m[i][i] - m[j][j] - m[k][k]) * 2
    q = [0.0] * 4
    q[0] = (m[k][j] - m[j][k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (m[j][i] + m[i][j]) / s
    q[1 + k] = (m[k][i] + m[i][k]) / s
    return q


def spoon_start(cup: dict, spoon: dict) -> tuple[list, list]:
    """(pos, quat) of a spoon whose bowl lies on the cup bottom just past the cup axis and whose handle rests on the
    rim on the side facing the arm base. Body z = handle axis (up), body x = handle width, kept in the lean plane so the
    jaws (closing across the width, sim/stir_skill._width_axis) can take it."""
    cx, cy = cup["xy"]
    r, h, t = cup["inner_radius"], cup["height"], cup["wall"]
    az = math.atan2(-cy, -cx)                                # towards the arm base
    e = (math.cos(az), math.sin(az))
    bw, bd, bl = spoon["bowl"]
    # bowl 0.4 r past the axis, away from the handle: where sim/stir_skill.handle_line expects it (a bowl pushed
    # against the far wall stands the spoon up so steeply that its edge-on handle rolls off the rim sideways)
    bowl = (cx - BOWL_PAST_AXIS * r * e[0], cy - BOWL_PAST_AXIS * r * e[1], t + bd / 2 + 0.006)
    rim = (cx + (r - spoon["handle_thickness"]) * e[0], cy + (r - spoon["handle_thickness"]) * e[1], h + 0.003)
    d = [rim[i] - bowl[i] for i in range(3)]
    n = math.sqrt(sum(v * v for v in d))
    z = [v / n for v in d]
    side = [z[1], -z[0], 0.0]                                # horizontal, perpendicular to the handle
    ns = math.hypot(side[0], side[1])
    y = [v / ns for v in side]
    x = [y[1] * z[2] - y[2] * z[1], y[2] * z[0] - y[0] * z[2], y[0] * z[1] - y[1] * z[0]]
    tip_local = -spoon["handle_length"] / 2 - bl / 2 + 0.004  # sim/build_scene.shape_parts 'spoon'
    pos = [bowl[i] - tip_local * z[i] for i in range(3)]
    return pos, _quat_from_axes(x, y, z)


def compose(task: dict, site: str | None = None) -> dict:
    """The full layout (dict): the site layout + the task objects + the stir block. No physics."""
    from sites import get_site
    st = get_site(site or task.get("site"))
    lay = json.loads(st.layout.read_text())
    c, s = task["cup"], task["spoon"]
    cx, cy = c["xy"]
    cup = {"name": c["name"], "shape": "cup", "radius": c["inner_radius"], "height": c["height"], "wall": c["wall"],
           "segments": c["segments"], "pos": [cx, cy, c["height"] / 2], "yaw": 0, "mass": c["mass"], "rgba": c["rgba"],
           "liquid_rgba": c["liquid_rgba"], "liquid_level": c["liquid_level"], "randomize": c["randomize"]}
    pos, quat = spoon_start(c, s)
    spoon = {"name": s["name"], "shape": "spoon", "handle_length": s["handle_length"], "handle_width": s["handle_width"],
             "handle_thickness": s["handle_thickness"], "bowl": s["bowl"], "pos": [round(v, 5) for v in pos],
             "quat": [round(v, 6) for v in quat], "mass": s["mass"], "rgba": s["rgba"], "follow": c["name"],
             "face_base": True}
    lay["objects"] = list(lay.get("objects", [])) + [cup, spoon]
    if task.get("rest_pose"):
        lay["rest_pose"], lay["_rest_pose_about"] = list(task["rest_pose"]), task.get("_rest_pose_about", "")
    lay["cameras"] = list(lay["cameras"]) + [{k: v for k, v in cam.items() if not k.startswith("_")}
                                             for cam in task.get("cameras", [])]
    lay["stir"] = {"cup": c["name"], "tool_tip": f"{s['name']}_tip", "tool_top": f"{s['name']}_top",
                   "tool": s["name"], "inner_radius": c["inner_radius"], "rim_height": c["height"], "wall": c["wall"],
                   "liquid_level": c["liquid_level"], "spoon_length": round(spoon_length(s), 4)}
    lay["_notes"] = [f"GENERATED by learning/stir/twin.py from the {st.name} site layout ({st.layout.relative_to(REPO)}) "
                     f"+ learning/stir/task.json: do not edit, re-run it."] + list(lay.get("_notes", []))
    lay["stir_twin"] = {"site": st.name, "base_layout": str(st.layout.relative_to(REPO)),
                        "task": str(TASK.relative_to(REPO))}
    return lay


def settle(lay: dict, seconds: float) -> dict:
    """Let the spoon come to rest in MuJoCo (arm at its rest pose) and write the settled spoon pose into lay."""
    import tempfile

    import mujoco
    import numpy as np

    import sys
    sys.path.insert(0, str(REPO / "sim"))
    import build_scene
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "stir_settle.json"
        p.write_text(json.dumps(lay))
        build_scene.build(p, Path(d) / "gen", report=False)
        m = mujoco.MjModel.from_xml_path(str(Path(d) / "gen" / "workspace.xml"))
    dat = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, dat, m.key("rest").id)
    dat.ctrl[:] = dat.qpos[m.jnt_qposadr[m.actuator_trnid[:, 0]]]
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, dat)
    spoon = next(o for o in lay["objects"] if o.get("shape") == "spoon")
    b = dat.body(spoon["name"])
    spoon["pos"] = [round(float(v), 5) for v in b.xpos]
    q = np.array(b.xquat)
    spoon["quat"] = [round(float(v), 6) for v in q / np.linalg.norm(q)]
    spoon["_settled"] = f"pose after {seconds} s of MuJoCo from the analytic rim rest (learning/stir/twin.spoon_start)"
    return lay


def make_layout(path: Path | None = None, site: str | None = None, task: dict | None = None,
                settle_s: float | None = None) -> Path:
    task = task or load_task()
    lay = compose(task, site)
    lay = settle(lay, task.get("settle_s", 1.5) if settle_s is None else settle_s)
    path = Path(path or default_out(lay["stir_twin"]["site"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lay, indent=1) + "\n")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--site", default=None, help="table layout to start from (default: task.json 'site' = office)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    print(make_layout(a.out, a.site))
