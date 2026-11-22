"""Run the task in the MuJoCo twin through the executor with the scripted planner.

    NORMA_CORE_REPO=... uv run --no-project --with numpy --with scipy --with pillow --with mujoco \
        --with opencv-contrib-python --with "mcp>=1.12,<2" python -m tasks.blue_to_bin.sim_trials --trials 10 [--seed 1]
        [--perturb K]  # on trial K, push the lid 2.5 cm while the arm hovers above it (tests the moved-object replan)

The lid starts uniformly within +-3 cm (x and y) of its measured spot (layout "randomize"); the tub stays at its
measured spot. Simulator ground truth is used ONLY to report whether the predicate was right, never by the planner.
Writes tasks/blue_to_bin/results/sim_scripted.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
RESULTS = Path(__file__).resolve().parent / "results"


PAD_FIX_M = 0.0035


def apply_pad_fix(inward_m: float = PAD_FIX_M, out_dir: Path | None = None) -> Path:
    """PROPOSED TWIN CHANGE, applied at runtime only (sim/build_scene.py is not edited): move both fingertip pads
    `inward_m` towards the jaw centre so the twin's jaw gap spans ~49 mm open .. ~1 mm closed like the real jaws
    (profile: gap = 2 x jaw_travel_each x norm). The unpatched twin closes to an 8 mm gap and cannot pinch a 2 mm wall.
    The patched model is built into a temporary directory; sim/generated is untouched."""
    import tempfile
    sim = ROOT / "sim"
    if str(sim) not in sys.path:
        sys.path.append(str(sim))
    import build_scene
    if getattr(build_scene, "_pad_fix", None) is not None:
        return build_scene._pad_fix
    orig_parts, out = build_scene.add_gripper_parts, Path(out_dir or tempfile.mkdtemp(prefix="twin_padfix_"))

    def add_gripper_parts(root, layout):
        orig_parts(root, layout)
        for name, sign in (("left_pad", 1.0), ("right_pad", -1.0)):
            g = root.find(f'.//geom[@name="{name}"]')
            p = [float(v) for v in g.get("pos").split()]
            p[0] += sign * inward_m
            g.set("pos", build_scene.fmt(p))
    build_scene.add_gripper_parts = add_gripper_parts
    build_scene.generated_dir = lambda layout_file, _o=out: _o / Path(layout_file).stem
    build_scene._pad_fix = out
    return out


def make_sim_executor(rec_root: Path, spec=None, pad_fix: bool = False, layout: str | None = None):
    """Executor on the site's twin (SITE, default home) with this task's locator and adapter skill (what the
    integrator would wire)."""
    from sites import get_site
    layout = layout or get_site().name
    if pad_fix:
        apply_pad_fix()
    from executor.backends.sim import SimBackend
    from executor.core import Executor
    from executor.kinematics import Kinematics
    from executor.recording import RecorderLink

    from .locate import BlueLocator
    from .skills import install
    from .task import Spec
    spec = spec or Spec.load()
    kin = Kinematics()
    b = SimBackend(layout=layout)
    # max_frame_age 2 s: twin renders are slow on a loaded laptop (the real backend keeps the executor default)
    ex = Executor(b, kin, recorder=RecorderLink("local", root=rec_root), locator=BlueLocator(spec), layout=layout,
                  table_memory_path=rec_root / "table_memory.json", max_frame_age=2.0)
    install(ex)
    return ex, b, kin, spec


def truth_world(b, kin) -> dict:
    t = b.truth()["objects"]
    return {k: np.round(kin.to_world(v["xyz"]), 4).tolist() for k, v in t.items()}


def place_cap(b, xy_sim, yaw=0.0):
    import mujoco
    import build_scene
    arm = b.arm
    with arm.lock:
        adr = arm.model.jnt_qposadr[arm.model.joint("blue_cap_free").id]
        dof = arm.model.jnt_dofadr[arm.model.joint("blue_cap_free").id]
        arm.data.qpos[adr:adr + 2] = xy_sim
        arm.data.qpos[adr + 3:adr + 7] = build_scene.yaw_quat(yaw)
        arm.data.qvel[dof:dof + 6] = 0
        mujoco.mj_forward(arm.model, arm.data)


def classify(rep: dict, truth_ok: bool, bin_moved_mm: float) -> str:
    if rep["success"] and truth_ok:
        return "success"
    if rep["success"] and not truth_ok:
        return "FALSE POSITIVE (predicate said in bin, truth says not)"
    if truth_ok:
        return "false negative (in the bin, predicate did not confirm)"
    if bin_moved_mm > 10:
        return f"bin knocked ({bin_moved_mm:.0f} mm)"
    return "failed: " + rep["why"][:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--perturb", type=int, action="append", default=[], help="trial number(s) with a pushed lid")
    ap.add_argument("--twin-pad-fix", action="store_true", help="apply the proposed fingertip-pad change (see apply_pad_fix)")
    ap.add_argument("--frames-dir", help="save overview + wrist frames after the key steps of every trial here")
    ap.add_argument("--out", default=str(RESULTS / "sim_scripted.json"))
    ap.add_argument("--layout", help="twin layout name or file (default: the site's; its overview camera must match scene_calib)")
    a = ap.parse_args()
    import tempfile
    rec_root = Path(tempfile.mkdtemp(prefix="blue_to_bin_rec_"))
    ex, b, kin, spec = make_sim_executor(rec_root, pad_fix=a.twin_pad_fix, layout=a.layout)
    from .plan import Runner
    from .task import MEASURED, truth_in_bin
    cap0 = json.loads(Path(spec.source.get("file", MEASURED)).read_text())["blue_cap"]["centre_xy"]
    trials = []
    for k in range(1, a.trials + 1):
        b.arm.rng = np.random.default_rng(a.seed * 1000 + k)
        b.arm.reset(randomize=True)
        time.sleep(0.5)
        t0 = truth_world(b, kin)
        on_step = None
        if k in a.perturb:
            def on_step(s, _done=[False]):   # noqa: B006  (once per trial)
                if s.step == "approach above" and not _done[0]:
                    _done[0] = True
                    cap = np.array(truth_world(b, kin)["blue_cap"])
                    new_w = cap[:2] + [0.0, 0.025]
                    place_cap(b, (kin.R.T @ np.r_[new_w, 0])[:2])
                    print(f"  perturb: lid pushed 25 mm to {np.round(new_w, 3).tolist()}", flush=True)
        def hook(s, f=on_step, k=k):
            if f:
                f(s)
            print(f"  [{s.attempt}] {s.step:28s} {s.status:13s} {s.seconds:5.1f}s {s.termination[:90]}", flush=True)
            if a.frames_dir and s.step in ("close (verify hold)", "lift", "move above bin", "release", "park") and s.attempt:
                d = Path(a.frames_dir) / f"trial{k:02d}"
                d.mkdir(parents=True, exist_ok=True)
                for role, fr in b.frames().items():
                    (d / f"a{s.attempt}_{len(list(d.iterdir())):02d}_{s.step.split(' (')[0].replace(' ', '_')}_{role}.jpg").write_bytes(fr.jpeg)
        rep = Runner(ex, spec, on_step=hook).run()
        t1 = truth_world(b, kin)
        ok = truth_in_bin(spec, t1["blue_cap"], t1["bin_tub"])
        bin_moved = 1000 * float(np.linalg.norm(np.subtract(t1["bin_tub"][:2], t0["bin_tub"][:2])))
        row = {"trial": k, "start_cap_world": t0["blue_cap"], "offset_mm": np.round(1000 * (np.array(t0["blue_cap"][:2]) - cap0), 1).tolist(),
               "perturbed": k in a.perturb, "predicate_success": rep["success"], "truth_in_bin": ok, "end_truth": t1,
               "bin_moved_mm": round(bin_moved, 1), "attempts": rep["attempts"], "seconds": rep["seconds"],
               "proposals": rep["proposals"], "why": rep["why"], "mode": classify(rep, ok, bin_moved), "report": rep}
        trials.append(row)
        print(f"trial {k}: {row['mode']}  attempts {row['attempts']}  {row['seconds']} s  start offset {row['offset_mm']} mm", flush=True)
    n_ok = sum(1 for t in trials if t["mode"] == "success")
    summary = {"trials": len(trials), "successes": n_ok, "success_rate": round(n_ok / max(1, len(trials)), 3),
               "predicate_agreed_with_truth": sum(1 for t in trials if t["predicate_success"] == t["truth_in_bin"]),
               "false_positives": sum(1 for t in trials if t["predicate_success"] and not t["truth_in_bin"]),
               "seconds_median": float(np.median([t["seconds"] for t in trials])) if trials else None,
               "seconds_max": max((t["seconds"] for t in trials), default=None),
               "attempts_hist": {n: sum(1 for t in trials if t["attempts"] == n) for n in range(1, spec.max_attempts + 1)},
               "modes": {m: sum(1 for t in trials if t["mode"] == m) for m in sorted({t["mode"] for t in trials})},
               "step_seconds_median": step_medians(trials), "spec": spec.to_dict(), "seed": a.seed,
               "twin_pad_fix_m": PAD_FIX_M if a.twin_pad_fix else 0.0, "layout": a.layout}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "trials": trials}, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "spec"}, indent=1))
    ex.end_task("trials done")


def step_medians(trials) -> dict:
    acc: dict[str, list] = {}
    for t in trials:
        for s in t["report"]["steps"]:
            acc.setdefault(s["step"], []).append(s["seconds"])
    return {k: round(float(np.median(v)), 2) for k, v in acc.items()}


if __name__ == "__main__":
    main()
