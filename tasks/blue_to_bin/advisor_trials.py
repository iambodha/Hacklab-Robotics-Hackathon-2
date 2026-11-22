"""Claude as the advisor (executor/advisor/claude.py: `claude -p` headless, MCP = the executor's tools only) on the twin.

    python -m tasks.blue_to_bin.advisor_trials --trials 3 [--model sonnet] [--port 8920] [--twin-pad-fix]

Per trial: reset the twin (lid within +-3 cm), the scripted planner parks and does the jaw check, then the console's
task start hands over to Claude with tasks/blue_to_bin/advisor_prompt.md (filled with the measured numbers). The
outcome is judged independently afterwards (overview + gripper evidence, task.judge; twin ground truth reported).
Needs a logged-in `claude` CLI (or ANTHROPIC_API_KEY in the environment). Writes results/sim_claude.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent


def task_prompt(spec) -> str:
    from .task import plan_grasp, release_point
    g = plan_grasp(spec, [0.26, -0.13, 0.0])
    return (HERE / "advisor_prompt.md").read_text().format(
        cap_d_mm=spec.cap_diameter * 1000, cap_h_mm=spec.cap_height * 1000, wall_mm=spec.cap_wall * 1000,
        grip_mm=spec.max_grip_width * 1000, cap_mid_mm=spec.cap_height * 500, bin_x=spec.bin_xy[0], bin_y=spec.bin_xy[1],
        rim_d_mm=spec.bin_rim_diameter * 1000, rim_h_mm=spec.bin_rim_height * 1000, pinch_mm=(spec.cap_r - spec.cap_wall / 2) * 1000,
        hover=g.z_hover, zg=g.z_grasp, force=spec.grip_force, ztest=g.z_grasp + spec.test_lift,
        zrel=float(release_point(spec, g)[2]), zlow=spec.bin_rim_height + spec.release_depth, rim_r_mm=spec.bin_rim_diameter * 500)


def held_seen(ex, since_mono: float, spec) -> bool:
    """Gripper evidence from the record: was the lid ever seen lifted >= 3 cm while the jaws were closed?"""
    G = ex.kin.P.gripper
    for o in list(ex._obs.values()):
        if o.t_mono < since_mono or (o.joints.gripper_open or 1) >= G.open_enough:
            continue
        if any(e.name == "blue_cap" and e.visible and e.xyz_world[2] - spec.cap_height / 2 > 0.03 for e in o.objects):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--port", type=int, default=8920)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--twin-pad-fix", action="store_true")
    ap.add_argument("--out", default=str(HERE / "results" / "sim_claude.json"))
    a = ap.parse_args()
    from sites import get_site
    os.environ.setdefault("SIM_LAYOUT", get_site().name)
    os.environ.setdefault("ROBOT_SIM_LAYOUT", get_site().name)
    if a.twin_pad_fix:
        from .sim_trials import apply_pad_fix
        apply_pad_fix()
    from http.server import ThreadingHTTPServer
    from executor.console.server import Console, make_handler
    from .plan import Runner
    from .serve import wire
    from .sim_trials import truth_world
    from .task import Spec, judge, truth_in_bin
    spec = Spec.load()
    c = Console("sim", a.port, local_rec_port=a.port + 1)
    wire(c, spec)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(c))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    ex, b = c.ex, c.ex.backend
    prompt = task_prompt(spec)
    rows = []
    for k in range(1, a.trials + 1):
        b.arm.rng = np.random.default_rng(a.seed * 1000 + k)
        b.arm.reset(randomize=True)
        time.sleep(0.5)
        t0w = truth_world(b, ex.kin)
        Runner(ex, spec).prepare()
        t_start, m0 = time.monotonic(), ex.now()
        task = c.start_task(prompt, "claude", a.model)
        while c.advisor.running and time.monotonic() - t_start < a.timeout:
            time.sleep(2)
        if c.advisor.running:
            c.stop_task()
        why = c.task.get("why")
        obs = ex.observe()
        est = next((o for o in obs.objects if o.name == "blue_cap" and o.visible), None)
        grip = ex.motion.read()[ex.kin.grip_id]["norm"]
        v = judge(spec, est, {"held_after_lift": held_seen(ex, m0, spec), "open_after_release": grip >= ex.kin.P.gripper.open_enough})
        t1w = truth_world(b, ex.kin)
        log = c.advisor.log_path
        n_calls = sum(1 for line in open(log) if '"tool_call"' in line) if log and Path(log).exists() else None
        rows.append({"trial": k, "start": t0w, "end": t1w, "predicate": v, "truth_in_bin": truth_in_bin(spec, t1w["blue_cap"], t1w["bin_tub"]),
                     "advisor_end": why, "seconds": round(time.monotonic() - t_start, 1), "advisor_log": str(log),
                     "tool_calls": n_calls, "recording": task.get("recording_id")})
        print(f"trial {k}: predicate {v['success']} truth {rows[-1]['truth_in_bin']} advisor: {why} ({rows[-1]['seconds']} s, "
              f"{n_calls} tool calls) log {log}", flush=True)
    out = {"model": a.model, "prompt_system": c.advisor.system if c.advisor else None, "prompt_task": prompt, "trials": rows,
           "successes": sum(1 for r in rows if r["predicate"]["success"] and r["truth_in_bin"])}
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k not in ("trials", "prompt_system")}, indent=1)[:3000])


if __name__ == "__main__":
    main()
