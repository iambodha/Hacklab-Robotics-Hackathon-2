"""The executor console with this task's adapters installed, optionally running the scripted planner.

    python -m tasks.blue_to_bin.serve --backend sim  [--port 8920] [--run dry|pick|full] [--twin-pad-fix]
    EXECUTOR_REAL_MOTION=1 python -m tasks.blue_to_bin.serve --backend real --port 8920 [--run dry|pick|full]

Same console and API as `python -m executor serve` (http://127.0.0.1:PORT), plus:
  - locator = tasks.blue_to_bin.locate.BlueLocator (executor/perception.make_locator finds no perception.locate.locate)
  - skill   = descend_guarded (tasks.blue_to_bin.skills)
  - --run   = the scripted planner in a background thread (every motion is a proposal, visible in the console);
              the report goes to tasks/blue_to_bin/results/<backend>_<mode>_<time>.json
On the real backend nothing moves until the console's "arm enabled" toggle is switched on (EXECUTOR_REAL_MOTION=1 is
also required); the planner waits for that (its first proposals are rejected with "motion disabled" otherwise).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULTS = Path(__file__).resolve().parent / "results"


def wire(console, spec) -> None:
    """Install the task adapters on the console's current executor (call again after a backend switch)."""
    from .locate import BlueLocator
    from .skills import install
    ex = console.ex
    ex.locator = BlueLocator(spec)
    install(ex)
    if ex.backend.name == "sim":
        ex.max_frame_age = 2.0            # twin renders are slow on a loaded laptop; real keeps the default 1 s


def run_planner(console, spec, mode: str, wait_enable_s: float = 600.0) -> dict:
    from executor.contracts import to_dict
    from .plan import Runner
    ex = console.ex
    t0 = time.monotonic()
    while not ex.backend.can_move and time.monotonic() - t0 < wait_enable_s:
        time.sleep(0.5)                   # real backend: wait for the operator's arm-enabled toggle
    if not ex.backend.can_move:
        return {"error": "arm never enabled"}
    ex.start_task(f"blue_to_bin-{mode}", prompt=f"blue_to_bin scripted planner, mode {mode}", advisor="script")
    try:
        rep = Runner(ex, spec, mode=mode, on_step=lambda s: console.events.add("task_step", **to_dict(vars(s)))).run()
    finally:
        ex.end_task(f"blue_to_bin {mode} finished")
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{ex.backend.name}_{mode}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(rep, indent=1, default=str))
    console.events.add("task_report", mode=mode, success=rep["success"], why=rep["why"], file=str(out))
    print(f"blue_to_bin {mode}: success={rep['success']} stage_ok={rep.get('stage_ok')} why={rep['why']} -> {out}", flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="sim", help="sim | real | fake")
    ap.add_argument("--port", type=int, default=8920)
    ap.add_argument("--run", choices=["dry", "pick", "full"])
    ap.add_argument("--twin-pad-fix", action="store_true")
    ap.add_argument("--tcp-offset-mm", default="0,0,0",
                    help="dx,dy[,dz] world mm added to every commanded TCP point near the objects (dry run / touch_table)")
    ap.add_argument("--grip-force", type=float, help="0.1..1 jaw torque limit while closing (default Spec.grip_force)")
    a = ap.parse_args()
    from sites import get_site
    if a.backend == "sim":
        os.environ.setdefault("SIM_LAYOUT", get_site().name)
        os.environ.setdefault("ROBOT_SIM_LAYOUT", get_site().name)
        if a.twin_pad_fix:
            from .sim_trials import apply_pad_fix
            apply_pad_fix()
    from http.server import ThreadingHTTPServer
    from executor.console.server import Console, make_handler
    from .task import Spec
    off = tuple(float(v) / 1000 for v in a.tcp_offset_mm.split(","))
    spec = Spec.load(tcp_offset=(off + (0.0, 0.0, 0.0))[:3], **({"grip_force": a.grip_force} if a.grip_force else {}))
    if a.backend == "real" and spec.source.get("bin_from_site", get_site().name) != get_site().name:
        raise SystemExit(f"SITE={get_site().name}: the bin position is {spec.source['bin_from_site']}'s; measure it on this "
                         "table first (python -m tasks.blue_to_bin.measure --live 10)")
    print(f"spec: lid {spec.cap_diameter * 1000:.0f} x {spec.cap_height * 1000:.0f} mm ({spec.grasp_mode()} grasp), bin "
          f"{list(spec.bin_xy)} rim {spec.bin_rim_height * 1000:.0f} mm, tcp offset {[round(v * 1000) for v in spec.tcp_offset]} mm, "
          f"grip force {spec.grip_force}", flush=True)
    c = Console(a.backend, a.port, local_rec_port=a.port + 1)
    wire(c, spec)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(c))
    srv.daemon_threads = True
    print(f"blue_to_bin console -> http://127.0.0.1:{a.port} (backend {a.backend}, locator {c.ex.locator.name})", flush=True)
    if a.run:
        threading.Thread(target=run_planner, args=(c, spec, a.run), daemon=True, name="blue_to_bin").start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        c.ex.cancel()
        c.ex.end_task("console stopped")


if __name__ == "__main__":
    main()
