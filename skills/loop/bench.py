"""PROTECTED. Sim benchmark for ONE arm, run inside that arm's sandbox (its own planner code, the protected
evaluator and synthetic world). The orchestrator runs it in the incumbent and candidate sandboxes on identical
trials and decides with skills.experiment.decide from the incumbent checkout.

When the MuJoCo twin provides a wipe scene (sim/twin_wipe.py with run_trial(params, trial) -> dict), it is run
too; until then the result says "twin: not available".

    python -m skills.loop.bench --params '{...}' --trials trials.json --w-true 0.03 [--w-true 0.04] --out arm.json
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
from pathlib import Path

from ..experiment import SimWipeBackend, Trial, _arm
from ..model import SkillVersion


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--trials", required=True)
    ap.add_argument("--w-true", type=float, action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    v = SkillVersion("wipe", "bench", None, json.loads(a.params))
    trials = [Trial(**{**t, "region": tuple(t["region"])}) for t in json.loads(Path(a.trials).read_text())]
    rows = []
    for w in a.w_true:
        be = SimWipeBackend(w_true=w)
        for t in trials:
            try:
                r = _arm(be.run(v, t, execution_id=f"bench-w{w * 1000:.0f}-t{t.index}"))
            except Exception as e:                       # a crash in the arm's code is a fault, not a pass
                r = {"value": None, "success": None, "strokes": 0, "faults": [f"{type(e).__name__}: {e}"],
                     "w_obs_mm": None, "w_valid": None, "duration_s": 0.0, "evidence": []}
            rows.append({"w_true": w, "trial": dataclasses.asdict(t), **r})
    twin = Path("sim/twin_wipe.py")
    twin_res = "not available"
    if twin.exists():
        spec = importlib.util.spec_from_file_location("twin_wipe", twin)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        twin_res = [mod.run_trial(v.params, dataclasses.asdict(t)) for t in trials]
    out = {"rows": rows, "twin": twin_res,
           "mean": sum(r["value"] for r in rows if r["value"] is not None and not math.isnan(r["value"])) / max(1, len(rows))}
    Path(a.out).write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
