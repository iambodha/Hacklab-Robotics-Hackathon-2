"""PROTECTED. Replay test, run INSIDE a sandbox:

1. Reproduce: read the inspection frames from the recording through recorder.ReplaySource (at the
   inspect:before / inspect:after markers) and re-run the evaluator; the Outcome must equal the recorded one.
2. Counterfactual on the SAME stain: rebuild the dirt map from the replayed before image, wipe it in the synthetic
   world with the sandbox's plan for the recorded task (true width = the evaluator's width from the trace, or a
   sweep if that was not identifiable), render, evaluate. This predicts what this code + params would have done.

    python -m skills.loop.replay_check --bundle B --params '{"w_hat": 0.03, ...}' --out result.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ..eval import evaluate, synth
from ..eval.dirt import classify
from ..eval.geometry import Board, Camera, rectify
from ..model import SkillVersion
from ..wipe import WipeSkill
from .bundle import load

SWEEP = (0.02, 0.03, 0.04, 0.05)


def frames(trace: dict) -> tuple[np.ndarray, np.ndarray, float, float]:
    from recorder import Recording, ReplaySource
    rec = Recording(trace["recording"]["path"])
    marks = {m.get("label"): m["t"] for m in rec.markers()}
    src = ReplaySource(rec)
    out = []
    for label in ("inspect:before", "inspect:after"):
        src.seek(marks[label])
        src.pause()
        out.append(synth.from_jpeg(src.frame("overview")))
    return out[0], out[1], marks["inspect:before"], marks["inspect:after"]


def widths(trace: dict) -> list[float]:
    w = (trace.get("evaluation") or {}).get("width") or {}
    return [w["w_observed_mm"] / 1000] if w.get("valid") else list(SWEEP)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--params", required=True, help="JSON params of the version to replay")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    trace = load(a.bundle)
    board = Board(*trace["board"])
    cam = Camera.from_scene_calib(trace["camera"])
    H = cam.H_img_world()
    before, after, tb, ta = frames(trace)
    strokes = [tuple(map(tuple, s)) for s in trace["executed_strokes"]]
    ev = evaluate(before, after, H=H, board=board, execution_id="replay", strokes=strokes, target=trace["task"]["region"])
    rec_val = trace["outcome"]["value"]
    same = (ev.outcome.success == trace["outcome"]["success"] and
            (math.isnan(rec_val) and math.isnan(ev.outcome.value) or abs(ev.outcome.value - rec_val) < 1e-5))   # records round to 6 decimals
    res = {"reproduced": bool(same), "replayed_value": ev.outcome.value, "recorded_value": rec_val,
           "frames_t": [tb, ta], "counterfactual": []}
    # counterfactual on the recorded stain
    rb, vb = rectify(before, H, board, 1000.0)
    dirt = classify(rb, vb)["dirt"].astype(np.float32)
    sk = WipeSkill()
    params = json.loads(a.params)
    v = SkillVersion("wipe", "replay", None, params)
    plan = sk.plan(v, trace["task"])
    val = sk.validate(plan)
    for w in widths(trace):
        world = synth.SynthWorld(board=board, dirt=dirt.copy())
        img_b = world.render(cam, seed=1)
        world.wipe([(s.start, s.end) for s in plan.strokes], w)
        img_a = world.render(cam, seed=2)
        e = evaluate(img_b, img_a, H=H, board=board, execution_id=f"cf-{w}", target=trace["task"]["region"])
        res["counterfactual"].append({"w_true_assumed": w, "reduction": e.outcome.value, "success": e.outcome.success})
    res["plan"] = plan.summary()
    res["plan_valid"] = val.ok
    Path(a.out).write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
