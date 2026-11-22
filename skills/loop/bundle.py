"""Trace bundles: everything the improvement agent gets about one failing run, in one small folder.

    bundle/trace.json     recording id + markers, task, incumbent skill version, proposals/results, Outcome,
                          evaluator stats (incl. the width estimate), executed strokes, board + camera, logs
    bundle/evidence/      the evaluator's evidence image(s)

Marker convention a recording must carry for replay (the executor emits these; see README "needs"):
    inspect:before / inspect:after   overview frame used by the evaluator, at the marker's time
    skill:plan, proposal, result, outcome   with their records in `data`
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from .._contracts import ExecutionResult, Outcome, Status, to_dict
from ..eval import evaluate, synth
from ..eval.geometry import Board
from ..wipe import WipeSkill

OVERVIEW_HASH = "9ea4cacd5be94751e1eea7be90bcf628"


def build(out: str | Path, *, recording: str | Path | None, outcome: Outcome, stats: dict, version: dict, task: dict,
          executed: list, results: list, board: Board, camera: dict, setup_id: str, logs: list | None = None) -> Path:
    out = Path(out)
    (out / "evidence").mkdir(parents=True, exist_ok=True)
    ev = []
    for e in outcome.evidence:
        p = Path(e)
        if p.suffix in (".png", ".jpg") and p.exists():
            shutil.copy(p, out / "evidence" / p.name)
            ev.append(f"evidence/{p.name}")
    rec = None
    if recording is not None:
        from recorder import Recording
        r = Recording(recording)
        rec = {"id": r.id, "path": str(Path(r.path).resolve()), "markers": r.markers()}
    trace = {"bundle_version": 1, "created": time.time(), "skill": version["skill"], "setup_id": setup_id,
             "incumbent": version, "task": task, "recording": rec, "results": [to_dict(x) for x in results],
             "outcome": to_dict(outcome), "evaluation": stats, "executed_strokes": [[list(a), list(b)] for a, b in executed],
             "board": [board.x0, board.y0, board.x1, board.y1], "camera": camera, "evidence": ev, "logs": logs or []}
    (out / "trace.json").write_text(json.dumps(trace, indent=1, default=str))
    return out


def load(bundle: str | Path) -> dict:
    return json.loads((Path(bundle) / "trace.json").read_text())


def synthetic_failure(root: str | Path, w_true: float = 0.03, w_hat: float = 0.06, seed: int = 7,
                      budget: int = 4) -> Path:
    """A recorded wipe where the skill OVERESTIMATES the tissue footprint (w_hat 6 cm, true 3 cm): wide-spaced
    lanes miss strips along the region edges and the Outcome fails. Written as a real recorder session (overview
    frames + markers) plus a trace bundle. Returns the bundle path."""
    from recorder.store import SessionWriter
    root = Path(root)
    sk = WipeSkill()
    v = sk.seed_version(w_hat=w_hat)
    task = {"region": list(synth.TARGET), "budget": budget}
    plan = sk.plan(v, task)
    world = synth.SynthWorld().add_stain_rect(synth.TARGET, seed=seed)
    before = world.render(seed=seed, shadow=(0.25, -0.05, 0.05, 0.3))
    executed = [(s.start, s.end) for s in plan.strokes]
    world.wipe(executed, w_true)
    after = world.render(seed=seed + 1)

    w = SessionWriter("synthetic-wipe-failure", "skills loop: overestimated tissue footprint", root=root / "recordings",
                      station="synthetic")
    t = w.t0
    w.event("start", mono=t)
    w.event("marker", mono=t + 0.1, label="skill:plan", data={"skill": "wipe", "version": v.to_dict(), "plan": plan.summary()})
    w.frame(OVERVIEW_HASH, synth.to_jpeg(before), t + 0.5, w.wall0 + 0.5, {"index": 0, "w": 299, "h": 224})
    w.event("marker", mono=t + 0.5, label="inspect:before", data={"cam": "overview"})
    results = []
    for i, s in enumerate(plan.strokes):
        res = ExecutionResult(f"exec-{i}", f"prop-{i}", Status.SUCCEEDED, {"from_xy_world": list(s.start), "to_xy_world": list(s.end)},
                              {"path_mm": 0.0}, "reached", [{"name": "execute", "status": "SUCCEEDED"}], w.id, [], None, "wipe")
        results.append(res)
        w.event("marker", mono=t + 1 + i, label="proposal", data={"stroke": i, "from": list(s.start), "to": list(s.end)})
        w.event("marker", mono=t + 1.5 + i, label="result", data=to_dict(res))
    ta = t + 2 + len(plan.strokes)
    w.frame(OVERVIEW_HASH, synth.to_jpeg(after), ta, w.wall0 + (ta - t), {"index": 1, "w": 299, "h": 224})
    w.event("marker", mono=ta, label="inspect:after", data={"cam": "overview"})
    cam = synth.default_camera()
    ev = evaluate(synth.from_jpeg(synth.to_jpeg(before)), synth.from_jpeg(synth.to_jpeg(after)), H=cam.H_img_world(),
                  board=synth.BOARD, execution_id="synthetic-wipe", strokes=executed, target=synth.TARGET,
                  evidence=[f"recording:{w.id}@0.5", f"recording:{w.id}@{ta - t:.1f}"], out_dir=root / "evidence")
    w.event("marker", mono=ta + 0.1, label="outcome", data={**to_dict(ev.outcome), "stats": ev.stats})
    w.close()
    return build(root / "bundle", recording=w.dir, outcome=ev.outcome, stats=ev.stats, version=v.to_dict(), task=task,
                 executed=executed, results=results, board=synth.BOARD, camera=synth.OVERVIEW_CALIB, setup_id="setup-synthetic",
                 logs=["synthetic: wipe run with every stroke SUCCEEDED, Outcome below target"])
