"""Run one advisor over the benchmark items and score it.

    uv run --no-project --with google-genai==3.3.0 --with opencv-python-headless --with numpy \
        python -m models.bench.run --advisor gemini            # needs GEMINI_API_KEY
    ... --advisor claude [--model claude-sonnet-5-5]           # Claude Code CLI, headless
    ... --advisor cv                                           # classical stain detector, stain items only
    ... --advisor mock                                         # noisy oracle: checks the pipeline, not a model

Writes results/<advisor>.jsonl (one row per item, raw predictions + per-item metrics) and prints a summary
(score.summarise). Every model call is also in models/logs/ like any other GeminiER call.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from models import perception_bridge as pb  # noqa: E402
from models.bench import score  # noqa: E402
from models.gemini_er import DEFAULT_MODEL, FakeTransport, GeminiER, GeminiError, Image, ParseError  # noqa: E402

DATA = HERE / "data"
RESULTS = HERE / "results"
GRASP_TASK = "pick up the object on the table and lift it"
GRASP_CRITERIA = "an object is held between the gripper jaws and lifted off the table"
STAIN_TASK = "wipe the brown stain off the wooden table"
STAIN_CRITERIA = "no brown stain or residue remains anywhere on the table top"


def load_items(kinds=None) -> list[dict]:
    items = [json.loads(l) for l in (DATA / "items.jsonl").read_text().splitlines() if l.strip()]
    return [it for it in items if not kinds or it["kind"] in kinds]


def images_of(it: dict) -> list[Image]:
    roles = it.get("roles") or [it["camera"]] * len(it["images"])
    return [Image.from_path(DATA / "frames" / n, role=r) for n, r in zip(it["images"], roles)]


def _call_info(er: GeminiER) -> dict:
    c = er.last
    return {"latency_s": c.latency_s, "cost_usd": c.cost_usd, "prompt_tokens": c.prompt_tokens,
            "output_tokens": c.output_tokens, "thinking_tokens": c.thinking_tokens, "call_id": c.call_id}


def run_item(er: GeminiER, it: dict) -> dict:
    imgs = images_of(it)
    row = {"item": it["id"], "kind": it["kind"], "calls": [], "pred": {}, "error": None}
    try:
        if it["kind"] in ("markers", "objects"):
            pts = er.point(imgs[0], it["targets"])
            row["calls"].append(_call_info(er))
            row["pred"]["points"] = [{"label": p.label, "px": [p.u, p.v]} for p in pts]
            if it["kind"] == "objects":
                bxs = er.boxes(imgs[0], it["targets"])
                row["calls"].append(_call_info(er))
                row["pred"]["boxes"] = [{"label": b.label, "box": [b.u0, b.v0, b.u1, b.v1]} for b in bxs]
        elif it["kind"] == "grasp":
            j = er.judge(imgs, GRASP_TASK, GRASP_CRITERIA)
            row["calls"].append(_call_info(er))
            row["pred"] = {"done": j.done, "confidence": j.confidence, "failure": j.failure, "evidence": j.evidence}
        elif it["kind"] == "stain":
            j = er.judge(imgs, STAIN_TASK, STAIN_CRITERIA)
            row["calls"].append(_call_info(er))
            row["pred"] = {"done": j.done, "confidence": j.confidence, "failure": j.failure, "evidence": j.evidence,
                           "residual": [[b.u0, b.v0, b.u1, b.v1] for b in j.residual]}
    except (GeminiError, ParseError) as e:
        row["error"] = str(e)
        if er.last:
            row["calls"].append(_call_info(er))
    row["metrics"] = score.item_metrics(it, row)
    return row


# ---- non-model advisors --------------------------------------------------------------------------------------
def cv_stain(it: dict) -> dict:
    """Brown blob detector: HSV threshold on the table region, largest component."""
    import cv2
    t0 = time.monotonic()
    img = cv2.imread(str(DATA / "frames" / it["images"][0]))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 110, 30), (22, 255, 170))
    m[:70] = 0                                                   # above the table's far edge in this view
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 12]
    res = []
    if keep:
        i = max(keep, key=lambda i: st[i, cv2.CC_STAT_AREA])
        x, y, w, h = st[i, :4]
        res = [[float(x), float(y), float(x + w), float(y + h)]]
    row = {"item": it["id"], "kind": "stain", "error": None, "pred": {"done": not res, "confidence": 1.0, "residual": res},
           "calls": [{"latency_s": round(time.monotonic() - t0, 4), "cost_usd": 0.0}]}
    row["metrics"] = score.item_metrics(it, row)
    return row


def oracle_transport(items_by_prompt: dict, seed: int = 0) -> FakeTransport:
    """Answers from the labels with pixel noise and occasional wrong verdicts: exercises parsing and scoring."""
    rng = random.Random(seed)

    def reply(model, parts, config):
        it = items_by_prompt["current"]
        W, H = it["size"]
        prompt = parts[-1]
        nz = lambda px: [round(min(1000, max(0, (px[1] + rng.gauss(0, 3)) / H * 1000))),  # noqa: E731
                         round(min(1000, max(0, (px[0] + rng.gauss(0, 3)) / W * 1000)))]
        if prompt.startswith("Point"):
            if it["kind"] == "markers":
                return json.dumps([{"point": nz(g["px"]), "label": "marker"} for g in it["gt"]["points"]])
            return json.dumps([{"point": nz([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]), "label": k}
                               for k, b in it["gt"]["boxes"].items() if b])
        if prompt.startswith("Detect"):
            return json.dumps([{"box_2d": [b[1] / H * 1000, b[0] / W * 1000, b[3] / H * 1000, b[2] / W * 1000], "label": k}
                               for k, b in it["gt"]["boxes"].items() if b])
        done = it["gt"].get("held") if it["kind"] == "grasp" else it["gt"]["done"]
        if rng.random() < 0.15:
            done = not done
        box = it["gt"].get("residual_box")
        res = [{"box_2d": [box[1] / H * 1000, box[0] / W * 1000, box[3] / H * 1000, box[2] / W * 1000], "label": "stain"}] \
            if box and not done else []
        return json.dumps({"done": done, "confidence": 0.8, "evidence": "oracle", "camera": it["camera"], "residual": res})
    return FakeTransport(reply, usage={"prompt_tokens": 1400, "output_tokens": 60, "thinking_tokens": 0})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--advisor", choices=["gemini", "claude", "cv", "mock"], required=True)
    ap.add_argument("--model")
    ap.add_argument("--kinds", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--thinking", default="low")
    a = ap.parse_args()
    items = load_items(a.kinds or (["stain"] if a.advisor == "cv" else None))[: a.limit]
    ctx = {"current": None}
    if a.advisor == "gemini":
        er = GeminiER(a.model or DEFAULT_MODEL, thinking_level=a.thinking, source_tag="bench")
    elif a.advisor == "claude":
        from models.claude_cli import ClaudeCLITransport
        er = GeminiER(a.model or "claude", transport=ClaudeCLITransport(), source_tag="bench")
    elif a.advisor == "mock":
        er = GeminiER("mock-oracle", transport=oracle_transport(ctx), source_tag="bench-mock")
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for it in items:
        ctx["current"] = it
        row = cv_stain(it) if a.advisor == "cv" else run_item(er, it)
        row["advisor"], row["model"] = a.advisor, (a.model or (er.model if a.advisor != "cv" else "hsv-threshold"))
        rows.append(row)
        print(f"{it['id']:55s} {json.dumps(row['metrics'])[:110]}" + (f"  ERROR {row['error'][:80]}" if row["error"] else ""))
        if row["error"] and "no GEMINI_API_KEY" in row["error"]:
            print("stopping: no Gemini key")
            break
    name = a.advisor + (f"-{a.model}" if a.model else "")
    with open(RESULTS / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summ = score.summarise(rows)
    (RESULTS / f"{name}.summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    if pb.locator() is None:
        print("note: no mm numbers, perception unavailable:", pb.last_error())


if __name__ == "__main__":
    main()
