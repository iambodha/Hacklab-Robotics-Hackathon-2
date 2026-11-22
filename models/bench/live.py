"""Live demo items: the blue cap and the bin, Gemini ER pointing vs perception's colour locate.

    uv run --no-project --with google-genai==3.3.0 --with opencv-contrib-python --with numpy --with scipy \
        python -m models.bench.live [--grab] [--no-gemini]

Frames (data/live/, committed: 4 x ~20 kB) were grabbed from the recorder, GET :8900/live/frame?cam=...,
on 22 Nov 01:32 with the joint readings of the still arm (wrist_1.joints.json, from GET :8900/live). In that
scene the cap is already INSIDE the clear bin (the task's end state), so the two boxes overlap.
--grab saves a fresh set (overview_<n>.jpg, wrist_<n>.jpg + joints); new frames need new hand labels in LABELS.

Per item and target: pixel error to the hand-labelled box centre, inside-box hit, and mm error of the table
projections (z=0, same plane for truth and prediction, so it measures pointing on the table).
  gemini   er.point + er.boxes with targets ["blue cap", "bin"] (2 calls per frame)
  colour   perception.locate.segment(color="blue") centroid + locate_object(color="blue") footprint centre.
           The bin is clear plastic: no colour class, so colour has no answer for it.
  hybrid   perception.locate_object(bbox=<Gemini's box>) (GrabCut in the box): Gemini recognises, perception
           measures. This is how the advisor is meant to be used.
Writes results/live.jsonl and results/live.summary.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from models import perception_bridge as pb  # noqa: E402
from models.bench.score import _pct  # noqa: E402
from models.gemini_er import DEFAULT_MODEL, GeminiER, GeminiError, GenAITransport, Image, ParseError  # noqa: E402

LIVE = HERE / "data" / "live"
RESULTS = HERE / "results"
RECORDER = "http://127.0.0.1:8900"
TARGETS = ["blue cap", "bin"]
# Hand labels (u0, v0, u1, v1) on the 299x224 frames, read off an 8x / 5x pixel grid, about +-3 px.
OVERVIEW = {"blue cap": [185, 96, 200, 119], "bin": [173, 82, 208, 120]}
LABELS = {"overview_1": OVERVIEW, "overview_2": OVERVIEW, "overview_3": OVERVIEW,
          "wrist_1": {"blue cap": [51, 54, 98, 104], "bin": [18, 25, 120, 122]}}
MARGIN = 3.0


def grab(n: int) -> None:
    for cam in ("overview", "wrist"):
        with urllib.request.urlopen(f"{RECORDER}/live/frame?cam={cam}", timeout=5) as r:
            (LIVE / f"{cam}_{n}.jpg").write_bytes(r.read())
    with urllib.request.urlopen(f"{RECORDER}/live", timeout=5) as r:
        bus = next(b for b in json.load(r)["buses"].values() if b.get("role") == "follower")
    joints = {k: {"raw": v["pos"], "lo": v["range_min"], "hi": v["range_max"]}
              for k, v in bus.items() if k.startswith("j") and isinstance(v, dict)}
    (LIVE / f"wrist_{n}.joints.json").write_text(json.dumps(joints))
    print(f"saved overview_{n}.jpg, wrist_{n}.jpg (+ joints); add their labels to LABELS")


def centre(b):
    return [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]


def table_xy(cam, px, joints):
    r = pb.pixel_to_world(cam, px[0], px[1], 0.0, joints)
    return r["xyz"][:2] if r.get("ok") else None


def score(cam, px, box, joints) -> dict:
    c = centre(box)
    a, b = table_xy(cam, px, joints), table_xy(cam, c, joints)
    return {"px": [round(px[0], 1), round(px[1], 1)], "err_px": round(math.dist(px, c), 2),
            "hit": box[0] - MARGIN <= px[0] <= box[2] + MARGIN and box[1] - MARGIN <= px[1] <= box[3] + MARGIN,
            "err_mm": round(1000 * math.dist(a, b), 1) if a and b else None}


def on_table(loc, cam, img, joints, margin_m=0.02):
    """The image with every pixel whose ray misses the calibrated table top (+margin) greyed out."""
    from perception import geometry as g
    T, intr, W, H = loc.camera(cam, joints)
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    P = g.pixel_to_plane(T, intr, W, H, np.c_[uu.ravel(), vv.ravel()].astype(float), 0.0)[:, :2]
    lo, hi = loc.sc.table_extent
    inside = (~np.isnan(P).any(1) & (P >= lo - margin_m).all(1) & (P <= hi + margin_m).all(1)).reshape(H, W)
    out = img.copy()
    out[~inside] = 128
    return out


def match(target, items):
    return next((x for x in items if target.split()[-1] in x.label.lower()), None)


def run_frame(name: str, er: GeminiER | None) -> dict:
    cam = name.split("_")[0]
    path = LIVE / f"{name}.jpg"
    jf = LIVE / f"{name}.joints.json"
    joints = json.loads(jf.read_text()) if cam == "wrist" and jf.exists() else None
    img = cv2.imread(str(path))
    gt = LABELS[name]
    row = {"item": f"live/{name}", "camera": cam, "gt": gt, "gemini": {}, "colour": {}, "colour_table": {},
           "hybrid": {}, "calls": [], "error": None}
    loc = pb.locator()
    # colour: perception's HSV segmentation + footprint locate, on the whole image and on the table area only
    from perception.locate import segment
    for who, im in (("colour", img), ("colour_table", on_table(loc, cam, img, joints))):
        mask, how = segment(im, color="blue")
        if mask is not None:
            ys, xs = np.nonzero(mask)
            r = row[who]["blue cap"] = score(cam, [float(xs.mean()), float(ys.mean())], gt["blue cap"], joints)
            r["mask_px"] = int(mask.sum())
            lo = loc.locate_object(cam, im, color="blue", joints=joints) if loc else {}
            gt_xy = table_xy(cam, centre(gt["blue cap"]), joints)
            if lo.get("ok") and gt_xy:
                r["locate_centre_xy"] = lo["centre_xy"]
                r["locate_err_mm"] = round(1000 * math.dist(lo["centre_xy"], gt_xy), 1)
        row[who]["bin"] = {"answer": None, "reason": "clear plastic: no colour class in perception.locate.HSV_RANGES"}
    if er is None:
        return row
    try:
        pts = er.point(Image.from_path(path, role=cam), TARGETS)
        row["calls"].append(_call(er))
        bxs = er.boxes(Image.from_path(path, role=cam), TARGETS)
        row["calls"].append(_call(er))
    except (GeminiError, ParseError) as e:
        row["error"] = str(e)[:400]
        if er.last:
            row["calls"].append(_call(er))
        return row
    for t in TARGETS:
        p, b = match(t, pts), match(t, bxs)
        g = score(cam, [p.u, p.v], gt[t], joints) if p else {"missed": True}
        if b:
            bb = [b.u0, b.v0, b.u1, b.v1]
            g["box"] = [round(x, 1) for x in bb]
            g["box_centre"] = score(cam, centre(bb), gt[t], joints)
            if loc:
                lo = loc.locate_object(cam, img, bbox=bb, joints=joints)
                gt_xy = table_xy(cam, centre(gt[t]), joints)
                if lo.get("ok") and gt_xy:
                    row["hybrid"][t] = {"centre_xy": lo["centre_xy"], "method": lo["method"],
                                        "err_mm": round(1000 * math.dist(lo["centre_xy"], gt_xy), 1)}
        row["gemini"][t] = g
    return row


def _call(er):
    c = er.last
    return {"latency_s": c.latency_s, "cost_usd": c.cost_usd, "prompt_tokens": c.prompt_tokens,
            "output_tokens": c.output_tokens, "thinking_tokens": c.thinking_tokens, "call_id": c.call_id,
            "error": bool(c.error)}


def summarise(rows) -> dict:
    out = {"frames": len(rows), "gemini_errors": sum(1 for r in rows if r["error"])}
    for who in ("gemini", "colour", "colour_table", "hybrid"):
        for t in TARGETS:
            got = [r[who][t] for r in rows if isinstance(r[who].get(t), dict) and r[who][t].get("answer", 1) is not None
                   and not r[who][t].get("missed")]
            if not got:
                continue
            k = f"{who}/{t}"
            mm = [g.get("err_mm") for g in got] if who != "hybrid" else [g["err_mm"] for g in got]
            out[k] = {"n": len(got), "err_mm_p50": _pct(mm, 0.5), "err_mm_max": max((m for m in mm if m is not None), default=None)}
            if who != "hybrid":
                out[k].update(err_px_p50=_pct([g["err_px"] for g in got], 0.5), hit_rate=round(sum(g["hit"] for g in got) / len(got), 3))
            if who.startswith("colour") and t == "blue cap":
                out[k]["locate_err_mm_p50"] = _pct([g.get("locate_err_mm") for g in got], 0.5)
    calls = [c for r in rows for c in r["calls"]]
    out["calls"] = len(calls)
    out["latency_p50_s"] = _pct([c["latency_s"] for c in calls if not c["error"]], 0.5)
    out["cost_total_usd"] = round(sum(c["cost_usd"] or 0 for c in calls), 5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grab", type=int, metavar="N", help="save a fresh frame set number N and exit")
    ap.add_argument("--no-gemini", action="store_true")
    ap.add_argument("--wait-429", type=float, default=120.0)
    a = ap.parse_args()
    if a.grab:
        return grab(a.grab)
    if str(pb.PERCEPTION_ROOT) not in sys.path:
        sys.path.append(str(pb.PERCEPTION_ROOT))
    er = None if a.no_gemini else GeminiER(DEFAULT_MODEL, transport=GenAITransport(wait_429_s=a.wait_429),
                                           thinking_level="low", source_tag="bench-live")
    rows = []
    for name in LABELS:
        r = run_frame(name, er)
        rows.append(r)
        print(r["item"], json.dumps({k: r[k] for k in ("gemini", "colour", "colour_table", "hybrid")})[:900],
              ("ERROR " + r["error"][:160]) if r["error"] else "")
        if r["error"] and "per day" in r["error"]:
            print("stopping: Gemini free-tier daily cap")
            er = None
    tag = "live" + ("-colour" if a.no_gemini else "")
    (RESULTS / f"{tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    s = summarise(rows)
    (RESULTS / f"{tag}.summary.json").write_text(json.dumps(s, indent=1) + "\n")
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
