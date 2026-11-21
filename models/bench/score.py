"""Per-item metrics and the summary table for bench/run.py results.

Localisation error is reported in the item's own pixels and rescaled to a 299-px-wide frame (`px299`, the
Station resolution) so the 598x448 real-run frames and the 299x224 recorder frames compare. mm errors come
from perception's calibration (overview camera, table plane) where a world truth exists.
"""
from __future__ import annotations

import math

from models import perception_bridge as pb

MATCH_PX299 = 30.0           # a predicted point further than this from every truth is a false point
BOX_MARGIN = 5.0             # px, label uncertainty when testing "inside the box"


def _d(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _label_match(target: str, label: str) -> bool:
    t, l = target.lower().strip(), label.lower().strip()
    return bool(l) and (t in l or l in t)


def _mm(cam: str, px, xyz) -> float | None:
    if xyz is None or cam != "overview":
        return None
    w = pb.pixel_to_world(cam, px[0], px[1])
    return round(1000 * math.hypot(w["xyz"][0] - xyz[0], w["xyz"][1] - xyz[1]), 1) if w.get("ok") else None


def item_metrics(it: dict, row: dict) -> dict:
    if row.get("error"):
        return {"error": True}
    W = it["size"][0]
    s = 299.0 / W
    pred, gt, kind = row["pred"], it["gt"], it["kind"]
    if kind == "markers":
        pts = [p["px"] for p in pred.get("points", [])]
        used, errs, mms = set(), [], []
        for g in gt["points"]:
            cands = [(i, _d(p, g["px"]) * s) for i, p in enumerate(pts) if i not in used]
            i, e = min(cands, key=lambda c: c[1]) if cands else (None, None)
            if i is None or e > MATCH_PX299:
                continue
            used.add(i)
            errs.append(round(e, 2))
            m = _mm(it["camera"], pts[i], g.get("xyz_world"))
            if m is not None:
                mms.append(m)
        return {"n_gt": len(gt["points"]), "matched": len(errs), "err_px299": errs, "err_mm": mms,
                "false_points": len(pts) - len(used)}
    if kind == "objects":
        out = {"err_px299": [], "hits": 0, "present": 0, "missed": 0, "hallucinated": 0, "box_iou": []}
        for target, box in gt["boxes"].items():
            p = next((p for p in pred.get("points", []) if _label_match(target, p["label"])), None)
            b = next((b for b in pred.get("boxes", []) if _label_match(target, b["label"])), None)
            if box is None:
                out["hallucinated"] += int(p is not None)
                continue
            out["present"] += 1
            if p is None:
                out["missed"] += 1
            else:
                c = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                out["err_px299"].append(round(_d(p["px"], c) * s, 2))
                out["hits"] += int(box[0] - BOX_MARGIN <= p["px"][0] <= box[2] + BOX_MARGIN and
                                   box[1] - BOX_MARGIN <= p["px"][1] <= box[3] + BOX_MARGIN)
            if b is not None:
                out["box_iou"].append(round(iou(b["box"], box), 3))
        return out
    truth = gt["held"] if kind == "grasp" else gt["done"]
    d = pred.get("done")
    m = {"truth": truth, "pred": d, "correct": d == truth, "false_success": d is True and truth is False,
         "unknown": d is None}
    if kind == "stain" and not truth and gt.get("residual_box"):
        res = pred.get("residual") or []
        if res:
            b = max(res, key=lambda r: iou(r, gt["residual_box"]))
            m["residual_iou"] = round(iou(b, gt["residual_box"]), 3)
            c, cg = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), ((gt["residual_box"][0] + gt["residual_box"][2]) / 2,
                                                         (gt["residual_box"][1] + gt["residual_box"][3]) / 2)
            m["residual_err_px299"] = round(_d(c, cg) * s, 2)
            if gt.get("world"):
                m["residual_err_mm"] = _mm("overview", c, gt["world"]["centre_xyz"])
        else:
            m["residual_iou"] = 0.0
    return m


def _pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 3)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def summarise(rows: list[dict]) -> dict:
    by = lambda k: [r for r in rows if r["kind"] == k and not r["metrics"].get("error")]  # noqa: E731
    calls = [c for r in rows for c in r.get("calls", [])]
    lat = [c.get("latency_s") for c in calls]
    cost = [c.get("cost_usd") for c in calls if c.get("cost_usd") is not None]
    out = {"items": len(rows), "errors": sum(1 for r in rows if r.get("error")), "calls": len(calls),
           "latency_p50_s": _pct(lat, 0.5), "latency_p95_s": _pct(lat, 0.95),
           "cost_total_usd": round(sum(cost), 5) if cost else None,
           "cost_per_call_usd": round(sum(cost) / len(cost), 6) if cost else None}
    mk = by("markers")
    if mk:
        e = [x for r in mk for x in r["metrics"]["err_px299"]]
        mm = [x for r in mk for x in r["metrics"]["err_mm"]]
        n = sum(r["metrics"]["n_gt"] for r in mk)
        out["markers"] = {"n_gt": n, "found": round(sum(r["metrics"]["matched"] for r in mk) / n, 3) if n else None,
                          "err_px299_p50": _pct(e, 0.5), "err_px299_p90": _pct(e, 0.9),
                          "err_mm_p50": _pct(mm, 0.5), "err_mm_max": max(mm) if mm else None,
                          "false_points": sum(r["metrics"]["false_points"] for r in mk)}
    ob = by("objects")
    if ob:
        e = [x for r in ob for x in r["metrics"]["err_px299"]]
        pres = sum(r["metrics"]["present"] for r in ob)
        out["objects"] = {"present": pres, "hit_rate": round(sum(r["metrics"]["hits"] for r in ob) / pres, 3) if pres else None,
                          "missed": sum(r["metrics"]["missed"] for r in ob),
                          "hallucinated": sum(r["metrics"]["hallucinated"] for r in ob),
                          "err_px299_p50": _pct(e, 0.5), "err_px299_p90": _pct(e, 0.9),
                          "box_iou_mean": _mean([x for r in ob for x in r["metrics"]["box_iou"]])}
    for kind in ("grasp", "stain"):
        rs = by(kind)
        if not rs:
            continue
        neg = [r for r in rs if r["metrics"]["truth"] is False]
        d = {"n": len(rs), "accuracy": round(sum(r["metrics"]["correct"] for r in rs) / len(rs), 3),
             "negatives": len(neg),
             "false_success_rate": round(sum(r["metrics"]["false_success"] for r in neg) / len(neg), 3) if neg else None,
             "unknown": sum(r["metrics"]["unknown"] for r in rs)}
        if kind == "stain":
            d["residual_iou_mean"] = _mean([r["metrics"].get("residual_iou") for r in neg])
            d["residual_err_mm_p50"] = _pct([r["metrics"].get("residual_err_mm") for r in neg], 0.5)
        out[kind] = d
    return out
