"""Is the cap free? The wrist-roll (j7) load signature of the strokes, as measured on the real arm on 22 Nov.

A closed cap needs the most torque in its first stroke (break-away), less while the thread turns, and much less once
the thread lets go: attempt 10 (slow, closed cap) per-stroke mean j7 load 104, 84, 74 .. 88, then 70, 71, 69 .. 45.
The signal: a stroke's mean j7 load at most `twist.free_load_ratio` x the reference (the largest mean of the first
`twist.free_ref_strokes` strokes), for `twist.free_confirm_strokes` strokes in a row, and only after
`twist.free_min_rotation_deg` of rotation in the attempt. Fitted and checked on all 12 real attempts in
docs/learning/CAP_FREE.md (it stops attempt 10 at stroke 12 and never fires in attempts 1-8, where the cap was on).

Pure standard library: the real-arm server, the learning rule and the replay test use it on stroke records
(the "strokes" of the twist phase records: achieved_deg, mean_j7_load, abort_reason).
"""
from __future__ import annotations

KEYS = ("twist.free_load_ratio", "twist.free_ref_strokes", "twist.free_confirm_strokes", "twist.free_min_rotation_deg")


def free_signal(strokes: list[dict], cfg: dict) -> dict:
    """{"free", "ref_load", "threshold", "below": [stroke numbers under the threshold after the minimum rotation],
    "began_at": the first stroke of the confirming run (the thread let go there), "confirmed_at": the stroke that
    confirmed it, "rotation_deg"}. Strokes that ended in a fault or moved < 1/2 of their command are not evidence
    (a stall also lowers the load)."""
    ratio, n_ref = float(cfg["twist.free_load_ratio"]), int(cfg["twist.free_ref_strokes"])
    n_conf, min_rot = int(cfg["twist.free_confirm_strokes"]), float(cfg["twist.free_min_rotation_deg"])
    good = [s for s in strokes if not s.get("fatal") and s.get("mean_j7_load") is not None
            and float(s.get("achieved_deg") or 0) >= 0.5 * float(s.get("commanded_deg") or 0) > 0]
    out = {"free": False, "ref_load": None, "threshold": None, "below": [], "began_at": None, "confirmed_at": None,
           "rotation_deg": round(sum(max(0.0, float(s.get("achieved_deg") or 0)) for s in strokes), 1)}
    if len(good) <= n_ref:
        return out
    ref = max(float(s["mean_j7_load"]) for s in good[:n_ref])
    thr = ratio * ref
    out.update(ref_load=round(ref, 1), threshold=round(thr, 1))
    rot, run = 0.0, []
    for s in strokes:
        rot += max(0.0, float(s.get("achieved_deg") or 0))
        if s not in good[n_ref:]:
            continue
        if rot >= min_rot and float(s["mean_j7_load"]) <= thr:
            out["below"].append(s["stroke"])
            run.append(s["stroke"])
            if len(run) >= n_conf and not out["free"]:
                out.update(free=True, began_at=run[0], confirmed_at=s["stroke"])
        else:
            run = []
    return out


def text(sig: dict) -> str:
    if sig["ref_load"] is None:
        return "j7 free signal: too few strokes to judge"
    if sig["free"]:
        return (f"j7 free signal: FREE from stroke {sig['began_at']} (mean j7 load <= {sig['threshold']:.0f} = ratio x "
                f"break-away {sig['ref_load']:.0f}, confirmed at stroke {sig['confirmed_at']})")
    return f"j7 free signal: not free (break-away {sig['ref_load']:.0f}, threshold {sig['threshold']:.0f}" + (
        f", under it at strokes {sig['below']})" if sig["below"] else ")")
