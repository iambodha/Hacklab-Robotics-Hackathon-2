"""Judge a stir from the twin's ground truth: did the spoon go round inside the mug, cleanly, without moving the mug or
losing the spoon? Pure functions over the records sim/sim_arm_mcp.py keeps (stir_summary over the stirring window,
the mug pose before/after, the spoon in the jaws before/after); thresholds and weights live in learning/stir/task.json
"judge", each with its reason.

    verdict = judge(window, extras, task["judge"], geometry)
    verdict = {"stirred": bool, "outcome": str, "score": float, "quality": float, "gates": {...}, "metrics": {...}}

Gates (all must pass to count as stirred): loops >= loops_min, wall contact share, arm-on-cup contact, mug moved,
mug tilt, spoon slip in the jaws, spoon still in the mug at the end, bowl under the drink. Quality (0..1): coverage
(mean bowl radius / usable radius), depth (how low the bowl circles), pace (loops/s), immersion, clean (no contact).
"""
from __future__ import annotations

import math

# first failing gate in this order names the outcome (the most serious first)
OUTCOMES = ("stirred", "no_grasp", "aborted", "dropped_spoon", "tipped_cup", "pushed_cup", "arm_hit_cup",
            "scraped_wall", "slipped", "stirred_air", "too_few_loops")
GATE_OUTCOME = {"spoon_in_cup": "dropped_spoon", "cup_tilt": "tipped_cup", "cup_moved": "pushed_cup",
                "arm_cup": "arm_hit_cup", "wall_contact": "scraped_wall", "slip": "slipped", "immersion": "stirred_air",
                "loops": "too_few_loops"}


def loops_about_centre(pts, min_radius: float) -> tuple[float, float]:
    """(net loops, mean radius) of a bowl path (list of [x, y(, z)] in the cup frame) around its own centre. Net, not
    absolute: back-and-forth jitter cancels. A path whose mean radius is under min_radius counts 0 loops."""
    if len(pts) < 10:
        return 0.0, 0.0
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    radius = sum(math.hypot(x - cx, y - cy) for x, y in zip(xs, ys)) / len(xs)
    if radius < min_radius:
        return 0.0, radius
    total, prev = 0.0, math.atan2(ys[0] - cy, xs[0] - cx)
    for x, y in zip(xs[1:], ys[1:]):
        a = math.atan2(y - cy, x - cx)
        total += (a - prev + math.pi) % (2 * math.pi) - math.pi
        prev = a
    return abs(total) / (2 * math.pi), radius


def judge(window: dict | None, extras: dict, cfg: dict, geometry: dict) -> dict:
    """window: stir_summary(since=<bookmark at the start of the circles>) at the end of the circles, None when the
    skill never reached them. extras: grasped (bool), aborted (str|None), stir_s (s of circling), cup_moved_m,
    cup_tilt_max_deg, slip_m, spoon_in_cup (bool), arm_cup_s (whole attempt). geometry: inner_radius, rim_height, wall,
    liquid_level (the layout's 'stir' block)."""
    m = {"grasped": bool(extras.get("grasped")), "aborted": extras.get("aborted") or None}
    if not m["grasped"] or window is None:
        outcome = "no_grasp" if not m["grasped"] else "aborted"
        return {"stirred": False, "outcome": outcome, "score": 0.0, "quality": 0.0, "gates": {}, "metrics": m,
                "evidence": [f"{outcome}: {m['aborted'] or 'the jaws never held the spoon'}"]}
    pts = window.get("pts") or []
    loops, path_r = loops_about_centre(pts, cfg["loop_min_radius_m"])
    stir_s = max(float(extras.get("stir_s") or 0.0), 1e-6)
    r_in = geometry["inner_radius"]
    usable = max(r_in - cfg["bowl_half_width_m"], 1e-6)
    axis_r = sum(math.hypot(p[0], p[1]) for p in pts) / len(pts) if pts else 0.0
    h, wall = geometry["rim_height"], geometry.get("wall", 0.004)
    liquid_depth = geometry.get("liquid_level", 0.6) * h - wall           # drink depth above the inner bottom
    heights = [p[2] + h / 2 - wall for p in pts if len(p) > 2]            # bowl centre above the inner bottom
    bowl_h = sum(heights) / len(heights) if heights else None
    m.update(loops=round(loops, 2), path_radius_m=round(path_r, 4), radius_about_axis_m=round(axis_r, 4),
             stir_s=round(stir_s, 2), pace_loops_per_s=round(loops / stir_s, 3),
             immersion_frac=round(min(1.0, window.get("tip_immersed_s", 0.0) / stir_s), 3),
             wall_frac=round(min(1.0, window.get("tool_wall_s", 0.0) / stir_s), 3),
             bottom_frac=round(min(1.0, window.get("tool_bottom_s", 0.0) / stir_s), 3),
             bowl_height_m=None if bowl_h is None else round(bowl_h, 4),
             arm_cup_s=round(float(extras.get("arm_cup_s", 0.0)), 3),
             cup_moved_m=round(float(extras.get("cup_moved_m", 0.0)), 4),
             cup_tilt_max_deg=round(float(extras.get("cup_tilt_max_deg", 0.0)), 1),
             slip_m=round(float(extras.get("slip_m", 0.0)), 4), spoon_in_cup=bool(extras.get("spoon_in_cup")))
    gates = {"spoon_in_cup": m["spoon_in_cup"],
             "cup_tilt": m["cup_tilt_max_deg"] <= cfg["cup_tilt_max_deg"],
             "cup_moved": m["cup_moved_m"] <= cfg["cup_moved_max_m"],
             "arm_cup": m["arm_cup_s"] <= cfg["arm_cup_max_s"],
             "wall_contact": m["wall_frac"] <= cfg["wall_contact_max_frac"],
             "slip": m["slip_m"] <= cfg["slip_max_m"],
             "immersion": m["immersion_frac"] >= cfg["immersion_min_frac"],
             "loops": loops >= cfg["loops_min"]}
    w = cfg["weights"]
    terms = {"coverage": min(1.0, axis_r / usable),
             "depth": 0.0 if bowl_h is None else max(0.0, min(1.0, 1 - bowl_h / max(liquid_depth, 1e-6))),
             "pace": min(1.0, m["pace_loops_per_s"] / cfg["target_pace_loops_per_s"]),
             "immersion": m["immersion_frac"],
             "clean": max(0.0, 1 - m["wall_frac"] - m["bottom_frac"])}
    quality = sum(w[k] * terms[k] for k in w) / sum(w.values())
    failed = [g for g, ok in gates.items() if not ok]
    if m["aborted"]:
        outcome = "aborted"
    else:
        outcome = next((GATE_OUTCOME[g] for g in GATE_OUTCOME if g in failed), "stirred")
    score = max(0.0, quality - cfg["gate_penalty"] * (len(failed) + (1 if m["aborted"] else 0)))
    ev = [f"{m['loops']} loops in {m['stir_s']} s (bowl radius {m['radius_about_axis_m'] * 100:.1f} cm about the mug "
          f"axis, usable {usable * 100:.1f} cm)",
          f"bowl under the drink {m['immersion_frac']:.0%}, on the wall {m['wall_frac']:.0%}, on the bottom "
          f"{m['bottom_frac']:.0%} of the stir",
          f"mug moved {m['cup_moved_m'] * 1000:.1f} mm, tilted up to {m['cup_tilt_max_deg']} deg; arm on mug "
          f"{m['arm_cup_s']} s; spoon slid {m['slip_m'] * 1000:.1f} mm in the jaws; spoon in mug at the end: "
          f"{m['spoon_in_cup']}"]
    if failed:
        ev.append("failed gates: " + ", ".join(failed))
    return {"stirred": not failed and not m["aborted"], "outcome": outcome, "score": round(score, 3),
            "quality": round(quality, 3), "terms": {k: round(v, 3) for k, v in terms.items()}, "gates": gates,
            "metrics": m, "evidence": ev}
