"""Docs material from a practice run: the learning curve (SVG), a curated copy of the records and the side-view image.

    python -m learning.stir.report data/stir/<run> [--docs docs/stir]

Writes <docs>/learning-curve.svg, <docs>/data/result.json (the run's result without the bulky paths),
<docs>/data/SUMMARY.md, <docs>/data/attempts.json (one line per attempt: changes, hypothesis, outcome, metrics,
lesson) and <docs>/stir_side.jpg (the best attempt's side view, mid-stir). Pure standard library.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COLORS = {"stirred": "#2a7d4f", "fail": "#c0392b", "best": "#1f4e8c", "grid": "#d9d9d9", "text": "#333333"}


def curve_svg(rows: list[dict], validation: dict | None = None, w: int = 720, h: int = 340) -> str:
    """Score per attempt (dots: green = stirred, red = a failed gate, labelled with the outcome) and the best so far
    (step line). Optional validation means on the right."""
    left, right, top, bottom = 56, 200 if validation else 24, 34, 64
    pw, ph = w - left - right, h - top - bottom
    n = len(rows)
    x = lambda i: left + (i + 0.5) * pw / max(n, 1)
    y = lambda s: top + ph * (1 - s)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
           f'font-family="Helvetica, Arial, sans-serif" font-size="11">',
           f'<rect width="{w}" height="{h}" fill="white"/>',
           f'<text x="{left}" y="20" font-size="14" fill="{COLORS["text"]}">Stirring practice in the twin: judge score '
           f'per attempt</text>']
    for s in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{left}" x2="{left + pw}" y1="{y(s):.1f}" y2="{y(s):.1f}" stroke="{COLORS["grid"]}"/>')
        out.append(f'<text x="{left - 8}" y="{y(s) + 4:.1f}" text-anchor="end" fill="{COLORS["text"]}">{s:g}</text>')
    best, pts = 0.0, []
    for i, r in enumerate(rows):
        if r["outcome"] == "stirred":
            best = max(best, r["score"])
        pts.append((x(i), y(best)))
    path = " ".join(f"{'M' if k == 0 else 'L'}{px - pw / max(n, 1) / 2:.1f},{py:.1f} L{px + pw / max(n, 1) / 2:.1f},{py:.1f}"
                    for k, (px, py) in enumerate(pts))
    out.append(f'<path d="{path}" fill="none" stroke="{COLORS["best"]}" stroke-width="2" opacity="0.7"/>')
    for i, r in enumerate(rows):
        c = COLORS["stirred"] if r["outcome"] == "stirred" else COLORS["fail"]
        out.append(f'<circle cx="{x(i):.1f}" cy="{y(r["score"]):.1f}" r="4.5" fill="{c}"/>')
        out.append(f'<text x="{x(i):.1f}" y="{top + ph + 16}" text-anchor="middle" fill="{COLORS["text"]}">'
                   f'{r["attempt"]}</text>')
        if r["outcome"] != "stirred":
            out.append(f'<text transform="translate({x(i) + 3:.1f},{y(r["score"]) - 8:.1f}) rotate(-50)" '
                       f'font-size="9" fill="{c}">{r["outcome"].replace("_", " ")}</text>')
    out.append(f'<text x="{left + pw / 2}" y="{h - 26}" text-anchor="middle" fill="{COLORS["text"]}">attempt</text>')
    out.append(f'<text x="{left}" y="{h - 8}" fill="{COLORS["text"]}" font-size="10">'
               f'<tspan fill="{COLORS["stirred"]}">●</tspan> stirred cleanly   '
               f'<tspan fill="{COLORS["fail"]}">●</tspan> a gate failed (score - 0.5 per gate)   '
               f'<tspan fill="{COLORS["best"]}">━</tspan> best so far</text>')
    if validation:
        vx = left + pw + 30
        out.append(f'<text x="{vx}" y="{top + 4}" fill="{COLORS["text"]}" font-size="12">Validation</text>')
        out.append(f'<text x="{vx}" y="{top + 20}" fill="{COLORS["text"]}" font-size="10">randomised mug, same seeds</text>')
        for k, (label, key) in enumerate((("baseline", "baseline"), ("learned", "learned"))):
            v = validation[key]
            yy = top + 48 + k * 70
            col = COLORS["fail"] if key == "baseline" else COLORS["stirred"]
            out.append(f'<text x="{vx}" y="{yy}" fill="{col}" font-size="12" font-weight="bold">{label}</text>')
            out.append(f'<text x="{vx}" y="{yy + 16}" fill="{COLORS["text"]}">stirred {v["stirred"]}/{v["runs"]}, '
                       f'score {v["mean_score"]}</text>')
            out.append(f'<text x="{vx}" y="{yy + 32}" fill="{COLORS["text"]}">radius {v["mean_radius_cm"]} cm, '
                       f'{v["mean_pace"]} loops/s</text>')
    out.append("</svg>")
    return "\n".join(out)


def attempts_table(practice_dir: Path) -> list[dict]:
    from learning.stir.store import StirStore
    out = []
    for n, a in sorted(StirStore(practice_dir).attempts().items()):
        j = a["judges"][-1] if a["judges"] else {}
        stir = next((p["measurements"] for p in a["phases"] if p["phase"] == "stir"), {})
        out.append({"attempt": n, "changed": a["begin"]["changed"], "hypothesis": a["begin"]["hypothesis"],
                    "phases": [f"{p['phase']}:{p['status']}" for p in a["phases"]],
                    "stir": {k: stir.get(k) for k in ("radius_m", "radius_asked_m", "lean_deg", "turns",
                                                      "points_per_turn")},
                    "outcome": (a["end"] or {}).get("outcome"), "score": (a["end"] or {}).get("score"),
                    "terms": j.get("terms"), "metrics": j.get("metrics"), "evidence": j.get("evidence"),
                    "lesson": (a["end"] or {}).get("lesson")})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run")
    ap.add_argument("--docs", default=str(REPO / "docs" / "stir"))
    ap.add_argument("--tag", default="", help="suffix for the curve and data folder (e.g. run1)")
    a = ap.parse_args(argv)
    run, docs = Path(a.run), Path(a.docs)
    sfx = f"-{a.tag}" if a.tag else ""
    data = docs / "data" / a.tag if a.tag else docs / "data"
    data.mkdir(parents=True, exist_ok=True)
    res = json.loads((run / "result.json").read_text())
    val = (res.get("validation") or {}).get("summary")
    (docs / f"learning-curve{sfx}.svg").write_text(curve_svg(res["rows"], val))
    slim = {k: v for k, v in res.items() if k != "run_dir"}
    slim["run"] = run.name
    (data / "result.json").write_text(json.dumps(slim, indent=1))
    (data / "attempts.json").write_text(json.dumps(attempts_table(run / "practice"), indent=1))
    shutil.copy(run / "practice" / "SUMMARY.md", data / "SUMMARY.md")
    best = (res.get("best") or {}).get("attempt")
    img = run / "practice" / f"attempt_{best}" / "stir_side.jpg"
    if img.exists():
        shutil.copy(img, docs / f"stir_side{sfx}.jpg")
    print(docs)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(REPO))
    main()
