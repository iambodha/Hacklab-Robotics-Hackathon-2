"""Kitting twin trials on a site's table: the exported learned profile (profiles/kit_learned.json, what run_real.py
uses) on the held-out seeds of learn.held_out, nominal and randomized, with the sheet at the site's placement
(spec.site_sheet_pose: sites/<site>/tasks.json "kitting", else the home pose).

    SITE=office KIT_RENDERER=cpu python -m tasks.kitting.site_trials [--nominal 2] [--randomized 6] [--out DIR]

Same seeds, profile and randomization as the home held-out comparison (docs/kitting/RESULTS.md: learned nominal 2/2
kits, randomized 36.1 % placed), so the numbers compare table to table. Writes DIR/results.json and per-object rows.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np


def run(site: str | None, nominal: int, randomized: int, out: Path, log=print) -> dict:
    from . import profile as P
    from .learn import TEST_SEED0, metrics
    from .sim_env import DR, KitSim, run_episode
    out.mkdir(parents=True, exist_ok=True)
    P.use_dir(out / "skillcfg", "twin")
    params = P.load(P.PROFILE)
    sim = KitSim(base=site)
    log(f"site {sim.site}: sheet {sim.nominal_sheet.as_list()}, renderer {sim.render_backend}")
    res, t0 = {"site": sim.site, "sheet_pose": sim.nominal_sheet.as_list(), "profile": str(P.PROFILE.name),
               "renderer": sim.render_backend, "seeds_from": TEST_SEED0}, time.time()
    for label, dr, n in (("nominal", DR.none(), nominal), ("randomized", DR(), randomized)):
        eps, rows = [], []
        for seed in range(TEST_SEED0, TEST_SEED0 + n):
            ep = run_episode(sim, params, np.random.default_rng(seed), dr)
            eps.append(ep)
            rows += [{"seed": seed, "object": o.name, "outcome": o.outcome, "diagnosis": o.diagnosis,
                      "error_mm": o.truth.get("error_mm")} for o in ep["objects"]]
            log(f"{label} seed {seed}: " + ", ".join(f"{o.name} {o.outcome}" for o in ep["objects"])
                + (f" ABORTED {ep['aborted']}" if ep.get("aborted") else "") + f" [{time.time() - t0:.0f} s]")
        m = metrics(eps, P.CLASSES)
        res[label] = {**m, "placed": sum(r["outcome"] == "placed" for r in rows),
                      "attempted": sum(e.get("expected_objects", len(e["objects"])) for e in eps), "objects": rows}
        log(f"{label}: placed {res[label]['placed']}/{res[label]['attempted']}, kits {m['kits_complete']}/{m['episodes']}, "
            f"err {m['mean_error_mm']} mm")
        if eps:
            from PIL import Image
            Image.fromarray(sim.render("overview_cam")).save(out / f"{label}_last_overview.jpg", quality=88)
    res["minutes"] = round((time.time() - t0) / 60, 1)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    sim.close()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default=None, help="default: SITE, home if unset")
    ap.add_argument("--nominal", type=int, default=2)
    ap.add_argument("--randomized", type=int, default=6)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or Path(tempfile.gettempdir()) / f"kit_site_trials_{a.site or os.environ.get('SITE', 'home')}"
    r = run(a.site, a.nominal, a.randomized, out)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "objects"})
                      for k, v in r.items()}, indent=1, default=str))


if __name__ == "__main__":
    main()
