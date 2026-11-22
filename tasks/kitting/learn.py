"""The continuous-improvement loop for kitting, entirely in the twin.

    nice -n 10 python -m tasks.kitting.learn [--iterations 10] [--episodes 8] [--out DIR] [--export]

Per iteration:
  1. evaluate the active skill config (target twin) on a FIXED set of randomized kits (same seeds every iteration, so
     the curve shows the parameters, not the dice); every kit is an episode record (kit_episodes.py);
  2. the learning rule reads the per-object diagnoses (sim_env.diagnose) and, per object class, proposes at most
     MAX_CHANGES_PER_CLASS parameter changes with a hypothesis (RULES below: which symptom moves which parameter);
  3. each proposal goes through learning/skillcfg.py: propose -> twin A/B check (the class's objects on fresh seeds,
     before vs after, paired) written into the proposal -> apply (a new version) or reject. A rejected direction halves
     its step; a parameter whose step falls below its minimum is left alone for that symptom.
Stops early when no class has a symptom left. Then a held-out test (new seeds) compares the generic profile with the
learned one, and --export writes tasks/kitting/profiles/kit_learned.json for the real arm.

Ground truth is used by the judge only (outcome, diagnosis), never by the policy (sim_env.py docstring).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/Volumes/Dell HDD/Robotics-Hackathon")) / "r5-kitsim"
MAX_CHANGES_PER_CLASS = 2      # per class and iteration (the bottle allows 3 per attempt; classes are independent)
MIN_SHARE = 0.10               # a symptom counts when it hits at least this share of a class's objects
PLACE_ABOVE_GRASP_MM = 1.0     # place_z is never set below grasp_z + this (object bottom 1 mm above the paper)
EVAL_SEED0, CHECK_SEED0, TEST_SEED0 = 10_000, 50_000, 90_000

# symptom -> [(parameter, direction, first step, minimum step)], tried in order. Direction +1 = increase.
RULES: dict[str, list[tuple[str, int, float, float]]] = {
    "unreachable": [("hover_above_mm", -1, 5.0, 1.25)],
    "grasp_too_high": [("grasp_z_mm", -1, 3.0, 0.5)],            # the pads closed above most of the object
    "closed_empty": [("grasp_z_mm", -1, 2.0, 0.5), ("yaw_deg", +1, 45.0, 11.25)],
    "descent_blocked": [("grasp_z_mm", +1, 2.0, 0.5)],           # touched the table above the grasp height
    "pushed_on_descent": [("yaw_deg", +1, 45.0, 11.25), ("grasp_z_mm", +1, 2.0, 0.5)],
    "slipped_in_carry": [("grip_force", +1, 0.1, 0.025), ("grasp_z_mm", -1, 2.0, 0.5)],
    "rotated_in_grasp": [("yaw_deg", +1, 45.0, 11.25), ("grip_force", +1, 0.1, 0.025)],
    "bounced_on_release": [("place_z_mm", -1, 8.0, 1.0)],        # released too high: it bounced / rolled
    "dragged_on_retreat": [("release_open", +1, 0.2, 0.05)],
    "knocked_neighbour": [("release_open", -1, 0.2, 0.05)],      # the opening jaws hit a placed neighbour
}
WHY = {
    "unreachable": "an approach or carry could not be completed; reduce approach clearance within the envelope",
    "grasp_too_high": "the fingertips closed above most of the object, so the jaws shut on nothing",
    "closed_empty": "the jaws closed empty with the fingertips low enough: the object squeezed out",
    "descent_blocked": "the descent touched down before the grasp height",
    "pushed_on_descent": "a fingertip landed on the object and pushed it",
    "slipped_in_carry": "the object slid out of the jaws during the lift or carry",
    "rotated_in_grasp": "the object turned in the jaws when they closed (closing along its length)",
    "bounced_on_release": "the object fell from the release height and bounced or rolled off the cell centre",
    "dragged_on_retreat": "the object moved when the gripper left",
    "knocked_neighbour": "a neighbour was moved",
}


# --- metrics -----------------------------------------------------------------------------------------------------
def metrics(eps: list[dict], classes) -> dict:
    objs = [o for e in eps for o in e["objects"]]
    # Aborted location attempts still count as six attempted assignments, never disappear from success rate.
    attempted = sum(e.get("expected_objects", len(e["objects"])) for e in eps)
    out = {"episodes": len(eps), "objects": attempted, "executed_objects": len(objs), "kits_complete": sum(1 for e in eps if e["objects"] and
                                                                               all(o.outcome == "placed" for o in e["objects"])),
           "aborted": sum(1 for e in eps if e.get("aborted"))}
    placed = [o for o in objs if o.outcome == "placed"]
    out["success_rate"] = round(len(placed) / max(1, attempted), 3)
    out["mean_error_mm"] = round(float(np.mean([o.truth["error_mm"] for o in placed])), 2) if placed else None
    rel = [o for o in objs if o.outcome in ("placed", "off_cell")]
    out["mean_error_released_mm"] = round(float(np.mean([o.truth["error_mm"] for o in rel])), 2) if rel else None
    out["sheet_error_mm"] = round(float(np.mean([e["sheet_error_mm"] for e in eps if e.get("sheet_error_mm") is not None])), 2) \
        if any(e.get("sheet_error_mm") is not None for e in eps) else None
    pc = {}
    for c in classes:
        co = [o for o in objs if o.cls == c]
        cp = [o for o in co if o.outcome == "placed"]
        tags = {}
        for o in co:
            for t in o.diagnosis:
                tags[t] = tags.get(t, 0) + 1
        pc[c] = {"objects": len(co), "placed": len(cp), "success_rate": round(len(cp) / max(1, len(co)), 3),
                 "mean_error_mm": round(float(np.mean([o.truth["error_mm"] for o in cp])), 2) if cp else None,
                 "outcomes": {k: sum(1 for o in co if o.outcome == k) for k in sorted({o.outcome for o in co})},
                 "symptoms": dict(sorted(tags.items(), key=lambda kv: -kv[1]))}
    out["per_class"] = pc
    return out


# --- the learning rule -------------------------------------------------------------------------------------------
class Rule:
    """Symptom -> parameter change, with per-(class, parameter, direction) step memory."""

    def __init__(self):
        self.step: dict[tuple, float] = {}

    def candidates(self, cls: str, m: dict, params: dict) -> list[tuple[str, str, float, str]]:
        """[(symptom, param, new value, hypothesis)] for one class, most frequent symptom first."""
        from . import profile as P
        n = max(1, m["objects"])
        out, used = [], set()
        for tag, count in m["symptoms"].items():
            if tag not in RULES or count / n < MIN_SHARE:
                continue
            for name, sign, first, minimum in RULES[tag]:
                if name in used:
                    continue
                step = self.step.get((cls, name, sign), first)
                if step < minimum:
                    continue
                lo, hi = P.PARAMS[name][:2]
                cur = params[name]
                new = float(np.clip(cur + sign * step, lo, hi))
                if name == "place_z_mm":
                    new = max(new, params["grasp_z_mm"] + PLACE_ABOVE_GRASP_MM)
                if name == "yaw_deg" and P.PARAMS and cur + sign * step > hi:
                    new = float(hi)
                if abs(new - cur) < 1e-9:
                    continue
                hyp = (f"{cls}: {count}/{m['objects']} objects showed '{tag}' ({WHY[tag]}); {name} {cur:g} -> {new:g} "
                       f"should remove it")
                out.append((tag, name, round(new, 4), hyp))
                used.add(name)
                break
        return out

    def rejected(self, cls: str, name: str, sign: int, step_used: float):
        self.step[(cls, name, sign)] = step_used / 2


def ab_check(sim, cls: str, before: dict, after: dict, seeds: list[int], dr, tag: str) -> dict:
    """Paired twin trials of one class's objects (the rest of the kit stays on its spots): before vs after."""
    from .sim_env import run_episode
    names = [o["name"] for o in sim.spec.objects() if o["class"] == cls]
    res = {}
    for label, params in (("before", before), ("after", after)):
        objs, aborted = [], 0
        for s in seeds:
            ep = run_episode(sim, params, np.random.default_rng(s), dr, only=names)
            aborted += bool(ep.get("aborted"))
            if getattr(sim, "episode_store", None) is not None:
                from .kit_episodes import record_episode
                from . import profile as P
                record_episode(sim.episode_store, ep, P.flat(params),
                               f"Paired {label} trial: {cls}, diagnosis {tag}", stage="ab_check", seed=s,
                               parameter_source=label, hypothetical=(label == "after"))
            objs += ep["objects"]
        placed = [o for o in objs if o.outcome == "placed"]
        res[label] = {"placed": len(placed), "of": len(names) * len(seeds), "executed": len(objs), "aborted": aborted, "symptom": sum(1 for o in objs if tag in o.diagnosis),
                      "mean_error_mm": round(float(np.mean([o.truth["error_mm"] for o in placed])), 2) if placed else None,
                      "mean_drop_mm": float(np.mean([p["measurements"]["truth_drop_mm"] for o in objs for p in o.phases
                          if "truth_drop_mm" in p["measurements"]])) if any("truth_drop_mm" in p["measurements"] for o in objs for p in o.phases) else None}
    b, a = res["before"], res["after"]
    better_err = a["mean_error_mm"] is not None and (b["mean_error_mm"] is None or a["mean_error_mm"] <= b["mean_error_mm"] - 0.5)
    less_drop = (tag == "bounced_on_release" and a["mean_drop_mm"] is not None and b["mean_drop_mm"] is not None
                 and a["mean_drop_mm"] < b["mean_drop_mm"] - 2 and a["placed"] > 0)
    ok = a["placed"] > b["placed"] or (a["placed"] == b["placed"] and (a["symptom"] < b["symptom"] or better_err or less_drop)
                                       and (a["mean_error_mm"] or 0) <= (b["mean_error_mm"] or 1e9) + 1.0)
    ok = ok and not a["aborted"] and not b["aborted"]
    res["result"] = "pass" if ok else "fail"
    res["summary"] = (f"placed {b['placed']}/{b['of']} -> {a['placed']}/{a['of']}, '{tag}' {b['symptom']} -> {a['symptom']}, "
                      f"mean error {b['mean_error_mm']} -> {a['mean_error_mm']} mm, drop {b['mean_drop_mm']} -> {a['mean_drop_mm']} mm (seeds {seeds[0]}..{seeds[-1]})")
    return res


def select_best(curve, P, SC, log=print):
    """Select on fixed evaluation seeds, before reading the held-out learned score."""
    best = max(curve, key=lambda r: (r["success_rate"], -(r["mean_error_mm"] if r["mean_error_mm"] is not None else float("inf"))))
    changed = P.current()[0] != best["params"]
    if changed:
        result = SC.revert("twin", best["version"], "evaluation-model-selection",
                           "Highest fixed-evaluation success; tie-break lower placed error; no held-out selection")
        if not result["ok"]:
            raise RuntimeError(f"selection revert failed: {result}")
        log(f"SELECT evaluation iteration {best['iteration']} (v{best['version']}) -> active v{P.current()[1]}")
    return {"selected_iteration": best["iteration"], "selected_version": best["version"],
            "active_version": P.current()[1], "criterion": "highest evaluation success, tie lower placed error"}, changed


# --- the loop ----------------------------------------------------------------------------------------------------
def run(iterations: int, episodes: int, check_seeds: int, out: Path, test_episodes: int, log=print, resume: bool = False) -> dict:
    from . import profile as P
    from .kit_episodes import KitEpisodeStore, record_episode
    from .sim_env import DR, KitSim, run_episode
    out.mkdir(parents=True, exist_ok=True)
    if not resume and ((out / "results.json").exists() or (out / "curve.jsonl").exists()):
        raise ValueError("Choose a fresh output directory; implicit resume would mix evaluation histories")
    P.use_dir(out / "skillcfg", "twin")
    SC = P.skillcfg()
    store = KitEpisodeStore(out / "episodes")
    sim = KitSim()
    sim.episode_store = store
    dr = DR()
    rule = Rule()
    curve, pending_note, t0 = [], "generic profile: one setting for every class (tasks/kitting/profile.py PARAMS)", time.time()
    if resume:
        curve = [json.loads(line) for line in (out / "curve.jsonl").read_text().splitlines() if line.strip()]
        if not curve or [r["iteration"] for r in curve] != list(range(len(curve))):
            raise ValueError("Resume requires consecutive completed iterations starting at zero")
        for proposal in SC.proposals(t="twin"):
            evidence = " ".join(proposal.get("evidence", []))
            import re
            match = re.search(r"iteration (\d+)", evidence)
            if proposal["status"] == "rejected" and match and int(match[1]) < len(curve):
                for k, delta in proposal["changes"].items():
                    _, cls, name = k.split(".")
                    diff = delta["after"] - delta["before"]
                    rule.rejected(cls, name, 1 if diff > 0 else -1, abs(diff))
        t0 -= curve[-1]["minutes"] * 60
        pending_note = "Resumed after audit; current version is authoritative; see JOURNAL for rollback history"
        log(f"RESUME at iteration {len(curve)}, active v{P.current()[1]}")
    eval_seeds = [EVAL_SEED0 + k for k in range(episodes)]
    for it in range(len(curve), iterations + 1):
        params, version = P.current()
        eps, ids = [], []
        for k, s in enumerate(eval_seeds):
            ep = run_episode(sim, params, np.random.default_rng(s), dr)
            eps.append(ep)
            ids.append(record_episode(store, ep, P.flat(params), pending_note, iteration=it, seed=s))
            if k == 0 and it in (0, iterations):
                save_views(sim, out / "views" / f"iter{it:02d}_seed{s}")
        m = metrics(eps, P.CLASSES)
        row = {"iteration": it, "version": version, "params": params, "episode_ids": ids, **m,
               "minutes": round((time.time() - t0) / 60, 1)}
        log(f"iter {it} v{version}: placed {m['success_rate']:.0%}, kits {m['kits_complete']}/{m['episodes']}, "
            f"err {m['mean_error_mm']} mm | " + " ".join(f"{c} {m['per_class'][c]['success_rate']:.0%} {m['per_class'][c]['symptoms']}"
                                                         for c in P.CLASSES) + f" [{row['minutes']} min]")
        applied, rejected = [], []
        if it < iterations:
            for ci, cls in enumerate(P.CLASSES):
                cands = rule.candidates(cls, m["per_class"][cls], params[cls])[:MAX_CHANGES_PER_CLASS]
                for tag, name, new, hyp in cands:
                    cur, _ = P.current()
                    before, after = cur, json.loads(json.dumps(cur))
                    after[cls][name] = new
                    evidence = [f"iteration {it} episodes {ids[0]}..{ids[-1]}: {cls} symptoms {m['per_class'][cls]['symptoms']}",
                                f"{cls} outcomes {m['per_class'][cls]['outcomes']}"]
                    p = SC.propose({P.key(cls, name): new}, hyp, evidence, author="kitting-rule", t="twin")
                    if not p["ok"]:
                        log(f"  proposal refused: {p['errors']}")
                        continue
                    pid = p["proposal"]["id"]
                    seeds = [CHECK_SEED0 + 100 * it + 10 * ci + j for j in range(check_seeds)]
                    chk = ab_check(sim, cls, before, after, seeds, dr, tag)
                    SC.update_proposal(pid, lambda r, c=chk: r["checks"].update(twin_ab=c))
                    sign = 1 if new > before[cls][name] else -1
                    if chk["result"] == "pass":
                        a = SC.apply(pid, by="kitting-loop (twin: auto-apply after a passed check)")
                        if not a.get("ok"):
                            raise RuntimeError(f"apply failed: {a}")
                        applied.append(f"{cls}.{name} {before[cls][name]:g}->{new:g}")
                        log(f"  APPLIED v{a.get('version')}: {cls}.{name} {before[cls][name]:g} -> {new:g} ({chk['summary']})")
                    else:
                        SC.reject(pid, by="kitting-loop", why=chk["summary"])
                        rule.rejected(cls, name, sign, abs(new - before[cls][name]))
                        rejected.append(f"{cls}.{name} {before[cls][name]:g}->{new:g}")
                        log(f"  rejected: {cls}.{name} {before[cls][name]:g} -> {new:g} ({chk['summary']})")
        row.update(applied=applied, rejected=rejected)
        curve.append(row)
        (out / "curve.jsonl").open("a").write(json.dumps(row, default=str) + "\n")
        store.write_summary(curve)
        pending_note = ("skill config after iteration " + str(it) + ": " + ("; ".join(applied) if applied else "no change")
                        + (f" (rejected by the twin check: {'; '.join(rejected)})" if rejected else ""))
        if it < iterations and not applied and not any(rule.candidates(c, m["per_class"][c], params[c]) for c in P.CLASSES):
            log("  no symptom left with an untried change: stopping")
            break
    selection, restored = select_best(curve, P, SC, log)
    (out / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    if restored:
        params, version = P.current()
        eps, ids = [], []
        for k, seed in enumerate(eval_seeds):
            ep = run_episode(sim, params, np.random.default_rng(seed), dr)
            eps.append(ep)
            ids.append(record_episode(store, ep, P.flat(params), "Best evaluated profile restored before held-out testing",
                                      iteration=len(curve), seed=seed))
            if k == 0:
                save_views(sim, out / "views" / f"iter{len(curve):02d}_seed{seed}")
        row = dict(iteration=len(curve), version=version, params=params, episode_ids=ids,
                   **metrics(eps, P.CLASSES), minutes=round((time.time()-t0)/60, 1), applied=[], rejected=[],
                   selected_from=selection["selected_iteration"])
        curve.append(row)
        with (out / "curve.jsonl").open("a") as f:
            f.write(json.dumps(row, default=str)+"\n")
        store.write_summary(curve)
    test = held_out(sim, P, dr, test_episodes, log) if test_episodes else None
    store.write_summary(curve)
    res = {"curve": curve, "test": test, "final_params": P.current()[0], "final_version": P.current()[1],
           "minutes": round((time.time() - t0) / 60, 1), "out": str(out), "dr": vars(dr), "renderer": sim.render_backend, "selection": selection}
    (out / "results.json").write_text(json.dumps(res, indent=1, default=str))
    sim.close()
    return res


def held_out(sim, P, dr, n: int, log) -> dict:
    """Generic vs learned profile on seeds never used for evaluation or checks; with and without randomization."""
    from .sim_env import DR, run_episode
    seeds = [TEST_SEED0 + k for k in range(n)]
    out = {"seeds": [seeds[0], seeds[-1]]}
    learned = P.current()[0]
    for label, params in (("generic", P.generic()), ("learned", learned)):
        for dlabel, d in (("randomized", dr), ("nominal", DR.none())):
            eps = [run_episode(sim, params, np.random.default_rng(s), d) for s in (seeds if dlabel == "randomized" else seeds[:2])]
            if getattr(sim, "episode_store", None) is not None:
                from .kit_episodes import record_episode
                for seed, ep in zip(seeds if dlabel == "randomized" else seeds[:2], eps):
                    record_episode(sim.episode_store, ep, P.flat(params),
                                   f"Held-out {label} profile, {dlabel}: compare without fitting these seeds",
                                   stage="held_out", seed=seed, parameter_source=label, condition=dlabel)
            m = metrics(eps, P.CLASSES)
            out[f"{label}_{dlabel}"] = m
            log(f"test {label} {dlabel}: placed {m['success_rate']:.0%}, kits {m['kits_complete']}/{m['episodes']}, "
                f"err {m['mean_error_mm']} mm, " + ", ".join(f"{c} {m['per_class'][c]['success_rate']:.0%}" for c in P.CLASSES))
    return out


def save_views(sim, prefix: Path):
    """Small overview + top views of the finished kit (docs)."""
    from PIL import Image
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sim.render("overview_cam")).save(str(prefix) + "_overview.jpg", quality=88)
    Image.fromarray(sim.render("top_cam")).save(str(prefix) + "_top.jpg", quality=88)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=8, help="randomized kits per evaluation")
    ap.add_argument("--check-seeds", type=int, default=4, help="paired kits per A/B check")
    ap.add_argument("--test-episodes", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--resume", action="store_true", help="explicitly continue completed curve rows using the current task config")
    ap.add_argument("--export", action="store_true", help="write tasks/kitting/profiles/kit_learned.json")
    a = ap.parse_args()
    if min(a.episodes, a.check_seeds) < 1 or min(a.iterations, a.test_episodes) < 0:
        ap.error("episodes/check-seeds must be positive, iterations/test-episodes nonnegative")
    out = a.out or DATA_ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S")
    logf = (out.mkdir(parents=True, exist_ok=True) or out / "log.txt").open("a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    res = run(a.iterations, a.episodes, a.check_seeds, out, a.test_episodes, log, resume=a.resume)
    if a.export:
        from . import profile as P
        c0, c1 = res["curve"][0], res["curve"][-1]
        prov = {"run": str(out), "selection": res.get("selection"), "renderer": res["renderer"], "iterations": len(res["curve"]) - 1, "twin_version": res["final_version"],
                "eval_episodes_per_iteration": a.episodes,
                "first": {k: c0[k] for k in ("success_rate", "mean_error_mm", "kits_complete", "episodes")},
                "last": {k: c1[k] for k in ("success_rate", "mean_error_mm", "kits_complete", "episodes")},
                "per_class_last": {c: {k: c1["per_class"][c][k] for k in ("success_rate", "mean_error_mm")} for c in P.CLASSES},
                "held_out_test": {k: {"success_rate": v["success_rate"], "mean_error_mm": v["mean_error_mm"],
                                      "kits_complete": v["kits_complete"], "episodes": v["episodes"]}
                                  for k, v in (res["test"] or {}).items() if isinstance(v, dict)},
                "changes": [x for r in res["curve"] for x in r.get("applied", [])]}
        log(f"exported {P.export(provenance=prov)}")


if __name__ == "__main__":
    main()
