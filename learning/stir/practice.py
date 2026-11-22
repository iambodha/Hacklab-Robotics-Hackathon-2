"""Practise stirring in the twin: the learning loop that taught the real arm the bottle cap, on a mug and a spoon.

    python -m learning.stir.practice [--attempts 16] [--validate 5] [--out DIR] [--write-profile]

Each attempt: choose a change (the policy below reads the last verdicts) -> record it as a skill-config version of
target "twin" (learning/skillcfg.py: proposal with reason and evidence, applied at once, as twin proposals are) ->
episode_begin (params, hypothesis) -> the stir skill (sim/stir_skill.py) with a phase record per step -> the twin
judge (learning/stir/judge.py, from ground truth) -> episode_end (outcome, lesson, next change). The best clean
configuration becomes learning/profiles/stir_cup.json (--write-profile): each learned value with the attempts and the
evidence behind it. --validate N then runs N randomised mug positions with the code baseline and with the learned
values on the same seeds.

Bulky output (records, images, the skill config history) goes to DATA_ROOT/stir/<run> (DATA_ROOT defaults to the
gitignored data/). Never touches the Station or the real arm: the twin is sim/sim_arm_mcp.py's in-process MuJoCo arm.

The policy is a scripted hill climb with diagnosis rules, not an LLM: it proves the records, the config versions and
the judge end to end, and it is what an operator would do with the verdicts (docs/stir/README.md).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("DATA_ROOT") or REPO / "data") / "stir"
PROFILE = REPO / "learning" / "profiles" / "stir_cup.json"

# The learnable stir settings (skill-config keys, envelopes in skills/strategy_defaults.json)
KEYS = ("sim.stir.radius_m", "sim.stir.turns", "sim.stir.height_m", "sim.stir.step_dt_s", "sim.stir.wall_margin_m",
        "sim.stir.lean_extra_deg", "sim.stir.grasp_from_top_m", "sim.stir.ring_points")
SHORT = {k: k.split(".")[-1] for k in KEYS}

# The search: (quality term it serves, key, step, why). One key at a time, continued while the score improves.
SEARCH = (
    ("coverage", "sim.stir.radius_m", +0.003, "a wider circle moves more of the drink (coverage)"),
    ("depth", "sim.stir.height_m", -0.006, "a lower circle mixes what settles at the bottom (depth)"),
    ("pace", "sim.stir.step_dt_s", -0.01, "shorter steps on the circle stir faster (pace)"),
)
WALL_STEP = -0.001        # radius capped by the handle at the rim -> let the handle closer to the rim, 1 mm at a time
GRASP_STEP = 0.005        # m along the handle per grasp correction
SLOWER = 0.01             # s per step added after the spoon slid, the mug moved or the arm could not follow
CAPPED_TOL = 0.0005       # a stirring radius this much under the asked one = capped by the wall margin
IMPROVE_EPS = 0.005       # a score must beat the best by this much to count as better (twin repeat noise ~0.002)
MAX_CHANGES = 3           # as the bottle's episodes: change at most 3 parameters per attempt


# --- the policy ---------------------------------------------------------------------------------------------------
class Policy:
    """Hill climb over SEARCH from the current config, with diagnosis rules for failures.

    After each attempt: a clean stir that beats the best becomes the best and the same move is tried again; a move that
    did not pay (worse, or a failed gate) is undone (back to the best) and its limit recorded, and the next search
    direction starts. A failure that did not come from a search move gets the diagnosis rule's fix. When every
    direction has found its limit, the best is repeated (a consistency check)."""

    def __init__(self, cfg: dict, envelope: dict):
        self.cfg, self.env = dict(cfg), envelope
        self.best: tuple[float, dict, int] | None = None   # (score, cfg, attempt)
        self.best_phases: dict = {}
        self.dim = 0                                       # index into SEARCH
        self.last: dict | None = None                      # {"key", "kind"} of the last move
        self.limits: dict[str, str] = {}                   # key -> what stopped it (evidence for the profile)

    def _clip(self, key, v):
        lo, hi = self.env[key]
        v = min(hi, max(lo, v))
        return round(v, 4) if isinstance(v, float) else v

    def _step(self, key, step):
        new = self._clip(key, self.cfg[key] + step)
        return None if new == self.cfg[key] else {key: new}

    def next(self, n: int, verdict: dict | None, phases: dict) -> tuple[dict, str]:
        """(changes, hypothesis) for the attempt after attempt n (verdict None: the first attempt)."""
        if verdict is None:
            self.last = {"key": None, "kind": "baseline"}
            return {}, ("Baseline: the r4 stir skill's code values as they are, on the office mug (8 cm, 9.5 cm tall). "
                        "Expect a clean but small, slow, high stir.")
        if verdict["stirred"] and (self.best is None or verdict["score"] > self.best[0] + IMPROVE_EPS):
            self.best, self.best_phases = (verdict["score"], dict(self.cfg), n), phases
            return self._search_or_repeat(phases, again=self.last["kind"] == "search", n=n)
        if self.best is None:                                   # nothing clean yet: repair
            fix, why = self._diagnose(verdict, phases)
            if fix:
                self.last = {"key": next(iter(fix)), "kind": "fix"}
                return self._apply(fix), why
            return self._search_or_repeat(phases, again=False, n=n)
        back = {k: v for k, v in self.best[1].items() if self.cfg.get(k) != v}
        if self.last["kind"] == "search":                       # the move did not pay: undo it, next direction
            k = self.last["key"]
            self.limits[k] = (f"attempt {n}: {SHORT[k]}={self.cfg[k]} -> {verdict['outcome']}, score {verdict['score']}"
                              f" (best {self.best[0]})")
            self.dim += 1
            why = (f"{SHORT[k]}={self.cfg[k]} did not pay ({verdict['outcome']}, score {verdict['score']} vs best "
                   f"{self.best[0]}): back to {SHORT[k]}={self.best[1][k]}")
            self._apply(back)
            nxt, why2 = self._search_move(self.best_phases)
            if nxt and len({**back, **nxt}) <= MAX_CHANGES:
                self.last = {"key": next(iter(nxt)), "kind": "search"}
                return {**back, **self._apply(nxt)}, f"{why}; then {why2}"
            self.last = {"key": None, "kind": "revert"}
            return back, why + "."
        if not verdict["stirred"]:                              # the best config itself failed this time
            fix, why = self._diagnose(verdict, phases)
            if fix:
                self.last = {"key": next(iter(fix)), "kind": "fix"}
                return self._apply({**back, **fix}), why
        if back:
            self.last = {"key": None, "kind": "revert"}
            return self._apply(back), f"Back to the best configuration (attempt {self.best[2]})."
        return self._search_or_repeat(self.best_phases, again=False, n=n)

    def _search_or_repeat(self, phases, again, n):
        ch, why = self._search_move(phases, again=again)
        if ch:
            self.last = {"key": next(iter(ch)), "kind": "search"}
            return self._apply(ch), why
        self.last = {"key": None, "kind": "repeat"}
        return {}, (f"Every search direction has found its limit: repeat the best configuration (attempt "
                    f"{self.best[2] if self.best else n}) to check that the result holds.")

    def _search_move(self, phases, again=False):
        """(changes, why) of the next search move from the current config, or (None, "") when all are exhausted."""
        cfg = self.cfg
        while self.dim < len(SEARCH):
            term, key, step, why = SEARCH[self.dim]
            if term == "coverage":
                stir = phases.get("stir", {})
                if stir.get("radius_asked_m", 0) - stir.get("radius_m", 0) > CAPPED_TOL:
                    # the handle's margin to the rim limits the circle, not the asked radius
                    new = self._clip("sim.stir.wall_margin_m", cfg["sim.stir.wall_margin_m"] + WALL_STEP)
                    if new != cfg["sim.stir.wall_margin_m"]:
                        return {"sim.stir.wall_margin_m": new}, (
                            f"the circle was capped at {stir.get('radius_m', 0) * 100:.1f} cm (asked "
                            f"{stir.get('radius_asked_m', 0) * 100:.1f} cm) by the handle's margin to the rim: let the "
                            f"handle 1 mm closer to the rim (wall_margin {cfg['sim.stir.wall_margin_m']} -> {new}): "
                            f"{why}.")
                    self.dim += 1
                    continue
            new = self._clip(key, cfg[key] + step)
            if new == cfg[key]:
                self.dim += 1
                continue
            return {key: new}, f"{'keep going: ' if again else ''}{SHORT[key]} {cfg[key]} -> {new}: {why}."
        return None, ""

    def _diagnose(self, verdict, phases) -> tuple[dict | None, str]:
        o, m = verdict["outcome"], verdict.get("metrics", {})
        if o == "no_grasp":
            sign = 1 if self.cfg["sim.stir.grasp_from_top_m"] < 0.04 else -1
            return self._step("sim.stir.grasp_from_top_m", sign * GRASP_STEP), (
                "The jaws closed on nothing: take the handle at another height.")
        if o == "arm_hit_cup":
            return self._step("sim.stir.grasp_from_top_m", -GRASP_STEP), (
                f"The jaws touched the mug ({m.get('arm_cup_s')} s): hold the handle higher, further from the rim.")
        if o in ("slipped", "pushed_cup", "tipped_cup", "dropped_spoon", "aborted"):
            return self._step("sim.stir.step_dt_s", SLOWER), (
                f"{o.replace('_', ' ')} ({'; '.join(verdict.get('evidence', [])[-1:])}): stir more slowly.")
        if o == "scraped_wall":
            return self._step("sim.stir.wall_margin_m", -WALL_STEP), (
                f"The spoon scraped the wall {m.get('wall_frac', 0):.0%} of the stir: keep the handle further from the rim.")
        if o == "stirred_air":
            return self._step("sim.stir.height_m", -0.005), "The bowl was above the drink: stir lower."
        if o == "too_few_loops":
            return self._step("sim.stir.turns", 1), f"Only {m.get('loops')} loops: one more turn."
        return None, f"{o}."

    def _apply(self, changes: dict) -> dict:
        self.cfg.update(changes)
        return changes


# --- one attempt in the twin --------------------------------------------------------------------------------------
def _spoon_in_jaws(S):
    """The spoon's top end in the gripper frame (m): constant while the spoon is held rigidly."""
    import numpy as np
    arm = S.arm
    tcp, R = arm.ik.fk(arm.measured_q())
    with arm.lock:
        top = arm.data.site(arm.layout["stir"]["tool_top"]).xpos.copy()
    return (R.T @ (top - tcp)).tolist(), np.asarray(tcp)


def _jpeg(S, cam, scale=1):
    import io

    from PIL import Image
    im = Image.fromarray(S.arm.frame(cam))
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


async def run_attempt(S, store, cfg: dict, hypothesis: str, *, seed=None, click_noise_m=0.0, images=True,
                      config_version=None) -> tuple[int, dict, dict]:
    """Reset the scene, run the stir skill with cfg, judge it. Returns (attempt, verdict, phase measurements)."""
    import numpy as np

    import stir_skill
    from learning.stir import judge as J
    from learning.stir.twin import load_task
    arm = S.arm
    task = load_task()
    geo = arm.layout["stir"]
    if seed is not None:
        arm.rng = np.random.default_rng(seed)
    arm.reset(randomize=seed is not None)
    await asyncio.sleep(0.6)                                  # let the scene settle in real time
    with arm.lock:
        cup0 = arm.data.body(geo["cup"]).xpos.copy()
    rng = np.random.default_rng(None if seed is None else seed + 1000)
    click = cup0[:2] + (rng.normal(0, click_noise_m, 2) if click_noise_m else 0)   # a pixel_to_table click
    n = store.begin(cfg, hypothesis, seed=seed, cup_xy=[round(float(v), 4) for v in cup0[:2]],
                    click_xy=[round(float(v), 4) for v in click], skillcfg_twin_version=config_version)
    start = arm.stir_mark()
    marks, phases, held = {}, {}, {}

    def on_phase(name, status, info):
        marks[name] = arm.stir_mark()
        meas = dict(info)
        if name == "lead_in":
            held["start"] = _spoon_in_jaws(S)[0]
        if name == "stir" and "lead_in" in marks:
            held["end"] = _spoon_in_jaws(S)[0]
            win = arm.stir_summary(since=marks["lead_in"])
            phases["_window"] = win
            phases["_stir_s"] = win["sim_s"]          # sim seconds: the twin's own clock (it may run under real time)
            meas["window"] = {k: v for k, v in win.items() if k != "pts"}
            if images:
                meas["image"] = store.save_image(n, "stir_side.jpg", _jpeg(S, "stir_cam"))
        phases[name] = {"status": status, **info}
        store.phase(n, name, status, meas)

    status, log = await stir_skill.StirSkill(arm, S.move_tcp, S.gripper).run(
        float(click[0]), float(click[1]), None, geo["rim_height"], geo["inner_radius"], geo["spoon_length"],
        int(cfg["sim.stir.turns"]), float(cfg["sim.stir.radius_m"]), grasp_from_top=cfg["sim.stir.grasp_from_top_m"],
        stir_height=cfg["sim.stir.height_m"], step_dt=cfg["sim.stir.step_dt_s"],
        ring_points=int(cfg["sim.stir.ring_points"]), wall_margin=cfg["sim.stir.wall_margin_m"],
        lean_extra_deg=cfg["sim.stir.lean_extra_deg"], on_phase=on_phase)
    await asyncio.sleep(0.5)
    whole = arm.stir_summary(since=start)
    with arm.lock:
        cup1 = arm.data.body(geo["cup"]).xpos.copy()
        tip = arm.data.site(geo["tool_tip"]).xpos.copy()
    rel = tip - cup1
    in_cup = bool(math.hypot(rel[0], rel[1]) < geo["inner_radius"] and rel[2] < geo["rim_height"] / 2)
    slip = (float(np.linalg.norm(np.subtract(held["end"], held["start"]))) if "end" in held and "start" in held
            else 0.0)
    stir_ph = phases.get("stir", {})
    aborted = None
    if stir_ph.get("status") in ("abort", "fail"):
        aborted = stir_ph.get("reason") or stir_ph["status"]
    elif phases.get("straighten", {}).get("status") in ("abort", "fail"):
        aborted = "straighten: " + str(phases["straighten"].get("reason"))
    extras = {"grasped": phases.get("grasp", {}).get("status") == "ok", "aborted": aborted,
              "stir_s": phases.get("_stir_s"), "cup_moved_m": float(np.linalg.norm(cup1[:2] - cup0[:2])),
              "cup_tilt_max_deg": whole["cup_tilt_max_deg"], "slip_m": slip, "spoon_in_cup": in_cup,
              "arm_cup_s": whole["arm_cup_s"]}
    verdict = J.judge(phases.get("_window") if "_window" in phases else None, extras, task["judge"], geo)
    store.judge(n, verdict)
    if "_window" in phases:
        store.save_json(n, "path.json", phases["_window"]["pts"])
    if images:
        store.save_image(n, "after_overview.jpg", _jpeg(S, "overview_cam"))
    phases["_log"] = log
    phases["_status"] = status
    return n, verdict, phases


def lesson(verdict: dict, changed: dict, prev_best: float | None) -> str:
    ch = ", ".join(f"{SHORT[k]} {c['before']}->{c['after']}" for k, c in changed.items()) or "no change"
    t, m = verdict.get("terms", {}), verdict.get("metrics", {})
    if not verdict["stirred"]:
        return f"{ch}: {verdict['outcome']} ({verdict['evidence'][-1]})."
    better = prev_best is None or verdict["score"] > prev_best + IMPROVE_EPS
    return (f"{ch}: stirred cleanly, score {verdict['score']} ({'better than' if better else 'no better than'} the "
            f"best {prev_best if prev_best is not None else '-'}); coverage {t.get('coverage')}, depth {t.get('depth')}, "
            f"pace {t.get('pace')} ({m.get('pace_loops_per_s')} loops/s), {m.get('loops')} loops.")


# --- the skill config (twin target, scratch dir per run) ------------------------------------------------------------
def config_apply(changes: dict, reason: str, evidence: list[str]) -> int:
    from learning import skillcfg as SC
    if not changes:
        return SC.version("twin")
    r = SC.propose(changes, reason, evidence, author="stir practice (learning/stir/practice.py)", t="twin")
    if not r["ok"]:
        raise RuntimeError("; ".join(r["errors"]))
    a = SC.apply(r["proposal"]["id"], by="stir practice loop (twin: auto-apply)")
    if not a["ok"]:
        raise RuntimeError("; ".join(a["errors"]))
    return a["version"]


def current_cfg() -> dict:
    from learning import skillcfg as SC
    v = SC.active("twin")["values"]
    return {k: v[k] for k in KEYS}


def envelope() -> dict:
    from learning import skillcfg as SC
    reg = SC.registry()
    return {k: (reg[k]["lo"], reg[k]["hi"]) for k in KEYS}


async def practice(S, store, attempts: int, log=print) -> list[dict]:
    pol = Policy(current_cfg(), envelope())
    rows, verdict, phases, n = [], None, {}, 0
    for i in range(attempts):
        prev_best = pol.best[0] if pol.best else None
        changes, hyp = pol.next(n, verdict, phases)
        ev = [f"stir attempt {n}: {verdict['outcome']}, score {verdict['score']}"] if verdict else ["code baseline"]
        version = config_apply(changes, hyp, ev)
        cfg = current_cfg()
        assert all(cfg[k] == v for k, v in changes.items()), (cfg, changes)
        n, verdict, phases = await run_attempt(S, store, cfg, hyp, config_version=version)
        att = store.attempts()[n]
        les = lesson(verdict, att["begin"]["changed"], prev_best)
        store.end(n, verdict["outcome"], les, verdict["score"])
        row = {"attempt": n, "outcome": verdict["outcome"], "score": verdict["score"],
               "best": max(verdict["score"] if verdict["stirred"] else 0, prev_best or 0),
               "changed": {SHORT[k]: c["after"] for k, c in att["begin"]["changed"].items()},
               "loops": verdict["metrics"].get("loops"), "radius_cm": round(100 * verdict["metrics"].get(
                   "radius_about_axis_m", 0), 2), "pace": verdict["metrics"].get("pace_loops_per_s"),
               "terms": verdict.get("terms"), "config_version": version,
               "rtf": (phases.get("_window") or {}).get("real_time_factor"), "cfg": cfg}
        rows.append(row)
        log(json.dumps({k: v for k, v in row.items() if k != "cfg"}))
    if verdict and verdict["stirred"] and (pol.best is None or verdict["score"] > pol.best[0] + IMPROVE_EPS):
        pol.best = (verdict["score"], dict(pol.cfg), n)          # the last attempt was the best: count it
    return rows, pol


async def validate(S, store, cfg: dict, seeds, label: str, click_noise_m: float, log=print) -> list[dict]:
    out = []
    for seed in seeds:
        n, v, _ = await run_attempt(S, store, cfg, f"validation ({label}): randomised mug, seed {seed}", seed=seed,
                                    click_noise_m=click_noise_m, images=False)
        store.end(n, v["outcome"], f"{label} on seed {seed}: {v['outcome']}, score {v['score']}", v["score"])
        out.append({"seed": seed, "outcome": v["outcome"], "stirred": v["stirred"], "score": v["score"],
                    "loops": v["metrics"].get("loops"), "radius_cm": round(100 * v["metrics"].get(
                        "radius_about_axis_m", 0), 2), "pace": v["metrics"].get("pace_loops_per_s")})
        log(json.dumps({"validation": label, **out[-1]}))
    return out


# --- the learned profile ------------------------------------------------------------------------------------------
def write_profile(rows: list[dict], pol: Policy, baseline: dict, run_dir: Path, validation: dict | None,
                  path: Path = PROFILE) -> dict:
    from learning.stir.twin import load_task, spoon_length
    task = load_task()
    score, best, n_best = pol.best
    entries = {}
    for k in KEYS:
        if best[k] == baseline[k]:
            continue
        used = [r["attempt"] for r in rows if r["cfg"][k] == best[k] and r["outcome"] == "stirred"]
        tried = sorted({r["cfg"][k] for r in rows})
        ev = (f"{SHORT[k]} {baseline[k]} -> {best[k]}: the best clean stir (attempt {n_best}, score {score}) used "
              f"{best[k]}; values tried {tried}.")
        if k in pol.limits:
            ev += f" Limit: {pol.limits[k]}."
        entries[k] = {"value": best[k], "baseline": baseline[k], "attempts": used or [n_best],
                      "date": time.strftime("%Y-%m-%d"), "evidence": ev, "source": "twin"}
    c, s = task["cup"], task["spoon"]
    prof = {
        "profile": "stir_cup", "version": 1,
        "skill": "stir a drink with a spoon standing in a mug: stir_spoon (sim/stir_skill.py)",
        "learned_on": time.strftime("%Y-%m-%d"),
        "robot": "ElRobot (sim/robots/elrobot.json) - MuJoCo twin of the office table (sim/layouts/stir_office.json)",
        "source": (f"learning/stir/practice.py: {len(rows)} twin attempts; records in DATA_ROOT/stir/{run_dir.name}/ "
                   f"(episodes.jsonl, SUMMARY.md); curated copy in docs/stir/data/"),
        "learned_in": "twin",
        "how_to_read": ("docs/stir/README.md. Each entry: value (the default in force), baseline (the code value it "
                        "replaces), attempts (the twin attempts that used it and stirred cleanly), evidence, date, "
                        "source = twin: learned in simulation only, not yet on the real arm (docs/stir/REAL_RUN.md)."),
        "object": {"cup_inner_radius_m": c["inner_radius"], "cup_height_m": c["height"], "cup_wall_m": c["wall"],
                   "cup_xy_m": c["xy"], "liquid_level": c["liquid_level"], "spoon_length_m": spoon_length(s),
                   "spoon_handle_width_m": s["handle_width"], "site": task["site"]},
        "history": [{"version": 1, "date": time.strftime("%Y-%m-%d"), "by": "stir-skill session (twin practice)",
                     "change": f"first stir profile: {len(rows)} twin attempts, best = attempt {n_best}"}],
        "entries": entries,
    }
    if validation:
        prof["validation"] = validation
    path.write_text(json.dumps(prof, indent=1) + "\n")
    return prof


def summarise_validation(res: list[dict]) -> dict:
    ok = [r for r in res if r["stirred"]]
    return {"runs": len(res), "stirred": len(ok), "mean_score": round(sum(r["score"] for r in res) / max(1, len(res)), 3),
            "mean_radius_cm": round(sum(r["radius_cm"] for r in ok) / max(1, len(ok)), 2),
            "mean_pace": round(sum(r["pace"] or 0 for r in ok) / max(1, len(ok)), 3),
            "outcomes": {o: sum(1 for r in res if r["outcome"] == o) for o in sorted({r["outcome"] for r in res})}}


def setup_env(run_dir: Path):
    """Environment for an in-process twin: the stir layout, no recorder, the twin config in run_dir/skillcfg, the code
    baseline as the starting point (no learned profile)."""
    sys.path.insert(0, str(REPO))
    from learning.stir import twin
    lay = REPO / "sim" / "layouts" / "stir_office.json"
    if not lay.exists() or lay.stat().st_mtime < twin.TASK.stat().st_mtime:
        twin.make_layout(lay)
    os.environ["SIM_LAYOUT"] = os.environ["ROBOT_SIM_LAYOUT"] = str(lay)
    os.environ.setdefault("ROBOT_SIM_RECORD", "0")
    os.environ["SKILLCFG_DIR"] = str(run_dir / "skillcfg")
    os.environ["SKILLCFG_TARGET"] = "twin"
    os.environ.setdefault("STATION_HOST", "station-disabled-in-tests.invalid")
    import recorder  # noqa: F401  (pin the recorder package before sim/ (with its own recorder.py) joins sys.path)
    sys.path.append(str(REPO / "sim"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--attempts", type=int, default=16)
    ap.add_argument("--validate", type=int, default=0, help="randomised validation runs (baseline and learned)")
    ap.add_argument("--click-noise", type=float, default=0.01, help="sd (m) of the mug-centre click in validation")
    ap.add_argument("--out", default=None)
    ap.add_argument("--write-profile", action="store_true")
    ap.add_argument("--start", action="append", default=[], metavar="KEY=VALUE",
                    help="start from this skill-config value instead of the baseline (e.g. sim.stir.turns=1)")
    a = ap.parse_args(argv)
    run_dir = Path(a.out or DATA_ROOT / time.strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SKILL_PROFILES"] = "none"   # practise from the code baseline, not from an earlier profile
    setup_env(run_dir)
    import sim_arm_mcp as S
    from learning.stir.store import StirStore
    baseline = current_cfg()
    if a.start:
        start = {k: json.loads(v) for k, v in (x.split("=", 1) for x in a.start)}
        config_apply(start, "operator start setting (practice --start)", ["command line"])
    t0 = time.time()
    rows, pol = asyncio.run(practice(S, StirStore(run_dir / "practice"), a.attempts))
    out = {"run_dir": str(run_dir), "baseline": baseline, "rows": rows,
           "best": {"attempt": pol.best[2], "score": pol.best[0], "cfg": pol.best[1]} if pol.best else None,
           "limits": pol.limits, "practice_s": round(time.time() - t0)}
    if a.validate and pol.best:
        seeds = list(range(1, a.validate + 1))
        vstore = StirStore(run_dir / "validation")
        vb = asyncio.run(validate(S, vstore, baseline, seeds, "baseline", a.click_noise))
        vl = asyncio.run(validate(S, vstore, pol.best[1], seeds, "learned", a.click_noise))
        out["validation"] = {"seeds": seeds, "click_noise_m": a.click_noise, "baseline": vb, "learned": vl,
                             "summary": {"baseline": summarise_validation(vb), "learned": summarise_validation(vl)}}
    (run_dir / "result.json").write_text(json.dumps(out, indent=1))
    if a.write_profile and pol.best:
        write_profile(rows, pol, baseline, run_dir, out.get("validation", {}).get("summary"))
        print(f"wrote {PROFILE}", flush=True)
    print(json.dumps({"summary": {k: out[k] for k in ("run_dir", "best", "limits", "practice_s")},
                      "validation": out.get("validation", {}).get("summary")}), flush=True)
    S.arm.running = False


if __name__ == "__main__":
    main()
