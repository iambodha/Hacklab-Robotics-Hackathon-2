"""Offline checks of a skill-config proposal (learning/skillcfg.py), before anyone applies it.

  ik_probe   (seconds) the last real cap poses from episodes.jsonl (locate phases): grasp and hover lean, reachability
             and the approach mode twist_cap would pick, with the config before vs after. Keys: ik.*, twist.approach,
             twist.side_lean_deg, twist.max_lean_deg, twist.hovers_m, twist.grasp_below_top_m.
  twin_run   (minutes) learning/sim_loop.py --real-geometry, before vs after, each in a scratch skill-config directory;
             outcomes, rotation, the cap's true rotation and the approach modes. Optional, usually started in the
             background: `python -m learning.skillcheck twin <proposal id>` writes its result into the proposal.

Results go into proposal["checks"][name] = {"result": pass|fail|skipped|running|error, "summary": text, ...}.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import episodes as E
from . import skillcfg as SC

REPO = Path(__file__).resolve().parents[1]
PROBE_KEYS = ("twist.approach", "twist.side_lean_deg", "twist.max_lean_deg", "twist.hovers_m", "twist.grasp_below_top_m")


def _ik_related(keys) -> bool:
    return any(k.startswith("ik.") or k in PROBE_KEYS for k in keys)


def _twin_related(keys) -> bool:
    return any(not k.startswith(("ik.", "executor.")) for k in keys)


def last_cap_poses(learn_dir=None, n: int = 3) -> list[dict]:
    """The last n distinct cap poses (5 mm apart) the operator gave twist_cap on the real arm, newest first."""
    p = Path(learn_dir or os.environ.get("BOTTLE_LEARN_DIR") or E.DEFAULT_DIR) / "episodes.jsonl"
    if not p.exists():
        return []
    out = []
    for line in reversed(p.read_text().splitlines()):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        m = r.get("measurements") or {}
        if r.get("type") != "phase" or r.get("phase") != "locate" or "cap_top_z" not in m:
            continue
        pose = {"attempt": r["attempt"], "x": m["x"], "y": m["y"], "cap_top_z": m["cap_top_z"]}
        if all(max(abs(pose[k] - o[k]) for k in ("x", "y", "cap_top_z")) > 0.005 for o in out):
            out.append(pose)
        if len(out) >= n:
            break
    return out


_KIN = None


def _kin():
    global _KIN
    if _KIN is None:
        import numpy as np
        from executor.kinematics import Kinematics
        _KIN = Kinematics(T_world_base=np.eye(4))       # base frame = the real_arm_mcp / twist_cap frame
    return _KIN


def _lean(K, xyz, seed, cfg, yaw=0.0):
    import numpy as np
    xyz = np.asarray(xyz, float)
    with K.lock:
        q, lean = SC.lean_search(lambda l, rs: K._solve_rot(xyz, K._target_rot(xyz, l, yaw), seed, rs),
                                 cfg["ik.leans"], cfg["ik.restarts"], cfg["ik.search"])
    return q, lean


def predict(pose: dict, cfg: dict, seed) -> dict:
    """What twist_cap would do at this cap pose with config cfg (no motion): grasp lean, hover, approach mode.
    The grasp lean is solved from two seeds, the park pose and the hover pose (where the arm is on a repeat attempt,
    as twist_cap's reachability check runs from the current joints); the larger lean counts."""
    K = _kin()
    x, y, top = pose["x"], pose["y"], pose["cap_top_z"]
    z_grasp = top - cfg["twist.grasp_below_top_m"]
    max_lean = cfg["twist.max_lean_deg"]
    out, q_h = {"attempt": pose["attempt"]}, None
    for h in cfg["twist.hovers_m"]:
        q, lh = _lean(K, (x, y, top + h), seed, cfg)
        if lh is not None and abs(lh) <= max_lean:
            out.update(hover_m=h, hover_lean_deg=lh)
            q_h = q
            break
    leans = [_lean(K, (x, y, z_grasp), s, cfg)[1] for s in ([seed] + ([q_h] if q_h is not None else []))]
    g = None if None in leans else max(leans, key=abs)
    out.update(grasp_lean_deg=g, grasp_leans_by_seed=leans, reachable=g is not None and abs(g) <= max_lean)
    if not out["reachable"]:
        return {**out, "mode": None}
    ap = cfg["twist.approach"]
    side = ap == "side" or (ap == "auto" and abs(g) > cfg["twist.side_lean_deg"])
    out["mode"] = "side" if side else "top"
    if not side and q_h is None:
        out.update(reachable=False, mode=None, why="no reachable hover")
    return out


def ik_probe(before: dict, after: dict, poses: list[dict]) -> dict:
    if not poses:
        return {"result": "skipped", "summary": "no real cap poses in episodes.jsonl (locate phases) to probe"}
    import numpy as np
    K = _kin()
    seed, _ = K.solve(np.asarray(K.P.poses.park, float), np.zeros(len(K.ids)), 0.0)     # arriving from park
    seed = np.zeros(len(K.ids)) if seed is None else seed
    t0, rows = time.time(), []
    for p in poses:
        rows.append({"pose": p, "before": predict(p, before, seed), "after": predict(p, after, seed)})
    lost = [r["pose"]["attempt"] for r in rows if r["before"]["reachable"] and not r["after"]["reachable"]]

    def fmt(d):
        if not d["reachable"]:
            return f"unreachable (grasp lean {d['grasp_lean_deg']})"
        return f"{d['mode']}, grasp lean {d['grasp_lean_deg']} deg" + (
            f", hover +{d['hover_m'] * 100:.1f} cm at {d['hover_lean_deg']} deg" if d.get("hover_m") else "")
    summary = "; ".join(f"attempt {r['pose']['attempt']} cap [{r['pose']['x']:.3f}, {r['pose']['y']:.3f}, "
                        f"{r['pose']['cap_top_z']:.3f}]: {fmt(r['before'])} -> {fmt(r['after'])}" for r in rows)
    return {"result": "fail" if lost else "pass", "rows": rows, "seconds": round(time.time() - t0, 1),
            "summary": (f"LOSES reach at attempts {lost}: " if lost else "") + summary}


# --- twin run ----------------------------------------------------------------------------------------------------
def check_python() -> list[str]:
    """The interpreter command for the twin run: SKILLCFG_CHECK_PYTHON, else this one if it has the twin's packages,
    else uv with them (the real-arm server's environment has no opencv/scipy)."""
    if os.environ.get("SKILLCFG_CHECK_PYTHON"):
        return os.environ["SKILLCFG_CHECK_PYTHON"].split()
    if all(importlib.util.find_spec(m) for m in ("mujoco", "cv2", "PIL", "mcp", "scipy")):
        return [sys.executable]
    return ["uv", "run", "--no-project", "-q", "--with", "numpy", "--with", "scipy", "--with", "pillow", "--with",
            "mujoco", "--with", "opencv-contrib-python", "--with", "mcp>=1.12,<2", "--with", "httpx", "python"]


def twin_run(values: dict, workdir: Path, attempts: int = 1, timeout: float = 900) -> dict:
    """learning.sim_loop --real-geometry with config `values` (as the twin's current config, in a scratch dir)."""
    workdir.mkdir(parents=True, exist_ok=True)
    d = SC.defaults()
    SC._write_json(workdir / "cfg" / "twin.json", {"target": "twin", "version": 0,
                                                   "values": {k: v for k, v in values.items() if d.get(k) != v}})
    env = {**os.environ, "SKILLCFG_DIR": str(workdir / "cfg"), "SKILLCFG_TARGET": "twin", "BOTTLE_GEMINI_JUDGE": "0",
           "ROBOT_SIM_RECORD": "0", "MUJOCO_GL": os.environ.get("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")}
    t0 = time.time()
    try:
        p = subprocess.run(check_python() + ["-m", "learning.sim_loop", "--attempts", str(attempts), "--real-geometry",
                                             "--learn-dir", str(workdir / "learn"), "--layout", str(workdir / "layout.json")],
                           cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timed out after {timeout:.0f} s"}
    if p.returncode != 0:
        return {"ok": False, "error": p.stderr[-800:]}
    summary = json.loads([l for l in p.stdout.splitlines() if l.startswith('{"summary"')][-1])["summary"]
    recs = [json.loads(l) for l in (workdir / "learn" / "episodes.jsonl").read_text().splitlines() if l.strip()]
    app = [r for r in recs if r["type"] == "phase" and r["phase"] == "approach"]
    return {"ok": True, "seconds": round(time.time() - t0), "outcomes": [a["outcome"] for a in summary],
            "rotation_deg": [a["rotation_achieved_deg"] for a in summary],
            "cap_turned_deg": round(summary[-1]["truth"]["turned_deg"], 1) if summary else 0.0,
            "approach": [{"status": r["status"], "mode": r["measurements"].get("mode"),
                          "lean_deg": r["measurements"].get("lean_deg")} for r in app]}


def _before_after(p: dict) -> tuple[dict, dict]:
    cur = SC.active(p["target"])["values"]
    before = {**cur, **{k: c["before"] for k, c in p["changes"].items()}}
    return before, {**cur, **{k: c["after"] for k, c in p["changes"].items()}}


def twin_check(pid: str, attempts: int = 1) -> dict:
    p = SC.proposal(pid)
    before, after = _before_after(p)
    scratch = Path(tempfile.mkdtemp(prefix=f"skillcheck-{pid}-", dir=os.environ.get("SKILLCFG_SCRATCH")))
    try:
        b = twin_run(before, scratch / "before", attempts)
        a = twin_run(after, scratch / "after", attempts)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if not (a["ok"] and b["ok"]):
        res = {"result": "error", "summary": "twin run failed: " + (b.get("error") or a.get("error") or "")[-300:],
               "before": b, "after": a}
    else:
        def fmt(r):
            modes = ",".join(f"{x['mode']}@{x['lean_deg']}" for x in r["approach"]) or "no approach"
            return f"{'/'.join(r['outcomes'])}, rotation {r['rotation_deg']} deg, cap turned {r['cap_turned_deg']} deg ({modes})"
        worse = (sum(a["rotation_deg"]) < 0.5 * sum(b["rotation_deg"]) and a["cap_turned_deg"] < b["cap_turned_deg"]) \
            or (a["outcomes"].count("aborted") > b["outcomes"].count("aborted"))
        res = {"result": "fail" if worse else "pass", "before": b, "after": a,
               "summary": f"twin ({attempts} attempt(s), real geometry): before {fmt(b)} -> after {fmt(a)}"}
    SC.update_proposal(pid, lambda r: r["checks"].__setitem__("twin_run", res))
    return res


def start_twin_check(pid: str, attempts: int = 1) -> str:
    """Run twin_check in a detached process (log in SKILLCFG_DIR/checks/<id>.log). Returns the log path."""
    log = SC.root() / "checks" / f"{pid}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    SC.update_proposal(pid, lambda r: r["checks"].__setitem__(
        "twin_run", {"result": "running", "summary": f"started {time.strftime('%H:%M:%S')}, log {log}"}))
    with open(log, "a") as f:
        subprocess.Popen(check_python() + ["-m", "learning.skillcheck", "twin", pid, "--attempts", str(attempts)],
                         cwd=REPO, env={**os.environ, "SKILLCFG_DIR": str(SC.root())}, stdout=f, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    return str(log)


def run_checks(pid: str, twin: bool | str = False, poses: list[dict] | None = None) -> dict:
    """The checks for a proposal's keys; twin: False | "async" | "sync" (the twin run). Returns the proposal."""
    p = SC.proposal(pid)
    keys = list(p["changes"])
    before, after = _before_after(p)
    checks = {"validation": {"result": "pass", "summary": "types, envelopes and safety class OK"}}
    if _ik_related(keys):
        try:
            checks["ik_probe"] = ik_probe(before, after, last_cap_poses() if poses is None else poses)
        except Exception as e:  # noqa: BLE001  (a broken probe must not block the proposal record)
            checks["ik_probe"] = {"result": "error", "summary": f"{type(e).__name__}: {e}"}
    SC.update_proposal(pid, lambda r: r["checks"].update(checks))
    if twin and _twin_related(keys):
        if twin == "sync":
            twin_check(pid)
        else:
            start_twin_check(pid)
    return SC.proposal(pid)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(prog="python -m learning.skillcheck")
    ap.add_argument("what", choices=("twin", "ik"))
    ap.add_argument("id")
    ap.add_argument("--attempts", type=int, default=1)
    a = ap.parse_args()
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    r = twin_check(a.id, a.attempts) if a.what == "twin" else run_checks(a.id)
    print(json.dumps(r, default=str)[:4000])
