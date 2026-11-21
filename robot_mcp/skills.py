"""skills — verified, cancellable tasks an agent can start on a robot.

Each skill runs inside an Operation (see core.py) under the robot's ownership
lock. It reports progress via op.report(), honors cancellation via
op.check_cancel(), and returns a structured result that distinguishes
"command sent" / "target reached" / "outcome verified".

Seed poses (HOME, grasp grid, drop path) come from last year's identical arm —
re-teach on this year's setup where noted.
"""
from __future__ import annotations

import json

from . import config, grasp_model, motion
from .core import Observation, Operation, RobotAdapter

# Taught home pose (normalized) from last year's rig. Ranges this year are
# within ~4% per joint, so it transfers — still, verify once on hardware.
HOME = {"j1": 0.526, "j2": 0.027, "j3": 0.443, "j4": 0.981,
        "j5": 0.532, "j6": 0.581, "j7": 0.45, "j8": 0.78}

DROP_PATH = config.DATA_DIR / "drop_path.json"

SKILLS: dict[str, dict] = {}


def skill(name: str, description: str, args: dict | None = None):
    def deco(fn):
        SKILLS[name] = {"fn": fn, "description": description, "args": args or {}}
        return fn
    return deco


def describe(name: str) -> dict:
    s = SKILLS[name]
    return {"name": name, "description": s["description"], "args": s["args"]}


async def _final_state(adapter: RobotAdapter) -> dict:
    obs: Observation = await adapter.joint_state()
    return obs.joint_map()


@skill("home", "Creep to the taught home pose (safe default; jaws open unless "
               "gripper='closed').", {"gripper": "open|closed|hold"})
async def home(adapter: RobotAdapter, op: Operation, gripper: str = "open") -> dict:
    target = dict(HOME)
    if gripper == "closed":
        target[adapter.gripper_joint] = motion.GRIPPER_CLOSED
    elif gripper == "hold":
        target.pop(adapter.gripper_joint, None)
    op.report("creeping to home pose")
    ok = await motion.creep(adapter, op, target)
    return {"success": ok, "verified": ok, "detail": "at home pose" if ok else "did not converge",
            "final_state": await _final_state(adapter)}


@skill("goto", "Creep to named absolute joint targets (normalized 0..1), "
               "overload-safe. args: {'targets': {'j2': 0.4, ...}}",
       {"targets": "dict joint->norm"})
async def goto(adapter: RobotAdapter, op: Operation, targets: dict) -> dict:
    unknown = [k for k in targets if k not in adapter.joint_names]
    if unknown:
        raise ValueError(f"unknown joints {unknown}")
    op.report(f"creeping to {targets}")
    ok = await motion.creep(adapter, op, {k: float(v) for k, v in targets.items()})
    return {"success": ok, "verified": ok, "detail": "target reached" if ok else "did not converge",
            "final_state": await _final_state(adapter)}


async def _approach(adapter: RobotAdapter, op: Operation, u: float, v: float) -> dict:
    op.report("creeping to home first")
    await motion.creep(adapter, op, HOME)
    pose = grasp_model.grasp_pose(u, v)
    pose[adapter.gripper_joint] = motion.GRIPPER_OPEN
    if "j7" in pose:
        pose["j7"] = 0.45                        # gripper rotation for the grasp
    op.report(f"creeping to approach pose (u={u}, v={v}), jaws open")
    ok = await motion.creep(adapter, op, pose)
    if not ok:
        raise RuntimeError("approach did not converge")
    return pose


async def _lift_and_carry_home(adapter: RobotAdapter, op: Operation) -> None:
    """Two-stage, overload-safe lift, then carry the object home still gripping.
    Never raise the shoulder (j2) while the arm is extended — tuck j3 first."""
    op.report("lift 1: tuck elbow (j3) in + wrist (j4) up")
    cur = (await adapter.joint_state()).joint_map()
    tuck = dict(cur)
    tuck["j3"] = max(0.10, cur["j3"] - 0.30)
    tuck["j4"] = min(0.95, cur["j4"] + 0.30)
    await motion.creep(adapter, op, tuck)
    op.report("lift 2: raise shoulder (j2) to clear the ground")
    cur = (await adapter.joint_state()).joint_map()
    clear = dict(cur)
    clear["j2"] = min(cur["j2"], 0.15)
    await motion.creep(adapter, op, clear)
    op.report("carrying home (jaws stay closed)")
    cur = (await adapter.joint_state()).joint_map()
    hg = dict(HOME)
    hg[adapter.gripper_joint] = cur[adapter.gripper_joint]
    await motion.creep(adapter, op, hg)


@skill("pick", "Pick an object at sheet position (u: left0..right1, "
               "v: top0..bottom1) using the taught grasp grid, then verify the "
               "grip (jaws stall => object held) and carry it home. Returns "
               "verified=False if the jaws closed on nothing — re-observe and "
               "retry with a corrected (u,v).",
       {"u": "0..1 left->right on the sheet", "v": "0..1 top->bottom on the sheet"})
async def pick(adapter: RobotAdapter, op: Operation, u: float, v: float) -> dict:
    await _approach(adapter, op, u, v)
    op.report("closing jaws until they clamp")
    gripped, jaw = await motion.close_jaws(adapter, op)
    if not gripped:
        op.report(f"jaws closed on nothing (j8={jaw:.2f}) — grasp missed")
        await motion.open_jaws(adapter, op)
        await motion.creep(adapter, op, HOME)
        return {"success": False, "verified": False,
                "detail": f"missed grasp: jaws fully closed at {jaw:.2f}; "
                          f"re-observe and retry with corrected (u,v)",
                "final_state": await _final_state(adapter)}
    op.report(f"gripped (j8 stalled at {jaw:.2f})")
    await _lift_and_carry_home(adapter, op)
    return {"success": True, "verified": True,
            "detail": f"object gripped (j8={jaw:.2f}) and carried to home",
            "final_state": await _final_state(adapter)}


@skill("place", "Carry a held object to the drop box along the taught drop "
                "path, open the jaws to release, back off the box and return "
                "home. Verified = jaws reached open position (the drop itself "
                "should be confirmed visually with observe).")
async def place(adapter: RobotAdapter, op: Operation) -> dict:
    if not DROP_PATH.exists():
        raise RuntimeError(f"no drop path at {DROP_PATH} — teach it first")
    wps = json.loads(DROP_PATH.read_text())
    poses = {w["label"]: w["pose"] for w in wps}
    order = [w["label"] for w in wps]
    open_th = 0.6
    g = adapter.gripper_joint
    fwd = [l for l in order if poses[l][g] < open_th]
    drop_label = next((l for l in order if poses[l][g] >= open_th), None)

    cur = (await adapter.joint_state()).joint_map()
    grip = cur[g]
    op.report("carrying to the drop box")
    for label in fwd:
        op.check_cancel()
        op.report(f"-> {label}")
        p = dict(poses[label])
        p[g] = grip                              # keep holding
        await motion.creep(adapter, op, p)
    if drop_label:
        op.report(f"-> {drop_label}: settle over the box, then open jaws")
        p = dict(poses[drop_label])
        p[g] = grip
        await motion.creep(adapter, op, p)
        op2 = dict(poses[drop_label])
        op2[g] = 0.95
        await motion.creep(adapter, op, op2)
    # Back out of the box: tuck elbow (pulls arm off the box + relieves the
    # shoulder), then raise the shoulder, then home — same recipe as the lift.
    if fwd:
        ob = dict(poses[fwd[-1]])
        ob[g] = 0.95
        await motion.creep(adapter, op, ob)
    cur = (await adapter.joint_state()).joint_map()
    tuck = dict(cur)
    tuck["j3"] = max(0.08, cur["j3"] - 0.28)
    tuck["j4"] = min(0.95, cur["j4"] + 0.18)
    tuck[g] = 0.95
    op.report("backing off the box")
    await motion.creep(adapter, op, tuck)
    cur = (await adapter.joint_state()).joint_map()
    clear = dict(cur)
    clear["j2"] = min(cur["j2"], 0.20)
    clear[g] = 0.95
    await motion.creep(adapter, op, clear)
    hg = dict(HOME)
    hg[g] = 0.95
    await motion.creep(adapter, op, hg)
    released = (await adapter.joint_state()).joint_map()[g] > open_th
    return {"success": released, "verified": released,
            "detail": "jaws opened over the box and arm returned home; "
                      "confirm the drop visually with observe()",
            "final_state": await _final_state(adapter)}


@skill("pick_and_place", "pick(u,v) then place() — the full tabletop task. "
                         "Stops before placing if the grasp is not verified.",
       {"u": "0..1 left->right", "v": "0..1 top->bottom"})
async def pick_and_place(adapter: RobotAdapter, op: Operation, u: float, v: float) -> dict:
    res = await pick(adapter, op, u, v)
    if not res["success"]:
        res["detail"] += " — skipped place"
        return res
    place_res = await place(adapter, op)
    return {"success": place_res["success"], "verified": place_res["verified"],
            "detail": f"pick: {res['detail']} | place: {place_res['detail']}",
            "final_state": place_res["final_state"]}
