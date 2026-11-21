"""motion — overload-safe movement primitives built on a RobotAdapter.

Ported from last year's proven `smooth_pick.py` recipes:

  * creep()    — step toward a target in small increments with pauses. The j2
                 shoulder OVERLOADS if commanded a big jump under gravity; small
                 steps + auto-relief (retract j3 on stall) avoid servo faults.
  * close_jaws — close the gripper until fully closed OR it stalls on an
                 object (= a real grip). Stall position => grasp verification.
"""
from __future__ import annotations

import asyncio

from . import config
from .core import Operation, RobotAdapter

GRIPPER_OPEN = 0.78
GRIPPER_CLOSED = 0.18        # fully-closed target; jaws stall earlier on an object
GRIP_MARGIN = 0.05           # j8 settling above CLOSED+margin => something is held


async def creep(adapter: RobotAdapter, op: Operation | None, target: dict[str, float],
                step: float | None = None, dt: float | None = None,
                tol: float = 0.02, max_iter: int = 400, relieve: bool = True) -> bool:
    """Creep every joint toward `target` in `step` increments. If progress
    stalls (error stops decreasing => shoulder overload), retract j3 to
    relieve, then resume. Returns True when within tol."""
    step = config.CREEP_STEP if step is None else step
    dt = config.CREEP_DT if dt is None else dt
    prev, stall = None, 0
    for it in range(max_iter):
        if op:
            op.check_cancel()
        obs = await adapter.joint_state()
        cur = obs.joint_map()
        err = max(abs(cur[n] - target.get(n, cur[n])) for n in adapter.joint_names)
        if err <= tol:
            return True
        if prev is not None and err >= prev - 0.004:
            stall += 1
        else:
            stall = 0
        prev = err
        if relieve and stall >= 4:
            if op:
                op.report("stall/overload — retracting j3 to relieve")
            for _ in range(6):
                if op:
                    op.check_cancel()
                obs = await adapter.joint_state()
                cur = obs.joint_map()
                goals = {"j3": max(0.05, cur["j3"] - step)} if "j3" in cur else {}
                await adapter.move_norm(goals, max_step=step)
                await asyncio.sleep(dt)
            stall, prev = 0, None
            continue
        goals = {n: cur[n] + max(-step, min(step, target.get(n, cur[n]) - cur[n]))
                 for n in adapter.joint_names}
        # send only joints that still need to move — adapter holds the rest
        goals = {n: g for n, g in goals.items() if abs(g - cur[n]) > 1e-4}
        if goals:
            await adapter.move_norm(goals, max_step=step)
        await asyncio.sleep(dt)
        if op and it % 20 == 0:
            op.report(f"creeping: max joint error {err:.3f}")
    return False


async def close_jaws(adapter: RobotAdapter, op: Operation | None,
                     target: float = GRIPPER_CLOSED, step: float | None = None,
                     dt: float | None = None, max_iter: int = 60) -> tuple[bool, float]:
    """Close the gripper joint until fully closed or it stalls on an object.
    Returns (gripped, final_jaw_norm)."""
    step = config.CREEP_STEP if step is None else step
    dt = config.CREEP_DT if dt is None else dt
    g = adapter.gripper_joint
    prev, stall = None, 0
    for _ in range(max_iter):
        if op:
            op.check_cancel()
        cur = (await adapter.joint_state()).joint_map()[g]
        if cur <= target:
            return False, cur                       # fully closed = jaws hit nothing
        if prev is not None and abs(cur - prev) < 0.004:
            stall += 1
            if stall >= 2:
                return cur > target + GRIP_MARGIN, cur   # stalled on something
        else:
            stall = 0
        prev = cur
        await adapter.move_norm({g: max(target, cur - step)}, max_step=step)
        await asyncio.sleep(dt)
    return False, prev if prev is not None else target


async def open_jaws(adapter: RobotAdapter, op: Operation | None,
                    value: float = GRIPPER_OPEN) -> None:
    await adapter.move_norm({adapter.gripper_joint: value}, max_step=config.MAX_STEP)
