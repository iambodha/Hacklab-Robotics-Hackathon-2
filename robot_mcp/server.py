"""robot_mcp MCP server — the tool surface an AI agent drives the robot with.

Design notes
------------
* The agent talks in TASKS (start_skill -> get_operation -> observe), not raw
  joint updates. Low-level move_joints exists for commissioning/recovery.
* Skills run as Operations under the robot's motion lock: exactly one motion
  source commands each arm at a time. stop_robot cancels + holds position.
* Safety lives here in the server process (adapters clamp [0,1], clamp step
  per call, refuse uncalibrated joints) — never in the model.
* "verified" in a result means a sensor check passed (jaws stalled on the
  object), NOT that a command was sent. If the view is needed, observe().

Run:  python -m robot_mcp        (stdio; usually launched via .mcp.json)
"""
from __future__ import annotations

import io
import logging

from mcp.server.fastmcp import FastMCP, Image

from . import config, skills
from .adapters import MockAdapter, StationAdapter
from .core import (Observation, get_operation as _get_op,
                   running_operation, start_operation)

logging.basicConfig(level=logging.INFO)
mcp = FastMCP("robot-mcp")

_ROBOTS: dict | None = None


def _robots() -> dict:
    global _ROBOTS
    if _ROBOTS is None:
        _ROBOTS = {"elrobot": StationAdapter()}
        if config.ENABLE_MOCK:
            _ROBOTS["mock"] = MockAdapter()
    return _ROBOTS


def _robot(robot_id: str):
    try:
        return _robots()[robot_id]
    except KeyError:
        raise ValueError(f"unknown robot {robot_id!r}; available: {list(_robots())}") from None


def _state_text(obs: Observation, robot_id: str) -> str:
    lines = [f"robot={robot_id}  units=normalized[0..1]  image_source={obs.image_source}",
             f"{'joint':>6} {'norm':>6} {'raw':>6} {'range':>13} cal  role"]
    for j in obs.joints:
        cal = "y" if j.calibrated else "n"
        lines.append(f"{j.name:>6} {j.pos_norm:6.3f} {str(j.raw):>6} "
                     f"[{j.range_min:>5},{j.range_max:>5}] {cal:>3}  {j.role}")
    return "\n".join(lines)


def _downscale(jpeg: bytes) -> bytes:
    try:
        from PIL import Image as PILImage
    except ImportError:
        return jpeg
    with PILImage.open(io.BytesIO(jpeg)) as im:
        im = im.convert("RGB")
        if im.width <= config.IMG_MAX_W:
            return jpeg
        buf = io.BytesIO()
        im.resize((config.IMG_MAX_W, round(im.height * config.IMG_MAX_W / im.width))) \
          .save(buf, format="JPEG", quality=72)
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Discovery

@mcp.tool(annotations={"readOnlyHint": True})
def list_robots() -> dict:
    """List the robots this server can drive, with their joint space, gripper
    joint, camera availability, units and supported skills."""
    out = {}
    for rid, r in _robots().items():
        caps = r.capabilities()
        caps["skills"] = list(skills.SKILLS)
        out[rid] = caps
    return out


@mcp.tool(annotations={"readOnlyHint": True})
def list_skills(robot_id: str = "elrobot") -> dict:
    """Skills the robot can run via start_skill: name, description, arguments.
    Prefer skills over raw move_joints — they are overload-safe and verify
    their outcome."""
    _robot(robot_id)
    return {name: skills.describe(name) for name in skills.SKILLS}


# ---------------------------------------------------------------------------
# Perception

@mcp.tool(annotations={"readOnlyHint": True})
async def observe(robot_id: str = "elrobot") -> list:
    """Fresh camera frame(s) + the current joint state. Call this at the start
    of every supervisory cycle and after any skill — never reason from a stale
    frame. Returns labeled images (if the robot has cameras) plus a table of
    joint positions (normalized 0..1, raw ticks, calibrated range, role)."""
    robot = _robot(robot_id)
    obs = await robot.observe(images=True)
    out: list = []
    for i, jpeg in enumerate(obs.images):
        out.append(f"Camera {i}:")
        out.append(Image(data=_downscale(jpeg), format="jpeg"))
    if not obs.images:
        out.append("(no camera frames available — joint state only)")
    out.append(_state_text(obs, robot_id))
    return out


@mcp.tool(annotations={"readOnlyHint": True})
async def read_joint_state(robot_id: str = "elrobot") -> str:
    """Joint positions only, read from the motor bus — cheap, and independent
    of the camera feed (which can freeze). Use for quick progress checks."""
    return _state_text(await _robot(robot_id).joint_state(), robot_id)


# ---------------------------------------------------------------------------
# Task execution (async operations)

@mcp.tool()
async def start_skill(robot_id: str, skill: str, arguments: dict | None = None) -> dict:
    """Start a skill as a background operation; returns an operation_id to poll
    with get_operation. One operation per robot at a time — a second start
    while one is running is rejected. Use observe() afterwards to confirm the
    physical outcome, even when verified=True."""
    robot = _robot(robot_id)
    if skill not in skills.SKILLS:
        raise ValueError(f"unknown skill {skill!r}; call list_skills()")
    running = running_operation(robot.robot_id)
    if running is not None or robot.motion_lock.locked():
        raise ValueError(
            f"robot {robot.robot_id} is busy"
            + (f" (operation {running.id} running)" if running else "")
            + " — poll get_operation or call stop_robot first")
    op = start_operation(robot, skill, skills.SKILLS[skill]["fn"], arguments or {})
    return {"operation_id": op.id, "status": op.status, "robot_id": robot.robot_id,
            "skill": skill}


@mcp.tool(annotations={"readOnlyHint": True})
def get_operation(operation_id: str) -> dict:
    """Status of a started skill: running | done | failed | cancelled, progress
    message, trace, and (on completion) a structured result with
    success/verified/detail/final_state."""
    op = _get_op(operation_id)
    if op is None:
        raise ValueError(f"unknown operation {operation_id!r}")
    return op.as_dict()


@mcp.tool()
async def cancel_operation(operation_id: str) -> dict:
    """Ask a running skill to stop at its next step boundary (checked between
    creep increments — typically <1s). The arm holds its current pose when the
    skill unwinds; it does not torque-off."""
    op = _get_op(operation_id)
    if op is None:
        raise ValueError(f"unknown operation {operation_id!r}")
    if op.status == "running":
        op.cancel.set()
        return {"operation_id": op.id, "status": "cancelling"}
    return {"operation_id": op.id, "status": op.status}


@mcp.tool(annotations={"destructiveHint": True})
async def stop_robot(robot_id: str = "elrobot") -> dict:
    """Immediate stop: cancels the running skill (if any) and commands every
    joint to hold its current position. Use for any unexpected motion."""
    robot = _robot(robot_id)
    running = running_operation(robot.robot_id)
    if running is not None:
        running.cancel.set()
    await robot.hold()
    return {"robot_id": robot.robot_id, "stopped": True,
            "cancelled_operation": running.id if running else None}


# ---------------------------------------------------------------------------
# Low-level motion (commissioning & manual recovery — prefer skills normally)

@mcp.tool(annotations={"destructiveHint": True})
async def move_joints(robot_id: str, targets: dict) -> str:
    """Move named joints to ABSOLUTE normalized targets in [0,1], e.g.
    {"j3": 0.4}. Joint roles: j1 base, j2 shoulder, j3 elbow, j4-j6 wrist,
    j7 gripper rotation, j8 gripper jaws. Each joint moves at most
    ROBOT_MAX_STEP (0.25) per call — larger requests are clamped, not
    rejected; unspecified joints hold position. For real tasks prefer
    start_skill, which creeps overload-safely and verifies outcomes."""
    robot = _robot(robot_id)
    if robot.motion_lock.locked():
        raise ValueError(f"robot {robot_id} is busy — stop_robot first or wait")
    async with robot.motion_lock:
        obs = await robot.move_norm({k: float(v) for k, v in targets.items()},
                                    max_step=config.MAX_STEP)
    return "moved.\n" + _state_text(obs, robot_id)


@mcp.tool(annotations={"destructiveHint": True})
async def set_gripper(robot_id: str, value: float) -> str:
    """Set the gripper jaws: 0.0 = closed, 1.0 = open (normalized; on this arm
    open ~0.78, a real grip stalls ~0.22-0.45). To pick something up reliably,
    prefer the pick skill — it creeps and detects the stall."""
    robot = _robot(robot_id)
    if robot.gripper_joint is None:
        raise ValueError(f"robot {robot_id} has no gripper joint")
    return await move_joints(robot_id, {robot.gripper_joint: float(value)})


# ---------------------------------------------------------------------------
# Teaching (re-tune taught poses for this year's setup)

@mcp.tool(annotations={"destructiveHint": True})
async def teach_grid_point(robot_id: str, label: str, u: float, v: float) -> str:
    """Record the CURRENT pose as the grasp pose for sheet position (u,v)
    under `label` (e.g. TL, MM, BR). Drive the arm to the grasp pose first
    (teleop or move_joints), then call this. Saved to a local override file —
    the pick skill's grid interpolator uses it immediately."""
    from . import grasp_model
    obs = await _robot(robot_id).joint_state()
    pose = {j.name: j.pos_norm for j in obs.joints}
    grasp_model.teach_point(label, float(u), float(v), pose)
    return (f"recorded {label.upper()} at (u={u}, v={v}) -> {grasp_model.LOCAL_GRID.name}. "
            f"Pose: {pose}")


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
