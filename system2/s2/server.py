"""elrobot-system2: the MCP server System 2 (Claude, ChatGPT, ...) uses. It has no motion tools.
System 2 looks, names things, and submits goals; the local runner (System 1) moves the arm.

  python -m s2.server            (stdio; spawns the arm server given by S2_ARM_CMD)
"""
from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP, Image

from .arm import ArmClient
from .config import Config
from .goals import GoalError, parse_goal
from .render import annotate, to_camera_pixels
from .runner import Runner, Trace
from .scorers import JevScorer
from .world import World


class App:
    runner: Runner
    world: World
    cfg: Config
    arm: ArmClient
    jev: JevScorer | None


app = App()


def log(*a):
    print('elrobot-system2:', *a, file=sys.stderr, flush=True)


@asynccontextmanager
async def lifespan(_server):
    cfg = Config()
    cfg.validate()
    trace = Trace(cfg.trace_dir)
    jev = JevScorer(cfg.jev_url, cfg.jev_timeout) if cfg.policy in ('jev', 'shadow') else None
    if jev:
        try:
            log('Open-Jev ready:', await jev.health())
        except Exception as e:
            log(f'WARNING Open-Jev not reachable at {cfg.jev_url} ({e}); '
                f'{"falling back to geometric" if cfg.jev_fallback == "geometric" else "goals will fail"}')
    async with ArmClient(cfg.arm_command[0], cfg.arm_command[1:], cfg.arm_env(), cfg.camera_roles) as arm:
        app.cfg, app.arm, app.jev, app.world = cfg, arm, jev, World()
        app.runner = Runner(arm, app.world, cfg, jev, trace)
        log(f'arm connected ({len(arm.tools)} tools), policy={cfg.policy}, trace={trace.path}')
        try:
            yield
        finally:
            app.runner.cancel('server shutting down')
            await app.runner.wait(5)
            if jev:
                await jev.aclose()


server = FastMCP('elrobot-system2', lifespan=lifespan)


def _dump(obj) -> str:
    return json.dumps(obj, indent=1, default=str)


@server.tool(annotations={'readOnlyHint': True})
async def look() -> list:
    """FRESH camera images (overview, then wrist) with a red pixel grid, a blue box + id around every
    detection, green names on tracked entities (orange = belief, not currently seen) and a magenta
    cross at the gripper TCP. Followed by the world summary as JSON."""
    await app.runner.observe()
    out: list = []
    for role in ('overview', 'wrist'):
        data = annotate(app.world, role)
        if data:
            out += [f'{role} camera', Image(data=data, format='jpeg')]
    out.append(_dump(app.world.summary()))
    return out


@server.tool(annotations={'readOnlyHint': True})
async def scene() -> str:
    """Fresh world summary without images: gripper pose and jaws, tracked entities (position, footprint,
    whether currently seen, held), untracked detections with ids, and the current goal's status."""
    await app.runner.observe()
    return _dump({**app.world.summary(), 'goal': app.runner.status()})


@server.tool()
async def track(name: str, detection: str, height_m: float, kind: str = 'object', depth_m: float | None = None) -> str:
    """Name a detection from the latest look()/scene() so goals can refer to it. kind: 'object'
    (something to pick up or move) or 'container' (something to put things in or avoid).
    height_m: its height above the table (e.g. 0.02 for a 2 cm cube, 0.09 for the box); positions
    are estimated at mid-height, so this matters. Containers are measured from where their near wall
    meets the table; depth_m (front to back, seen from the overview camera) defaults to their width.
    Re-tracking a name replaces it."""
    if not name.isidentifier() or name == 'gripper':
        raise ValueError('name must be an identifier other than "gripper"')
    if not 0.002 <= height_m <= 0.3:
        raise ValueError('height_m must be between 0.002 and 0.3')
    if depth_m is not None and not 0.02 <= depth_m <= 0.5:
        raise ValueError('depth_m must be between 0.02 and 0.5')
    e = app.world.track(name, detection, height_m, kind, depth_m)
    return _dump({name: e.summary(app.world.t)})


@server.tool()
async def mark_point(name: str, x: float | None = None, y: float | None = None, z: float | None = None,
                     camera: str | None = None, u: float | None = None, v: float | None = None) -> str:
    """Name a fixed point for goals: either base-frame metres (x, y, z), or a pixel (u, v) of the
    latest look() image of `camera` ('overview' or 'wrist') projected onto the plane at height z.
    Base frame: origin = arm base centre on the table, +z up, +y away from the arm towards the
    overview camera, +x = the arm's right (image-LEFT in the overview camera)."""
    if not name.isidentifier() or name == 'gripper':
        raise ValueError('name must be an identifier other than "gripper"')
    if camera is not None:
        if u is None or v is None or z is None:
            raise ValueError('camera points need u, v and z')
        p = app.world.pixel_to_table(camera, *to_camera_pixels(u, v), z)
    elif None in (x, y, z):
        raise ValueError('give x, y, z or camera, u, v, z')
    else:
        p = [x, y, z]
    e = app.world.mark_point(name, p)
    return _dump({name: e.summary(app.world.t)})


@server.tool()
async def forget(name: str) -> str:
    """Stop tracking an entity."""
    if app.runner.busy:
        raise RuntimeError('a goal is running; cancel it before forgetting entities')
    app.world.entities.pop(name, None)
    return f'tracked: {sorted(app.world.entities)}'


@server.tool(annotations={'destructiveHint': True})
async def submit_goal(description: str, conditions: list[dict], constraints: list[dict] | None = None,
                      timeout_s: float = 90, max_steps: int = 40, replace: bool = False) -> str:
    """Hand System 1 a goal: conditions that must all become true, constraints on the way there.
    It runs in the background; follow it with goal_status(wait_s=...). The arm moves in short,
    bounded steps chosen from fresh observations. Fails fast with a reason if it stalls, loses
    sight of something, breaks a constraint, or no motion can help.

    conditions (all must hold):
      {"type":"reach","a":"gripper","b":<entity>,"offset":[dx,dy,dz],"tol":0.01,"tol_z":0.01}  (or "xyz":[x,y,z] instead of b)
      {"type":"gripper","state":"open"|"closed"|"holding","object":<entity for holding>}
      {"type":"within","a":<held object>,"b":<container>,"margin":0.01}
      {"type":"distance","a":<entity>,"b":<entity>,"axes":"xy"|"xyz","min":m,"max":m}
      {"type":"height","a":<entity>,"min":m,"max":m}
      {"type":"yaw_aligned","a":"gripper","b":<object>,"tol_deg":8}
    constraints:
      {"type":"min_height","z":m} {"type":"region","x":[lo,hi],"y":[lo,hi],"z":[lo,hi]}
      {"type":"avoid","entity":<name>,"clearance":m} {"type":"keep_holding","object":<name>}
      {"type":"top_down","value":true|false} {"type":"max_step","m":m}
    "holding" approaches the object from above, lines up, and closes; "within" with a held object
    carries it over the container and lets go. replace=True cancels a running goal first."""
    if app.runner.busy:
        if not replace:
            raise RuntimeError(f'goal {app.runner.goal.id} is running; pass replace=True or cancel_goal() first')
        app.runner.cancel('replaced by a new goal')
        await app.runner.wait(30)
    if app.world.frame is None:
        await app.runner.observe()
    try:
        goal = parse_goal({'description': description, 'conditions': conditions, 'constraints': constraints or [],
                           'timeout_s': timeout_s, 'max_steps': max_steps}, app.world)
    except GoalError as e:
        raise ValueError(f'invalid goal: {e}')
    app.runner.submit(goal)
    return _dump({'goal': goal.id, 'status': 'running', 'policy': goal.policy})


@server.tool(annotations={'readOnlyHint': True})
async def goal_status(wait_s: float = 0) -> str:
    """Status of the current (or last) goal: running | succeeded | failed | cancelled, the reason,
    each condition's latest measurement, and the last few actions with their observed effect.
    wait_s (<= 120) blocks until the goal ends or the time passes, so you do not have to poll."""
    if wait_s > 0 and app.runner.busy:
        await app.runner.wait(min(float(wait_s), 120.0))
    return _dump(app.runner.status())


@server.tool(annotations={'destructiveHint': True})
async def cancel_goal() -> str:
    """Stop the running goal after the current short motion; the arm then holds still."""
    if not app.runner.busy:
        return _dump(app.runner.status())
    app.runner.cancel()
    await app.runner.wait(30)
    return _dump(app.runner.status())


@server.tool(annotations={'readOnlyHint': True})
async def system_status() -> str:
    """Which System 1 policy is active, Open-Jev health and latency, runner statistics, arm tools."""
    out = {'policy': app.cfg.policy, 'jev_url': app.cfg.jev_url if app.jev else None,
           'runner': app.runner.stats, 'arm_tools': sorted(app.arm.tools), 'trace': str(app.runner.trace.path)}
    if app.jev:
        try:
            out['jev'] = {'health': await app.jev.health(), **app.jev.stats()}
        except Exception as e:
            out['jev'] = {'error': f'{type(e).__name__}: {e}'}
    return _dump(out)


@server.tool(annotations={'destructiveHint': True})
async def reset_scene(randomize: bool = True) -> str:
    """SIMULATION ONLY: the operator puts the arm at rest and the objects back on the table
    (randomly placed when randomize). Cancels any goal and forgets every tracked entity."""
    app.runner.cancel('scene reset')
    await app.runner.wait(30)
    await app.arm.reset_scene(randomize)
    app.world.entities.clear()
    app.world.gripper.jaw_cmd = None
    app.runner.yaw = None
    await app.runner.observe()
    return 'scene reset; call look() and track() again.\n' + _dump(app.world.summary())


def main():
    server.run()


if __name__ == '__main__':
    main()
