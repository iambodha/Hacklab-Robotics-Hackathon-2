# robot_mcp — task-level MCP server for the robot arms

An MCP server that lets an AI agent (Claude, Codex, …) drive the **NormaCore
ElRobot 8-DOF arm** — and later a **LeRobot SO-100/101** — at the level of
*tasks with verified outcomes*, not raw joint updates.

The agent's loop:

```
observe() -> start_skill("pick", {u, v}) -> get_operation() -> observe()
                              |
                     missed grasp? re-observe, correct (u,v), retry once
```

Motion is executed by a **local executor** (this server) — the model chooses
tasks and recovery strategy, it is never in the per-step control loop.

## Architecture

```
AI agent (any MCP client)
   |  MCP tools (stdio, or HTTP + token: --http [--lan])
robot_mcp.server
   |-- core.py        Observation types, RobotAdapter interface, Operations
   |-- skills.py      home / goto / pick / place / pick_and_place (verified)
   |-- motion.py      overload-safe creep, stall-detect gripper close
   |-- adapters.py    StationAdapter (real arm) | MockAdapter (software)
   |-- station_lib.py norma-core Station client (normfs :8888)
   |-- grasp_model.py taught-grid bilinear "IK"  (u,v) -> grasp pose
   +-- data/          seed grasp grid + drop path (re-teach on new setups)
```

Five responsibilities, kept separate on purpose:

1. **Adapters** translate a normalized joint space `[0,1]` to native units
   (servo ticks via each joint's calibrated range). A LeRobot adapter can
   implement the same interface — tasks stay portable at the task level.
2. **Task execution**: skills have preconditions, bounded motion, and return
   `operation_id` immediately so observation/cancellation stay responsive.
3. **Control ownership**: `robot.motion_lock` guarantees one motion source
   per arm — a skill, a direct move, or a stop. Teleop must be stopped first
   (`scripts/teleop.py stop`) before the agent drives.
4. **Outcome verification**: results distinguish *command sent* / *target
   reached* / *verified*. `pick` verifies via the jaw stall (closing to
   empty = missed grasp); `place` verifies jaw release and asks you to
   `observe()` to confirm the drop visually.
5. **Bounded recovery**: a missed grasp returns `verified: False`; the agent
   re-observes, corrects `(u,v)`, retries once. Stale sensing or a fault →
   the operation fails with the cause.

## Tool surface

| Tool | Purpose |
|---|---|
| `list_robots()` | Robots, joint space, gripper joint, units, safety envelope |
| `observe(robot_id)` | Fresh camera frames + joint table (roles, ranges, cal) |
| `read_joint_state(robot_id)` | Joints only, from the motor bus (never freezes on a camera drop) |
| `list_skills(robot_id)` | Skills + arguments |
| `start_skill(robot_id, skill, arguments)` | Start a task → `operation_id` |
| `get_operation(operation_id)` | Status, progress, trace, verified result |
| `cancel_operation(operation_id)` | Stop at the next step boundary (<~1s) |
| `stop_robot(robot_id)` | Cancel + hold current position (never torque-off) |
| `move_joints(robot_id, targets)` | Commissioning: absolute norm targets, step-clamped |
| `set_gripper(robot_id, value)` | Jaws open/close (1.0/0.0) |
| `teach_grid_point(robot_id, label, u, v)` | Record current pose as a grasp-grid point |

## Safety envelope (enforced in this process, not by the model)

- Targets clamped to `[0,1]` and to `ROBOT_MAX_STEP` (0.25) per call.
- Skills creep in `ROBOT_CREEP_STEP` (0.03) increments with pauses; a stall
  auto-retracts `j3` to relieve the shoulder (`j2` overloads if raised while
  the arm is extended — learned on last year's identical arm).
- Uncalibrated joints (`range_min == range_max`) refuse motion.
- The gripper jaws are `j8`; `j7` is gripper rotation. Don't swap them.

## Hardware facts (ElRobot follower)

- 8 DOF, joints `j1..j8`, motor ids 1..8. Follower bus `5B61033301`
  (12.4 V); leader `5B61035034` is never commanded.
- Move = `ST3215SyncWriteCommand` to register `0x2A`; present position = `0x38`.
- Joint state comes from `st3215/inference` — camera-independent.
- Images from `inference/normvla` (combined frame) or `usbvideo/<hash>`.

## Setup

```bash
pip install -r robot_mcp/requirements.txt
git clone --depth 1 https://github.com/norma-core/norma-core.git   # repo root, gitignored
# station running: scripts/run-station.sh   (normfs :8888)
python -m robot_mcp                 # stdio server (a local MCP client launches it)
python -m robot_mcp --http [--lan]  # HTTP + token, for any MCP client by URL (see below)
```

`mcp-config.example.json` shows the `.mcp.json` block for Claude Code.
Env vars: `STATION_HOST`, `NORMA_CORE_REPO`, `ROBOT_BUS_SERIAL`,
`ROBOT_CAM_QUEUES`, `ROBOT_MAX_STEP`, `ROBOT_CREEP_STEP`, `ROBOT_CREEP_DT`,
`ROBOT_IMG_MAX_W`, `ROBOT_MCP_ENABLE_MOCK`.

Develop without hardware: `ROBOT_MCP_ENABLE_MOCK=1` adds a `mock` robot that
simulates the same joint space (and an object between the jaws).

## Connecting any model (MCP over HTTP)

MCP is an open protocol, so nothing here is Claude-specific: any MCP client works, whatever
model sits behind it. Locally a client can launch the server over stdio (above). To share one
arm with several tools or laptops, run it over HTTP with a shared token:

```bash
python -m robot_mcp --http          # this machine only:  http://127.0.0.1:8765/mcp
python -m robot_mcp --http --lan    # other laptops too:  http://<laptop-ip>:8765/mcp
```

It prints the URL and the token. The token is reused across restarts (stored in
`robot_mcp/data/http_token.local.json`, gitignored) or set with `ROBOT_MCP_TOKEN`.
`curl http://<ip>:8765/health` checks reachability without a token.

| Client | Config |
|---|---|
| Claude Code (tested) | `claude mcp add --transport http robot http://<ip>:8765/mcp --header "Authorization: Bearer <token>"` |
| Python `mcp` client, any model in your own loop (tested) | `streamablehttp_client("http://<ip>:8765/mcp", headers={"Authorization": "Bearer <token>"})` |
| Codex CLI (`~/.codex/config.toml`, not tested here) | `[mcp_servers.robot]` with `url = "http://<ip>:8765/mcp"` and `bearer_token_env_var = "ROBOT_MCP_TOKEN"` |
| Cursor (`.cursor/mcp.json`, not tested here) | `{"mcpServers": {"robot": {"url": "http://<ip>:8765/mcp", "headers": {"Authorization": "Bearer <token>"}}}}` |
| Gemini CLI (`settings.json`, not tested here) | `{"mcpServers": {"robot": {"httpUrl": "http://<ip>:8765/mcp", "headers": {"Authorization": "Bearer <token>"}}}}` |

Check the untested ones against each tool's current docs; key names change between versions.

Security, hackathon grade:
- Every request needs `Authorization: Bearer <token>` (401 otherwise).
- Plain HTTP: someone sniffing the same Wi-Fi can read the token. Trusted networks only.
- Station's own ports (8888 API, 8889 web UI) have no authentication and stay on localhost;
  only this server's clamped, step-limited tool surface is exposed.
- All clients share one server process, so the per-arm motion lock still allows only one
  motion at a time. It does not stop two agents taking turns on the arm; agree who drives.
- Venue Wi-Fi often blocks laptop-to-laptop traffic; test `/health` from a teammate first.

## First run on hardware

1. `python3 scripts/teleop.py stop` — exclusive control.
2. `list_robots()` then `observe()` — check ranges are calibrated (`cal=y`).
3. `start_skill(elrobot, home)` — watch it creep to the home pose.
4. `teach_grid_point` the 9 labels (TL…BR) if the seed grid doesn't line up —
   move the arm via teleop to each grasp pose, then call the tool.
5. `start_skill(elrobot, pick_and_place, {"u":0.5,"v":0.5})` — the demo task.
   Deliberately offset `u`/`v` to watch a missed grasp get reported.

## Roadmap

- **LeRobot adapter** — `LeRobotAdapter` for SO-100/101 (same normalized
  task interface; its own calibration + camera stack).
- **Real IK** — the grid interpolator covers a fixed sheet; add analytic or
  learned IK (the MuJoCo model in the workspace already has joint ranges)
  for free-space targets.
- **OpenCV localization** — detect the object in the scene camera → `(u,v)`
  automatically, replacing hand-picked coordinates.
- **Learned-policy executor** — ACT/SmolVLA as another skill behind the same
  operation interface, with the same verified-outcome contract.
- **Drop-path teaching tool** — `place` uses a seed path; add a waypoint
  recorder like `teach_grid_point`.
