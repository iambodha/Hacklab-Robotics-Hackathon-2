# sim: MuJoCo twin of the hackathon table, driven like the real arm

> **Real arm:** the same operator runs on the physical follower arm. See [REAL_ARM.md](REAL_ARM.md).

A simulated ElRobot follower on the hackathon table (grey box + black cube), behind **the same
MCP tools as `arm_mcp/`** (`look`, `joint_state`, `set_torque`, `move_to`, `gripper`). An agent
practises here exactly as it would on hardware: two camera JPEGs and the motor-bus joint table.
It gets no object poses, no simulator state and no IK.

| File | What |
|---|---|
| `layout.json` | Table, bins, objects, cameras, rest pose. **Estimated from photos**: tape-measure and edit. |
| `build_scene.py` | URDF → MJCF (8 position servos, URDF limits, 2.94 N m) + scene → `generated/workspace.xml`, report and previews. |
| `sim_arm_mcp.py` | The MCP server (`elrobot-sim`). Physics runs in real time on a background thread. |
| `OPERATOR.md` | System prompt for Claude as the arm operator (task: black cube → grey box). |
| `run_server.sh` / `run_agent.sh` | Start the server; start Claude (in WSL) with only these tools. |
| `run_agent_windows.ps1` | Same, using the **Windows** Claude Code login; the server runs in WSL over `wsl.exe`. |
| `recorder.py` | Records every session and serves the **live dashboard** at http://localhost:8891. |
| `make_replay.py` | Turns a session into `side.mp4` + `replay.html` (video, joint traces, command log). |
| `judge.py` | Human-side ground truth (where the cube is, whether it is in the box). |
| `selftest.py` | Harness self-test: a privileged-IK pick-and-place through the tool functions. |

## Robot-knowledge tools (ROBOT_SIM_KINEMATICS=1, default)
Besides the `arm_mcp` joint tools, the server offers what HANDOFF.md task 4 plans for the real arm.
They draw only on the arm's own URDF and camera calibration, never on object state:
`tcp_pose()` (FK of the measured joints), `move_tcp(x, y, z, top_down, yaw_deg)` (IK, executed through the
same bounded `move_to` steps; it travels high and falls back to a tilted gripper where jaws-down is
unreachable), and `pixel_to_table(camera, u, v, height)` (pixel ray → table point). `look()` images are
upscaled 2× with a labelled pixel grid (`ROBOT_SIM_IMG_SCALE`, `ROBOT_SIM_GRID`). With
`ROBOT_SIM_KINEMATICS=0` only the hardware-identical joint tools remain.

Results (Claude Code, `OPERATOR.md`, 2026-11-21): **Sonnet** completed the pick-and-place twice in a row,
with the default cube placement in about 62 s and a randomized placement in about 95 s, including recovering
from a misjudged first grasp. **Haiku** was faster per step but misjudged the box position and dropped the
cube off the table. Use `--model sonnet` (the `Start-Sim-Agent.cmd` default).

## How it matches the real arm
- **Joints:** `n = (q - lower)/(upper - lower)` for j1..j7 (the direct mapping that
  `camera_pose/check_joint_mapping.py` supports). **j8 jaws: 1 = open.** Raw ticks are
  `2048 + q·4096/2π`, and the table format is identical to `arm_mcp`.
- **Motion:** `move_to` creeps `step` per `dt`, refuses requests above `ROBOT_MAX_DELTA` (0.35)
  and aborts on a tracking error above `ROBOT_TRACK_TOL` (250 ticks) or a motor fault. `gripper`
  has the same delta limit, so close in steps. A stall on an object is a grip.
- **Servos:** torque-limited, no gravity compensation. A servo held at its limit for 2 s trips
  (`status 0x20`, limp) and re-engages once it is unloaded, like the j2 overloads on hardware.
- **Cameras** (299×224 JPEG, as Station sends them): the wrist cam sits at the CAD pose from
  `camera_pose/wrist_camera.json` (jaws at the bottom of the frame), and the overview cam is at
  the front of the table. The labels reuse the real queue hashes `322ab551` / `8c697f6e`.
  The FOVs are guesses until the intrinsics are calibrated.
- **Gripping:** box pads on both jaws (the CAD jaw meshes are visual only), ~35 N squeeze.
- Sim-only extra: `operator_reset_scene(randomize)`, which does what a human resetting the table would do.

## Run (WSL Ubuntu)
```bash
export NORMA_CORE_REPO=/path/to/full/norma-core   # needs hardware/elrobot (URDF + meshes)
python3 sim/build_scene.py                        # optional; the server builds on first start
ROBOT_SIM_GUI=1 bash sim/run_server.sh            # MCP server on stdio + MuJoCo viewer
bash sim/run_agent.sh                             # Claude as operator, robot tools only
python3 sim/judge.py                              # did it work? (reads the ground-truth log)
python3 sim/selftest.py --randomize               # sanity check of physics + tools
```
`run_agent.sh` starts Claude Code with `--tools ""` (no shell or file tools), `--strict-mcp-config`
(only `elrobot-sim`), an empty working directory, and `OPERATOR.md` appended to the system prompt.
The ground truth goes to `ROBOT_SIM_TRUTH_LOG` (default `~/.cache/elrobot-sim/truth.jsonl`),
which the agent cannot read.

### From Windows (Claude Code logged in on Windows, not in Ubuntu)
```powershell
powershell -ExecutionPolicy Bypass -File sim\run_agent_windows.ps1             # interactive
powershell -ExecutionPolicy Bypass -File sim\run_agent_windows.ps1 -p "Start."  # headless
```
Windows `claude` gets an MCP config whose command is `wsl.exe -d Ubuntu -- … run_server.sh`, so the
simulator, its MuJoCo window (WSLg) and the recorder run in Ubuntu. The live dashboard opens in your
Windows browser, and the replay opens when Claude exits. Options: `-SimDistro`, `-SimRecordDir`,
`-SimServerEnv "ROBOT_SIM_PYTHON=… NORMA_CORE_REPO=…"`, `-SimRunSh`, `-SimPort`, `-SimNoBrowser`;
everything else is passed to `claude`.

### Watching and replaying a session (human side only)
- **Live:** http://localhost:8891 shows the side-camera stream, joint bars (measured vs commanded, load,
  fault status), and every tool call with its arguments, its result and the images the agent was shown.
- **Recorded:** `sessions/<timestamp>/` (or `ROBOT_SIM_RECORD_DIR`) holds frames at 10 fps,
  `state.jsonl` and `events.jsonl`, the agent's `looks/`, and, after the run, **`side.mp4` + `replay.html`**.
  The replay keeps the video, the joint-trace chart and the command log in sync (click either to seek), shows
  "what the agent saw", plays at 1–16× and can skip idle thinking time. Rebuild it with `python3 sim/make_replay.py <session>`.
- The side camera, dashboard, replay and ground truth are not exposed through MCP; the agent only gets `look()`.

In the Robotics-Hackathon D: workspace: `bash scripts/run.sh sim-arm-mcp | sim-agent | sim-replay | sim-judge | sim-selftest | build-workspace`,
or double-click `Start-Sim-Agent.cmd` (Windows Claude, recordings in `outputs\sim-sessions`).

## SO-101 adaptation (`ROBOT_MODEL=so101`)
The same simulation, 5-DOF IK, camera ray projection, `pick_at` / `drop_at` skills, and real-arm MCP stack also run on the **6-motor SO-101 follower arm** (`j1..j5` + `j6` gripper):
- Scene & camera calibration: `sim/layout_so101.json`, `camera_pose/wrist_camera_so101.json` (calibrated to `Innomaker-U20CAM-1080p-S1`, 398×224).
- MCP servers & prompts: `sim/so101_sim_mcp.py` (`so101-sim`, `OPERATOR_SO101.md`) and `sim/so101_real_mcp.py` (`so101-real`, `OPERATOR_SO101_REAL.md`).
- Station transport: connects automatically to Station's NormFS WebSocket (`ws://127.0.0.1:8000/api` or `:8889/api`) or TCP (`:8888`), reading both `st3215/inference` and `usbvideo/<hash>` directly.
```bash
ROBOT_MODEL=so101 MUJOCO_GL=egl uv run --no-project --with 'mcp>=1.12,<2' --with mujoco --with numpy --with pillow python3 sim/selftest.py
ROBOT_MODEL=so101 python3 arm_mcp/health.py
bash sim/run_so101_agent.sh
bash sim/run_so101_agent_real.sh
```

## Known gaps
Object/bin sizes and positions, camera FOVs and the overview camera pose are estimates; link
masses come from the CAD (≈0.6 kg arm); the j1/j5/j7 signs are unverified on hardware
(`camera_pose/HANDOFF.md`). Calibrate these before trusting sim-to-real transfer of poses.
