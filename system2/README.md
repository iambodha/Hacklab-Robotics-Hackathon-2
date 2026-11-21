# system2: a planner (System 2) and a reflexive controller (System 1) for the ElRobot arm

Claude (or any MCP client) is **System 2**: it looks, names things, and submits **goals**, i.e.
measurable conditions on the world plus constraints. It has no motion tools. A local runner is
**System 1**: from fresh camera and joint feedback it repeatedly picks one short, bounded motion
from candidates generated around the arm's current pose, executes it, and observes the result,
until the goal holds or it can say precisely why not. Nothing is a named skill: "pick up the cube"
is whatever sequence of goals System 2 composes.

```text
 operator: "put the cube in the box"
        │
 System 2 (Claude via MCP: elrobot-system2)   look · scene · track · mark_point
        │ submit_goal / goal_status / cancel_goal      (no motion tools)
        ▼
 ┌─────────────────────────── s2.server (this package) ───────────────────────────┐
 │  World model ◄── perception (OpenCV, both cameras, calibration) ◄── fresh frame │
 │      │                                                                  ▲       │
 │  Runner loop: evaluate conditions → generate candidates (+ kinematic    │       │
 │  prediction each) → rank: geometric | Open-Jev | shadow → freshness &   │       │
 │  guards → execute ONE bounded motion → observe ─────────────────────────┘       │
 └──────────────────────────────────┬─────────────────────────────────────────────┘
                                    │ MCP client (stdio)
                      arm server: sim/sim_arm_mcp.py (MuJoCo twin, default) or sim/real_arm_mcp.py
                      (elrobot-real, S2_ARM=real): look, joint_state, gripper, tcp_pose, move_tcp, camera_info
```

| File | Role |
|---|---|
| `s2/server.py` | The MCP server System 2 uses (`look`, `scene`, `track`, `mark_point`, `forget`, `submit_goal`, `goal_status`, `cancel_goal`, `system_status`, `reset_scene`). |
| `s2/goals.py` | The goal language: conditions (`reach`, `gripper`, `within`, `distance`, `height`, `yaw_aligned`) and constraints (`min_height`, `region`, `avoid`, `keep_holding`, `top_down`, `max_step`), each a measurable predicate with a cost. |
| `s2/candidates.py` | System 1's action space: short TCP moves, jaw and wrist changes generated from the current geometry, each with its predicted effect; hard guards. |
| `s2/scorers.py` | Rankers behind one interface: `geometric` (lowest predicted error, the baseline), `jev` (Open-Jev-2B ranks the candidates), `shadow` (geometric acts, Jev logged). |
| `s2/runner.py` | The loop, and every way a goal ends: succeeded (met on consecutive fresh observations), timeout, step budget, stall, cannot observe, nothing predicted to help, constraint broken (e.g. dropped object), motor fault, cancelled. Writes `traces/<time>/decisions.jsonl`. |
| `s2/perception.py`, `s2/world.py` | Detections (dark or saturated blobs on the light table, arm excluded via calibration), tracking by name across both cameras, grasp/release inference, beliefs with age and evidence. |
| `s2/arm.py` | MCP client for the arm server. |
| `eval.py` | Closed-loop evaluation with a scripted System 2, scored only by the simulator's ground truth. |
| `export_jev_data.py` | Decision traces → Open-Jev training rows (to fine-tune System 1). |
| `PLANNER.md` | System prompt for Claude as System 2. |

## Install (Linux or Windows + WSL2 Ubuntu)

Requirements: `git`, [`uv`](https://docs.astral.sh/uv/getting-started/installation/), Python 3.12
(uv fetches it if missing), Claude Code (logged in) for System 2. For Open-Jev: an NVIDIA GPU
with about 5 GB free and ~12 GB of disk. In Ubuntu (WSL) from the repo root:

```bash
bash system2/setup/install.sh            # System 2 venv + simulator venv (+ ElRobot URDF if missing)
bash system2/setup/install_open_jev.sh   # optional: Open-Jev loader, torch, Open-Jev-2B + Qwen3.5-2B (pinned)
bash system2/doctor.sh                   # read-only check
```

Everything goes to `S2_HOME` (default `~/.local/share/elrobot-system2`). Versions are pinned in
`requirements.txt`, `setup/sim-requirements.txt` and the installer (Open-Jev commit, model and
base-weight revisions, torch 2.9.0 cu128). Set `NORMA_CORE_REPO` to reuse an existing norma-core
checkout for the URDF.

## Run

```bash
# Linux / inside WSL (Claude installed there):
bash system2/run_agent.sh -p "Put the black cube in the grey box."
S2_POLICY=jev bash system2/run_agent.sh             # Open-Jev as System 1 (started in the background)
```

```powershell
# Windows, with Claude Code logged in on Windows (the server, simulator and Open-Jev run in WSL):
powershell -ExecutionPolicy Bypass -File system2\run_agent_windows.ps1 -p "Put the black cube in the grey box."
powershell -ExecutionPolicy Bypass -File system2\run_agent_windows.ps1 -Policy shadow
```

Claude gets only the `elrobot-system2` tools: no shell, files, or other MCP servers, and an empty
working directory. The MuJoCo window opens (`-NoGui` / `ROBOT_SIM_GUI=0` to skip), the simulator's
live dashboard is at http://localhost:8891, and each session is recorded under `sim/sessions/`
(replay.html on exit). System 1's decisions are in `system2/traces/<time>/decisions.jsonl`.

To use the server from another MCP client, run `bash system2/run_server.sh` over stdio (see
`mcp-config.example.json`).

### System 2 on another machine's model (Gemma over a Cloudflare tunnel)

The team's OpenAI-compatible operator runner (`sim/run_agent_openai.py`) drives System 2 too. Your friend
serves a vision model (e.g. `ollama pull gemma3:12b`, then `cloudflared tunnel`; the exact commands are in
`sim/model_endpoint.env.example`); this machine runs the arm, System 1 and the runner:

```bash
cp sim/model_endpoint.env.example sim/model_endpoint.env   # MODEL_BASE_URL=https://<name>.trycloudflare.com/v1, MODEL_NAME=...
bash system2/run_agent_model.sh --check
bash system2/run_agent_model.sh -p "Put the black cube in the grey box."
```

Goal-level tools suit small models: a whole pick is a few calls (track, submit_goal, goal_status), and
the closed-loop work never waits on the tunnel. Only the model runs remotely; nothing moves on its timing.

### The real follower arm

`S2_ARM=real` makes `run_server.sh` spawn `sim/run_real_server.sh` (elrobot-real) instead of the twin, with
the hardware grip thresholds (`S2_GRIP_MIN_OPEN=0.045`, one-call gripper). Set it up exactly as in
`sim/REAL_ARM.md` (Station, cam_stream, `ROBOT_WRIST_CAM` / `ROBOT_OVERVIEW_CAM`, health check) first. System 2
reads the same camera hashes. The overview camera is uncalibrated on hardware, so perception uses the wrist
camera only: move above the area with a `reach` goal to an `xyz` point (e.g. [0, 0.2, 0.24], like survey()),
`look()`, then `track` wrist detections (containers too). Not yet run on hardware: start with `look`/`scene`
only, then single goals with someone at the arm.

## Choosing System 1

| `S2_POLICY` | What decides | Use it for |
|---|---|---|
| `geometric` (default) | lowest predicted condition error | the working baseline |
| `shadow` | geometric; Open-Jev's choice and probabilities logged next to it | collecting comparisons safely |
| `jev` | Open-Jev-2B ranks the feasible candidates; guards still apply. Falls back to geometric if the service is down (`S2_JEV_FALLBACK=fail` to stop instead) | experiments |

Measured on this hackathon laptop (RTX 3060 Laptop, 6 GB): Open-Jev-2B uses 4.4 GB and takes
2.2–2.5 s per decision for 11–15 candidates. **Zero-shot it is not a good System 1 yet:** it
spreads probability almost evenly (top choice 0.10–0.21) and ranked the geometric best move 3rd
to 10th in static probes. Plain-language effects ("this gets 1.2 cm closer to the goal") helped
markedly over numbers, which is how candidates are described now. The path to a useful learned
System 1 is fine-tuning on this domain:

```bash
python system2/eval.py --episodes 50                                     # geometric runs -> traces
python system2/export_jev_data.py system2/traces/* --out data/elrobot-s1 # successful decisions -> rows
cd ~/.local/share/elrobot-system2/Open-Jev && ../open-jev-venv/bin/python -m jev.train \
  --model Qwen/Qwen3.5-2B --revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
  --data /path/to/data/elrobot-s1 --output runs/elrobot-s1 --steps 500
```

That first dataset is imitation of the geometric policy; outcome-weighted labels (what actually
reduced the error) are the next step. Training has not been run here; a 6 GB GPU is likely too small.

## Evaluate

```bash
python system2/eval.py --episodes 10 --task pick_place   # randomized cube; scripted System 2
python system2/eval.py --episodes 8 --task grasp --policy shadow
cd system2 && python -m pytest -q tests                  # unit tests, no simulator needed
```

Run these with the System 2 venv's python and `ROBOT_SIM_PYTHON` pointing at the simulator venv
(`run_server.sh` sets both). `report.json` lands next to the traces.

## Limits (measured, not hypothetical)

- **Perception** assumes a light, plain table: objects are dark or saturated blobs. Cube positions
  were within 1 mm and yaw within 2° (mod 90°) of ground truth in the twin; real cameras need the
  calibration in `camera_info` to be right.
- **Containers** are measured where their near wall stands on the table; their front-to-back depth
  is not observable from the overview camera and defaults to their width (`track(..., depth_m=)`).
- **Releasing while tilted:** jaws-down is unreachable over the box. The twin's `move_tcp` now uses the
  real arm's least-lean IK (it leans only as much as it must), which made the cube drop out reliably;
  orientation-free IK left it resting on the lower jaw. A release with the jaw axis steeper than 30°
  is still reported as unverified rather than done.
- **Hardware:** supported through elrobot-real (`S2_ARM=real`) but not yet tested on the arm. Its wrist
  camera intrinsics are assumed (fovy 72°). White boxes on the white table are not detected (perception
  looks for dark or coloured blobs).
