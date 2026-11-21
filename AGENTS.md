# AGENTS.md — SO-101 & ElRobot Simulation + MCP Stack

## Workspace & Hardware Notes

- Full `norma-core` checkout with URDF/STL assets and generated Python protobufs (`target/gen_python`): `../norma-core` or `../normacorehack/norma-core` (auto-discovered by `arm_mcp/station.py` and `sim/build_scene.py`, or set `NORMA_CORE_REPO`).
- **SO-101 follower arm**:
  - 6 ST3215 motors (`j1..j5` 5-DOF arm + `j6` moving jaw; `0.0` = closed, `1.0` = open; `0.55` = ~6 cm grasp opening).
  - Connected bus serial on this machine: `5A46084639` (`/dev/ttyACM0`, auto-detected when `ROBOT_MODEL=so101`).
  - Wrist camera: `Innomaker-U20CAM-1080p-S1` (`398x224` JPEG from Station, upscaled 2x to `796x448` with a 50 px red grid in `look()`).
  - Station WebSocket transport (`ws://127.0.0.1:8000/api` or `:8889/api`) is supported directly by `arm_mcp/station.py` (`WsStationClient`) without requiring `-t` (`:8888`) or `cam_stream.py` (`:8890`), while remaining compatible with both.
  - ST3215 overload latch (`status == 0x20`): on SO-101 (`max_torque=500`, `I=8`), stretching far out can latch `0x20`, which causes firmware to ignore new `0x2A` goal writes until `Torque_Enable (0x28)` is cycled `0 -> goal=raw -> 1` on the latched motor. `sim/real_arm_mcp.py` handles this automatically in `_drive`.

## Verification & Run Commands

```bash
# 1. Build SO-101 MuJoCo twin & reachability report
MUJOCO_GL=egl uv run --no-project --with mujoco --with numpy --with pillow \
  python3 sim/build_scene.py --layout sim/layout_so101.json --out sim/generated_so101

# 2. Run SO-101 physics + MCP pick-and-place self-test
ROBOT_MODEL=so101 MUJOCO_GL=egl uv run --no-project --with 'mcp>=1.12,<2' --with mujoco --with numpy --with pillow \
  python3 sim/selftest.py

# 3. Read-only live hardware & camera health check (never moves the arm)
ROBOT_MODEL=so101 python3 arm_mcp/health.py

# 4. Launch MCP servers (stdio) or operator agents
bash sim/run_so101_server.sh          # so101-sim stdio MCP server
bash sim/run_so101_real_server.sh     # so101-real stdio MCP server
bash sim/run_so101_agent.sh           # operator policy on SO-101 sim
bash sim/run_so101_agent_real.sh      # operator policy on real SO-101 arm

# 5. Persistent Streamable HTTP bridge for step-by-step VLM/CLI policy execution
uv run --no-project --with 'mcp>=1.12,<2' --with pillow --with numpy --with mujoco --with uvicorn \
  python3 sim/mcp_bridge.py serve --target sim --port 8771
uv run --no-project --with 'mcp>=1.12,<2' \
  python3 sim/mcp_bridge.py call --port 8771 look --img-dir /tmp/views
```
