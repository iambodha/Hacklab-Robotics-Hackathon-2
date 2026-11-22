# arm_mcp: joint-level MCP server for the ElRobot arm

A small MCP server for the real arm. It focuses on reading true state and moving
without jumps. It complements `robot_mcp/` (task-level skills).

| Tool | What it does |
|---|---|
| `look` | Fresh JPEGs from every live camera plus the live joint table. A frame is rejected if the relay says it's older than `ROBOT_MAX_CAM_AGE` (1 s), or if its bytes match the previous capture; a camera that stays stale is reported as `STALE`. |
| `joint_state` | Joint table read from the motor bus (`st3215/inference`): raw ticks, calibrated range, in-range flag, torque, status, load, temperature. |
| `set_torque(enabled)` | When enabling, writes goal = present position first, so nothing jumps. |
| `move_to(targets, step, dt)` | Creeps to absolute normalized targets. Joints you don't list hold their present raw position. Refuses if torque is off, if a commanded joint is outside its calibrated range (clamping it would snap the joint), or if the move is larger than `ROBOT_MAX_DELTA` (0.35). Aborts and holds on a tracking error (`ROBOT_TRACK_TOL` = 250 ticks) or a motor fault status. |
| `gripper(open_fraction)` | Drives the jaws (`j8`). Stalling on an object counts as a grip, so there is no tracking abort. |

Scene perception tools (read-only; read `calibration/scene_calib.json`, see `perception/README.md`):

| Tool | What it does |
|---|---|
| `calibration_status` | Markers, intrinsics source, overview camera pose, how the arm base was found, hold-one-out mm, mm per pixel. |
| `locate(camera, u, v, bbox, color, height_m, label)` | Pixel, bbox or colour in a camera image → world table coordinates (centre, size, yaw) with sigma in mm. |
| `triangulate(overview_u, overview_v, wrist_u, wrist_v)` | 3D point from the same point seen in both cameras, with ray gap and sigma. |
| `project_point(x, y, z)` | World point → pixel in each camera, plus an in-view flag. |
| `scene_map` | Table, markers, arm base, both cameras' positions and footprints, TCP, objects located this session. |
| `annotated_view(camera)` | Live frame with a 5 cm grid, world axes, markers, wrist footprint, TCP and located objects; the border ruler is in 299x224 px. |

They need `opencv-contrib-python` and pin `mcp<2` (mcp 2.x renamed FastMCP):
`uv run --no-project --with "mcp<2" --with pillow --with numpy --with opencv-contrib-python python arm_mcp/server.py`.

`python arm_mcp/health.py` is a read-only check of bus reads, torque and calibration ranges.

## Running

The server needs the Station (normfs `:8888`), the `cam_stream` relay (`:8890`,
`scripts/cam_stream.py` in the workspace) and a `norma-core` checkout. It finds
`norma-core` through `NORMA_CORE_REPO`, or next to or inside this repo. In the
D: workspace it runs as `bash scripts/run.sh arm-mcp`. See
`mcp-config.example.json` for a Windows/WSL client config.

Environment: `STATION_HOST`, `ROBOT_BUS_SERIAL` (default `5B61033301`), `ROBOT_CAM_URL`,
`ROBOT_MAX_CAM_AGE`, `ROBOT_MAX_DELTA`, `ROBOT_TRACK_TOL`, `ROBOT_GRIPPER_JOINT`.

## Findings from bring-up (2026-11-21)

- **Frozen `observe` images:** the `inference/normvla` frame (camera plus joints) is
  published only when exactly one bus has torque on (`st3215-bus: auto`). With
  torque off, every reader gets the same old frame. This is why `look` uses the
  `cam_stream` relay and reads joints from the motor bus.
- **Bus desync:** after the Station had been running a while, every motor read
  failed with "Motor ID mismatch: expected N, got N+1". All positions read 0, and
  anything that normalizes them gives nonsense. Restarting the Station fixed it.
  `health.py` reports these read errors.
- **Calibration drift:** at bring-up, j1 (raw 543 vs 1066–3030), j2 (3180 vs 982–3114)
  and j6 (1297 vs 1688–2408) were outside the Station's stored ranges, and torque
  was off on every motor. The j6 and j8 ranges in the Station (1688–2408, 1898–2198)
  also differ from the ones in `LOG.md` (1006–3090, 1249–2847). Recalibrate (or
  move those joints into range) before commanding motion; `move_to` refuses until then.
