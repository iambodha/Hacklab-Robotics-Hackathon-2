# Generaloid journal (agent r5-generaloid)

Times are local (22 Nov 2026, overnight run). Newest entries last.

## Start
- Read `prompts/r5-common.md`, the brief, `sim/sim_arm_mcp.py`, `sim/real_arm_mcp.py`, `sim/robot_profile.py`,
  `sim/station_io.py`, `arm_mcp/station.py`, `sim/build_scene.py`, `sim/robots/*`.
- Read the SO-101 work on `origin/simulation_s101` (twin + MCP adaptation, on an older base: it edits
  `sim/build_scene.py` and `sim/sim_arm_mcp.py` heavily) and `origin/robot-profiles` (already merged into
  `integration`: `sim/robots/so101.json` + `sim/robot_profile.py`).
- Baseline test run started (full command from `r5-common.md`), output in
  `/Volumes/Dell HDD/Robotics-Hackathon/r5-generaloid/baseline.txt`.

## Findings that shape the design
- `sim/sim_arm_mcp.py` and `sim/real_arm_mcp.py` are scripts: importing them builds the robot object and a
  `FastMCP` with every tool registered at import time. Importing `real_arm_mcp` does **not** open a Station
  connection (arm_mcp connects lazily on the first tool call), so an import test of real mode is safe.
- The twin server is ElRobot-only (joint index 7 = jaws, `j1..j8`, `grasp_tcp` on `Gripper_Base_v1_1`).
  The real server is already profile-driven (`ROBOT_PROFILE=so101`).
- `origin/simulation_s101` makes the twin SO-101 capable by threading `robot_model == 'so101'` branches through
  `build_scene.py` and `sim_arm_mcp.py`. Other agents edit those files tonight, so that patch cannot be merged
  here.

## Decisions
- **D1. Wrap, do not fork, the ElRobot servers.** The `elrobot` connector imports `sim_arm_mcp` (sim) or
  `real_arm_mcp` (real) and *adopts* their registered tools into the Generaloid server (same function objects,
  same schemas, same docs). Tool parity with `sim/sim_arm_mcp.py` is then true by construction, and r5-skills'
  edits to those files flow through without a merge.
- **D2. A generic twin for every other arm.** SO-101 needs a twin, but the ElRobot twin is hard-wired. Instead of
  editing it, `generaloid/twin/generic.py` is a connector-driven twin (joint count, gripper index and direction,
  TCP site, cameras, safety envelope all come from the connector). The SO-101 connector supplies the scene
  builder ported from `origin/simulation_s101` (norma-core's own `so101_robot_follower.urdf` with meshes, base
  turned so +y = ahead, the branch's jaw pads, TCP and wrist-camera pose). A third arm writes a connector and gets
  a twin without touching the server.
- **D3. Station host for scripts.** The five scripts read `STATION_HOST` inline
  (`os.environ.get("STATION_HOST", "localhost")`) instead of importing Generaloid: they run standalone from
  `scripts/` where the repo root is not on `sys.path`.
