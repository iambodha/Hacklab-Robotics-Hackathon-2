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

## Baseline
- Full suite on the worktree before any Generaloid change: **2 failed, 285 passed, 1 skipped** (13 min 39 s).
  Failures are the known ones: `executor/tests/test_wiring.py::test_preflight_names_the_marker_under_a_table_hit`
  and `learning/tests/test_sim_loop.py::test_three_attempts_learn_to_open_the_cap` (cap_lift_mm 0.0 < 8).

## Build log
- Scripts: the five `scripts/*.py` Station clients read `STATION_HOST` (commit bf9b266).
- Core: `generaloid/core/station.py` (Station I/O layer + the one guarded connect helper), `core/server.py`
  (connector + mode -> FastMCP, tools adopted unchanged), `core/hooks.py` (comms hook). Connector interface
  `connectors/base.py`, profile-backed declarations `connectors/profile.py`, `elrobot` connector, CLI.
- `python -m generaloid tools --connector elrobot --mode sim` lists exactly the 18 tools of `sim/sim_arm_mcp.py`;
  `--mode real` lists the 27 tools of `sim/real_arm_mcp.py` (import only; STATION_HOST unresolvable, nothing connects).
- **D4. Station.bus() returns `arm_mcp.server.stn`**, not `arm_mcp.station`: arm_mcp/server.py imports its station
  module by the bare name `station`, so `arm_mcp.station` is a second module object and patching it would not
  reach the real server. `Station.install()` runs after `real_arm_mcp` is imported (that import sets
  `ROBOT_BUS_SERIAL` from the profile, and arm_mcp reads it at import).
- **D5. sim/ is appended to sys.path, not prepended**: `sim/recorder.py` would shadow the `recorder/` package.
  Mounting `sim_arm_mcp` / `real_arm_mcp` (which prepend sim/ themselves) happens once per process.
- Generic twin (`generaloid/twin/generic.py`, `twin/ik.py`) and the `so101` connector. The SO-101 scene uses
  norma-core's `so101_robot_follower.urdf` (meshes included) through `sim/build_scene.build_xml`, with the arm and
  gripper builders swapped in for one call (no edit to build_scene.py).
- **D6. Generic twin keeps the gripper goal during arm moves** (sim_arm_mcp resets unlisted joints' goals to their
  present position, so a stalled grip goes slack; origin/simulation_s101 had fixed the same thing). Gripper calls
  are exempt from the max-delta refusal (one call, like the real server's gripper tool).
- SO-101 twin: joint-1 axis lands at the table origin (shoulder body at [0, 0, 0.0648]); jaws-down IK solves
  (0, .19, .06), (.1, .2, .05), (-.08, .15, .03) with lean 0 and (.15, .17, .12) with lean 10 deg, 0.15-0.55 s each.
  In-process pick of the 2 cm cube: HOLDING, cube lifted to z = 0.115 m. Images: `img/so101_*.jpg`.

## MCP client checks (twin, `python -m generaloid.checks`, server as a stdio subprocess)
- `pick --connector elrobot`: cube 0.0099 -> 0.110 m, picked.
- `pick --connector so101`: cube 0.0099 -> 0.116 m, gripper reported HOLDING, picked.
- `twist --connector elrobot` with sim_loop's attempt-3 params (0.45 grip, 6 x 90 deg) from a fully closed cap:
  533 deg of gripper rotation, the cap turned 323 deg, not free (needs 1.25 turns = 450 deg). sim_loop only opens it
  on its third attempt, after two earlier attempts loosened it. Changed the check to 10 strokes, as the real demo
  takes needed (10-11 strokes).
