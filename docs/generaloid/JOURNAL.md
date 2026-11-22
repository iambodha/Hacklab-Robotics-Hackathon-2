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

## 2026-11-22 08:42 CET — Codex takeover from Devin
- Read the common brief first, all four branch commits (including WIP 16d6651), the journal and the end of
  `/Users/bodha/Documents/b0dha/Robotics-Hackathon/logs/r5-generaloid.log`. Devin stopped when its quota expired.
- Reviewed core, connector declarations, generic twin, CLI, fake Station and WIP tests. No uncommitted changes
  existed at takeover. Required architecture/CLI/connector/migration/pitch docs were missing.
- Independently inspected `origin/simulation_s101:sim/build_scene.py` and `origin/robot-profiles:sim/robots/so101.json`.
  The SO-101 twin and real profile use different URDFs and base frames; real hardware remains unverified.
- Started the common full-suite baseline with Station disabled; output is `takeover-baseline.txt` under the external
  r5-generaloid data directory. Started the WIP connector suite separately (only short twin mounts/IK, no long sim).
- Review fixes planned: expose the correct frame in real SO-101 server instructions; reject cached legacy servers
  from a different connector; clean up generic twin resources; make scripted checks return failure on bad outcomes.
- Never started real serving or connected to a real Station. Real-mode tests only use import and FakeStation.
  No commits will be made, per this run's explicit instruction; the integrator will commit the working tree.

## 2026-11-22 08:48 CET — Review and sandbox validation
- WIP connector suite: **27 passed** before fixes; **30 passed** after mode-frame, cached-real-server and Station
  policy regression tests. The policy test uses a private environment mapping; it never enables live tests in
  the process environment.
- Full common baseline in this sandbox: **278 passed, 6 failed, 1 skipped, 3 errors** (106.63 s). Eight failure/error
  cases depend on rendering and fail with `mujoco.cgl.cgl.CGLError: invalid CoreGraphics connection`; the remaining
  failure is the known executor marker-name assertion. This differs from Devin's less-restricted run above.
- Added a sim-only, test-only headless launcher which labels its images as rendering disabled. Normal serving is
  untouched. `GENERALOID_CHECK_HEADLESS=1` selects it for scripted checks and the result labels that substitution.
  This validates MCP control and physics, not camera appearance or visual perception.
- Scripted MCP pick: ElRobot cube **9.9 -> 110.0 mm**; SO-101 cube **9.9 -> 115.7 mm**, HOLDING. Both exit 0.
  Evidence: external data directory `takeover-pick-elrobot.json`, `takeover-pick-so101.json` and corresponding
  `takeover_pick_*/truth.jsonl`. Bottle twist is running as the only long simulation.
- `nice -n 10` reports `setpriority: Operation not permitted` but executes the command; heavy checks are serialized.
  `ps` is also denied by the sandbox; used tool session completion and output logs instead. No escalation requested.
- Finished architecture, CLI, connector guide, migration and pitch drafts. No legacy server/package changes.
- Further review found the SO-101 scene cache could reuse a prior layout when a different older file was selected.
  Cache now fingerprints source paths/mtime/size, including the URDF; added an older-layout regression test.
  Generic twin now closes its thread/renderer, bounds motion chunks by connector max delta and reports failure
  instead of success when bounded motion does not converge. Check CLI now exits nonzero on failed pick/release.
- Found a further inherited twin-clock bug: servo overload checks gated on every tenth step of each wall-clock
  batch never run in normal one/two-step batches. Fixed only the new generic twin to account for every batch's
  simulated duration, with a short deterministic regression test. Legacy sim is untouched (merge note for owner).

## 2026-11-22 08:51 CET — Bottle result and final verification
- Final focused suite: **32 passed in 27.84 s**, including the new cache and clock tests. Documentation link check
  and `git diff --check` pass.
- Scripted MCP bottle check exited 0 after ten strokes: gripper achieved **886.9 deg**, cap truth reports
  **652.7 deg turned**, **free=true**, lift **-0.4 mm**. Thus the simulated threads released, but the cap was not
  lifted clear. `check_cap` honestly reported `unclear: holding after lifting only 0 mm`; the episode's sensed
  `cap_free` remains false. The client's truth-based `opened=true` means released threads, not removed cap.
  Evidence is `takeover-twist-elrobot.json`, `takeover_twist_elrobot/truth.jsonl` and its isolated episode log.
- Repeating SO-101 MCP pick after the generic overload-clock fix, then running the combined common + Generaloid suite.

## 2026-11-22 08:55 CET — Handoff
- Post-fix SO-101 MCP pick passed again: cube **9.9 -> 115.6 mm**, HOLDING.
- Final combined common + Generaloid suite: **310 passed, 6 failed, 1 skipped, 3 errors in 145.74 s**.
  Compared the summary identifiers with the baseline: exactly the same six failures and three errors; no new
  failures. The 32 Generaloid tests also pass together with the common suite.
- Documentation complete: README, architecture diagram, CLI, connector guide, migration plan, one-page pitch and
  VALIDATION report. All local documentation links and `git diff --check` pass.
- Reviewed inherited commits bf9b266, 372621f, 3225cae and WIP 16d6651. **No new commits**, as instructed.
- Takeover edits are confined to `generaloid/` excluding comms and `docs/generaloid/`. Inherited external edits are
  only the five authorized script Station-host substitutions plus their `os` imports. No existing sim/learning/
  executor/robot_mcp/arm_mcp hooks were needed. All existing launch scripts/configs remain unchanged.
- Integrator: commit the working tree with the reviewed inherited work; merge r5-comms independently against
  `register(server, connector)`, and r5-skills normally (ElRobot adopts their existing server tools). Keep the legacy
  overload-clock issue, SO-101 frame/calibration gap, camera-render check outside this sandbox, and real cap lift
  verification as explicit follow-ups. Neither hardware qualification nor a real cap-removal test was performed.
