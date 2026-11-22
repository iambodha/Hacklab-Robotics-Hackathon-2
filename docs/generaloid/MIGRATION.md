# Release migration plan

This branch intentionally adds a facade and leaves existing packages in place. Other agents own learning/skills and comms work. Release should reconcile those branches before physically moving anything. The table is a proposed destination map, not a statement that files have moved.

| Existing source | Proposed eventual home | Transition requirement |
|---|---|---|
| `norma-core/`, `station-bin` | External NormaCore runtime; adapter in `generaloid/core/station.py` | Keep Station as I/O owner, never vendor or replace its bus/camera services |
| `arm_mcp/station.py`, `sim/station_io.py`, `perception/station_io.py` | `generaloid/core/station/` adapters | Consolidate connect, queue subscription/cache and host policy; preserve old import shims |
| `sim/real_arm_mcp.py` | `generaloid/connectors/elrobot/legacy_real.py`, then generic real backend | First retain all registered schemas and existing semantics; separate model-independent tools from robot-specific skills later |
| `sim/sim_arm_mcp.py` | `generaloid/twin/elrobot.py` | Retain ElRobot parity and session recorder; replace import-time construction with a factory before supporting multiple mounts |
| `sim/robot_profile.py`, `sim/robots/*` | `generaloid/connectors/profiles/` | Preserve profile resolution and `ROBOT_PROFILE`; avoid breaking operator prompt/UI |
| `sim/build_scene.py`, sim layout/scene helpers | `generaloid/twin/scene/` | Pass robot builders explicitly instead of SO-101's temporary callback substitution; keep calibrated assets/layout selection |
| `sim/splat/` | `generaloid/twin/splat/` | Keep shell entry point shim, ArUco metric alignment and external DATA_ROOT paths |
| `learning/`, `skills/` | `generaloid/learning/`, `generaloid/skills/` | Coordinate with r5-skills; preserve episode format, versioned config and public tools |
| `executor/` | `generaloid/executor/` | Keep one async motion owner and proposal/approval semantics |
| `recorder/`, `sim/recorder.py` | `generaloid/recording/service/`, `generaloid/recording/twin.py` | Remove ambiguous bare `recorder` imports before moving; preserve recording formats and :8900 |
| `dashboard/` | `generaloid/dashboard/` | Preserve :8920, approval/config URLs and integration with recorder |
| `perception/`, `camera_pose/`, `calibration/` | `generaloid/perception/`, `generaloid/calibration/` | Preserve coordinate transforms, camera assets and import shims |
| `models/`, `executor/advisor/` | `generaloid/addons/advisors/` | Optional dependencies and API keys; serving core tools must not require an advisor |
| `robot_mcp/`, `arm_mcp/server.py` | Compatibility shims or Generaloid adapters | Audit tool consumers before deprecation; do not duplicate active motion ownership |
| r5-comms `generaloid/comms/` | Already in final namespace | Preserve `register(server, connector)` signature and absent-module behaviour |

## Safe release order

1. Integrate additive Generaloid and parallel changes. Keep the original entry points untouched. Run schema parity, fake Station, connector contracts and scripted twin checks.
2. Convert script-style servers to explicit construction/close factories. Remove global `ROBOT_PROFILE` capture, bare-module aliases (`station`, `recorder`) and import-time simulation threads. Support a single robot per process until this is tested.
3. Unify SO-101 twin and real URDF/frame/mapping and validate hardware calibration separately. Metadata saying unverified must remain until physical qualification.
4. Move one module at a time with compatibility shims. Replace `sys.path` manipulation with package-relative imports. Update `generaloid.modules` paths and launcher working directories alongside each move.
5. Re-run the original entry points plus Generaloid equivalents before retiring any shim. Protect stdout MCP, environment overrides, profile assets, recording paths and stable schemas.

## Merge surface of this branch

New files are under `generaloid/` (excluding comms) and `docs/generaloid/`. The only modifications outside them are the five owned scripts: `scene_cam.py`, `teleop.py`, `arm_state.py`, `camera.py`, `calibrate.py`. Each adds `os` and replaces hard-coded Station localhost with `os.environ.get("STATION_HOST", "localhost")`. There are **no hooks or edits in existing sim, learning, executor, robot_mcp or arm_mcp files**. Comms is reached through dynamic import in the new core. Existing launch/config files are unchanged.

No commit is created in the takeover run. Integrator should review the pre-existing WIP commits as well as the working-tree diff, then commit together. Do not add `.generated/`, session outputs or external-disk test artifacts.
