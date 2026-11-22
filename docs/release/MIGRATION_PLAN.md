# Generaloid migration plan

**Plan only; no file moves in this release.** The 22 November update takes precedence over the original restructure request. r6-sites owns site resolution and the related implementation/launchers; r6-office owns the new table assets under sites/office. This pass leaves those paths and all real-arm entry points intact.

The [Generaloid migration design](../generaloid/MIGRATION.md) remains the source mapping. The existing facade in `generaloid.modules` exposes implementations without relocating them.

| Current source | Eventual destination | Work required before moving |
| --- | --- | --- |
| External `norma-core/`, Station binary | Remain external; `generaloid/core/station.py` adapts I/O | Preserve Station as bus/camera owner; do not vendor the runtime |
| `arm_mcp/station.py`, `sim/station_io.py`, `perception/station_io.py` | `generaloid/core/station/` | Consolidate connection/subscription policy and preserve import shims |
| `sim/real_arm_mcp.py` | ElRobot connector backend | Remove import-time globals; retain exact tools and schemas |
| `sim/sim_arm_mcp.py`, `sim/build_scene.py` | `generaloid/twin/` | Factories instead of import-time threads; explicit robot builders; remove bare module aliases |
| `sim/robot_profile.py`, `sim/robots/` | Connector profiles | Reconcile r6 site/calibration selection; preserve ROBOT_PROFILE |
| `sim/splat/` | `generaloid/twin/splat/` | Keep shell shim and external DATA_ROOT outputs |
| `learning/`, `skills/`, `tasks/` | Generaloid learning/skills/task modules | Preserve JSONL/config formats and runtime approval; reconcile task registry extension |
| `executor/` | `generaloid/executor/` | Preserve one motion owner, proposal checks and console URLs |
| `recorder/`, `sim/recorder.py` | Generaloid recording service/twin | Resolve ambiguous bare recorder imports first; retain format/API |
| `perception/`, `camera_pose/`, `calibration/` | Generaloid geometry + site assets | Complete r6 site registry first; preserve frame conventions and provenance |
| `dashboard/`, `models/` | Generaloid dashboard / optional advisors | Keep URLs and optional dependency/API-key boundary |
| `robot_mcp/`, `arm_mcp/server.py` | Compatibility entry points | Inventory client configs before deprecation |
| `generaloid/comms/`, `sites/` | Existing namespace / site data | No relocation needed |

For each later move: snapshot tool schemas and imports; move one module; install old-path shims; check old/new entry points in offline/fake mode; run the full suite and MCP parity checks before proceeding. Use a branch and reviewable rename commits when git writes are available. Do not combine source relocation with calibration changes.

Protect `sim/run_real_server.sh`, `scripts_integrator/real_server_home.sh` and `real_operator/.devin/mcp_config.json`. Some integrator/operator files live outside this checkout; inspect them at deployment without starting the live server. Source-checkout packaging remains intentional until path-based assets and bare imports can support a tested wheel.
