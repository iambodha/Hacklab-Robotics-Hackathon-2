# Generaloid documentation

Start with the [project overview](../README.md), [offline quickstart](release/QUICKSTART.md), [architecture](generaloid/ARCHITECTURE.md) and [known issues](KNOWN_ISSUES.md). The [night's story](STORY.md) is written for judges and the video agent; measurements link to their provenance.

| Area | Guide and evidence |
| --- | --- |
| [Generaloid](generaloid/README.md) | [Connector guide](generaloid/WRITING_A_CONNECTOR.md), [CLI](generaloid/CLI.md), [validation](generaloid/VALIDATION.md), [images](generaloid/img/), [journal](generaloid/JOURNAL.md) |
| [Learning](learning/README.md) | [Twelve attempts/data](learning/data/attempts.json), [curve](learning/learning-curve.svg), [profiles](learning/PROFILES.md), [free-cap evidence](learning/CAP_FREE.md), [audit](learning/AUDIT.md), [validation](learning/VALIDATION.md), [journal](learning/JOURNAL.md) |
| [Kitting](kitting/README.md) | [Measured results](kitting/RESULTS.md), [images](kitting/images/), [episode evidence](kitting/evidence/), [report](kitting/REPORT.md), [journal](kitting/JOURNAL.md), [printable A4](../tasks/kitting/print/kit_a4_v1.pdf). Simulation only. |
| [Communication](comms/README.md) | [Protocol](comms/PROTOCOL.md), [scenario](comms/SCENARIO.md), [security](comms/SECURITY.md), [tunnel notes](comms/CLOUDFLARE.md), [evidence](comms/evidence/README.md), [journal](comms/JOURNAL.md). Twin/localhost validation only. |
| [Twin](twin/README.md) | [Original simulator README](../sim/README.md), [real/twin comparison](../sim/real_compare/), [robot profiles](../sim/robots/README.md) |
| [Splat](splat/README.md) | [Pipeline + measured synthetic reconstruction](../sim/splat/README.md); splats are appearance, not collision geometry |
| [Calibration](calibration/README.md) | [Wrist handoff](../camera_pose/HANDOFF.md), [calibration artifacts](../calibration/), historical evidence from the home setup |
| [Perception](perception/README.md) | [Detailed geometry and API](../perception/README.md), calibrated rays, ArUco and uncertainty |
| [Recorder](recorder/README.md) | [Capture/replay format and API](../recorder/README.md), every camera frame plus motor state |
| [Dashboard](dashboard/README.md) | Learning inspection :8920; distinct from the executor console :8910 and recorder :8900 |
| [Executor](executor/README.md) | [Operator workflow](../RUN.md), proposals/validation/one motion owner, [blue-to-bin](../tasks/blue_to_bin/README.md) |
| [History](history/INTEGRATION.md) | Sanitized integration board, source [working log](../LOG.md), rounds and incident timeline |
| [Release](release/QUICKSTART.md) | [Validation](release/VALIDATION.md), [hygiene](release/HYGIENE.md), [local paths](release/LOCAL_PATHS.md), [migration plan](release/MIGRATION_PLAN.md), [moves](release/MOVES.md), [journal](release/JOURNAL.md) |

Optional model work lives in [models/POLICY_PLAN.md](../models/POLICY_PLAN.md), [benchmark](../models/bench/) and [System 2](../system2/README.md). Earlier MCP surfaces: [arm_mcp](../arm_mcp/README.md), [robot_mcp](../robot_mcp/README.md). Those historical READMEs describe their own point in development; the top-level README and release validation qualify current claims.

Site work (`home`, `office`) is **in progress: r6**; calibration must be selected for the actual table. Prior journals and evidence contain local paths as provenance, not portable launch instructions. Use the release setup guide for new checkouts.
