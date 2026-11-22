# Generaloid

Generaloid is a hardware-agnostic, model-agnostic robot MCP platform around **NormaCore Station**. A robot plugs in as a connector. Any MCP client can drive it; no model SDK, Gemini key or human approval service is required to serve tools.

This is an additive package: existing servers and launch scripts remain in place. ElRobot adopts the existing real/twin tools unchanged. SO-101 adds a connector-driven MuJoCo twin. Neither existing robot server was edited.

- [Architecture](ARCHITECTURE.md): Station I/O, connectors, modules and optional comms.
- [CLI and validation](CLI.md): launch, inspect, test and reproduce the MCP checks.
- [Write a connector](WRITING_A_CONNECTOR.md): SO-101 worked example.
- [Migration plan](MIGRATION.md): release-agent file mapping and compatibility requirements.
- [Pitch](PITCH.md): why this matters, with the bottle-cap evidence.
- [Validation](VALIDATION.md): test counts, MCP outcomes and sandbox limits.
- [Journal](JOURNAL.md): decisions, takeover, measurements and limitations.

## Status

| Connector | Twin | Real backend | Qualification |
|---|---|---|---|
| `elrobot` | Existing calibrated scene and learning tools, schemas preserved | Existing real-arm server through guarded Station adapter | Prior hardware demonstrations; this run used twins and fake Station only |
| `so101` | NormaCore follower URDF, generic IK, cameras and motion tools | Existing profile-driven real server | Fake-bus tests only; profile has unverified calibration fields and a different frame from the twin |

The facade at `generaloid.modules` exposes the twin, splat pipeline, learning, skills, executor, optional advisor, recorder, dashboard, perception and calibration in their current locations. The release agent can move those packages later without competing with parallel work tonight.
