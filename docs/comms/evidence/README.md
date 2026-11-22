# Verification captures — 22 Nov 2026 takeover

These four captures replace the unreviewed WIP captures with runs of the repaired executable sources.

| Capture | What actually ran | Duration |
| --- | --- | --- |
| `twin_demo_*` | Two separate ElRobot MuJoCo MCPs; B explicitly stands in for SO-101 | 20.83 s |
| `generaloid_demo_*` | Separate Generaloid ElRobot and SO-101 sim connectors | 6.99 s |

Both exercise help → accept → progress → own twin gripper action → progress → result. One oversized gripper
waypoint is refused; smaller waypoints change the joint state. The holding stage is scripted; neither run
verifies a physical bottle hold. JSONL logs contain no join/session tokens. Both servers bind loopback only.

The Generaloid capture and integration tests ran in a temporary combined package: sibling `r5-generaloid` at
`16d6651` plus this worktree's final `generaloid/comms/`, with other repo entries symlinked to this worktree.
No sibling source was edited. The original worktree intentionally lacks `generaloid.core` until integration.

Final combined comms suite: **74 passed in 46.92 s**, including auth/timeout/reconnect, malformed input, standalone
hub/client subprocesses, two twin MCP processes and the distinct connector integration. Earlier local run:
70 passed / 1 skipped before the last three regressions were added; all three pass in the final combined run.

The prescribed full common regression suite has identical before/after outcomes:
**278 passed, 6 failed, 1 skipped, 3 errors** (99.40 s before, 150.86 s after).
Five failures and the three errors involve the unavailable macOS CGL graphics connection; the remaining failure
is the pre-existing table-collision preflight expectation. These are outside comms. Full outputs remain at
`/Volumes/Dell HDD/Robotics-Hackathon/r5-comms/takeover-*.txt`.

Reproduction commands and limits: [README](../README.md), [JOURNAL](../JOURNAL.md), [scenario](../SCENARIO.md).
