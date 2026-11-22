# Executor and operator console

The executor accepts `ActionProposal`s from a scripted planner, MCP/model advisor or human, validates freshness and bounds, optionally preflights in the twin, executes a skill in phases and records an independently checked result. Its single asynchronous motion owner prevents overlapping actions within that workflow. It does not arbitrate all independent legacy MCP processes; deployments must choose one active controller per arm.

[RUN.md](../../RUN.md) explains the console :8910, CLI and blue-to-bin demo. Its stack launcher also starts live-facing services, so it is not the offline quickstart. Start with [the simulation-only wrapper](../release/QUICKSTART.md); real deployment follows [the task checklist](../../tasks/blue_to_bin/REAL_RUN.md) with an operator.

Backends: fake, MuJoCo sim, Station and replay. Advisors propose; they do not bypass checks. Real motion has separate enable gates. The inherited marker-versus-table preflight assertion remains a [known failure](../KNOWN_ISSUES.md). Kitting's fast trainer is a task-local controller; only a single box transfer was checked through its executor simulation bridge, not an entire executor-driven training run.
