# Integration board — public history extract

Source: the integrator's external INTEGRATION.md, read 22 November 2026. This curated copy retains architecture, workstream ownership, milestones and incident lessons. Process IDs, machine paths, local session identifiers and secret-file instructions are omitted. It is historical, not deployment state. LOG.md at the repository root preserves the original 21 November bring-up record.

## Starting branches, 21 November

`robot-profiles` contained the most advanced ElRobot sim/real tools and profile constants, building on `simulation`. `simulation_s101` separately adapted the twin to SO-101. `system-two-approach` worked on the calibrated table twin; `calculate-camera-position` supplied wrist geometry; `MCP` supplied HTTP serving; `kerandia` supplied Station/relay work. These diverged branches motivated a coherent connector boundary.

## Round 1: separate foundations

| Workstream | Delivered / remaining at integration |
| --- | --- |
| Hardware bring-up | Station, two cameras, follower reads, calibrated ranges |
| Recorder | Read-only capture/replay API and viewer, 14/14 tests; proposed legacy replay wiring still pending |
| Perception | Pooled intrinsics, marker geometry, wrist/base estimation and MCP measurements; final calibration and subscription fix pending |
| Twin | Calibrated home scene, textures/lighting, live mirror, comparison tools; remaining selftests and offset fit |
| Executor | Proposal/validation/motion ownership, advisors and console scaffolding; real motion gated |
| Skills | Wipe planner/evaluator/adaptation, memory and sandbox experiment flow |
| Models | Gemini ER client/advisor, benchmark/export/autolabeller; initial key restrictions blocked real calls |

At 01:15 on 22 November the first integration had 160 passing tests and one skip.

## Round 2: wire and measure

- r2-wire unified contracts, perception, preflight, skills and optional Gemini advisor through the executor.
- r2-percfix fixed repeated Station subscriptions, refined camera/base calibration and localized the cap/bin.
- r2-console delivered the operator console :8910, CLI and stack launcher.
- r2-task measured blue-to-bin objects, trialed the task and wrote real-run checks.
- r2-gemini exercised model calls/benchmarks; r2-twinfix added selftests, offset fitting, mirror performance and the splat pipeline.

The integrated checkpoint reached 193 passes and one skip after console integration. Follow-ups handled partial Station frames, a single render-thread owner, and the twin locator import path. Wrist mount pitch/offset corrections and exported overview calibration were explicit integration work, not interchangeable defaults for every arm.

## Storage and service incidents

Station's queue defaults could fill the internal disk. The team constrained normfs file/queue sizes and moved bulky recordings/reconstruction data off the system disk. Moving Station's own normfs history to exFAT caused AppleDouble metadata to be parsed as invalid entries, so Station history returned to the internal disk. A uv cache prune required attention to the running recorder's cache lock. Separate macOS Documents permissions disrupted terminal processes. None of these are instructions to restart today's services.

## Rounds 3 and 4: bottle, dashboard, versioned learning

Round 3 (`3861509` integration base) added shared bottle tools/episodes and a learning dashboard :8920. A model operator ran the real attempts with the person at the table. The first round loosened the cap but did not remove it.

Round 4 (`f63acfd` bottle changes) used least-lean IK and top-down approach. The versioned self-tuning layer followed: bounded strategy proposals, offline checks, immutable safety class and human approval for the real target. The integration board explicitly deferred merging tuning work until live bottle operation had ended. The repo-wide pytest guard was added after test connections disturbed Station during an operator run.

## Round 5: Generaloid and review

Starting near `e3cdeaa`, four workstreams ran: r5-skills (learned bottle profile and guarded extraction), r5-kitsim (A4 twin kitting), r5-generaloid (Station-centred connectors) and r5-comms (robot inbox/hub). Devin quota exhaustion left WIP commits; Codex reviewed and completed those trees, documenting limitations rather than treating WIP as validated.

The release pass assembles their evidence and docs on main without source moves. The updated brief records already-pushed main `65d5984`, with r6-sites and r6-office developing the new table/site registry in parallel. See each topic's journal and [release validation](../release/VALIDATION.md) for counts from the actual merged tree rather than adding independent workstream totals.

## Deployment lessons retained

Camera hashes changed repeatedly; roles/site configuration must replace copied IDs. A 299×224 overview yields only a few millimetres per pixel; calibration quality matters. `follow()` subscriptions must be shared. Keep recordings and model artifacts outside git, Station local, and exactly one controller active on an arm. Historical lists of running services and calibrated ranges must be rechecked for the selected site by its operator.
