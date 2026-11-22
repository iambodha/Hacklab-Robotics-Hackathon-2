# The night a robot learned to open a bottle

This is a production narrative for the video agent, grounded in [LOG.md](../LOG.md), [the integration history](history/INTEGRATION.md), [git commits](../learning/SELF_TUNING.md) and the topic journals. Times are CET on 21–22 November 2026 unless marked otherwise. Raw agent log clocks sometimes disagree; use event order and recorded attempt timestamps rather than inventing a continuous minute-by-minute story.

## 21 November: first give the agent eyes and hands

The track was to integrate NormaCore Station into an AI coding agent so it could control a robot by looking through cameras. Station already owned the motor buses and camera queues. The team first read the real follower, identified the two cameras, calibrated the arms and demonstrated teleop. The leader's inconsistent auto-calibration required a manual sweep. LOG.md records the observations and limits, including the lower image resolution delivered by Station.

The next layer exposed bounded motion and camera observations as MCP tools. A model could ask for a move, but local code checked ranges, tracking and faults. A simulated robot provided the same kind of interface without hardware.

**Visual cue:** Station camera image beside the joint table, then an MCP tool call and the corresponding twin view. Do not substitute a synthetic camera image without labeling it.

## The IK lean bug: reaching a point was not enough

An early IK fallback found a reachable point with the gripper tilted as much as 76° from vertical. That could make a mathematically valid move useless for a grasp. Commit `0cd5133` on 21 November changed the real-arm search to incrementally try lean angles instead of accepting an orientation-free solution. Its recorded offline cases improved from 23–77° to 0–11°; these are the commit's own test observations, not new release measurements.

Bottle round 2 exposed the remaining search-order issue: a nearby-joint solution at roughly 60° could win before restarts found a less tilted pose. Commit `f63acfd` at 03:55 on 22 November made the least-tilted reachable solution win and switched bottle approach to above the cap. Later real attempts recorded 32–35° lean. Pose choice and approach geometry mattered as much as force.

## Midnight integration, then a full disk

Round 1 agents built recorder, perception, twin, executor, skills and model advice in separate worktrees. At 01:15 their first integration had 160 passing tests and one skip. Round 2 connected those pieces: calibration and shared subscriptions, collision preflight, task planner, console, model benchmarks and splat alignment. A later integrated checkpoint reached 193 passes and one skip.

Storage interrupted the work. Station's queue history could exhaust the internal drive; a bottle-agent run reported “No space left on device” and even temporary files became unavailable. Large recordings moved to the external disk and queue sizes were bounded. Station's own normfs storage did not tolerate exFAT AppleDouble files, so its history had to remain on the internal disk. Keep these as two related storage problems, not one magical deletion that solved everything.

**Visual cue:** the disk-full message, then a simple split between small source/evidence and bulky recordings/reconstructions. The later r5 kitting drive disconnection is a separate incident.

## Bottle round 1: movement looked like success

Attempts 1–7 loosened the real cap without extracting it. The [curated twelve-attempt record](learning/data/attempts.json) and [curve](learning/learning-curve.svg) preserve the failures: slip, a refused excessive depth, a motor overload, and aborted runs. Attempt 3 achieved 201° gripper motion while the cap notch moved only about 12°. A short 15 mm check sometimes reported free while the jaws merely slid up a threaded cap.

This is the central observation: motor motion was not the task outcome. The longer extraction and visible air gap became essential evidence.

## Bottle round 2: geometry, load and a real opening

The bottle moved closer, about 0.20 m from the base. The approach came from above with less lean. Attempt 8 still slipped; attempt 9 used −6.5 mm depth and slow strokes and opened a cap already partly loosened. Attempt 10 began fully closed: roughly 10–11 strokes of 32° freed it, but the tool continued, timed out and needed a separate 60 mm lift.

At 04:34, commit `7f92050` addressed the stall: least-lean IK had run inline and blocked the event loop for about 15 seconds under load, leaving the Station stream silent. IK moved into a worker thread and silent streams could resubscribe. This was a transport/event-loop fix, not an increase in motor force.

For the demo, attempt 11 reset its connection before motion. Attempt 12 completed 11 medium strokes, 305.1° recorded gripper rotation and a 65 mm commanded lift; the operator saw an open bottle mouth. Do not claim all demo takes succeeded or use commanded lift as an independent measurement of cap height.

**Visual cue:** show failed attempts in sequence, then the open-mouth frame and longer lift. Source recordings live outside git; request the original take from the owner. The repo contains curated numbers and the SVG curve, not a rendered reconstruction of the real event.

## From one success to a platform

Round 4 made strategy changes versioned and reviewable. Round 5 separated four jobs: learned bottle defaults/guarded extraction, kitting in MuJoCo, Generaloid connector assembly, and inter-robot comms. Devin quota exhaustion left WIP, then Codex review corrected implementation and documentation. Their journals record each correction.

The bottle profile codifies 0.5 torque fraction, −6.5 mm depth, 32° slow strokes, a 60 mm extraction and a 12-stroke confirmation ceiling. A 110-second per-call budget supports continuation. It was derived from one series of changing conditions, not a controlled universal optimization. Replay found a slow-load signature; the medium-speed successful demo did not show it. Human/visual evidence still matters.

The twin is useful without pretending it is perfect. Connector tests lifted cubes on ElRobot and SO-101. A synthetic camera reconstruction aligned a splat to the twin's metric frame. The distinct robot twins negotiated help through a comms hub and exercised a gripper; they did not physically steady a bottle. Kitting improved held-out placements from 9/36 to 13/36, completing only the two identical nominal kits. Its external drive disconnected; the selected settings were reconstructed and freshly reverified, with 22 recovery episodes preserved.

The real-height bottle's physical twin still fails contact/extraction acceptance. Tell that plainly. The proof is a real bottle opening plus inspectable shared software, not a solved general-purpose robot or verified transfer across every arm.

## Morning handoff

Generaloid puts NormaCore Station at the centre, with replaceable robot connectors and model clients. The release adds an offline quickstart, an evidence-led README and explicit known issues. The new table's `office` calibration and `home`/`office` site registry are **in progress: r6**. No source relocation or live robot operation happened in this release pass. For reproducible checks and exact current counts, use [release validation](release/VALIDATION.md).
