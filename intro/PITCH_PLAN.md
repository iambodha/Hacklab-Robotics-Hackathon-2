# Pitch plan: 5-minute presentation + 3-minute showcase video

## The story in one line

**Station abstracts the hardware. Our MCP abstracts the skills. The agent just says what it wants.**
We sharpen new skills in a calibrated simulator, and they transfer to the real arm one to one.

## Narrative arc

1. **Layers.** Different robot arms are wrapped by the NormaCore Station API. Our MCP layer sits on top, and an
   agent sits on top of that. Each layer does one job, which is what makes the system easy to extend.
2. **Why a simulator.** Testing new behaviour on the real arm is expensive, risky and slow: every session needs
   calibration, a person at the arm, and a broken grasp can damage hardware. So we built a MuJoCo twin behind
   *exactly the same MCP tools*, calibrated to the real camera, table and lighting.
3. **A new skill, sharpened in sim.** We show one new skill improving over ~5 attempts (sped up).
4. **Sim-to-real reveal.** The final sim attempt fades out and lands exactly on top of real footage of the arm
   doing the same thing (camera views matched). The skill transfers to reality one to one.
5. **Skills, not improvisation.** Sorting rubbish and other tasks: the model does not invent motions, it reuses
   tools. So inference does not need a frontier model; cheap models can do the job (backed by our benchmark).
6. **Close.** Layers recap, what's next, team.

## Showcase video (3:00)

| Time | Visual | Voice-over (draft) |
|---|---|---|
| 0:00–0:20 | Real arm sorting the table on its own (run 7, sped up), overview + wrist camera | "This arm has never been told where anything is. It looks, plans, picks and checks — on its own." |
| 0:20–0:50 | Animation (`intro/`): ElRobot and SO-101 snap into a **Station API** block, our **MCP** layer docks on top, an **agent** on top of that | "NormaCore Station turns any supported arm into one clean API: motors and cameras as data, one command path, calibration and recording built in. On top we built an MCP layer of robot skills, so any agent can drive it." |
| 0:50–1:10 | Quick cuts: calibration, a person at the arm, a failed grasp on hardware | "But real hardware is slow to iterate on: every test needs calibration, supervision, and a bad move can break something." |
| 1:10–1:50 | Sim: attempts 1→5 of the new skill, sped up, each with a one-line caption of what went wrong | "So we practise in a twin of our table, behind the same tools the real arm uses. The agent tries, fails, adjusts — five attempts in under a minute." |
| 1:50–2:20 | Final sim attempt cross-fades and lands exactly on the real-arm footage | "Then we run the same skill on the real arm. Same camera, same tool calls — and it behaves the same." |
| 2:20–2:45 | Montage: sorting rubbish, other skills; benchmark chart (success rate and cost per model) | "Because the model reuses skills instead of inventing motions, it doesn't need to be the smartest model: [Haiku solved X/N tasks at $Y each]." |
| 2:45–3:00 | Layer diagram again, team names, repo, "built on NormaCore Station" | "Station abstracts the hardware, MCP abstracts the skills, the agent just says what it wants." |

## Presentation (5:00, 7 slides)

| # | Slide | Time | Content |
|---|---|---|---|
| 1 | Hook + problem | 0:30 | Models speak intent, servos speak registers. Short clip of the real sort. |
| 2 | Architecture | 0:45 | The layer animation: arms → Station API → our MCP → agent. |
| 3 | Station as the foundation | 0:30 | Hardware as data streams, one command API, calibration + teleop, recording all day (~7 GB), web UI, one binary. "We wrote zero driver code: all our time went into behaviour." |
| 4 | Sim → real | 1:15 | Why sim; the 5-attempt clip; the overlay reveal. This is our main selling point. |
| 5 | Results | 1:00 | Real arm: 3 items sorted with no coordinates given, 189 s, $0.37 (run 7). Sim benchmark: success rate and cost, Haiku vs Sonnet. |
| 6 | Safe and portable | 0:30 | Bounded steps, abort on tracking error, refuse out-of-range targets. Robot profiles (ElRobot, SO-101 offline), any model incl. Gemma. |
| 7 | Close | 0:30 | Recap line, next steps, team, repo. |

Backup slides for Q&A: full tool list, lessons from the hardware (USB brown-outs, calibration drift,
frozen frames, read timeouts), the per-run table from `sim/real_runs/2026-11-21/README.md`.

Draft deck (14 slides, to be cut down to the 7 above): https://claude.ai/artifact/JMLyXjXa8432rNQyS9TiNn

## Open decisions and risks

### Which skill to show in the sim-to-real segment
Unscrewing a lid is the most impressive, but risky: the skill does not exist yet, thread friction is hard to
model in MuJoCo, and j7 turns about one revolution, so it needs regrasping. If the real run fails, the overlay
moment falls apart.

Safer candidates that already exist and have been run with real objects:
- **`stir`**: IK-planned spoon circles inside a mug (there was a real "mug run").
- **`grab_tip`**: pull one tissue out of a pack by its tip.

**Decision needed:** lid (if it works in sim and on the arm by [time]) or fall back to `stir`.

### The "5 attempts" must be real
Use real recorded sessions (`sim/recorder.py`, `sim/make_replay.py`), not staged ones. Caption each failed
attempt with its real cause (missed height, spoon hit the rim, …). Be ready to answer "who is iterating —
the agent or you?".

### Making the overlay honest and 1:1
- The twin's camera is already calibrated to the real table (commit "calibrate the twin to the real table").
- Run the skill with the **same parameters** in sim and on the arm, then align in time using
  `actions.jsonl` / Station's recording. Real servos lag the sim, so sync on the start of each tool call.
- Do **not** drive the sim with the recorded real joint angles to get a perfect match: that would be faking it.
  A small, visible drift is fine and believable.
- Match object positions on the real table to the sim layout (`sim/layout.json`).

### The "cheap models are enough" claim needs data
Current data contradicts it: in sim, Haiku dropped the cube off the table; on the real arm (run 4) it only
partly finished the sort. Those runs were *before* `survey` / `center_over` / `pick_at` existed.

Plan: a small sim benchmark — N = 10–20 randomized scenes (`operator_reset_scene(randomize)`, scored with
`sim/judge.py`), Haiku vs Sonnet with the current skill tools. Report success rate and cost per task.
This also shows a second use of the sim: **cheap, safe evaluation**, not just training.
If Haiku is still clearly behind, soften the wording to what the numbers show.

### SO-101 in the animation
Fine to show SO-101 wrapped by Station as the pattern, but its profile has **never run on hardware** (offline
only). Say "profile ready", not "runs live".

### People in footage
Frames from real runs 2 and 4 show identifiable people. Use frames without faces in the video and slides.

## To-do

- [ ] Decide the sim-to-real skill (lid vs `stir`) — owner: [ ]
- [ ] Record 5 real sim attempts of that skill — owner: [ ]
- [ ] Record the same skill on the real arm, camera matched to the twin — owner: [ ]
- [ ] Build the overlay cross-fade — owner: [ ]
- [ ] Run the Haiku vs Sonnet sim benchmark, make the chart — owner: [ ]
- [ ] Finish the layer animation in `intro/` — owner: Evgenii
- [ ] Cut the deck to 7 slides, add clips — owner: [ ]
- [ ] Voice-over recording, final edit, submit the 3-min video — owner: [ ]
- [ ] Fill in team name, names, repo link
