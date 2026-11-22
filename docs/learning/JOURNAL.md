# r5-skills journal

Worktree `ws/r5-skills`, branch `r5-skills` (cut from `integration` at `e3cdeaa`). Times are local (22 Nov 2026).

## 08:10 Read the records and the r4 layer
- Read `prompts/r5-common.md`, `prompts/r4-selftune.md`, commits `dc7921c`, `f0b86a3`, `ced4561`,
  `learning/skillcfg.py`, `learning/skillcheck.py`, `learning/SELF_TUNING.md`, `learning/twist.py`,
  `learning/tools.py`, `learning/episodes.py`, the bottle sections of both MCP servers.
- Read all 367 records of `logs/learning/bottle/episodes.jsonl` (12 attempts: 12 begin, 331 phase, 12 judge pairs,
  12 end) and `SUMMARY.md`, plus the tool-call log `logs/real_sessions_home/actions.jsonl` (call durations).
- Baseline test run started (full command of r5-common).

Facts I took from the records (not from the brief) and that changed my design:
- **Per-stroke mean j7 load** (attempt 10, slow, closed cap): 104, 84, 74, 76, 76, 78, 88, 86, 76, 74, **70, 71**,
  69, 67, 67, 65, 57, 56, 59, 56, 55, 54, 45. The load falls below 0.7 x the break-away stroke at stroke 11.
  Attempt 12 (medium, demo take): 153, 136, 133, ... 135, 114: it never falls below 0.74 x. Attempt 9 (slow, a cap
  already turned in attempt 8): 83 -> 64. So the "it is free" signal is clear only at slow speed on a closed cap.
- **Call durations** (server side, `actions.jsonl`): attempt 8 `twist_cap` 172 s (no error), attempt 9 154 s
  (TimeoutError), attempt 10 271 s (TimeoutError), attempt 12 122 s (ok). So a 120 s client timeout is not a hard
  fact in these logs: attempt 12's 122 s call returned. I keep the brief's 120 s as the budget's upper envelope and
  note the contradiction in the docs.
- **Stroke cycle** (twist + release + turn back + regrasp), from the twist phase timestamps: slow 10.1 s
  (attempt 10), medium 8.5 s (attempt 12), 6.3-8.7 s earlier with 25 deg strokes. The approach + grasp inside the
  call takes ~25-35 s. So 11 slow strokes (111 s) plus the approach (~25 s) do **not** fit any 120 s budget; 11
  medium strokes are at the limit (attempt 12: 122 s).
- **Safety floor** at the learned depth: with the 22 mm cap at cap_top 0.205 m the floor is TCP 0.188 m and the
  learned grasp (-6.5 mm) puts the TCP at 0.1885 m: the learned depth is the deepest the floor allows (-7 mm). For
  the 20-21 mm cap of round 1 the floor allowed only -5 / -6 mm: attempt 4's -11 mm was refused for that reason.
- **check_cap could not tell**: attempts 1, 2, 7 reported cap_free=true after the 15 mm lift (gripper rose 9.3-9.5
  mm still holding) while the cap was still on the thread (the operator's images); attempts 3, 6, 8 rose 7.9-8.9 mm
  and reported "unclear". Neither the rise nor the load rise (44-124) separates the cases. A lift longer than the
  grip depth on the cap (~16 mm) does: the jaws slide off the top of a cap that is still on the thread, but carry a
  free cap all the way up. That is the 6 cm lift of attempts 9-12.

## 08:27 Design decisions
1. **Three layers, lowest wins last**: code baseline (today's values: behaviour unchanged with no profile) <
   learned profile `learning/profiles/bottle_cap.json` (git-versioned, provenance per value) < skill-config
   versions (runtime, proposals, human approval on the real arm). A skillcfg revert to version 0 = the profile.
   `SKILL_PROFILES=none` = the code baseline.
2. The episode parameters' defaults become skill-config keys `params.default.<name>`, so the learning rule can
   propose them through `propose_change` like any other key. `episode_begin` still inherits from the previous
   attempt, but a `params.default.*` value that changed in a config version since the previous attempt is taken
   over (and counted as a change), so an applied proposal reaches the next attempt.
3. **twist_speed: slow** (as the brief), backed by the records: the free signal is only visible at slow speed.
4. **Time budget with resume**: because 11 slow strokes cannot fit one call, `twist_cap` stops before the budget
   (`twist.time_budget_s`, 110 s = 120 s client timeout - 10 s for the reply) with the cap still gripped, and a
   second `twist_cap` call at the same cap **resumes** (no re-approach) until the attempt's stroke total.
5. Stop-on-free thresholds fitted on the real records, not guessed: ratio 0.70 of the break-away stroke, 2
   strokes in a row, only after 280 deg (0.8 x the ~350 deg thread). Replay: attempt 10 stops at stroke 12; no
   false positive in attempts 1-8.

## 08:43 Codex takeover
- Took over from Devin after quota exhaustion; reviewed WIP 02ef54d against integration, journal and log tail. No commits will be made per this run’s override.
- Started the required full baseline suite with Station disabled. Found missing learning/rules.py, missing extraction MCP endpoints, unsafe retreat after failed jaw opening, and nonconsecutive free-signal confirmation.
- Source logs live outside the worktree at hackathon/logs/learning/bottle; read-only. No live service or robot calls.

## 08:55 Review corrections and first validation
- Completed `rules.py`, wired `extract_cap` on both servers; pure replay of curated source shows overload -> 0.5 torque, attempt 8 bevel -> -6.5 mm/top, short lift -> more strokes/longer guarded lift, timeout -> 110 s calls.
- Fixed resume to release/backoff/regrasp before another forward stroke. Previously a resumed call would rotate forward again from the end pose.
- Free-load confirmation now rejects slip/failed strokes and resets the consecutive run across invalid strokes. Never treats load alone as proof of extraction.
- Retreat now refuses upward motion if jaw opening fails or holding persists. Lift planning also rejects IK branch jumps.
- Learned cap height must default to 22 mm: leaving MCP's old 20 mm default would refuse the learned -6.5 mm grasp against the cap-edge floor.
- Baseline at takeover: 272 passed, 12 failed, 3 errors, 1 skipped (102.53 s). Includes stale assumptions about pre-profile defaults/stamps and sandbox CoreGraphics renderer failures. Devin's earlier baseline was 286 passed, 1 failed, 1 skipped.
- First targeted run: 88 passed, 1 failed (isolated wipe package lacked new learning import). Fixed with `skills/config.py`, which uses versioned config normally and the identical checked-in defaults in skills-only improvement sandboxes. New learned tests: 8 passed.
- Sandbox blocks `nice` priority changes (`Operation not permitted`); reran a single headless twin job without nice. CoreGraphics unavailable: physics runs with frames disabled only in the offline probe. Initial learned-geometry probe aborted park-to-hover travel on tracking before any cap motion; investigating rather than loosening the safety guard.

## 08:57 Validation and remaining physical gap
- Targeted existing and learned suites: 97 passed (82.81 s). Expanded learned tests then passed 13/13 (3.80 s); added hard-load, changed-profile inheritance and shared-reader regressions afterward.
- Full intermediate suite: 290 passed, 6 failed, 3 errors, 1 skipped (142.68 s). Six WIP failures eliminated relative to takeover; remaining failures are 5 renderer-dependent test failures + 3 renderer setup errors and the known executor marker/table message assertion. Final full run underway.
- Added actual async motion deadline and bounded halt acknowledgement to both rigs. A deadline interruption forces reapproach on the next MCP call. Failed release blocks backoff; lifts enforce immutable hard loads/status in addition to load-rise/lag guards.
- Runtime defaults from a newly selected profile can invalidate old runtime bounds; conflicting overrides now fall back visibly to profile values. Episode records store the exact default snapshot so profile edits reach the next inherited attempt.
- Profile ceiling now 12 strokes: 10–11 opened in real recordings, but attempt 10 requires stroke 12 to confirm the low-load run. This conservative confirmation margin is explicitly documented; baseline remains 3 with SKILL_PROFILES=none.
- Headless MuJoCo probe at learned real geometry: park-to-hover tracking abort; prepositioned hover gives modeled 45 degree lean and contact 9 mm too high. Short fixture: 11 strokes achieved 339.4 degrees gripper motion but only ~270 degrees cap motion; lift stopped safely on 410 load rise after 4 mm measured rise. Did not weaken guards or retune physics to manufacture success. Added reproducible learning.validate_bottle_twin probe and documented that physical acceptance is unresolved.
- Deterministic kinematic screw/contact contract opens within the learned budget and extracts exactly 60 mm. This verifies skill/config behavior only, not physical twin accuracy.
- Audited executor and both MCP skill sections; named remaining strategy numbers in skills/strategy_defaults.json, retaining robot-profile safety/physical values. sim/stir_skill.py internal strategy is outside the owned-file boundary and remains an integration follow-up.
- Updated story, per-attempt data, detection caveats, configuration/rollback guide, audit and validation docs. Matplotlib used a temporary cache after sandbox refused home cache paths; a small learning-curve plot is being generated.

## 08:59 Final verification and handoff
- Final required full suite: **294 passed, 6 failed, 3 errors, 1 skipped**, 149.78 s. Remaining six failures: executor sim observation/pick/wipe (CoreGraphics), executor preflight marker/table message (known baseline), learning real-geometry run and proposal-to-twin run (CoreGraphics). Three learning loop fixture errors are the same unavailable renderer.
- After final registry/profile/alias corrections: focused learning config/store suite **52 passed** in 32.99 s. Final learned acceptance file **18 passed** in 6.51 s, including actual early-stop behavior, 60 mm deterministic extraction, real-record replay, async deadline halt, immutable hard loads, profile inheritance and shared executor/wipe config reads.
- `git diff --check` clean. Branch remains `r5-skills`. No commits, pushes, branch switches, live robot calls or Station connections.
- Small learning-curve SVG generated from the curated real records (44.8 kB); docs/learning totals ~176 kB. Sandbox font caches fell back to temporary storage. No logs/learning/bottle source files modified.

## REPORT r5-skills

**Status:** learned configuration, evidence rules, guarded extraction, early stop/deadline/resume and documentation implemented and tested offline. **Physical MuJoCo opening/extraction acceptance remains incomplete**, with measured failures and a reproducible nonzero-exit probe; do not call this full sim-to-real validation.

**Commits:** none in this continuation, per explicit git override. Inherited unreviewed WIP is 02ef54d; all follow-up changes are in the working tree, including new untracked files that the integrator must add.

**What works:** versioned bottle profile with per-value provenance; 12-stroke ceiling with confirmation margin; shared real/twin defaults; 60 mm guarded extraction tool on both servers; j7 free-signature replay (attempt 10 confirms at stroke 12, none in attempts 1–8); evidence-derived bounded proposals with real approval preserved; inherited defaults reflect approved changes; executor/wipe/twin-tool strategy values are named config; immutable safety refusals and motor-fault/deadline behavior covered by tests.

**What does not:** full physical twin demonstration at learned real geometry; camera rendering in this sandbox. The old short fixture also loses cap rotation relative to gripper motion. sim/stir_skill.py internals still require an audit outside this brief’s owned scope. Motor-only extraction evidence still needs visual confirmation of cap/bottle separation.

**Files touched:** learning/{skillcfg,episodes,tools,twist,capfree,rules,bottle_twin,validate_bottle_twin}.py, learning/SELF_TUNING.md, learning/profiles/bottle_cap.json, learning/tests/{conftest,test_learned_profile}.py; skills/config.py, skills/strategy_defaults.json, skills/wipe/{adapt,planner,skill}.py; executor/skills.py; bottle/skill sections of sim/{real_arm_mcp,sim_arm_mcp}.py; docs/learning/{README,PROFILES,CAP_FREE,AUDIT,VALIDATION,JOURNAL}.md, data/attempts.json, learning-curve.svg.

**Test counts before/after:** takeover 272 passed / 12 failed / 3 errors / 1 skipped; final required run 294 passed / 6 failed / 3 errors / 1 skipped. Latest focused suites 52 passed and 18 passed (overlapping selection). Devin’s pre-WIP report was 286 passed / 1 failed / 1 skipped, from an environment with rendering access.

**Merge notes:** include untracked files. Preserve existing public names and additive extract_cap/rig halt methods when integrating r5-generaloid. Existing check_cap remains the 15 mm compatibility tool; use extract_cap for the learned 60 mm action. No real auto-apply path was introduced. SKILL_PROFILES=none reproduces baselines; unset or bottle_cap activates the learned defaults. Review source profile changes as human-reviewed configuration, and use runtime proposal/approve/revert for subsequent tuning. Do not restart live MCP services until the operator has finished.

**Open work:** calibrate twin jaw/contact geometry and dynamics against measured real poses before claiming physical parity; audit the out-of-scope twin spoon helper; rerun renderer-dependent tests in a desktop-capable environment. See VALIDATION.md for reproduction and exact observations.
