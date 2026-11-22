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
