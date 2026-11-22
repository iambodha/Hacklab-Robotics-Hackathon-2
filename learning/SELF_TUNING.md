# Self-tuning: versioned skill config

The episode params (`episode_begin`) tune one attempt. One level up are the **skill strategy settings**: the IK lean
policy, how `twist_cap` approaches the cap, and the bounds of the episode params themselves. Before this layer they were
constants in the code, and only an engineer could change them, by editing code and restarting the server (commit
`f63acfd`). Now an agent or a program **proposes** a change with evidence. The system validates it, checks it offline,
applies it as a **new version** without a restart, and stamps the version into every attempt record.

Code: `learning/skillcfg.py` (registry, storage, read path, proposals, CLI), `learning/skillcheck.py` (offline checks).

## What is tunable

`skill_config()` (MCP) or `python -m learning.skillcfg show` lists every key with its value, envelope and readers.

| key | default | envelope | read by |
|---|---|---|---|
| `ik.leans` | profile `ik.leans_small + leans_large` (0, ±10, ±20, ±30, ±35, ±40, ±45, 50…70) | 1..32 unique ints in -80..80 | `real_arm_mcp._solve_tcp`, twin `ArmIK.solve_least_lean`, executor `Kinematics.solve` |
| `ik.search` | `least_lean` | `least_lean` \| `near_first` | real arm + twin (one shared function: `skillcfg.lean_search`) |
| `ik.restarts` | profile `ik.restarts` (12) | 0..32 | real arm, twin, executor |
| `executor.ik.search` | `near_first` | `least_lean` \| `near_first` | executor `Kinematics.solve` (all executor skills' plans) |
| `twist.approach` | `auto` | `auto` \| `top` \| `side` | `learning/twist.py` |
| `twist.side_lean_deg` | 90 (= always from above) | 0..90 | twist (`auto`: side above this lean) |
| `twist.hovers_m` | [0.05, 0.035, 0.025] | 1..6 values in 0.015..0.10 | twist |
| `twist.max_lean_deg` | 75 | 10..80 | twist |
| `twist.grasp_below_top_m` | 0.010 | 0..0.02 | twist (the safety floor still applies) |
| `twist.release_open` | 0.25 | 0.05..0.6 | twist |
| `params.bounds.<name>.lo/hi` | `episodes.PARAMS` | hard envelope per param (`skillcfg.PARAM_ENVELOPE`); must keep lo ≤ default ≤ hi | `episodes.validate` (so also `episode_begin`, `episode_end` next_change) |

Hard envelopes of the bounds: grip_close 0..1, grip_torque_limit 0.05..**SAFETY.max_grip_force** (0.6),
approach_z_offset_mm -20..10, twist_deg_per_stroke 5..180, strokes 1..48, lift_per_turn_mm 0..6, yaw_start_deg
-180..180, regrasp_backoff_deg 0..180, j7_load_abort 50..**SAFETY.j7_load_hard** (500).

**Safety keys (class `safety`)** are listed but can never be changed through this layer: `safety.*` (=
`episodes.SAFETY`: max grip force, j7/arm hard loads, below-cap-edge floor, max step, min step dt, slip ticks),
`workspace.min_tcp_z/max_tcp_z/max_radius`, `motion.load_abort`. Changing them still means a code/profile edit by a
human.

With no config file every value is the code default, so behaviour is exactly as before.

## Storage and versions

`SKILLCFG_DIR` (default `<hackathon root>/logs/skillcfg/`):

- `real.json`, `twin.json`: the current config of each **target**: `{"target", "version", "values": {non-default keys}}`.
- `changes.jsonl`: append-only. One record per version: version, previous, author, approved_by, reason, evidence,
  proposal id, check summaries, `changes: {key: {before, after}}`, full `values`, timestamp. Also rejected proposals.
- `proposals/<id>.json`: each proposal with its status (pending, applied, rejected) and check results.
- `checks/<id>.log`: output of a background twin check.

Readers call `skillcfg.active()` / `get()` at call time. The file is cached by mtime (as in `sim/real_joints.py`), so a
running server picks up a new version on its next call. A `Twister` takes one snapshot per attempt, so an attempt
never mixes versions. A stored value that falls outside its envelope (a hand edit) is ignored, the default is used
instead, and `skill_config()` shows it under IGNORED.

**Targets.** A process reads one target. `real_arm_mcp` always reads `real`. `sim_arm_mcp` and `learning/sim_loop.py`
read `twin`. The executor reads `real` on the Station backend and `twin` on every other backend. A process that
doesn't choose a target gets `real`, so nobody runs unapproved values by accident.

## The flow

```
propose_skill_change(changes, reason, evidence)            (MCP, executor API, or CLI)
  -> validate: known key, strategy class, type, envelope, bounds lo <= default <= hi   (refused: nothing stored)
  -> proposal record, status pending
  -> checks (learning/skillcheck.py)
       ik_probe   ik.* and twist approach keys: the last 3 real cap poses from episodes.jsonl (locate phases),
                  grasp lean (seeded from park and from the hover pose), hover, approach mode: before vs after.
                  FAIL if a pose that was reachable before is not after.                         (~10 s)
       twin_run   optional (twin_check=true): learning.sim_loop --real-geometry, 1 attempt, before vs after, in
                  scratch config dirs; outcomes, rotation, the cap's true rotation, approach mode/lean. Runs in
                  the background (minutes) and writes into the proposal. FAIL if clearly worse.
  -> twin target: applied at once if no check failed (a running twin check doesn't block)
     real target: stays pending until a human approves
```

**Who may apply what:**
- **Twin:** automatic (`sim_arm_mcp.propose_skill_change`, executor on a non-Station backend).
- **Real arm:** only a human, through the CLI: `python -m learning.skillcfg approve <id> --by <name>`. The real
  server's MCP tools can only read and propose; `skillcfg.apply` refuses the real target unless `human=True`, which only
  the CLI passes. A proposal with a failed check needs `--force`. A proposal whose keys changed after it was made
  (another version came in between) is refused: propose again.
- **Never auto-apply to the real arm.**

## Commands (run from the repo root)

```
python -m learning.skillcfg show [--target real|twin]     # values, envelopes, pending proposals, recent versions
python -m learning.skillcfg proposals [--status pending]
python -m learning.skillcfg proposal <id>
python -m learning.skillcfg check <id> [--twin]           # re-run the checks (--twin: the twin run, synchronously)
python -m learning.skillcfg approve <id> --by <name> [--force]
python -m learning.skillcfg reject <id> --by <name> [--why "..."]
python -m learning.skillcfg revert <real|twin> <version> --by <name> [--reason "..."]   # 0 = the code defaults
python -m learning.skillcfg propose KEY=JSON ... --target real|twin --reason "..." --evidence "..." [--no-check]
python -m learning.skillcfg log [--target real]
```

The IK probe needs numpy and mujoco: `uv run --no-project --with numpy --with mujoco --with pillow python -m learning.skillcfg ...`.
`approve`, `reject`, `show` and `revert` are pure stdlib. A revert never rewrites history: it writes a new version
whose values equal those of the version you name.

## Attribution

`episode_begin` writes `"skillcfg": {"target", "version", "overrides"}` into the record. The attempt `report.md` has
a "Skill config: real v3 (twist.approach="top")" line, `episode_history` shows it, `SUMMARY.md` has a `cfg` column, and
there is an "Outcomes per skill-config version" table (attempts, outcomes, mean rotation per version). Every executor
validation carries `checks["skill_config"] = {target, version, overrides}`, and `describe()` shows the version.

## Operator brief: paste into OPERATOR_BOTTLE.md

```markdown
### Skill strategy changes (skill_config / propose_skill_change)

Your episode params tune one attempt. If the evidence says the *strategy* is wrong, not the params, propose a skill
change. Call `skill_config()` to see the settings in force, their envelopes and any pending proposals.

Propose when (cite the attempt records as evidence):
- The approach phase shows `lean_deg` > 45 with `mode=side`, and attempts end in `slip`/`misaligned` with the jaws
  on the cap's bevel -> `propose_skill_change({"twist.approach": "top"}, ...)`.
- `twist_cap`/`tcp_pose` notes say "jaws-down unreachable ... gripper leans 60 deg", but other attempts or poses at the
  same place reached the cap at ~35-40 deg -> `{"ik.search": "least_lean"}`, or add finer leans to `ik.leans`.
- "no reachable hover above the cap" -> lower hover heights: `{"twist.hovers_m": [0.035, 0.025, 0.02]}`.
- You need a param value outside its bound (e.g. more strokes, "keep spinning until it opens") ->
  `{"params.bounds.strokes.hi": 36}`. The hard envelope still caps it.
- Jaws hit the cap when they release between strokes -> `{"twist.release_open": 0.35}`.

Rules:
- One idea per proposal, with a reason and evidence (e.g. `["attempt 7 approach lean_deg=60 mode=side",
  "attempt 7 lesson: jaws slid onto the bevel"]`). Safety limits cannot be changed: don't ask.
- On the REAL arm a proposal stays PENDING until the person at the table approves it. Tell them the proposal id and
  the command `python -m learning.skillcfg approve <id> --by <name>`. Never run approve, revert or reject yourself.
  Keep attempting with the current version meanwhile, or wait if the change matters for the next attempt.
- `twin_check=true` adds a twin before/after run (minutes, in the background; results in `skill_config()`).
- After a version change, keep the episode params unchanged for the first attempt, so the effect can be attributed
  (the record shows `skill config: real vN`).
```
