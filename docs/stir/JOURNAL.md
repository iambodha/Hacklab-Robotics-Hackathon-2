# stir-skill journal

Branch `stir-skill` (cut from `main` at `00b9219`), clone `ws/r5-release`. Times are local (22 Nov 2026). The brief:
build a "stir a cup" skill in the MuJoCo twin of the **office** table and practise it with the same learning loop
that taught the real arm to open the bottle cap. Twin only: no Station, no real arm.

## 12:25 Read what exists
- `sim/stir_skill.py` (r4): `StirSkill.run` already does the whole stir from a cup centre: refine the centre on the
  overview image (the red mug), grasp the spoon handle along its axis (jaws closing across its 12 mm width), swing
  it upright keeping the bowl in place, stir circles with the bowl, release, retreat. Exposed as the twin tool
  `stir_spoon` in `sim/sim_arm_mcp.py`; geometry and turns/radius read from skill-config keys `sim.stir.*`
  (`skills/strategy_defaults.json`).
- `sim/sim_arm_mcp._track_stir`: per physics step, bowl angle around the cup axis, path length, time inside, cup
  tilt; `stir_summary()` counts loops of the bowl about its own path centre.
- `sim/layout_stir.json`: the r4 stir scene on the *venue* table (6.8 cm mug, 16 cm spoon), arm folded back.
- The bottle loop: `learning/episodes.py` (records), `learning/skillcfg.py` (versioned config, profiles with
  provenance), `learning/sim_loop.py` (scripted policy on the twin), `learning/profiles/bottle_cap.json`.
- Decision: **build on `StirSkill`, don't write a second stir.** The real arm's `stir` tool (`sim/real_arm_mcp.py`)
  is a different, tool-tip-only skill (the operator puts the tool in the jaws); the twin skill is the one that grasps
  a spoon from a mug, so that's the one to practise. The learned values are the motion settings both share
  (radius, turns, depth, speed); REAL_RUN.md maps them.
- Decision: **own record store** (`learning/stir/store.py`) shaped like the bottle's (begin / phase / judge / end
  JSONL, attempt folders, SUMMARY.md) instead of generalising `learning/episodes.py`: that module's PARAMS, OUTCOMES
  and SAFETY are the bottle's, the real arm's MCP server imports it, and its tests pin it. Same record kinds, so the
  same reading habits apply.

## 12:28 Baseline on the r4 scene
`stir_spoon` on `sim/layout_stir.json` (home-era scene): DONE, 2.8 loops, cup untouched, 38 s. The skill works.

## 12:30 Office scene (learning/stir/task.json -> sim/layouts/stir_office.json)
- Task objects in `learning/stir/task.json` (sizes with reasons), composed onto the office site layout by
  `learning/stir/twin.py` into the committed `sim/layouts/stir_office.json` (select with `SIM_LAYOUT=stir_office`).
  `sites/office/layout.json` and every home file stay untouched (a test checks the site layouts carry no stir
  objects).
- Sizes: mug 8.0 cm outside (inner radius 3.6 cm, wall 4 mm), 9.5 cm tall, 60 % full; spoon 15.5 cm bowl-centre to
  handle end (14 cm handle 12 x 4 mm, 3.8 cm bowl). Mug 19 cm in front of the arm base on its centre line.
- **Problem 1: the spoon would not stay on the rim.** Placed analytically (bowl on the bottom against the far wall,
  handle on the rim), it slid 3-5 cm sideways in the first 0.25 s, every variant. Contact dump: the arm's
  `Joint_03` link was touching the spoon. The office layout's `rest_pose` is the real arm's pose during the
  calibration frames, reaching forward low over the table (gripper at y 0.35, z 0.095): its forearm lies across the
  mug spot. Fix: the stir layout starts from the r4 stir scene's folded pose (`task.json rest_pose`, documented; the
  real-arm checklist parks the arm there before the mug goes down). Then the spoon is stable (top moves < 0.1 mm
  over 3 s).
- Bowl placement: 0.4 x inner radius past the axis, which is what `stir_skill.handle_line` assumes. The generator
  lets MuJoCo settle the spoon for 1.5 s and writes the settled pose, so a reset starts at rest.

## 12:35 Office baseline: the mug centre was 4.6 cm off
`stir_spoon` on the office scene: "could not get hold of the spoon handle", 5 tries. `refine_cup` had moved the
mug centre 4.6 cm the wrong way. Cause: it back-projects pixels with a centred pinhole from `cam_fovy`, but the
office overview camera is built from its calibration (`K_normalized`): principal point 4 px right of and 6 px below
the centre. At the overview's grazing view that is ~4 cm on the table. Fix: use `build_scene.camera_K` (the matrix
MuJoCo renders with). After it: centre within 2 mm, grasp on try 1, 3.6 loops, mug unmoved.
Follow-up for the integrator (not fixed here, out of scope): `SimArm.pixel_ray` (the agent's `pixel_to_table`) has
the same centred-pinhole assumption, so on the office twin an agent's clicks are off by centimetres too.

## 12:40 Judge, store, policy
- Twin tracking (`_track_stir`): added contact timers (spoon on the mug walls, spoon on the bottom, any arm link on
  the mug), time with the bowl under the drink, the largest mug tilt, and bookmarks (`stir_mark`,
  `stir_summary(since=)`) so a phase's share is a difference of two bookmarks (the spoon rests on the rim before the
  grasp; that contact must not count against the stir).
- Judge (`learning/stir/judge.py`, thresholds and weights in `task.json "judge"`, each with its reason): gates
  (>= 3 loops, wall contact <= 5 % of the stir, no arm on the mug, mug moved <= 5 mm, tilt <= 5 deg, spoon slip in
  the jaws <= 1 cm, spoon in the mug at the end, bowl under the drink >= 90 %), then quality = coverage (mean bowl
  radius / usable radius), depth, pace, immersion, clean. Failed gates cost 0.5 each. Loops are **net** loops about
  the path centre (back-and-forth cancels) and a path under 3 mm radius counts none: the r4 summary counted 5.3
  "loops" for a spoon that only jittered after a failed grasp.
- Skill parameters: `StirSkill.run` takes stir height, step time, ring points, wall margin, extra lean and an
  `on_phase` callback; the defaults are the r4 code values (so `stir_spoon` behaves as before). New skill-config
  keys `sim.stir.height_m, step_dt_s, ring_points, wall_margin_m, lean_extra_deg, grasp_from_top_m` next to the
  existing `sim.stir.radius_m, turns` (baseline = code value, envelope, origin).
- Policy (`learning/stir/practice.Policy`): hill climb over coverage -> depth -> pace, one key at a time, undo a move
  that does not pay and record its limit, diagnosis rules for failures (no grasp: other grasp height; arm on mug:
  hold higher; slip/push/abort: slower; scrape: more margin; too few loops: one more turn). Every change is a
  skill-config version of target `twin` with reason and evidence (proposal + apply), as twin proposals are.

## 12:45 Smoke run (3 attempts) - two real findings, one bug
1. Baseline: 2.94 loops for 3 commanded turns: the servos lag the streamed circle by ~20 deg. Gate failed; the
   policy's rule added a turn (4 turns: 3.94 loops, stirred, score 0.57).
2. Radius 0.9 -> 1.2 cm: no stirring circle of 1.2 cm is reachable at the straightened lean (19 deg), so the skill
   leaned 4 deg more, which shrank the allowed circle to 0.67 cm and put the handle on the wall 42 % of the stir.
   Bigger is not reachable here; the policy records the limit and moves on.
3. Bug: the stir duration was wall-clock while the twin's counters are sim time, so a run with the twin under
   real time looked like "bowl under the drink 80 %". Durations now come from the twin's clock and every window
   logs its real-time factor (1.0 in these runs).

## 12:50 Full run started
`python -m learning.stir.practice --attempts 16 --validate 5 --write-profile --out data/stir/run1`.
