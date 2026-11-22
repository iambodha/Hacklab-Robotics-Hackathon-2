# r5-kitsim journal

Kitting task trained in the twin (branch `r5-kitsim`). Newest entries at the bottom. Times are local (CET).

## 2026-11-22 08:15 start
- Read `prompts/r5-common.md`, `prompts/r5-kitsim.md`, `prompts/r5-skills.md` (the agent that owns `learning/`).
- Baseline test run started (full suite of r5-common, `nice -n 10`).
- Surveyed: `sim/build_scene.py` (layout -> MJCF; textures only for the table and the markers), `sim/sim_arm_mcp.py`
  (real-time physics thread, grip force = fraction of the 0.4 N m jaw torque), `executor/` (skills
  `move_tcp`/`gripper`/`pick`/`place`, one proposal per motion), `tasks/blue_to_bin/` (the pick-and-place task that ran
  on the real arm: Runner, `descend_guarded`, REAL_RUN.md, the 3.5 mm pad fix), `learning/episodes.py` and
  `learning/skillcfg.py`.

### Decisions
1. **Learning episodes run in a fast headless MuJoCo loop, not through the real-time twin server.** The executor on
   the twin takes ~52 s per pick-and-place in real time (`tasks/blue_to_bin/results/sim_scripted.json`, median 51.8 s).
   A learning curve needs hundreds of picks. The kitting env loads the same `workspace.xml` (same URDF, servos,
   contact pads, cameras), uses the same IK (`build_scene.ArmIK.solve_least_lean`, which reads the shared skill
   config) and steps physics as fast as the CPU allows. The executor sim backend is used once, for the sim-to-real
   handoff check (the real runner on the twin).
2. **`learning/` is used, not edited.** `episodes.EpisodeStore.begin` validates against the bottle's `PARAMS` and
   `OUTCOMES`, and `skillcfg.registry()` is closed (unknown keys are refused). Kitting therefore:
   - subclasses `EpisodeStore` (same JSONL record schema `episode_begin | phase | judge | episode_end`, same
     `attempt_N/report.md`, `SUMMARY.md`) with kitting params / phases / outcomes;
   - registers its keys `kit.<class>.<param>` into the skillcfg registry at runtime (`tasks/kitting/profile.py`), then
     uses the real `propose / update_proposal / apply / reject / revert / active` flow in its own `SKILLCFG_DIR`
     (`DATA_ROOT/r5-kitsim/skillcfg`), so the bottle config is never touched. Merge note: an official
     `skillcfg.register_keys()` hook would replace this.
3. **Objects start on printed start spots on the sheet.** At 299x224 the overview camera sees a 4 mm screw as ~1.3 px
   wide, so it cannot localise it. Spots make object poses follow from the located sheet alone. The human placement
   error is part of the training randomization (sigma 3 mm, 6 deg).
4. **Pad fix in the kitting twin only.** The stock twin's jaws close to an 8 mm gap and cannot hold a 4 mm screw;
   `sim/kitting_scene.py` moves both pads 3.5 mm in (`tasks/blue_to_bin/sim_trials.PAD_FIX_M`, the change that task
   proposed) on its own XML. `sim/build_scene.py` is not edited.

## 08:20 sheet
- `tasks/kitting/kit_a4.json`: A4 landscape, markers DICT_4X4_50 ids 10-13 (50 mm, the corners), 6 target cells
  (42 mm, 3 x 2, far half), 6 start spots (near half), 2 screws (M4 x 25), 2 caps (30 x 12 mm), 2 boxes (30 x 20 x 15).
- `tasks/kitting/sheet.py` prints 3508 x 2480 px at 300 dpi -> PDF page 297.01 x 209.97 mm.

## 08:35 twin scene + sheet localisation
- `sim/kitting_scene.py` builds the twin; objects settle within 1.4 mm.
- `tasks/kitting/locate.py`: markers -> rays -> table plane -> 2D rigid fit. On the 299x224 twin overview the sheet is
  ~85 px wide and a marker 8-15 px. One `perception.aruco.detect` decode found only marker 10 (of 4); the set found
  changed with the interpolation (3x cubic: {10}; 3x linear: {10, 11}; 4x lanczos: {10, 11, 13}). So locate decodes
  8 times (3/4/6x, cubic/linear/lanczos) and takes the union, then refines the corners at full resolution.
- Three sheet poses: position error 0.4 / 0.6 / 0.2 mm, yaw error 0.09 / 0.03 / 0.07 deg, 2-3 markers found (the two
  near the arm are partly hidden by the objects and the folded arm).
