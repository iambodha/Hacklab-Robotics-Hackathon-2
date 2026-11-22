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

## 2026-11-22 08:42 CET — Codex takeover from Devin
- Read the common brief, both branch commits, the journal, and the end of Devin's quota-exhausted log. The WIP is unreviewed; no commits will be made in this run per the overriding brief.
- Started the full baseline suite with Station disabled. No live services will be contacted.
- Review found an undefined `KitSim.snapshot` call, missing real runner/tests/docs/profile, optimistic metrics for aborted kits, and motion failure paths that could release after failed descent. Fixing these before collecting evidence.
- Keep the fast physics implementation explicit: it is a task-local simulation controller, not evidence that the executor sim backend has passed. Real handoff must use the existing executor.

## 08:45 CET — review fixes and sandbox workaround
- Full baseline here: 278 passed / 1 skipped / 6 failed / 3 errors in 102 s. The additional sim failures/errors are `invalid CoreGraphics connection` (the sandbox cannot create a CGL context); the marker-under-table preflight failure is pre-existing. `nice -n 10` also reports Operation not permitted; continued at the available priority, one training job at a time.
- Added explicit `KIT_RENDERER=cpu`: projects the sheet texture and object top silhouettes from MuJoCo model/data camera extrinsics. It does not render arm occlusion, shadows, or full lighting; all grasp/contact/outcome physics remain MuJoCo. GL remains the default for an unrestricted host.
- Fixed undefined snapshot, output reuse, apply-result checking, count of aborted assignments, and failed lift/descent handling. Rejudge placements at episode end so later knocks cannot leave stale success labels.
- Early pilot shows baseline 50% object placement, with screws closing too high. Pilot output is exploratory, not the final curve (it started before the final judge fixes).
- The original release diagnosis was binary and saturated (40→32 mm still 'bounced'), preventing successive improvements. Added measured drop height as a paired A/B secondary criterion, only with non-decreasing placed count and <=1 mm worsening placed error. This is explicit rule design from pilot evidence, not an LLM policy or autonomous discovery of the learning rule.

## 08:51 CET — verified artifacts and executor boundary
- All initial 8 kitting tests passed, then the executor bridge smoke test passed separately (5.08 s): full box pick/carry/release through executor proposals, final footprint in the assigned cell.
- Added `executor_sim.KittingBackend`: same task MuJoCo model, physics advances on executor sleep, CPU frames, real normalized motor/read/torque-limit contract. No Station client, real-time twin singleton or live service. This handoff check deliberately disables duplicate twin preflight; it does not prove real collision preflight.
- Fixed the print-resolution dashed-line infinite loop using integer dash intervals. Printable PDF now has an exact A4 media box (841.8898×595.2756 pt), lossless artwork, and a 100 mm bar. Rendered the PDF with PyMuPDF and visually checked labels, all four markers, spot outlines and margins.
- Copied Devin's existing 08:23 GL overview into docs as a small reproducible perception fixture; it is explicitly labelled as the predecessor's render, not a fresh render from this sandbox.
- Started reviewed01 with 8 improvement iterations, 3 repeated evaluation kits, 2 paired check seeds, 6 held-out randomized kits (plus 2 nominal). This supersedes the small pilot; raw runs, printouts and scenes are on the external DATA_ROOT.

## 08:54 CET — expanded tests and proposal audit
- 12 kitting tests pass in 10.02 s. They cover spec/scene geometry, print-scale/marker decoding, 3 randomized camera localizations, a physics episode + EpisodeStore + profile flow, aborted-kit metrics, world-frame command conversion, executor full box transfer, saved GL image localization, invalid spec ids, and A/B abort rejection.
- The saved GL image is 640×480, not the calibrated deployment resolution. Its first test incorrectly interpreted those pixels at 299×224 and failed. Fixed the fixture conversion to 299×224; test now passes. `locate_sheet` now refuses resolution mismatches rather than silently returning a wrong pose.
- First post-change full run caught that fixture issue: 288 passed, 7 failed, 1 skipped, 3 errors. After fixing it and adding the A/B regression, started the final reviewed full suite. Baseline failures remain external GL/known preflight failures.
- Hardened A/B checks so any aborted before/after kit refuses a proposal, even if its missing diagnoses look improved. reviewed01 had already imported the previous code; audited every accepted proposal so far (v1–v7): all completed 4/4 intended object trials on BOTH sides, so none benefited from this bug. Repeat audit at completion.

## 08:57 CET — rollback on invalid A/B evidence
- Final full suite: **290 passed, 1 skipped, 6 failed, 3 errors** in 148.33 s. The failure/error identities exactly match the sandbox baseline; all 12 new tests pass in the full suite. Output: DATA_ROOT/r5-kitsim/final-tests-reviewed.txt.
- Audit caught v8 accepting screw grasp 14→13 mm with only 2/4 completed after-trials (one numerical abort). Immediately interrupted the batch. v9 (box release) had also landed in that incomplete iteration, so rolled the whole iteration back to known-valid v7 through the public `skillcfg.revert` flow (new version, no history deletion).
- Added an integrity-fail annotation to the original v8 proposal; preserved the old curve as `curve_before_audit.jsonl`. Retained rows 0–2 and all raw episode/proposal logs. Rerun iteration 3 with the abort guard, no cherry-picking of test seeds.
- Added explicit `--resume`: restores completed consecutive curve rows and rejection step sizes, uses authoritative active config, and retains previous episode IDs/history. Runtime counters include completed earlier row time but omit discarded interrupted checks; wall-clock duration is longer.

## 09:01 CET — repeatability and remaining limits
- Resumed iteration 3 at reverted v10 exactly reproduced the valid v7 evaluation: 9/18 placements, 4.63 mm placed-centre error. The hardened check rejected the numerically aborted 14→13 mm screw candidate. The valid box-release improvement was rechecked and applied as v11.
- New tests still all pass (12/12, 10.19 s) after ensuring every dry-mode gripper command stays fully open even when a learned profile uses partial release.
- Saved GL overview localization (downsampled to 299×224): all ids 10–13, centre error ~0.37 mm, yaw -0.072°, marker residual 1.42 mm, estimated pixel-noise sigma 0.54 mm. Inspected the overlay: cell boxes/spot crosses align. This nominal result does not cover real camera calibration uncertainty.
- Evaluation is not monotone: lower release reduced bounce, but full-kit rates initially went 33%→39%→33%→50%. The per-class A/B check can miss interactions with the rest of a full kit; report this rather than presenting accepted proposals as guaranteed improvements.

## 09:09 CET — evaluation-based model selection
- Eight tuning iterations ended at v13 with 7/18 fixed-evaluation placements (38.9%), worse than the best v11 profile (9/18, 4.53 mm). Exporting the last proposal would knowingly ship an inferior evaluated candidate.
- Interrupted before the learned held-out comparison was observed. Selected the best **evaluation** row, by success then lower placed error, and used public `skillcfg.revert` to restore v11 as v14. Preserved all nine curve rows. Resume adds a separate selected-profile evaluation row (iteration 9), then reruns the held-out comparison.
- Added this model-selection step to the reusable loop so a fresh run also exports the best evaluated profile, records the revert, and re-evaluates it before held-out testing. Added a regression test that prefers success over a deceptively smaller placed-only error and records the revert as a new version.

## 09:13 CET — external drive disconnected
- `/Volumes/Dell HDD` disappeared during held-out evaluation and the final test run. `ls /Volumes` shows only Macintosh HD; `diskutil` is unavailable in this sandbox (DiskManagement framework inaccessible). Asked asynchronously for reconnection.
- Worktree code and copied small images remain safe. Original raw history, selected v14 config, printouts and test logs are temporarily inaccessible. The checked-in-tree profile is still the earlier pilot, so it must NOT be mistaken for the selected final profile.
- Preparing `/tmp/r5-kitsim-rescue` as an authorized writable local fallback. Known selected values from prior console evidence are screw grasp 14 mm, cap release 19 mm, box release 21.5 mm, all other generic values. Any reconstruction will be labelled as such and verified; no missing raw logs will be invented.

## 09:15 CET — local reconstruction and fresh verification
- The held-out process printed selected randomized success **13/36 (36.1%)**, 5.58 mm mean placed error, before its open external-disk log handle failed with EIO. Generic was **9/36 (25%)**, 6.21 mm. No result/profile file could be finalized on that disconnected drive. The simultaneous full test run also exited on disk I/O; do not interpret that as a code regression.
- Reconstructed the selected profile using a new public proposal/apply flow in `/tmp/r5-kitsim-rescue/skillcfg` (fresh version lineage). Values are explicitly documented, not inferred from missing data. Re-running all 16 held-out kits with per-phase EpisodeStore records plus both generic/selected fixed-seed evaluations (6 more kits).
- Saved `historical_curve.json` in the worktree by transcribing completed console counts/errors. It is explicitly labelled as recovered aggregate evidence, NOT original per-episode records or a replacement for the missing raw history. The drive may still contain the original files when reconnected.
- Future A/B checks and held-out kits now record full episode phases as well as aggregate proposal/score data. Hypothetical parameter trials are labelled; their explicit params are authoritative while skillcfg stamps identify the active config at trial time.

## 2026-11-22 10:03 CET — final verified delivery
- Fresh local verification reproduced fixed evaluation 6/18 → 9/18 (5.70 → 4.53 mm); held-out randomized 9/36 → 13/36 (6.21 → 5.58 mm); nominal 6/12 → 12/12, selected completes 2/2 identical reset kits (4.83 mm). No randomized kit completed.
- Archived all 22 fresh recovery episodes (1,072 records, 22 begin/end pairs) as compressed evidence in the worktree. Reconstructed historical aggregates are separately labelled. The candidate profile now contains the selected, freshly reverified parameters, not the pilot.
- Final full suite under local DATA_ROOT: 291 passed, 6 failed, 3 errors, 1 skipped in 158.62 s; all 13 new tests pass and baseline failure identities are unchanged. Final test log retained in /tmp/r5-kitsim-rescue.
- Offline handoff CLI loaded the delivered profile and generated 48 dry commands; all gripper commands remain fully open. No Station/live-robot connection or motion was made.
- Delivered exact-A4 PDF/PNG, profile, measured JSON, plot, images, evidence archive, REAL_RUN checklist and REPORT. No commits. External drive remains unavailable; original tuning audit trail requires reconnection.
