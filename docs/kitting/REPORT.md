# REPORT r5-kitsim

Status: implementation and simulation evidence delivered; robust randomized transfer remains unqualified. No live
robot/Station services were contacted. No commits were made; all changes remain in the working tree for integration.
The branch's existing Devin commits (3cd8654, WIP e949a11) were reviewed and finished, with takeover in JOURNAL.

## Verified outcomes

| condition | generic | selected | placed centre error generic → selected |
|---|---:|---:|---:|
| fixed randomized evaluation | 6/18 | 9/18 | 5.70 → 4.53 mm |
| held-out randomized, 6 kits | 9/36 | 13/36 | 6.21 → 5.58 mm |
| nominal, 2 identical reset kits | 6/12 | 12/12, 2 complete kits | 5.20 → 4.83 mm |

No randomized kit completed. Errors above are conditional on placement. Sample sizes are small. The exported
profile is selected using fixed evaluation success (ties: placed error), not held-out results. Its changed values are
screw grasp 14 mm, cap release 19 mm, box release 21.5 mm; other class values stay generic.

Verified exact-A4 lossless PDF/PNG, all four marker IDs, three randomized cell-localization cases, saved GL image
localization, physical box transfer, EpisodeStore/profile flows, aborted-score handling, config selection/revert,
world-frame conversion and a full box transfer through the executor simulation backend. Offline real-run planning
loaded the exported profile with Station disabled and produced 48 fully-open dry commands.

## Tests

Baseline: **278 passed, 6 failed, 3 errors, 1 skipped** (102.26 s).
Final: **291 passed, 6 failed, 3 errors, 1 skipped** (158.62 s). All **13 new tests pass**; no new failure identity.
The full final log is `/tmp/r5-kitsim-rescue/final-tests.txt`.

Unchanged failures: three `executor/tests/test_sim_wiring.py` GL cases; marker-under-table preflight in
`executor/tests/test_wiring.py`; two GL-dependent `learning/tests/test_sim_loop.py` cases. The three setup errors are
also the existing bottle-loop fixture failing to create a CoreGraphics context. One intermediate run caught a new
GL-fixture resolution mistake; it was fixed and the test passes in the final suite.

## Review fixes and limitations

Fixed undefined snapshot, a print-resolution infinite dash loop, resolution mismatch acceptance, duplicate spec ids,
failed-motion paths, stale final placement judgments, aborted-kit score denominators, A/B acceptance after numerical
aborts, ignored config-apply errors and inferior-last-profile export. Numerical warnings invalidate episodes. One
invalid running-batch proposal was identified, annotated and reverted through public skillcfg; superseded history
was retained, not silently erased. See JOURNAL for the complete sequence.

The fast trainer is a task-local MuJoCo controller. The executor simulation bridge was checked for one box; this is
not complete controller equivalence or a full executor-based training run. CPU visual projection replaces unavailable
GL during training and omits arm occlusion/shadows. The saved GL fixture independently checks perception. Start spots
on the sheet replace arbitrary loose-object localization. Randomized success is still weak; follow REAL_RUN's
supervised single-cap checks before any real kit. No optional cloud operator was added.

## Sandbox and storage

`nice` priority changes, CGL rendering, DiskManagement and default font-cache writes were unavailable. Used normal
available priority, explicit CPU rendering, writable temporary font caches and local fallback storage. The external
Dell drive disconnected during evaluation and the later test run; original raw training logs/configs are temporarily
inaccessible. Historical aggregates were transcribed from completed console output and labelled. Selected values and
all final scores were freshly reproduced locally; **22 fresh episodes / 1,072 records** are preserved in
`docs/kitting/evidence/recovery_episodes.jsonl.gz`. Print artifacts, profile, graphs and measurements are in the worktree.

## Files and merge notes

Modified: `sim/kitting_scene.py`, `sim/layouts/kit_home.json`, and the existing kitting `sheet`, `spec`, `locate`, `plan`,
`profile`, `sim_env`, `learn` modules. Added: kitting `cpu_render`, `executor_sim`, `run_real`, `report`, REAL_RUN, print
artifacts, candidate profile; `tasks/tests/test_kitting.py`; all kitting docs/images/evidence/measurements. Only owned
paths were edited. `sim/build_scene.py`, `learning/` and existing tests are unchanged.

Add `--with reportlab` to the common uv test dependency list (PDF generation). Matplotlib is only needed to regenerate
the report. Reconcile the task's mutable public `skillcfg.registry()` extension with r5-skills' registration API; the
kitting schema is task-local and never edits bottle config. Keep the 3.5 mm inward / 6 mm pad-tip corrections isolated
until physical pad measurements confirm them. Remaining question: can real camera/base calibration and fingertip
geometry meet the small cells' tolerances? Reconnect the Dell drive to recover the original full tuning audit trail.
