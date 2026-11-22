# Local path inventory

The initial shipping-set audit found **40 occurrences across 31 files** of `/Users/bodha/...` or `/Volumes/Dell HDD/...`. This inventory covers every affected file, including historical evidence. We do not rewrite provenance or r6-owned implementation/launchers. Line numbers describe the inspected baseline, not a stable API. The release wrappers set portable paths explicitly.

For a new source checkout use `DATA_ROOT="$PWD/.local/data"`, `NORMA_CORE_REPO="$PWD/norma-core"`, `SKILLCFG_DIR="$DATA_ROOT/skillcfg"`, `BOTTLE_LEARN_DIR="$DATA_ROOT/bottle"`, `BOTTLE_SIM_LEARN_DIR="$DATA_ROOT/bottle-sim"`, `ACTIONS_LOG="$DATA_ROOT/actions.jsonl"`, and `RECORDER_DIR="$DATA_ROOT/recordings"`. These overrides do not relocate existing operator data.

| File (baseline lines) | Classification / portable use |
| --- | --- |
| `RUN.md` (16) | Historical example/default, local-only as written. Substitute your own NORMA_CORE_REPO / DATA_ROOT / checkout path; follow the release quickstart. |
| `dashboard/data.py` (14) | Local history defaults; override BOTTLE_LEARN_DIR and ACTIONS_LOG (optional OPERATOR_LOG). |
| `docs/comms/evidence/README.md` (26) | Historical run/evidence location, local-only; small curated evidence is shipped, full external recordings/logs are not. |
| `docs/comms/evidence/generaloid_demo_timeline.txt` (17) | Historical run/evidence location, local-only; small curated evidence is shipped, full external recordings/logs are not. |
| `docs/comms/evidence/twin_demo_timeline.txt` (17) | Historical run/evidence location, local-only; small curated evidence is shipped, full external recordings/logs are not. |
| `docs/generaloid/CLI.md` (7, 8) | Historical example/default, local-only as written. Substitute your own NORMA_CORE_REPO / DATA_ROOT / checkout path; follow the release quickstart. |
| `docs/generaloid/JOURNAL.md` (12, 77) | Historical run/evidence location, local-only; small curated evidence is shipped, full external recordings/logs are not. |
| `docs/generaloid/VALIDATION.md` (33) | Historical run/evidence location, local-only; small curated evidence is shipped, full external recordings/logs are not. |
| `docs/kitting/README.md` (53, 103) | Historical example/default, local-only as written. Substitute your own NORMA_CORE_REPO / DATA_ROOT / checkout path; follow the release quickstart. |
| `docs/kitting/evidence/offline_plan.json` (7) | Historical source/provenance or result path, local-only; not a file to fetch on a fresh machine. Generate new artifacts for your site. |
| `docs/learning/VALIDATION.md` (33) | Historical run/evidence location, local-only; small curated evidence is shipped, full external recordings/logs are not. |
| `learning/episodes.py` (23) | Local default; override BOTTLE_LEARN_DIR. Quickstart/checks use isolated stores. |
| `learning/operator/.devin/mcp_config.json` (5) | Historical MCP client config; local-only. Replace command path for your deployment. |
| `learning/operator/real_server.sh` (4) | Historical operator launcher; local-only. Do not invoke on a new machine; use reviewed deployment settings. |
| `learning/skillcfg.py` (38) | Local default; override SKILLCFG_DIR. Keep real/twin state separate. |
| `perception/README.md` (17) | Historical example/default, local-only as written. Substitute your own NORMA_CORE_REPO / DATA_ROOT / checkout path; follow the release quickstart. |
| `scripts/stack.sh` (18) | DATA_ROOT override; stack includes live-facing services and is not the offline quickstart. |
| `sim/kitting_scene.py` (136, 150) | Set DATA_ROOT or --out to a writable data directory. |
| `sim/layouts/home.json` (27, 205) | Historical source/provenance or result path, local-only; not a file to fetch on a fresh machine. Generate new artifacts for your site. |
| `sim/layouts/kit_home.json` (47, 713) | Historical source/provenance or result path, local-only; not a file to fetch on a fresh machine. Generate new artifacts for your site. |
| `sim/real_compare/20261122-004222/metrics.json` (2) | Historical source/provenance or result path, local-only; not a file to fetch on a fresh machine. Generate new artifacts for your site. |
| `sim/real_compare/20261122-005053/metrics.json` (2) | Historical source/provenance or result path, local-only; not a file to fetch on a fresh machine. Generate new artifacts for your site. |
| `sim/splat/README.md` (5) | Historical example/default, local-only as written. Substitute your own NORMA_CORE_REPO / DATA_ROOT / checkout path; follow the release quickstart. |
| `sim/splat/prepare.sh` (20) | Set DATA_ROOT; optional SPLAT_BRUSH selects your Brush binary. |
| `sim/splat/synth_video.py` (22) | Set DATA_ROOT to a writable data directory. |
| `sites/home/layout_home.json` (27, 205) | Historical source/provenance or result path, local-only; not a file to fetch on a fresh machine. Generate new artifacts for your site. |
| `tasks/blue_to_bin/REAL_RUN.md` (50, 51) | Historical example/default, local-only as written. Substitute your own NORMA_CORE_REPO / DATA_ROOT / checkout path; follow the release quickstart. |
| `tasks/blue_to_bin/results/sim_scripted.json` (71) | Historical source/provenance or result path, local-only; not a file to fetch on a fresh machine. Generate new artifacts for your site. |
| `tasks/blue_to_bin/results/sim_scripted_no_padfix.json` (77) | Historical source/provenance or result path, local-only; not a file to fetch on a fresh machine. Generate new artifacts for your site. |
| `tasks/kitting/REAL_RUN.md` (23, 27) | Historical example/default, local-only as written. Substitute your own NORMA_CORE_REPO / DATA_ROOT / checkout path; follow the release quickstart. |
| `tasks/kitting/learn.py` (33) | Set DATA_ROOT and explicit --out for a fresh experiment. |

The additional `/mnt/d/...`, Windows drive-letter paths, worktree-relative paths and camera hashes in older scripts/configs are also deployment-local. NormaCore finders have environment/repo-relative fallbacks; set NORMA_CORE_REPO explicitly. Files such as sim comparison JSON and layout metadata retain their creation source even when that source is unavailable. They are not prerequisites for simulation.

Do not copy old MCP client launcher paths or real-arm settings onto a different table. The protected real entry points stay in place; scripts_integrator/real_server_home.sh and real_operator/.devin/mcp_config.json may live in the external hackathon workspace rather than the files shipped by git. The integrator must inspect those externally before deployment. The r6 site registry will centralize home/office camera IDs and calibration paths; this release does not preempt it.
