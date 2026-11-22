# r6-sites journal

## 2026-11-22 — initial inspection
- Read r5-common.md with round 6 substitutions. No live Station clients or real-arm server will be started. Git changes remain uncommitted as requested.
- Started the exact shared baseline test suite with STATION_HOST disabled.
- Found frozen home calibration, layout, overview camera, wrist mount and intrinsics; office currently contains only site.env. Office calibration outputs are owned by r6-office.
- Registry will read JSON without loading site.env into the process, so explicit environment overrides retain precedence. Paths in manifests are repository-relative; log paths are relative to the hackathon data root (existing parent of ws worktrees).
- Office uses the existing frozen camera intrinsics and CAD wrist mount explicitly: same camera hardware, new USB queues. Its scene/layout/overview files will never fall back to home when missing.

## 2026-11-22 — registry and callers
- Added stdlib `sites.get_site()` with JSON manifests, no Station imports, no directory creation on lookup and no silent office-to-home scene fallback.
- Added office/site.json as explicitly required by build step 1; no office calibration outputs or site.env were modified.
- Routed perception, twin capture, real camera prefixes, recorder roles, dashboard camera list, benchmark and trace bundle through the registry. Preserved each named environment override.
- Routed episode defaults and dashboard log readers through the registry. Added two worktree-local launchers (scripts_integrator did not exist here) and removed overrides that pinned learning/operator to home.
- Extended the small path changes to executor/kinematics.py, dashboard/data.py, sim/robot_profile.py, sim/operator_ui.py and camera_pose/elrobot_camera.py: otherwise calibration transforms, log readers and real FK would keep selecting the previous setup.
- Site layouts build into sim/generated/site_<name>, preventing home/office layout.json basename collisions. Explicit SIM_LAYOUT / ROBOT_SIM_LAYOUT and explicit paths still win; explicit venue retains the synthetic venue scene. Default layout selection now follows the home site as requested.
- Retained the legacy `camera_pose/wrist_camera.json` layout token, which build_scene now resolves through the registry; old calibration producers and the existing wrist-mount contract test require this token. Custom pose_file paths still work.
- Baseline shared suite: 303 passed, 7 failed, 8 errors, 1 skipped (162.76 s). Failures include CoreGraphics rendering, the known table-hit preflight check, and kitting writes blocked outside this worktree's writable roots. No robot was accessed.
- First targeted run: 27 passed, 1 wrist token regression; corrected the source-token compatibility branch rather than changing the existing test.

## 2026-11-22 — comparison and regression fixes
- Read-only discovery against http://127.0.0.1:8890/api/cameras succeeded: exactly two live office queues, 299x224, 28.4 fps. No Station connection or service lifecycle action occurred.
- Full comparison run: 314 passed, 9 failed, 8 errors, 1 skipped (97.17 s), including all 13 new registry tests passing. Two added failures identified: frozen layout basename displayed as layout_home instead of home; skills loop's minimal test repo did not copy the new sites dependency.
- Corrected executor/twin.py to report the site name and use SITE for its default preflight (it otherwise hard-coded home even with SITE=office). Added the sites package to the existing skills loop test fixture's minimal-repo copy. These are narrow integration changes required by the registry dependency.
- Calib_scene's output defaults were already routed through perception.scene; additionally changed its wrist JSON reader and source label to the site mount. Real overview errors now name the resolved missing file.
- Added the mount as a simulator scene-build dependency; preserved explicit custom workspace XMLs in robot profiles.

## 2026-11-22T10:17+01:00 — recording isolation and validation
- Broader regression run passed: 175 tests across sites, perception, skills, recorder, dashboard and Generaloid (47.63 s).
- Wired the recorder/store.py startup default to the registry too. Kept the existing home checkout-relative `recordings/` default, separated office into `recordings/office/`, and preserved RECORDER_DIR. This small extra caller change closes the record-directory requirement; no recorder service was restarted.
- Added a subprocess integration test selecting SITE=office once and checking perception, twin capture, recorder roles, dashboard paths, episode defaults and ElRobot connector cameras/layout together without importing a real MCP server.
- Documented the full workflow, architecture, per-site/shared assets, site.env override pitfalls, discovery limitations and operator-only calibration/launcher commands.
