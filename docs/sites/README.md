# Sites: move the table, keep the experiment

A site is a physical table setup around **NormaCore Station**: the camera queue identities, marker calibration,
arm base pose and digital twin belong together. Robot geometry and motor safety remain robot-profile concerns.
The home setup is where the bottle-cap demo was learned on 21–22 November. Office has new USB queue identities
and its own calibration. Switching sites must never reinterpret home episode records as office evidence.

```text
SITE=home (default) / SITE=office
               |
        sites/<name>/site.json          explicit environment overrides win
               |
       sites.get_site() (stdlib; no Station client)
       /             |                  \
 perception       MuJoCo twin       real MCP / recorder / dashboard
       |             |                  |
 site scene, intrinsics, mount     site action log + episode store
               |
     NormaCore Station hardware and existing read-only camera relay
```

The registry is `sites/__init__.py`. Asset paths in `site.json` are relative to the repository root, so another
worktree resolves its own frozen assets. Recording directories are relative to the checkout, preserving the old home recorder default. Action/episode directories are relative to the hackathon root when the
checkout is under `ws/<branch>`; elsewhere they are relative to the checkout. Lookup does not create directories
or contact hardware. Unknown sites fail with the manifest path; missing office outputs never select home assets.
Services select a site at startup: restart them deliberately when changing SITE, with the robot under operator control.

## Switch

Operator runbook (stack restart, verification): [switching.md](switching.md). From the repository root:

```bash
export SITE=office
python -m sites check               # resolved files, cameras, log dirs; non-zero on missing files or stale overrides
python -m sites env                 # inspect resolved defaults; does not launch anything
python -m sites discover            # GET only, from the already-running relay
```

The real MCP operator launcher is `bash scripts_integrator/real_server_site.sh office`. The named
`real_server_home.sh` always selects home. `learning/operator/real_server.sh` uses SITE (home if unset) and this
worktree's files. These launchers start a real server: use them only during an authorized hardware session.
No launcher was executed during this change.

No need to source `site.env`: the manifests are authoritative. The old files remain usable for shell tools,
but sourcing home/site.env exports home camera overrides that will still win after changing SITE. Clear old
camera overrides in your shell when switching, or start a clean shell. `python -m sites env` prints shell-quoted
exports for legacy tools that need environment variables; inspecting it is read-only.

| Setting | Behavior |
| --- | --- |
| `SITE` | `home` if unset; selects `sites/<name>/site.json` |
| `PERCEPTION_{OVERVIEW,WRIST}_QUEUE` | Override the selected site's perception queues |
| `TWIN_{OVERVIEW,WRIST}_QUEUE` | Override twin capture queues |
| `ROBOT_{OVERVIEW,WRIST}_CAM` | Override real MCP camera hash prefixes |
| `RECORDER_CAM_ROLES` | Override recorder hash-prefix-to-role mapping |
| `ROBOT_OVERVIEW_CAL` | Override the real overview camera JSON |
| `ROBOT_REAL_LOG_DIR` | Override action log directory |
| `BOTTLE_LEARN_DIR` | Override episode store |
| `RECORDER_DIR` | Override recording storage directory |
| `EXECUTOR_SCENE_CALIB` | Existing executor-only scene override remains supported |
| `SIM_LAYOUT`, then `ROBOT_SIM_LAYOUT` | Override twin layout; explicit function/CLI argument wins over both |

The default twin now follows the selected home site. `SIM_LAYOUT=venue` explicitly selects the original
synthetic venue. `SIM_LAYOUT=home` and `SIM_LAYOUT=office` resolve registered layouts; existing custom named
layouts and paths still work. Generated models are separated into `sim/generated/site_home` and
`sim/generated/site_office`. Virtual simulator queue IDs remain synthetic; physical capture uses the site queues.

## What stays with a site

| Item | Home | Office |
| --- | --- | --- |
| Scene | `sites/home/scene_calib.json` | `sites/office/scene_calib.json` |
| Twin | `sites/home/layout_home.json` | `sites/office/layout.json` |
| Overview | `sites/home/real_overview_camera.json` | `sites/office/real_overview_camera.json` |
| Actions | `logs/real_sessions_home/actions.jsonl` | `logs/real_sessions_office/actions.jsonl` |
| Episodes | `logs/learning/bottle/` (existing records) | `logs/learning/office/bottle/` |
| Recordings (checkout-relative) | `recordings/` (existing default) | `recordings/office/` |

Home's frozen JSON files match the original calibration/, sim/layouts/home.json and camera_pose assets.
Historical captures, static synthetic benchmark fixtures, learned bottle layouts and archived JSON are evidence:
they keep their recorded identities and are not rewritten. The recorder defaults to the registry's `record_dir` at startup, and retains an explicit `RECORDER_DIR` override.
Existing recordings on the external drive stay put; changing SITE does not reconfigure an already-running recorder.

Office's manifest explicitly reuses home's frozen intrinsics and CAD wrist mount because these are the same
physical cameras and mount on different USB ports. This is separate from the table calibration: office's scene,
arm base, layout and overview extrinsics are never shared. If lenses, resolution or mount change, make site-local
copies and update the manifest before recalibrating intrinsics. Never calibrate into the shared frozen inputs.
The legacy layout token `camera_pose/wrist_camera.json` is resolved through the selected site's mount by the scene
builder, for compatibility with previous scene-contract producers. Explicit custom pose paths remain supported.

## Add a new site

1. **Plug in and identify.** Stop robot motion under operator control; plug cameras into their final ports. Keep the
   existing NormaCore Station and relay running. Run `python -m sites discover` (optional `--relay http://host:8890`).
   It lists live queues from GET `/api/cameras`, registered roles and snapshot links. For new queues it can use Pillow
   to compare bottom-edge symmetry: the wrist usually has two jaws at the bottom, the overview sees the table.
   These are tentative visual hints, not identity guarantees; verify the snapshots. No Station clients are opened.
2. **Create a folder and manifest.** Copy home/site.json to `sites/<name>/site.json`. Replace both full queue hashes,
   the three scene/layout/overview paths, and action/episode/record directory suffixes. Copy the appropriate wrist
   mount and intrinsics to the new folder and point the manifest there. Share these only when the hardware and optics
   are unchanged; treat shared files as read-only. Keep an optional site.env for external shell tools.
3. **Place markers.** Fix the arm base and markers on the table. Record sizes, IDs, orientation, table edges and arm
   base measurements. Use a site-local copy of `perception/scene_config.json` via `--config`; do not reuse the old
   table's foot or tape measurements. Record reference photographs.
4. **Calibrate.** Set `SITE=<name>`. In an authorized hardware session, run
   `python -m perception.calib_scene --live --frames 90 --config sites/<name>/scene_config.json`.
   Defaults write the site's scene_calib.json. Offline captured frames also work with `--frames-dir <capture>`.
   Check reprojection errors, held-out markers and wrist/base agreement before accepting. Keep previous files or
   create a new site revision if the same table is physically rearranged.
5. **Sync the twin.** Seed `sites/<name>/layout.json` with a copy of the appropriate layout (geometry and objects
   appropriate to this table), then run `SITE=<name> python sim/sync_from_calib.py`. It reads the site's scene and
   writes the site's layout. `SITE=<name> python sim/build_scene.py` rebuilds a separate model. Produce the matching
   `real_overview_camera.json` from those calibrated extrinsics (the office calibration task supplies office's file).
   Estimation with `sim/estimate_home.py` also follows SITE but remains an estimate; it must not replace stronger
   calibration. Its intermediate results go to `sites/<name>/estimate/`.
6. **Check before operation.** Inspect `python -m sites env`, compare relay snapshots against roles, compare twin
   renders with the real overview, and check marker reprojection and wrist-camera pose. Verify action and episode
   paths differ from home. Perform any physical checks only in an explicitly authorized hardware session.

## Validation and limitations

Offline tests live in `sites/tests/test_registry.py`: frozen home JSON equality, default home IDs and record paths,
office queue/path selection, all camera override families, log/calibration overrides, invalid site names, layout
precedence, generated model separation and mocked discovery HTTP calls. Existing tests also cover recorder roles,
calibration and connector contracts. The journal records exact test results and environment limitations.

Discovery was checked against the existing relay on 22 November: two live office queues, both 299×224 at about
28 fps. Matching registered queues is metadata confirmation; new-camera bottom-edge scores are only a heuristic.
A table edge or clutter can resemble jaws, and occluded jaws may be missed. Visual confirmation remains necessary.
The registry cannot create an office calibration that is still being computed: until r6-office outputs are merged,
office perception and twin startup will fail clearly on those missing files.
