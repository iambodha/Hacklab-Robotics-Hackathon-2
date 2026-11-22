# Switching to the office table

The arm now sits at the office table. One setting selects it everywhere: `SITE=office` (unset means `home`, the
table where the bottle cap was learned). Every service reads the site **once, at startup**, so switching means
stopping the stack, setting SITE, and starting it again with the robot under operator control.

## What to set

```bash
export SITE=office
```

Nothing else. Do not source `sites/home/site.env` (or an old shell that did): its camera variables
(`PERCEPTION_*_QUEUE`, `TWIN_*_QUEUE`, `ROBOT_*_CAM`, `RECORDER_CAM_ROLES`) override the site and would point the
stack at the home cameras. The same holds for `SIM_LAYOUT`/`ROBOT_SIM_LAYOUT=home`, `ROBOT_OVERVIEW_CAL`,
`EXECUTOR_SCENE_CALIB`, `ROBOT_REAL_LOG_DIR`, `BOTTLE_LEARN_DIR` and `RECORDER_DIR` left over from home work.
Start a clean shell if unsure; `python -m sites check` (below) flags these leftovers.

## What to start

```bash
scripts/stack.sh down                     # services started for home keep home's calibration until restarted
scripts/stack.sh up --site office         # or: SITE=office scripts/stack.sh up
scripts/stack.sh status                   # each service it started shows SITE=office
```

`stack.sh` exports SITE to the recorder, the executor console (perception, kinematics `T_world_base`, twin
preflight) and the twin mirror. The console's twin and the mirror use the office layout; recordings go to
`$DATA_ROOT/recordings/office`. If a service is already running for another site, `up` leaves it alone and prints
a warning.

The real-arm MCP server is started separately (authorized hardware sessions only):
`bash scripts_integrator/real_server_site.sh office`. It reads the office overview camera extrinsics
(`sites/office/real_overview_camera.json`), camera ids and writes `logs/real_sessions_office/actions.jsonl`.

## How to verify the office calibration is in use

Before starting anything (read-only, no Station or camera connections):

```bash
SITE=office python3 -m sites check
```

It prints every resolved file (scene calibration with its `created` stamp, twin layout, overview camera, wrist
mount, intrinsics), the camera queues each consumer will open, and the log/episode/recording directories. It exits
non-zero when a site file is missing or when an environment override still points at another site's files or cameras.

With the stack up:

```bash
curl -s http://127.0.0.1:8910/api/scene | python3 -m json.tool | grep -E '"(site|scene_calib|layout)"'
```

`site` must be `office`, `scene_calib` must end in `sites/office/scene_calib.json`, and `layout` is the twin the
console uses. In the console, the overview overlay (markers, world grid) must sit on the office markers; in the
twin mirror (http://127.0.0.1:8892), the twin's overview view must match the real overview camera.

## Site-specific task data

`tasks/blue_to_bin` keeps one measurement of the lid and bin per site: `measured.json` for home,
`measured_office.json` for office (`python -m tasks.blue_to_bin.measure` writes the selected site's file). Until
the office file exists, the task uses home's object sizes and warns that the bin position comes from home; the
real-arm task server refuses to start in that state. Measure the bin on the office table first.

## Switching back

`scripts/stack.sh down`, `unset SITE` (or `export SITE=home`), `scripts/stack.sh up`. Home outputs are unchanged.
