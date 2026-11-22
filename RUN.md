# Running the demo: "put the blue object in the bin"

One process moves the arm: the **executor**. Advisors (scripted planner, Claude, Gemini) only propose skills.
The executor validates each proposal, preflights it, runs it in phases and has an independent check report the
outcome. The console shows every one of those steps.

## 1. Start

```
scripts/run-station.sh            # NormaCore Station, if it is not running yet (or: scripts/stack.sh up --station)
scripts/stack.sh up               # recorder :8900, console :8910 (MuJoCo twin), twin mirror :8892
scripts/stack.sh status           # what is up, with pids and URLs
```

`up` is idempotent: anything already listening is left alone. Logs are in `logs/<service>.log`. Recordings go to
`RECORDER_DIR` (default `/Volumes/Dell HDD/Robotics-Hackathon/recordings`, `.../recordings/office` for the office).
The twin uses `SIM_LAYOUT` (default: the site's layout). `scripts/stack.sh down` stops only what the script started.

### Switching to the office table

The table is selected by `SITE` (`home` if unset): calibration, twin layout, overview camera, camera queues, action
logs and recordings all follow it. Services read it at startup, so restart them:

```
scripts/stack.sh down
SITE=office python3 -m sites check        # read-only: resolved office files, cameras, log dirs; flags stale overrides
scripts/stack.sh up --site office         # same as SITE=office scripts/stack.sh up
curl -s http://127.0.0.1:8910/api/scene | python3 -m json.tool | grep -E '"(site|scene_calib|layout)"'
```

The last line must show `"site": "office"` and `sites/office/scene_calib.json`. Do not keep home's camera variables
(from `sites/home/site.env`) in the shell: they override the site. Details: `docs/sites/switching.md`.

## 2. Open the console

http://127.0.0.1:8910 (loopback only, no build step)

- **Header pills:** Station, recorder, camera fps, arm torque and calibration, backend, advisor, setup id.
- **Task:** prompt, advisor (Scripted, Claude, Gemini or Manual), backend (Fake, Sim twin or Real arm), Start, Stop
  and Cancel motion.
- **Live view:** both cameras. The overview has an overlay with the world grid (5 cm), axes, reach limit,
  ArUco markers, perception's estimates (blue object, bin) and the planned TCP path.
- **Robot logic:** the advisor's prompts, its messages and tool calls, and one card per proposal. Each card shows
  rationale → validation checks with their numbers → twin preflight → phases with residuals → independent outcome.
  The strip above the cards is the run at a glance: pick → place → park.
- **Twin and recordings:** the live twin mirror, plus the recorder viewer for this run.

## 3. Run the task in the twin

In the console: Backend **Sim twin**, Advisor **Scripted**, then **Start task**. From a shell:

```
python3 -m executor task "Put the blue object in the bin" --backend sim --advisor scripted
python3 -m executor task "Put the blue object in the bin" --backend sim --advisor claude   # Claude Code headless
```

The task prints the same logic as the console. It exits with 0 when the advisor reports DONE. Ctrl-C stops the
task; the arm holds.

Other commands:

- `python3 -m executor status`: everything in the header.
- `observe`: objects in world coordinates.
- `skills`
- `runs`: recordings and executions with their outcomes.
- `cancel`

`--backend fake` runs the whole flow with simulated motors and synthetic estimates, without MuJoCo. It is a quick
check that everything is wired.

## 4. Then the real arm

Someone stands at the arm. Follow `tasks/blue_to_bin/REAL_RUN.md` for the dry-run order and the abort criteria.

```
scripts/stack.sh down && scripts/stack.sh up --backend real --real-motion
```

1. In the console, choose Backend **Real arm**. It is read-only until you tick **Arm enabled**. The Backend pill
   then reads ARMED, in red. Without `--real-motion`, the toggle is refused.
2. Check the pills: Station up, both cameras at about 30 fps, arm torque on and calibrated.
3. Check that Perception lists the blue object and the bin where they are on the table, and that the overlay sits
   on the markers.
4. Start the task. **Cancel motion** (or `python3 -m executor cancel`) stops commands at once; the arm holds.
   **Stop task** also ends the advisor.

From a shell: `python3 -m executor task "Put the blue object in the bin" --backend real --arm-enabled`.
