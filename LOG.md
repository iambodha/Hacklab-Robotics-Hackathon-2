# NormaCore track — working log

Hackathon: 2nd edition Berlin Robotics & AI Hackathon, 21–22 Nov 2026.
Track: "AI-Powered Robot Control — integrate the Station API into Codex or Claude so it can
control the robot by looking through the cameras."

All times below are local (CET) on 21 Nov 2026.

## Setup

To reproduce on another Mac (Apple Silicon), from this repo's root:

```bash
git clone --depth 1 https://github.com/norma-core/norma-core.git
mkdir station-bin station-run frames
curl -L -o station-bin/station.zip https://github.com/norma-core/norma-core/releases/download/v0.1.0-beta.10/station-macos-arm64.zip
unzip station-bin/station.zip -d station-bin
scripts/run-station.sh
```

Those four folders are gitignored. `station.yaml.example` is the config Station generated here.
The bus serials and camera queue hashes below are specific to this hardware.

On the machine this log was written on:

- Station v0.1.0-beta.10 (macOS ARM64) in `station-bin/`; config, data and log in `station-run/`.
- Start it with `scripts/run-station.sh` (loopback only; add `--lan` to expose it).
- Web UI http://localhost:8889, TCP API `localhost:8888`.
- `norma-core/` is a shallow clone without `hardware/`. The Python client is pure stdlib.

## Hardware

| Thing | ID | Notes |
|---|---|---|
| Leader arm | bus `5B61035034` | 7.6 V, 8 motors, moved by hand |
| Follower arm | bus `5B61033301` | 12.4 V, 8 motors, the one Claude drives |
| Overview camera | queue `usbvideo/5cd55831e21692d4f05fcaec75c351a5` | macOS ID `0x11200002c7f4a60`, looks across the table at the arm |
| Gripper camera | queue `usbvideo/e091eeda8dcda172482f039057691cdb` | macOS ID `0x11110002c7f4a60`, on the follower, looks down past the jaws |

- Leader/follower roles were inferred from supply voltage and confirmed by teleop working.
- A third camera never enumerated on USB.
- Both cameras capture at 320x240 (higher formats failed with "Failed to create capture session").

## Scripts (run from the project root with `python3`)

| Script | What it does |
|---|---|
| `scripts/camera.py` | Saves the latest frame from every camera queue to `frames/<first 8 chars of queue>.jpg` |
| `scripts/arm_state.py [seconds]` | Prints position, goal, torque, current, voltage, temperature and calibrated range for every motor on both buses |
| `scripts/teleop.py start\|stop [LEADER FOLLOWER]` | Starts or stops leader-to-follower mirroring |
| `scripts/calibrate.py start\|stop\|reset\|freeze BUS` | Auto-calibration start/stop; `reset` clears ranges for a manual sweep; `freeze` locks them |
| `scripts/calibrate_manual.py [BUS]` | Guided manual calibration: reset (with retries), you sweep by hand, freeze, compare spans. Defaults to the leader |
| `scripts/scene_cam.py [camera_id] [port]` | Live browser view of a camera at http://127.0.0.1:8890, relayed from Station |
| `scripts/run-station.sh [--lan]` | Starts Station |
| `scripts/grasp.py step J DEG [J DEG ...]` | **Main control tool.** Moves joints by degrees (servo direction), closed loop, aborts on contact; prints where the object moved in the gripper camera |
| `scripts/grasp.py goto J STEPS [J STEPS ...]` | Same, to absolute servo positions |
| `scripts/grasp.py probe J DEG ...` | Moves, reports the object's image shift, moves back |
| `scripts/arm.py pose \| hold \| grip 0..1 \| relax` | Read pose; hold current position with torque on; set gripper opening; torque off (arm drops!) |
| `scripts/vision.py` | Finds dark blobs (the object) in the gripper camera; prints centre, area, bbox |
| `scripts/record_demo.py [secs] [name]` | Records a teleop demo: follower joints at 10 Hz + both cameras at 2 Hz into `demos/<name>/` |

### `scripts/camera.py` — how camera frames come out of Station

Written 11:59. This is the "eyes" half of the track deliverable.

- Follow the queue `usbvideo/<hash>` with the Python client (`client.follow`).
- Each entry is a `usbvideo.RxEnvelope`; keep only those with type `ET_FRAMES`.
- `envelope.frames` is a `FramesPack`: `format` (width, height, kind) plus `frames_data`.
- Kind was `FF_JPEG` on both cameras, so `frames_data[-1]` is a ready JPEG. The script also handles raw `FF_NCHW` by writing a PNG, but that path is untested.
- Frames arrive at 299x224 because `resize_target: 224` in `station-run/station.yaml`. Raise it for sharper frames (needs a Station restart).
- The queue hashes are read from `station-run/station.log`; pass hashes as arguments to grab specific cameras.

First result: both cameras showed a small black cylindrical object on the white table, roughly
centred and a short distance in front of the open gripper jaws. A second similar object was near
the bottom-right of the overview frame.

## Calibration — what happened

- Follower: auto-calibration is repeatable (identical ranges on two runs). First attempt was stopped by hand because the arm hit the table.
- Leader: auto-calibration is not reliable. Joint 5's span came out as 2000, 1664 and 1350 steps on three runs, and it ended outside its own range each time.
- Fix that worked: manual calibration of the leader. `calibrate.py reset`, sweep every joint by hand to both ends, then `calibrate.py freeze`.
- `reset` half-applied on the first try (bus timeouts / motor ID mismatches). Re-run it and check `arm_state.py` shows `frozen False` on all 8 motors.
- After the manual sweep, joints 1–6 match the follower's span within about 4%. Joint 7 leader span is about 5% larger; joint 8 (gripper) leader span is about 25% smaller.

### To redo it

1. `python3 scripts/teleop.py stop`
2. Follower: `python3 scripts/calibrate.py start 5B61033301` (auto, about 2 minutes, needs clearance below table level).
3. Leader: `python3 scripts/calibrate_manual.py` and follow the prompts.
4. `python3 scripts/teleop.py start`

`calibrate_manual.py` packages the steps that were run by hand on 21 Nov. Its state-reading part
was run against the live arms; the full reset/sweep/freeze flow has not been run through the
script itself, because that would wipe the working calibration.

A snapshot of the working ranges is in `calibration/calibration-2026-11-21.txt`.

Calibrated ranges after the manual sweep (steps):

| Joint | Leader min–max | Follower min–max |
|---|---|---|
| 1 | 1079–3017 | 1066–3030 |
| 2 | 1001–3095 | 982–3114 |
| 3 | 905–3191 | 885–3211 |
| 4 | 900–3196 | 882–3214 |
| 5 | 139–3957 | 128–3968 |
| 6 | 1045–3051 | 1006–3090 |
| 7 | 5–4091 | 109–3987 |
| 8 | 1447–2649 | 1249–2847 |

## Teleop

- Mirroring maps each joint by percentage of its calibrated range, so mismatched ranges show up as joints moving different angles.
- With the ranges above, every follower joint tracked within about 3% of the leader.
- `teleop.py`'s status readout prints "no mirroring state published" even when mirroring is on. Not fixed; check Station's log or `arm_state.py` instead.

## How commands work (for the bridge)

- Motor state: follow `st3215/inference`; each motor carries its full register dump plus `range_min`, `range_max`.
- Move a joint: `st3215.Command(target_bus_serial=..., write=ST3215WriteCommand(motor_id, address=0x2A, value=<2 bytes LE>))`, wrapped in a `DriverCommand` of type `STC_ST3215_COMMAND`, sent with `send_commands`. Clamp to the calibrated range first.
- Mirroring and calibration use the same `send_commands` path (see the scripts).

## Direct control — what we learned (afternoon, 21 Nov)

- **Teleop must be off** (`teleop.py stop`) before scripts move the follower, or mirroring fights them. Then `arm.py hold` so it doesn't drop.
- **Plain goal writes stop short.** The follower servos run P-only (P=16, I=0, D=0), so a joint commanded 45 steps away settles about 30 short. `grasp.py` closes the loop: it keeps nudging the goal until the measured position arrives. Use it, not raw writes.
- **Contact abort:** `CUR_LIMIT = 260` mA in `arm.py`. Any joint 1–7 above that stops the move and holds position. A "did not fully arrive" note with a joint stuck short usually means something is touching the table.
- **The URDF model is not trustworthy for this arm.** `arm.py`'s FK/IK (`pose` tip xyz, `move`, `fwd`) uses `norma-core`'s `elrobot_follower.urdf`, but its joint directions don't match what the cameras show. Ignore the tip xyz; don't use `move`/`fwd`. Steer by camera with `grasp.py` instead.
- **Joint directions observed in the top-down grasp pose** (servo steps):
  - j1 (base) decrease → object moves left in the gripper image (about 5 px per degree at grasp height). j1 decrease also turns the arm towards the white box.
  - j2 (shoulder) increase → gripper goes **down**. Hover 1778, grasp height 1919, clearance 1640.
  - j3 (elbow) decrease → reaches further out.
- **Gripper:** `arm.py grip 0.8` opens. Close by writing goal `range_min + 40` to motor 8. Holding the object shows ~20–40 mA on motor 8; an empty close shows 0–2 mA and ends at ~0.03 opening.

### The grasp that worked

Approaching at a slant failed six times: the jaws closed just above the object every time.
A teleop demonstration (`demos/grasp1/`, gitignored, local only) showed the grasp is top-down,
with j4 bent to ~1220 instead of ~2160. Replaying that pose then worked first time:

1. `teleop.py stop`, `arm.py hold`
2. Fold into the top-down pose with the arm raised (the fold was done at j2 ≈ 1030, arm up; folding j4 with the arm low is untested): `grasp.py goto 2 1030`, then `grasp.py goto 4 1240 3 3166 6 2632`, then `grasp.py goto 2 1700`, then `grasp.py goto 1 2226 5 2116 7 1680 6 2551 3 3084`
3. `arm.py grip 0.8`, `grasp.py goto 2 1778` (hover)
4. Centre the object between the jaws in the gripper camera with small j1 steps (`grasp.py step 1 -1.2` etc.; `vision.py` gives the pixel centre; jaws' centre is x≈150 in a 299 px frame)
5. `grasp.py goto 2 1919` (descend), close the gripper, check motor 8 current > 15 mA
6. `grasp.py goto 2 1780` (lift)

Demo grasp moment: t≈111–118 s in `demos/grasp1/states.jsonl` (gripper 0.76 → 0.22).

### Placing on the white box

From the lifted grasp pose: `goto 2 1640` (clearance), base j1 decreased ~29° in steps
(`step 1 -12`, `step 1 -8 3 -10`, `step 1 -6 2 12`), lowered with j2 (`step 2 10`, `step 2 8 1 -3`)
until the box filled the gripper view, `arm.py grip 0.85`, then `step 2 -14` to lift clear.
The object landed on the open cardboard box near its edge closest to the arm.

## Current state when this session ended (21 Nov, ~13:00)

- Station running (`scripts/run-station.sh`), both arms calibrated and frozen.
- **Teleop OFF.** Follower torque ON, holding just above the white box, gripper open (0.85).
  Pose: j1 1828, j2 1820, j3 2974, j4 1222, j5 2113, j6 2548, j7 1682, j8 2604.
- The black object is on the white box.
- To hand control back to the leader: `python3 scripts/teleop.py start` (the follower will jump to the leader's pose, so put the leader in a similar pose first).

## Open items

- Build the Claude bridge (MCP server) around `grasp.py` + `vision.py` + `camera.py`: get frame, find object, step joints, grip.
- Automate the grasp recipe above into one script (centre with j1 from `vision.py`, descend, close, verify by current).
- Fix or replace the kinematic model so moves can be planned in centimetres (ArUco markers + OpenCV for camera height was proposed).
- Third camera not detected; frames are 299x224 (`resize_target: 224`).
