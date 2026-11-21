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

## Open items

- Build the Claude bridge (MCP server): get camera frame, read joints, move joints within range.
- No direct joint move from a script has been tested yet; only mirroring and calibration.
- Third camera not detected; camera resolution stuck at 320x240.
