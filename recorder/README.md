# recorder: record and replay everything the arm does

A local service that follows the running Station **read-only** and records motor state for every bus,
every camera frame (the JPEG Station sends), motor commands, and hot-plug events, all with timestamps.
It can replay any recording through an API, a browser viewer, or a Python drop-in for live Station
reads. It never writes to the arm: no torque, goals or calibration. `recorder/live.py` only calls
`follow` / `read_from_tail`.

```bash
export NORMA_CORE_REPO=/path/to/norma-core           # needed for live capture only
uv run --no-project python -m recorder serve          # API + live capture on http://127.0.0.1:8900
uv run --no-project python -m recorder record --name calib --seconds 20   # uses the server if it is up
uv run --no-project python -m recorder ls
uv run --no-project python -m recorder show 20261122-000849_live-20s      # rates, drops, sizes
uv run --no-project --with pytest python -m pytest recorder/tests -q      # parity tests skip without norma-core
```

Open `http://127.0.0.1:8900/` for live camera thumbnails and the recordings list. Each recording has
a viewer at `/recordings/<id>/viewer`.

Environment variables: `STATION_HOST` (default `localhost`), `ROBOT_BUS_SERIAL` (follower, default
`5B61033301`), `ROBOT_LEADER_SERIAL` (default `5B61035034`), `RECORDER_CAM_ROLES`
(default `28ff0ca8=wrist,9ea4cacd=overview`), `RECORDER_DIR` (default `<repo>/recordings`),
`STATION_LOG` (fallback for camera discovery), `RECORDER_URL` (the CLI uses it to reach the server).

## What Station exposes and what gets recorded

| Station queue | Rate seen | Recorded as |
|---|---|---|
| `st3215/inference` (`InferenceState`, all buses) | ~300-400 frames/s. Each frame refreshes **one** motor, so ~45-50 Hz per motor | `state.jsonl`, decimated to `--state-hz` (default 50) on Station's bus clock. `0` writes every frame |
| `usbvideo/<hash>` (`RxEnvelope`) | ~28-30 fps, one 299x224 JPEG (~23 KB) per envelope | `cams/<hash8>/<n>.jpg` + `index.jsonl`. Non-frame envelopes (connect, disconnect, recording start/end, error) become events |
| `commands` (`StationCommandsPack`) | only when a client sends | `commands.jsonl` (`src: "commands"`), decoded: kind, bus, address, per-motor values |
| `st3215/tx` (`TxEnvelope`) | only when the driver dispatches a command | `commands.jsonl` (`src: "st3215/tx"`) with Station stamps |
| `inference-states` (`InferenceRx`) | polled once a second | camera discovery (new `usbvideo/*` queues). Falls back to the Station log |

Hot-plug: a bus appearing or disappearing in `st3215/inference` logs `bus_connected` / `bus_disconnected`.
A bus whose stamps stop moving for 3 s logs `bus_stale`, then `bus_resumed`. A new camera queue logs
`camera_discovered`. No frames for 3 s logs `camera_lost`, and frames returning log `camera_live`. If no
state arrives for 6 s, the recorder logs `state_stalled` and re-follows every queue (Station restart).
None of these crash a recording.

## On-disk format (format_version 1)

`recordings/<YYYYmmdd-HHMMSS>_<name>/` (gitignored). The layout is close to `sim/recorder.py` sessions:

- **`session.json`**: `id, name, note, source: "real", format_version, started` (unix), `started_iso,
  stopped, duration, station, state_hz, joints: [j1..j8]`, `buses: {serial: {role, motor_ids}}`,
  `cameras: {hash8: {hash, hash8, role, frames, dropped, late, bytes, w, h}}`,
  `counts: {state_rows, state_frames_seen, events, markers, commands}`, `size_bytes`.
  It is rewritten every ~2 s while recording, so a crash still leaves a usable file. `stopped: null`
  means the recording is still active or was interrupted.
- **`state.jsonl`**: one row per emitted state frame:
  ```json
  {"t": 3.2101, "wall": 1795302536.12, "station_ns": 1795302536101000000, "station_mono_ns": 532..., 
   "buses": {"5B61033301": {"role": "follower", "n_motors": 8, "gripper": 0.5735,
      "j1": {"id": 1, "pos": 2010, "goal": 0, "speed": 0, "load": 0, "current": 0, "volt": 12.4, "temp": 33,
             "torque": 0, "moving": 0, "status": 0, "range_min": 1066, "range_max": 3030, "frozen": true,
             "error": null, "norm": 0.4788, "stamp_ns": 1795302536090000000, "regs": "030a00..."}, "...": "j2..j8"}},
   "norm": [...8], "goal": [...8], "load": [...8], "status": [...8], "torque": [...8]}
  ```
  - `pos`: present position 0..4095, sign-normalised like `arm_mcp.station.norm_raw`. `goal`: the raw
    u16 goal register, as `read_state` returns it.
  - `speed` and `current` are signed (bit 15 is the direction). `load` is signed in 0.1 % units
    (bit 10 is the direction). `current` is raw, about 6.5 mA per unit. `volt` is in volts, `temp` in °C.
  - `norm` and `gripper` are null when the range is not calibrated (`range_min >= range_max`).
    `gripper` is j8's `norm` clamped to 0..1, where higher means more open.
  - `error` is Station's error text for that motor read, null when the read was clean. `stamp_ns` is
    when Station last read that motor.
  - `regs` is the full register dump as hex, so nothing is lost and `read_state` can be rebuilt exactly.
  - The top-level `norm`/`goal`/`load`/`status`/`torque` arrays hold j1..j8 of the follower (or the
    first bus). They match the keys of sim `state.jsonl` rows.
- **`events.jsonl`**: `{"t", "wall", "type", ...}`. Types: `start`, `stop`, `marker` (`label`, `data`),
  `station_connected`, `bus_*`, `camera_*`, `state_stalled`.
- **`commands.jsonl`**: `{"t", "wall", "src", "kind", "bus", "address", "motor_id" | "motors", ...}`.
- **`cams/<hash8>/index.jsonl`**: `{"n", "t", "wall", "station_ns", "station_mono_ns", "index", "latency",
  "bytes", "w", "h"}`. `n` names `cams/<hash8>/<n>.jpg`. `index` is Station's frame counter, and gaps
  in it are counted as `dropped`. `latency` = arrival `wall` minus `station_ns`. Frames with latency
  above 0.25 s are counted as `late`.

Times: `t` is seconds since the recording started, measured on this process's monotonic clock at
arrival. `wall` is unix time at arrival. `station_ns` / `station_mono_ns` are Station's own
stamps (bus stamp for state, frame stamp for cameras).

## HTTP API (127.0.0.1:8900, JSON, CORS for http://localhost:* and http://127.0.0.1:*)

| Method & path | Returns |
|---|---|
| `GET /health` | `station_reachable`, buses, cameras (fps, age, live), errors, active recording |
| `GET /live` | latest motor-state row (`?regs=1` keeps the register hex) |
| `GET /live/frame?cam=<hash8\|role>` | latest JPEG |
| `POST /recordings` `{name, note}` | `201 {id, session}`. `409` if one is already active |
| `POST /recordings/<id>/stop` | final `session` |
| `POST /recordings/<id>/markers` `{label, data}` | `201 {event}` (timestamped). `<id>` may be `active` |
| `GET /recordings` | `{recordings: [session...], active}` |
| `GET /recordings/<id>` | `{session, summary, markers}`. The summary has rates, drops, latency, size |
| `GET /recordings/<id>/state?from=&to=&every=` | `{rows}`. `every` is the minimum spacing in seconds. `regs=1` keeps the hex |
| `GET /recordings/<id>/at?t=` | `{state, frames: {hash8: index row}}`, nearest (`causal=1` = last at/before t) |
| `GET /recordings/<id>/frame?cam=&t=` | nearest JPEG (headers `X-Frame-N`, `X-Frame-T`, `X-Frame-Cam`) |
| `GET /recordings/<id>/frames/<hash8>/<n>.jpg` | one frame, cacheable |
| `GET /recordings/<id>/index`, `/events`, `/commands` | frame time arrays, events, commands |
| `GET /recordings/<id>/viewer` | self-contained HTML viewer |

The viewer shows both cameras, joint traces (position solid, goal dashed, load, current), gripper
opening, a per-joint table, markers and lifecycle events, a scrubber, play/pause, and 0.25x–4x speed.
Keys: space plays/pauses, ←/→ steps 0.1 s, shift+←/→ steps 1 s.

Use `active` in place of an id in any `/recordings/<id>/...` path. For example, a script can tag the
current recording without knowing its id:

```bash
curl -XPOST localhost:8900/recordings/active/markers -H 'Content-Type: application/json' \
     -d '{"label":"grasp attempt","data":{"target":"cup"}}'
```

## Replay library

```python
from recorder import Recording, ReplaySource
rec = Recording("recordings/20261122-000849_live-20s")   # or just the id
rec.state_at(3.2)                    # nearest row; state_at(3.2, causal=True) = last row at/before 3.2
row, jpeg = rec.frame_at("wrist", 3.2)
rec.markers(); rec.stats()
for t, kind, payload in rec.iter(speed=2.0, realtime=True):   # kind: state | frame | event | command
    ...

src = ReplaySource(rec, speed=1.0, start=0.0, loop=False)      # replay clock starts now
st = await src.read_state()          # == arm_mcp.station.read_state(cli) shape: {1: {raw, goal, lo, hi, norm,
                                     #    in_range, torque, status, temp, volt, load}, ..., 8: {...}}
jpeg = src.frame("wrist")            # latest JPEG at the replay clock (never from the future)
src.cameras()                        # same fields as scripts/cam_stream.py /api/cameras
src.seek(10); src.pause(); src.resume(); src.set_speed(4); src.done
```

`ReplaySource.read_state` rebuilds each motor from the recorded register dump using the same
expressions as `arm_mcp.station.read_state`. `tests/test_station_parity.py` feeds identical dumps to
both and asserts the results are equal. Like the live function, it raises
`RuntimeError("motor bus …: no complete error-free read …")` when the row at the replay time is
missing a motor or has an error.

### Wiring it in (for the integrator; nothing in `arm_mcp/`, `robot_mcp/` or `sim/` is changed here)

Proposed convention: `ROBOT_SOURCE=replay:<recording dir or id>[?speed=1&start=0&loop=1&bus=SERIAL]`.

```python
# arm_mcp/server.py
from recorder import ReplaySource
_replay = ReplaySource.from_env()            # None unless ROBOT_SOURCE=replay:...

async def _read() -> dict:
    return await _replay.read_state() if _replay else await stn.read_state(await _client())

# look(): use _replay.cameras() / _replay.frame(hash) instead of the cam_stream HTTP calls
# _write(): refuse (or log and no-op) when _replay is set; a recording cannot be moved
```

`robot_mcp/station_lib.current_st3215()` can be fed the same way. It needs `(joints_norm, ranges,
raws)`, which `read_state()` contains: `norm`, `(lo, hi)`, `raw`. Scripts that only want data can
also use the HTTP API (`/recordings/<id>/at?t=`, `/frame`) without importing anything.

## Limits / known behaviour

- State rows are decimated (default 50 Hz) because Station publishes about 400 frames/s and each frame
  changes only one motor. `state_frames_seen` counts every frame received. `--state-hz 0` records all of
  them (about 1.5 KB/row/bus).
- `t` is arrival time. Under heavy machine load, Station delivers in bursts. Use `station_ns` for
  source timing.
- Recording size is about 1.4 MB/s with both cameras (≈ 29 MB per 20 s), almost all JPEG.
- Only JPEG camera frames are stored. Raw NCHW frames are skipped and reported in `/health` `errors`.
