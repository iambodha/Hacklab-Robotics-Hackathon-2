# Running the Claude operator on the real arm (follower only)

Claude drives the **follower arm** through the `elrobot-real` MCP server (`sim/real_arm_mcp.py`). It uses the
same tools as the simulator (`look`, `tcp_pose`, `move_tcp`, `pixel_to_table`, `gripper`, `move_to`, ...), so
anything that works in the sim (`sim/README.md`) runs here unchanged. **The leader arm is not needed**: leave it
unplugged or unpowered, and keep teleop off.

Claude gets only the robot tools: no shell, no file access and no other MCP servers. It sees the two camera images
and the joint table, nothing else.

## 0. Before you start
- One person stays **at the arm** for the whole run. To stop: Ctrl-C in the Claude window (the servos then hold
  their last goal), or cut the arm's 12 V. `set_torque(false)` makes the arm fall, so support it first.
- Clear the table around the arm, and keep hands out of the camera views while it moves.
- Hardware: the follower's servo bus (CH343 USB serial, bus serial `5B61033301` on our arm), the **wrist camera** and
  the **overview camera**. Put the servo adapter and the cameras on a **powered USB hub**. Without one the USB
  adapters brown out and re-enumerate when the motors draw current ("bus desync" in step 4).

## 1. One-time setup
```bash
git clone -b simulation https://github.com/iambodha/Hacklab-Robotics-Hackathon-2.git && cd Hacklab-Robotics-Hackathon-2
export NORMA_CORE_REPO=/path/to/norma-core      # a FULL clone (it needs hardware/elrobot for the URDF + meshes);
                                                # the repo-local norma-core/ is shallow and is not enough
# Python with mujoco, numpy, pillow, mcp<2 (run_real_server.sh uses `uv` automatically if ROBOT_REAL_PYTHON is unset)
python3 sim/build_scene.py                      # builds sim/generated/workspace.xml (the kinematic model); ~30 s
```
Claude Code must be installed and logged in on the machine you run it from (`claude` on the PATH).

## 2. Start Station and the camera relay
```bash
./station --config station.yaml -t --web        # normfs :8888 (the -t is required), web UI http://localhost:8889
python3 scripts/cam_stream.py                   # live cameras at http://localhost:8890 (needs Station running)
```
**Windows + WSL only:** first attach the devices to WSL from an admin PowerShell. Use `usbipd list` to find them,
`usbipd bind --busid X` once, then `usbipd attach --wsl --busid X` for the CH343 serial adapter and both KD-USB
cameras. If Station logs `Failed to open device for camera ...: -3`, the camera permissions are wrong:
```powershell
wsl -d Ubuntu -u root -- bash -c "chmod 666 /dev/bus/usb/001/*"   # resets when the cameras are re-plugged
```
Then restart Station. In the D: workspace: `bash scripts/run.sh restart-station` (restarts Station and cam_stream).

## 3. Tell the server which camera is which (per machine)
Open http://localhost:8890. The **wrist camera** is the one with the gripper jaws at the bottom of the frame. Note the
first 8 characters of each camera's hash and export them:
```bash
export ROBOT_WRIST_CAM=df52ffda ROBOT_OVERVIEW_CAM=c98bf1cf     # ours on the Windows laptop; yours will differ
export ROBOT_BUS_SERIAL=5B61033301 STATION_HOST=localhost ROBOT_CAM_URL=http://localhost:8890
```
The hashes can change when Station or the cameras restart, so check them again if `look()` reports `not live`.

## 4. Check the arm (read-only, never moves)
```bash
python3 arm_mcp/health.py
```
- `read_errors` must be `{}`. **`Motor ID mismatch` / "bus desync"** means the bus is out of sync: restart Station.
  If it keeps coming back, the USB adapter is resetting: use a powered hub and check the 12 V supply.
- `torque_on: true`, and every joint `in_range: true`. If a joint is **out of range**, Station's stored calibration
  is stale. Either recalibrate (`python3 scripts/calibrate_manual.py 5B61033301`: reset, then sweep every joint to
  both ends by hand) or point the server at a known-good calibration, without changing Station:
  ```bash
  export ROBOT_RANGES_FILE=$PWD/sim/real_ranges_5B61033301.json   # follower recalibration of 21 Nov 14:48
  ```
  Use that file only for the same arm (bus `5B61033301`).

To check the whole MCP path, run `bash scripts/run.sh real-check` in the D: workspace. It lists the tools, reads the
joints, the TCP and both cameras, and moves nothing.

## 5. Run Claude
**Linux / macOS / WSL** (Claude logged in there):
```bash
bash sim/run_agent_real.sh --model sonnet -p "Start."     # one-shot; drop -p for an interactive session
```
**Windows** (Claude logged in on Windows, server in WSL):
```powershell
powershell -ExecutionPolicy Bypass -File sim\run_agent_real_windows.ps1 -RealRunSh /mnt/d/Robotics-Hackathon/scripts/run.sh --model sonnet -p "Start."
```
or double-click `Start-Real-Agent.cmd` in the D: workspace. Use **Sonnet**: Haiku is faster but misjudged positions
in the sim.

The default task, in `sim/OPERATOR_REAL.md`, is "black cube into the grey box". For a different object or box, say
so in the prompt. For example:
```
-p "Task: pick up the black rectangular object and drop it into the open white box next to the arm. The overview camera is not calibrated: use the wrist camera. Start now."
```

## 6. Watching and afterwards
- Live cameras: http://localhost:8890. Station's web UI with the joints: http://localhost:8889.
- Every tool call (arguments, result, duration) goes to `sim/real_sessions/actions.jsonl`, or to `ROBOT_REAL_LOG_DIR`.
- When it ends, the servos hold their last goal. Move the arm home by hand or with `move_tcp`, or support it and
  call `set_torque(false)`.

## Known limits
- **Overview camera not calibrated.** `pixel_to_table("overview", ...)` refuses until `sim/real_overview_camera.json`
  exists, so Claude localizes with the wrist camera instead: it hovers high, finds the object, then descends.
  Calibrating the overview camera makes the first step much faster.
- The wrist camera's lens parameters are assumed (fovy 72° as in the sim), so positions are accurate to a few mm
  up close.
- `move_tcp` never lets the gripper point go below 1.2 cm (`ROBOT_MIN_TCP_Z`). It stops and holds on contact
  (load above `ROBOT_LOAD_ABORT`, default 550) or on a motor fault.
- Joint directions: j1, j5 and j7 were checked on the arm on 21 Nov. The mapping is in `real_arm_mcp.py` (`measured()`).

## Troubleshooting
| Symptom | Cause / fix |
|---|---|
| `no complete error-free read (bus desync? restart Station)` | Restart Station. If it recurs during moves, the USB adapter is browning out: use a powered hub. |
| `j6`/`j8` `outside its calibrated range`, every move refused | Stale Station calibration: `ROBOT_RANGES_FILE` or recalibrate (step 4). |
| `look()`: `camera ...: not live on the relay` | Wrong `ROBOT_WRIST_CAM`/`ROBOT_OVERVIEW_CAM` (step 3), or cam_stream isn't running. |
| Station: `Failed to open device for camera ...: -3` | USB permissions (step 2), then restart Station. |
| `torque is off on some motors` | Call `set_torque(true)`. It sets goal = present position first, so nothing jumps. |
| Arm sags or `still short by ...` | P-only servos under gravity. `move_tcp` corrects closed-loop and reports what's left. |
