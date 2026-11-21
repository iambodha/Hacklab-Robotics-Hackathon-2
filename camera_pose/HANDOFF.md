# Wrist camera pose: findings and handoff

Goal: the agent (via `robot_mcp`) should know where the gripper camera is and where it
points for any arm pose, so it can turn a pixel in the wrist image into a table
position, and project a 3D point into the image. The camera is not on the gripper axis,
so the jaw centre is not the image centre.

Branch: `calculate-camera-position`. Nothing in this branch moves the arm.

## Files

| File | What |
|---|---|
| `camera_pose/extract_wrist_camera.py` | Reads the official CAD and writes `wrist_camera.json`. Run it again only if the CAD changes. |
| `camera_pose/wrist_camera.json` | **Result:** camera pose and TCP in the URDF gripper link frame. Intrinsics are not filled in yet. |
| `camera_pose/check_joint_mapping.py` | Shows which normalized-to-radian joint convention matches the taught grasp grid. Also has a minimal URDF FK (`Urdf.fk`) you can reuse. |
| `scripts/cam_stream.py` | Live browser view of every Station camera at http://127.0.0.1:8890. It reads Station's queues, so the camera is not opened a second time. `/snapshot/<hash>` returns one JPEG. |

## Findings

### 1. Camera pose relative to the gripper (from the CAD, high confidence)
- The URDF `hardware/elrobot/simulation/elrobot_follower.urdf` has **no camera link**.
- The standard assembly `hardware/elrobot/STEP/elrobot_follower_parts/Gripper.stp` contains
  `CameraMount_square32mm` and `UVC_Camera`. No CAD library is installed, so the STEP is parsed directly.
- The STEP `Gripper_Base` was rigidly registered onto the URDF `Gripper_Base_v1_1` mesh:
  median error **0.07 mm**, 95th percentile 1.65 mm. The placement therefore carries over exactly.
- In the **`Gripper_Base_v1_1` link frame** (child of `rev_motor_07`, metres; +y points toward the fingertips,
  jaws close along x):
  - optical centre (front of lens barrel): **(0.0015, 0.0228, 0.0608)**
  - optical axis: **(0.0074, 0.9030, −0.4296)**, i.e. along the fingers and tilted **25.4°** toward the jaws
  - TCP (between the fingertip pads): **(0.0017, 0.0936, 0.0005)**
  - So the camera sits about 61 mm off the jaw plane and about 71 mm behind the TCP, laterally centred.
  - `T_link_camera` (4×4) is in the JSON. Camera axes use the OpenCV convention (x right, y down, z forward).
    Image-down points toward the jaws, which matches the live image (jaws at the bottom of the frame).
- Uncertainty: about ±3 mm for where the pinhole sits along the lens barrel. If the physical mount is
  not the 32 mm square one (the README also lists 27/28/34 mm and a generic mount), expect a few mm of difference.

### 2. Which camera is which (live Station, this robot)
- `usbvideo/322ab5514b8e14014c3743c1abfd4a07`: **wrist cam** (jaws visible at the bottom)
- `usbvideo/8c697f6e0c79a3d2fab2a4ea977ee738`: overview cam
- Frames are JPEG 299×224 because Station has `resize_target: 224`; the native aspect ratio is 4:3.
  **Store intrinsics normalised by image width and height** so a resize does not break them.

### 3. Normalized joint value to URDF radians
- Station's web viewer uses `upper - (lower + n*span)` (`station-viewer/src/devices/elrobot/config.ts`).
  For the symmetric joints this is wrong, because it gives values outside the limits. **Don't copy it.**
- `check_joint_mapping.py` result: with **direct `q = lower + n*(upper-lower)`** for j2, j3, j4 and j6, the nine taught grasp
  poses all land at TCP height 3.0 cm ± 1.7 cm (flat, as expected). Flipping j2, j3 or j4 makes the spread 4 to 5 times worse.
- **Unverified: the signs of j1, j5 and j7.** They barely change the TCP height. j7 is gripper roll, so it **rotates the camera**
  and matters most. Default to direct and verify on the robot (task 3).
- j8 (jaws) runs the opposite way: **n = 1 is open**. Each jaw moves inward by `0.0255*(1-n)` m.
- Better physics: ST3215 has 4096 ticks per revolution, so `q = anchor + (raw - raw_anchor) * 2π/4096`.
  This holds even when a calibrated range does not cover the full URDF span. Norm-based mapping stretches a short range.

### 4. Calibration changed during the day. Check it before trusting any pose
Live follower ranges (`5B61033301`) at about 14:00, compared with `calibration/calibration-2026-11-21.txt` from this morning:

| joint | live range (ticks) | morning snapshot |
|---|---|---|
| j6 | 1688–2408 (720) | 1006–3090 (2084, matches the URDF span) |
| j8 | 1898–2198 (300) | 1249–2847 |

The other joints matched. j6 is wrist pitch: if its range really shrank, the normalized→radian mapping for j6 is
off by tens of degrees and the camera direction will be wrong. Ask the operator whether calibration was re-run or reset.
Then use the tick-based mapping, anchored at the range centre, or recalibrate.

## Progress (21 Nov, afternoon)

**Task 1 done: `camera_pose/elrobot_camera.py`.** numpy-only library (`WristCamera`) with
`joints_to_q`, `link_pose`, `camera_pose`, `tcp`, `project`, `pixel_ray`, `pixel_to_table`, plus a CLI.

- `validate` (MuJoCo 4.0, in-memory URDF + camera at `T_link_camera`, nothing on disk changed):
  FK matches MuJoCo's body pose to 6e-16 on 12 poses (today's top-down grasp, all-mid, 10 random);
  60 random markers 0.12-0.30 m in front of the camera project within **0.72 px** of the rendered
  marker centroids (target < 2 px); pixel -> table -> pixel round trip 1e-13 px.
  Run: `uv run --no-project --with numpy --with mujoco python camera_pose/elrobot_camera.py validate`
- `pose` on the live arm (read-only, motor bus via `robot_mcp.station_lib`) gave camera 10 cm above
  the base plane looking 71 deg down and TCP 1.1 cm above it, which matches the wrist and overview
  images at that moment (gripper down at the table among objects).
- Loading the URDF in MuJoCo needs, in memory: `meshdir` = the simulation folder with
  `strippath="false"` (MuJoCo 4 keeps the `assets/` prefix), and `boundmass`/`boundinertia`
  because `Gripper_Gear_v1_1` has no mass.
- Joint mapping default is "ticks" anchored at range centre (finding 3). On this laptop's Station
  the calibrated spans match the URDF spans within 2%, so it equals the direct norm mapping.
- `locate`/`project` refuse to run without intrinsics unless `--hfov DEG` is given; results are then
  labelled ASSUMED. Task 2 fills `intrinsics` in `wrist_camera.json` and they're used automatically.
- Camera queues on the macOS laptop (Station `0x...` unique ids differ per machine):
  wrist `usbvideo/e091eeda8dcda172482f039057691cdb`, overview `usbvideo/5cd55831e21692d4f05fcaec75c351a5`.

Still open: task 2 (checkerboard), task 3 (joint signs, needs the arm), task 4 (MCP tools),
task 5 (table height, needs the arm).

## Tasks for the next agent (in order)

Coordinate with the human operator: **a teammate is driving the arm.** Everything below is read-only
unless it says otherwise. Never command the leader bus `5B61035034`. Follow `robot_mcp/README.md` safety rules.

1. **`camera_pose/elrobot_camera.py` (library, numpy only)**
   - Load the URDF and `wrist_camera.json`. FK base → `Gripper_Base_v1_1` (reuse `Urdf` from `check_joint_mapping.py`).
   - `joints_to_radians(norms or raw ticks, ranges, cfg)` with a per-joint sign and anchor config (defaults as in finding 3).
   - `camera_pose(q) -> T_base_camera`, `tcp(q)`, `project(points_base, q, W, H) -> pixels`,
     `pixel_ray(u, v, q, W, H)`, `pixel_to_table(u, v, q, W, H, table_z)` (intersect the ray with the table plane).
   - CLI: `pose` (reads live joints via `robot_mcp.station_lib` / motor bus), `locate U V`, `project X Y Z`.
   - Done when: in MuJoCo, add a camera at `T_link_camera` to body `Gripper_Base_v1_1` (MjSpec or in-memory XML;
     **don't** overwrite the shared `simulation/elrobot.xml`). Render it and check that projected cube corners
     land within about 2 px of where they appear in the render. The sim uses URDF radians directly, so this
     isolates the maths from joint-mapping errors.
2. **Intrinsics.** Print a checkerboard (e.g. 9×6 inner corners, measured square size) and hold it at
   10 to 40 cm from the wrist cam in about 15 poses. Grab frames from `http://127.0.0.1:8890/snapshot/322ab551…`,
   run `cv2.calibrateCamera`, and save normalised `fx/W, fy/H, cx/W, cy/H` and distortion into `wrist_camera.json`.
   This is a wide-angle camera, so expect visible barrel distortion; consider `cv2.fisheye` if the RMS error is above 1 px.
   Sanity check: draw the projected jaw fingertips (from the CAD and the current j8) onto a live frame. They should sit on the real jaws.
3. **Joint signs on hardware** (needs the operator, small moves only):
   with the arm still, nudge j7 by +0.05 through `robot_mcp` `move_joints` and check that the predicted image rotation
   matches the real one. Repeat for j5 and j1, using the overview cam or a tape mark on the table. Then resolve the j6 range (finding 4).
4. **Expose it in `robot_mcp`** (read-only tools, `readOnlyHint: True`; the server can't hot-reload, so restart the MCP client):
   - `wrist_camera_pose(robot_id)`: camera position and viewing direction in the base frame, plus TCP, computed from the
     **motor-bus** joint read (`read_joint_state` path, which never freezes on a camera drop).
   - `wrist_pixel_to_table(robot_id, u, v)`: table point in metres in the base frame, plus the offset from the current TCP.
   - `project_to_wrist(robot_id, x, y, z)`: pixel, or "outside view".
   - Optionally, `observe()` could overlay the projected TCP on the wrist image so the model can see where the jaws will close.
5. **Measure the table height** once: touch the TCP to the table, then `table_z = tcp(q)[2]`. Store it in the config.

## Environment notes (Windows + WSL)
- Run everything inside the **Ubuntu** WSL distro: `wsl -d Ubuntu`. The default distro is `docker-desktop`,
  so a plain `bash` from PowerShell fails with `execvpe(/bin/bash) failed`.
- `source /mnt/d/Robotics-Hackathon/env.sh` gives python3 with numpy, scipy, cv2, mujoco and mcp. It also sets `NORMA_CORE_REPO`
  to the full norma-core clone, which includes `hardware/`. The repo-local `norma-core/` is shallow and has no `hardware/`.
- From Git Bash, prefix `wsl.exe` calls with `MSYS_NO_PATHCONV=1`, or `/mnt/...` paths get mangled.
- Station is at `localhost:8888` (TCP) and `:8889` (web UI). The camera stream is at `:8890` (`python3 scripts/cam_stream.py`).
