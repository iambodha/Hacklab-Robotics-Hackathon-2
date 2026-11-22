# perception: calibrated cameras and table coordinates you can measure

Claude recognises things in the camera images and says where they are as a pixel, a bbox or a colour.
This package does the measuring: calibrated cameras, ArUco markers, the ElRobot URDF kinematics and
ray/plane geometry turn that into **world coordinates in metres with an error estimate in mm**.
The coordinates come from the geometry, not from a vision model's guess.

World frame (`prompts/scene-contract.md`): origin = follower `base_link` origin on the table, z up,
x = the arm's forward direction at j1 = 0, y = z × x (left as seen from the arm). Note that URDF
`base_link` reaches along **its +y**, so `T_world_base` is a −90° yaw.

## Commands

All of them run from the worktree root. Station must be running on `localhost:8888`.

```bash
export NORMA_CORE_REPO=/Users/bodha/Documents/b0dha/Robotics-Hackathon/repo/norma-core
UV="uv run --no-project --with opencv-contrib-python --with numpy"

$UV python -m perception.calib_scene --live --frames 90        # (re)calibrate the scene -> calibration/scene_calib.json
$UV python -m perception.calib_scene --live --wrist-poses 4    # + 4 hand-placed arm poses for the arm base (see below)
$UV python -m perception.calib_scene --live --extra-wrist-dir frames/calib-20261122-012214   # + an earlier wrist pose
$UV python -m perception.calib_scene --frames-dir frames/calib-YYYYMMDD-HHMMSS --dry-run   # offline rerun
$UV python -m perception.arm_base tape-guide                   # draw the 4 tape distances -> calibration/tape_guide.jpg
$UV python -m perception.locate --color blue --height auto --out /tmp/cap.jpg      # locate on a live frame
$UV python -m perception.locate --bbox 172 80 212 123 --height auto               # the bin (rim height estimated)
$UV --with pillow python -m perception.calib_intrinsics board  # printable ChArUco board (already in calibration/board/)
$UV python -m perception.calib_intrinsics live overview        # intrinsics with the printed board
$UV python -m perception.calib_intrinsics live wrist
$UV --with scipy --with mujoco --with pytest --with pillow python -m pytest -q tests   # 50 tests, synthetic data, ~20 s
```

Dependencies: `opencv-contrib-python`, `numpy` (plus `pillow` for the board PDF, `pytest` for the tests,
and `mcp<2` for the MCP server). The library itself only needs numpy and cv2. The FK goes through
`camera_pose/elrobot_camera.py`, which needs the URDF from `$NORMA_CORE_REPO/hardware/elrobot/simulation`
and reads the joint mapping from the robot profile (`sim/robot_profile.py`; mujoco if installed, else the JSON).

**Station subscriptions.** `station_py` has no unfollow: every `client.follow()` streams forever. All Station
reads here go through `perception.latest.Latest.of(client, queue)`: one long-lived subscription per (client,
queue) that keeps only the newest entry (`get()`, or `get(fresh=True)` for one that arrives after the call).
`perception.station_io.grab_jpegs`, `arm_mcp/station.bus_frame` (so every arm_mcp/real_arm_mcp joint read) and
the intrinsics capture use it. Other code should too: `from perception.latest import Latest`.

Outputs:

| File | What |
|---|---|
| `calibration/scene_calib.json` | the contract file, plus a `quality` block (per-marker reprojection px/mm, hold-one-out, wrist check, mm per pixel) |
| `calibration/overview_annotated.jpg`, `wrist_annotated.jpg` | what the maths sees: 5 cm grid, axes, markers (model red / detected green), ruler in 299x224 px |
| `calibration/arm_base_overview.jpg` | the arm foot: detected front edge (green), CAD foot at the final base (magenta), j1 axis (cyan) |
| `calibration/tape_guide.jpg` | the four tape distances A–D for pinning the arm base (below) |
| `calibration/evidence/*.jpg` | `locate` results: blue cap on the table (01:22), cap in the bin and the bin (live, 01:4x) |
| `calibration/intrinsics_<cam>.json` | board calibration (written by `calib_intrinsics`); also mirrored into `camera_pose/wrist_camera.json` for the wrist |
| `calibration/board/charuco_A4.{pdf,png}` | the board to print |
| `frames/calib-*`, `frames/intrinsics-*` | at most ~10–25 JPEGs per run (gitignored) so a calibration can be recomputed offline |

## How it works

| Module | Role |
|---|---|
| `latest.py` | One shared Station subscription per (client, queue), newest entry only (see above). |
| `station_io.py` | Frames from the Station `usbvideo/<hash>` queues, joint ticks from `st3215/inference` (via `arm_mcp/station.py`). Read-only. |
| `aruco.py` | Detection runs on a white-padded, 3x upscaled copy: marker 21 (one corner past the frame edge) and the far, small markers 22/23 (4x4 bits ~2.5 px wide) are found in ~100 % of frames instead of 0–40 %. Corners are edge-refined at full resolution (lines fitted to the black square's sub-pixel edges), which cuts OpenCV's 0.5–0.8 px inward bias to about 0.15 px; a clipped marker's corner is the intersection of its side lines, just outside the frame. Median over N frames. |
| `scene_fit.py` | One least-squares fit of the overview camera pose and every marker's (x, y, yaw) on **one shared table plane**, plus the focal length while there is no board calibration. Using coplanarity is what makes a 299x224 camera usable here. |
| `wrist.py` | Station ticks → URDF FK (robot-profile joint mapping) → CAD camera mount → `T_base_wristcam`. Solves the arm base (x, y, yaw) on the table from markers the wrist sees, from one or more arm poses, optionally with priors (the foot). |
| `arm_base.py` | The arm foot in the overview, the tape procedure, the evidence images. |
| `calib_scene.py` | Puts it together and writes the json. |
| `locate.py` | The API (below). |
| `annotate.py` | Overlay drawing. |
| `calib_intrinsics.py` | ChArUco board: generation, live capture via Station, `cv2.calibrateCamera`. |

### The world frame: where the arm base is
ElRobot `base_link` (CAD, `assets/base_link.stl`): the black foot is 100 mm wide, its front face is 56.6 mm
ahead of the `base_link` origin, and the **j1 axis is 42.5 mm ahead of it** (the world origin is `base_link`,
not the j1 axis; `arm.j1_axis_world` in the json). `calib_scene` tries these in order
(`perception/scene_config.json`, `arm_base.source = "auto"`; or force one of `tape`, `foot`, `wrist`):
1. **tape** (`arm_base.tape_mm`, below): ±1–2 mm, ±0.5°.
2. **foot + wrist** (current): the overview finds the foot's front bottom edge column by column (the table →
   dark crossing; the foot ends ~6 px lower in the image than the table's far edge behind it). Back-projected,
   that is the foot's front-centre point, ±2.6 mm across / ±4.9 mm along the view. Its measured width
   (100.6 mm against 100 mm in the CAD) checks the detection. The wrist camera's view of a marker through FK
   is fused in, mostly for the yaw; with the foot pinning the position, the yaw comes from where marker 20 sits
   in the wrist image. Each observation is weighted by its own sigma, and the covariance is inflated when they
   disagree.
3. **foot**: foot only, yaw assumed square to the line through the base markers (±3°).
4. **wrist**: wrist + FK only (one pose: ±56 mm statistical; poorly constrained along the view).
5. **midpoint**: between the two base markers (rough, ±5 cm).

`arm_base.base_z_m` is the height of the base's bottom above the table top (0 if it stands directly on the table).

#### Pin the arm base to ±2 mm with a tape measure (2 minutes)
1. Open `calibration/tape_guide.jpg` (or regenerate it: `python -m perception.arm_base tape-guide`). It shows
   four yellow segments, **A–D**, each from a **magenta** dot (a front corner of the black arm foot, at table
   level) to a **red** dot (a corner of the black square of a marker, not the white border):
   - **A**: front-LEFT foot corner (robot's view = towards marker 22) → nearest corner of marker **22** (~210 mm)
   - **B**: front-LEFT foot corner → nearest corner of marker **21** (far left, ~416 mm)
   - **C**: front-RIGHT foot corner (towards marker 23) → nearest corner of marker **23** (~247 mm)
   - **D**: front-RIGHT foot corner → nearest corner of marker **20** (far right, ~446 mm)

   The exact corners and expected lengths are in `scene_calib.json` → `arm.tape_segments`.
2. Hold the tape flat on the table and read to the millimetre. Move the bin out of the way for B if needed and
   put it back afterwards. Don't move the arm or the markers.
3. Write the four numbers (mm) into `perception/scene_config.json`, replacing each `null`:
   `"arm_base": {"source": "auto", "base_z_m": 0.0, "tape_mm": {"A": null, "B": null, "C": null, "D": null}}`
4. Rerun `python -m perception.calib_scene --live --frames 90` (table clear of hands, ~10 s). The log shows
   `arm base source: tape (...) + wrist (yaw)`, the sigma (expect ~1.4 mm / ~0.55°; tape alone 1.3 mm / 1.0°)
   and `tape_residual_mm`. A residual over ~3 mm means a misread distance or the wrong corner.

Why these four: each foot corner is fixed by two near-perpendicular distances (one sideways to a base marker,
one forward to a far marker), so 1 mm of tape error moves the corner ~1 mm, not ~6 mm as with two sideways
distances. The 100 mm between the two corners then fixes the yaw, to ~1° from the tape alone. The wrist view
of a marker tightens the yaw to ~0.5° (more with `--wrist-poses 4`).

### API (`perception/locate.py`, world metres)
```python
from perception.locate import Locator
loc = Locator()                                    # loads calibration/scene_calib.json
loc.pixel_to_table("overview", u, v, z=0)          # -> xyz, mm_per_px, sigma_mm (pixel + calibration parts)
loc.world_to_pixel("wrist", xyz, joints)           # -> pixel, depth, in_view
loc.triangulate((u1, v1), (u2, v2), joints)        # -> xyz, ray_gap_mm, sigma_mm
loc.locate_object("overview", img, bbox=(u0,v0,u1,v1), color="red", height_m=0.05)
                                                   # -> centre_xy, size_m [long, short], yaw_deg, sigma_centre_mm,
                                                   #    sigma_vs_arm_mm, polygon_xy, centre_shift_mm_per_cm_height, top
loc.locate_object("overview", img, color="blue", height_m=None)   # height estimated (round/square objects)
loc.scene_map(joints, objects)                     # table, markers, arm base, camera frustums, TCP, objects
```
`locate_object` segments the silhouette (a colour threshold, or GrabCut inside the bbox). It keeps the table
points whose projections at z = 0 **and** at z = height both fall inside the silhouette. For an upright
prism that is exactly its footprint, from a single view. In synthetic tests the centre comes out within 1 mm
at heights of 0–10 cm. The size reads up to 1 px large, because boundary pixels are counted in.
- **Colour only** (no bbox/pixel): the largest colour blob **standing on the table** wins. Its lowest pixel
  must look at the table (markers' extent + 5 cm), so the blue window at the top-right of the overview is
  ignored. The blob itself isn't clipped.
- **`height_m=None`**: the height is estimated, assuming a round or square footprint. The z = 0 footprint of a
  tall object is stretched along the view, and the height where its depth and width extents become equal is
  the estimate: ±5 mm for round objects, ±1 cm for square ones (synthetic tests). The centre sigma includes it.
- **`top`** (whenever height > 0): the rim of a container. For a tapered bin the prism footprint is the
  smaller base; the rim centre and width come from the silhouette's far edge and widest extent at the rim height.
- **`sigma_vs_arm_mm`** adds the arm-base (world origin) uncertainty to `sigma_centre_mm` (overview only; the
  wrist's sigma already contains it). Use it for grasp and drop tolerances.

## MCP tools (`arm_mcp/server.py`, all read-only)
`calibration_status`, `locate(camera, u, v | bbox, color, height_m | null)`, `triangulate(...)`,
`project_point(x, y, z)`, `scene_map`, `annotated_view(camera)`. Run the server with
`uv run --no-project --with "mcp<2" --with pillow --with numpy --with opencv-contrib-python python arm_mcp/server.py`.
mcp 2.x renamed FastMCP, so pin `<2`.

## Accuracy (live, 22 Nov 01:35: all four markers, pooled board intrinsics, arm base from foot + wrist)
Frames `frames/calib-20261122-013512` (table clear apart from the bin with the cap in it); the json was
regenerated offline from them at 01:50 after the table got cluttered (identical to the 01:35 live run within 0.1 mm).

| | before (00:40) | after (01:35) |
|---|---|---|
| markers used | 20, 22, 23 (21 missed at the frame edge) | **20, 21, 22, 23** |
| intrinsics | f = 148 ± 6 px fitted to the markers | board, pooled over both cameras: f = 143.3 px ± 1 %, k1 k2 |
| overview reprojection | 0.54 px | **0.30 px** |
| per marker px (mm on table) 20 / 21 / 22 / 23 | 0.55 (1.4) / – / 0.71 (4.6) / 0.25 (1.5) | 0.42 (1.3) / 0.30 (0.9) / 0.27 (2.1) / 0.16 (1.2) |
| **hold-one-out RMS** 20 / 21 / 22 / 23 | 15.2 / – / 7.6 / 2.0 mm (f refitted); 1.6 / – / 5.8 / 1.7 (f fixed, optimistic) | **1.5 / 1.1 / 2.7 / 2.5 mm** (board f) |
| arm base | wrist only: ±23 mm / ±2.5° (statistical), ticks mapping | foot + wrist: **±8 mm (x 7, y 4) / ±0.9°** |

- Hold-one-out (`accuracy.py`): refit without marker k, then measure marker k's 100 mm sides and 141 mm
  diagonals on the table through that fit. That is the error `pixel_to_table` makes on a flat object there.
  The far-end markers (22/23, 6.3 mm per pixel) are the worst at 2.5–2.7 mm.
- Corners: median of 90 frames, jitter 0.05–0.15 px per corner. One pixel on the table is 2.9 mm at markers
  20/21 (near the overview camera) and 6.3–6.4 mm at the base markers. **At 299x224 that is the floor.** A
  Station `resize_target` above 224 would scale it down directly (needs a Station restart; the integrator's call).
- Marker 22 sits 44 mm away from where the old calibration put it (20 and 23: 8 mm). The old fit had f from
  three markers and a poorly constrained marker 22 (held-out 7.6 mm).
- Arm base: the foot's measured width is 100.6 mm (CAD 100). The wrist-only estimate (one pose, marker 20) is
  57 mm from the foot (its own sigma: 56 mm, mostly along the view). Fused, the wrist's residual rises from 3.7
  to 7.1 px, and the sigma is inflated for that. The yaw is 1.8° off square to the base-marker line. The
  arm stands where its foot is; the tape procedure above brings this to ~1.4 mm / 0.55°.
- Objects (`locate`, `calibration/evidence/`): the blue cap on the table (01:22) is at **(0.285, −0.141) m**,
  56 mm wide, height estimated 3.3 cm. Frame-to-frame scatter is 0.6 mm; sigma_centre 5.6 mm (3.4 mm if its
  height is given), sigma_vs_arm 11–12 mm. The bin (bbox, rim height estimated 8.7 cm) has its base centre at
  (0.288, 0.095) and its rim centre at **(0.296, 0.088) m**, rim ~105 mm wide, sigma_vs_arm ~12 mm.

### Physical ground-truth check (do this once markers and the board are done)
1. Put a flat, high-contrast object (e.g. a 5 cm red square of paper) on the table where both cameras see it.
2. Measure with a tape the distance from its centre to the centres of two markers (e.g. 20 and 23), ±1 mm.
3. Call `locate(camera="overview", color="red")` (or a bbox), and compute the distances from `centre_xy`
   to those markers' `centre_xy` in `scene_map`.
4. The difference is the measured accuracy. Repeat in 3–4 spots (near the arm, near the camera, at the sides).
   Record it in `scene_config.json` under `measured_marker_distances_m` for future checks.

## What the user needs to do (in order of payoff)
1. **Tape the arm base** (2 minutes, above): ±8 mm → ±1.4 mm. Nothing else is as cheap and as big.
2. Keep the arm foot visible from the overview camera (nothing standing in front of it) when recalibrating.
   `calib_scene` refuses to replace a foot/tape base with a weaker one (`--force` to override).
3. Optional: with torque OFF, move the arm by hand during `calib_scene --live --wrist-poses 4`, so the wrist
   sees markers 20/21 from 4 poses (turn the base left/right, tilt the wrist). Nothing is commanded. This
   tightens the yaw and checks the FK. `calibration/wrist_pose_0122` (3 wrist frames + joints of the 01:22 pose,
   marker 20 in view) is what `--extra-wrist-dir` used for the committed calibration; it stays valid while
   the base doesn't move.

## Joint mapping
FK uses the robot profile's `kinematics.arm_mapping` (`sim/robots/elrobot.json`: `range_fraction`), the same
as the executor, `sim/real_arm_mcp.py` and the twin: `q = lower + n · (upper − lower)`, n = the fraction of the
calibrated range (`camera_pose/elrobot_camera.py`; tests compare it with `executor.kinematics` and
`sim/real_joints.py`). A joint whose calibrated range is missing or reset (< 200 ticks) falls back to ticks
around the servo centre 2048. If the arm is recalibrated on Station, rerun `calib_scene --live`. The joint
signs of j1/j5/j7 are still unverified (camera_pose/HANDOFF.md task 3).

## NormaCore integration
- **Used:** Station TCP client (`station_py`) and its queues: `usbvideo/<hash>` JPEG frames,
  `st3215/inference` joint ticks and calibrated ranges. The ElRobot follower URDF
  (`hardware/elrobot/simulation`), the STEP gripper assembly for the camera mount (via
  `camera_pose/extract_wrist_camera.py`), and the STL base for the footprint check. No device is opened directly.
- **Could go upstream:**
  - `perception/latest.py` (one subscription per queue, newest entry) and an `unfollow` in `station_py`.
  - `perception/station_io.py` as a small Station client helper (`grab_jpegs(cli, queue, n)`).
  - The wrist camera link (`T_link_cam` from the CAD) as a `<link>`/`<joint>` in `elrobot_follower.urdf`.
  - A per-camera intrinsics record next to the camera queue in Station's config, normalised by image
    size so `resize_target` changes don't break it.
  - The ChArUco capture loop as a Station web-UI page on :8889.
  - Scene markers as a Station "scene" queue that publishes `T_world_cam` and marker poses.
