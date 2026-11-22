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

$UV python -m perception.calib_scene --live                    # (re)calibrate the scene -> calibration/scene_calib.json
$UV python -m perception.calib_scene --live --wrist-poses 4    # + 4 hand-placed arm poses for the arm base (see below)
$UV python -m perception.calib_scene --frames-dir frames/calib-YYYYMMDD-HHMMSS --dry-run   # offline rerun
$UV --with pillow python -m perception.calib_intrinsics board  # printable ChArUco board (already in calibration/board/)
$UV python -m perception.calib_intrinsics live overview        # intrinsics with the printed board
$UV python -m perception.calib_intrinsics live wrist
$UV --with pytest --with pillow python -m pytest -q tests      # 31 tests, synthetic data, ~10 s
```

Dependencies: `opencv-contrib-python`, `numpy` (plus `pillow` for the board PDF, `pytest` for the tests,
and `mcp<2` for the MCP server). The library itself only needs numpy and cv2. The FK goes through
`camera_pose/elrobot_camera.py`, which needs the URDF from `$NORMA_CORE_REPO/hardware/elrobot/simulation`.

Outputs:

| File | What |
|---|---|
| `calibration/scene_calib.json` | the contract file, plus a `quality` block (per-marker reprojection px/mm, hold-one-out, wrist check, mm per pixel) |
| `calibration/overview_annotated.jpg`, `wrist_annotated.jpg` | what the maths sees: 5 cm grid, axes, markers (model red / detected green), ruler in 299x224 px |
| `calibration/intrinsics_<cam>.json` | board calibration (written by `calib_intrinsics`); also mirrored into `camera_pose/wrist_camera.json` for the wrist |
| `calibration/board/charuco_A4.{pdf,png}` | the board to print |
| `frames/calib-*`, `frames/intrinsics-*` | at most ~10–25 JPEGs per run (gitignored) so a calibration can be recomputed offline |

## How it works

| Module | Role |
|---|---|
| `station_io.py` | Frames from the Station `usbvideo/<hash>` queues, joint ticks from `st3215/inference` (via `arm_mcp/station.py`). Read-only. |
| `aruco.py` | Dictionary auto-detection. Corners are edge-refined (lines fitted to the black square's sub-pixel edges), which cuts OpenCV's 0.5–0.8 px inward bias on small markers to about 0.15 px (tested on rendered markers). Median over N frames. |
| `scene_fit.py` | One least-squares fit of the overview camera pose and every marker's (x, y, yaw) on **one shared table plane**, plus the focal length while there is no board calibration. Using coplanarity is what makes a 299x224 camera usable here. |
| `wrist.py` | Station ticks → URDF FK → CAD camera mount → `T_base_wristcam`. Solves the arm base (x, y, yaw) on the table from markers the wrist sees, from one or more arm poses. |
| `calib_scene.py` | Puts it together and writes the json. |
| `locate.py` | The API (below). |
| `annotate.py` | Overlay drawing. |
| `calib_intrinsics.py` | ChArUco board: generation, live capture via Station, `cv2.calibrateCamera`. |

### The world frame: where the arm base is
`calib_scene` tries these in order (`perception/scene_config.json`, `arm_base.source = "auto"`):
1. **measured**: `measured_dist_to_base_marker_centres_m = {"22": d1, "23": d2}`, tape distances from the
   centre of the arm's base (the j1 rotation axis) to the centres of the two markers beside the arm. Yaw is
   taken as perpendicular to the line through those two markers, facing the far markers (adjust with
   `measured_yaw_deg_from_perpendicular`).
2. **wrist** (current): the wrist camera sees a marker whose table pose the overview fit knows. FK closes
   the loop, and the base (x, y, yaw) is solved together with the wrist focal length. One pose gives about ±2 cm / ±2.5°.
   More hand-placed poses (`--wrist-poses N`) shrink that.
3. **midpoint**: between the two base markers (rough, ±5 cm).

`arm_base.base_z_m` is the height of the base's bottom above the table top (0 if it stands directly on the table).

### API (`perception/locate.py`, world metres)
```python
from perception.locate import Locator
loc = Locator()                                    # loads calibration/scene_calib.json
loc.pixel_to_table("overview", u, v, z=0)          # -> xyz, mm_per_px, sigma_mm (pixel + calibration parts)
loc.world_to_pixel("wrist", xyz, joints)           # -> pixel, depth, in_view
loc.triangulate((u1, v1), (u2, v2), joints)        # -> xyz, ray_gap_mm, sigma_mm
loc.locate_object("overview", img, bbox=(u0,v0,u1,v1), color="red", height_m=0.05)
                                                   # -> centre_xy, size_m [long, short], yaw_deg, sigma_centre_mm,
                                                   #    polygon_xy, centre_shift_mm_per_cm_height
loc.scene_map(joints, objects)                     # table, markers, arm base, camera frustums, TCP, objects
```
`locate_object` segments the silhouette (a colour threshold, or GrabCut inside the bbox). It keeps the table
points whose projections at z = 0 **and** at z = height both fall inside the silhouette. For an upright
prism that is exactly its footprint, from a single view. In synthetic tests the centre comes out within 1 mm
at heights of 0–10 cm. The size reads up to 1 px large, because boundary pixels are counted in.
If the height isn't known, the result says how far the centre moves per cm of height.

## MCP tools (`arm_mcp/server.py`, all read-only)
`calibration_status`, `locate(camera, u, v | bbox, color, height_m)`, `triangulate(...)`,
`project_point(x, y, z)`, `scene_map`, `annotated_view(camera)`. Run the server with
`uv run --no-project --with "mcp<2" --with pillow --with numpy --with opencv-contrib-python python arm_mcp/server.py`.
mcp 2.x renamed FastMCP, so pin `<2`.

## Accuracy (live, 22 Nov, ~01:00, markers 20/22/23)
- Corners: median of 90 frames, jitter 0.06–0.08 px per corner. Overview fit: 0.54 px RMS reprojection
  (1.4–4.6 mm on the table). f = 148 ± 6 px (estimated from the markers; fovy 74°). An independent
  estimate agrees: the wrist camera gives f = 144 px from FK + marker 20 (it looks like the same camera model).
- One pixel on the table is 2.9 mm at marker 20 (near the overview camera) and 6.3 mm at the base markers.
  **At 299x224 that is the floor.** A Station `resize_target` above 224 would scale it down directly
  (needs a Station restart; the integrator's call).
- Hold-one-out (`accuracy.py`): refit without marker k, then measure marker k's 100 mm sides and
  141 mm diagonals on the table through that fit. With f fixed the RMS is **1.6 / 5.8 / 1.7 mm** (markers
  20 / 22 / 23). Refitting f as well gives 15 / 7.6 / 2.0 mm, because the two distant markers alone can't
  pin f down. The board calibration fixes this.
- Wrist: marker 20 predicted through the overview calibration and FK lands 1.7 px / 13 mm from where the
  wrist detects it. This is the residual of the base solve, not an independent check.
- Cross-check of the arm base: the overview pixel where the arm meets the table maps to (0.02, 0.00).
  The base STL's front edge is 57 mm ahead of `base_link`, so the overview puts the origin at about
  x = −0.035, while the wrist solve puts it at 0. Expect about ±3 cm in x until the base is measured or
  `--wrist-poses` is run.

### Physical ground-truth check (do this once markers and the board are done)
1. Put a flat, high-contrast object (e.g. a 5 cm red square of paper) on the table where both cameras see it.
2. Measure with a tape the distance from its centre to the centres of two markers (e.g. 20 and 23), ±1 mm.
3. Call `locate(camera="overview", color="red")` (or a bbox), and compute the distances from `centre_xy`
   to those markers' `centre_xy` in `scene_map`.
4. The difference is the measured accuracy. Repeat in 3–4 spots (near the arm, near the camera, at the sides).
   Record it in `scene_config.json` under `measured_marker_distances_m` for future checks.

## What the user needs to do (in order of payoff)
1. **Move the black cable off the bottom-right marker** (id 21 presumably). Its border is crossed, so it
   isn't detected. A fourth marker roughly halves the overview error and gives the wrist a second marker.
2. **Board intrinsics**: print `calibration/board/charuco_A4.pdf` at 100 % on A4 (no "fit to page").
   Check that the 100 mm bar measures 100 mm and that 4 squares measure 120 mm. Glue it flat on cardboard.
   Run `... calib_intrinsics live overview` and hold the board 25–50 cm under the camera, moving through
   the centre, the corners and edges, tilted ±30–45°, near and far, until 25 views are taken. Repeat with
   `live wrist` at 15–40 cm (the jaws cover the bottom of the image). If the board measured differently,
   pass `--square-mm`. Then rerun `calib_scene --live`.
3. **Arm base**: either measure d1/d2 (j1 axis centre to the centres of markers 22 and 23) into
   `scene_config.json`, or, with torque OFF, move the arm by hand during
   `calib_scene --live --wrist-poses 4`, so the wrist sees markers 20/21 from 4 poses (turn the base
   left/right, tilt the wrist). Nothing is commanded.

## Once the arm is calibrated on this Station
The FK uses `q = mid(URDF limits) + (raw − mid(range)) · 2π/4096`. Today every follower range is centred
on raw 2048, and if the ranges are reset (`range_min == range_max`) `wrist.sane_joints` falls back to 2048.
If a new calibration moves a range's centre, the wrist pose changes by that offset. Rerun
`calib_scene --live` afterwards. The joint signs of j1/j5/j7 are still unverified (camera_pose/HANDOFF.md task 3).

## NormaCore integration
- **Used:** Station TCP client (`station_py`) and its queues: `usbvideo/<hash>` JPEG frames,
  `st3215/inference` joint ticks and calibrated ranges. The ElRobot follower URDF
  (`hardware/elrobot/simulation`), the STEP gripper assembly for the camera mount (via
  `camera_pose/extract_wrist_camera.py`), and the STL base for the footprint check. No device is opened directly.
- **Could go upstream:**
  - `perception/station_io.py` as a small Station client helper (`grab_jpegs(cli, queue, n)`).
  - The wrist camera link (`T_link_cam` from the CAD) as a `<link>`/`<joint>` in `elrobot_follower.urdf`.
  - A per-camera intrinsics record next to the camera queue in Station's config, normalised by image
    size so `resize_target` changes don't break it.
  - The ChArUco capture loop as a Station web-UI page on :8889.
  - Scene markers as a Station "scene" queue that publishes `T_world_cam` and marker poses.
