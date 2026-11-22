# Office calibration report (22 Nov 2026)

Capture: `frames/calib-20261122-101741` (integration worktree; 10 overview frames, 10 wrist frames at the rest pose,
6 hand-placed wrist poses x 5 frames, raw servo ticks). Everything below was recomputed offline from it; nothing
touched the robot. `scene_calib.live.json` is the integrator's live run, `scene_calib.json` the new offline run.

## Why only 2 of 6 wrist poses counted

| pose | joints (j1, j2, j3, j4, j6) | what the wrist sees | live run | now |
|---|---|---|---|---|
| 1 (= rest pose) | 1990, 3030, 1561, 1505, 2354 | marker 20 clipped by the right frame edge | - | - (not decodable) |
| 2 | 1660, 2939, 1424, 1468, 2592 | marker 20, whole | 20 | 20 |
| 3 | 1079, 2431, 1420, 2186, 2790 | marker 23, whole, close (fills 1/3 of the view) | - | **23** |
| 4 | 1080, 2095, 922, 986, 1425 | a marker cut by the left/top frame edges | - | - (not decodable) |
| 5 | 2381, 2642, 1851, 1544, 2331 | marker 21, whole | 21 | 21 |
| 6 | 2948, 2076, 1851, 2200, 2961 | a marker cut by the top/right frame edges | - | - (not decodable) |

Not blur (sharpness of pose 3 equals pose 2/5) and not resolution: the printed markers' black is banded by thin
bright horizontal streaks (toner), which split the adaptive threshold once a marker is large. A vertical box blur
(3 and 5 px; symmetric, so edge positions stay put) bridges them: `perception/aruco.py` `destripe`. Where both the
plain and the de-striped image decode a marker, the refined corners agree within 0.2-0.4 px. Poses 1, 4 and 6 show
the marker cut by the frame edge; ArUco cannot decode those, so they stay out.

The same banding cost the offline overview markers 22/23 (decoded in only 3-4 of the 10 saved frames; the live run
had 90): with the 3 px pass and a 30 % presence rule all four markers are measured offline.

## Old vs new

| | live (`scene_calib.live.json`) | new (`scene_calib.json`) |
|---|---|---|
| overview frames | 90 | 10 (the saved subset) |
| overview reproj | 0.537 px | 0.512 px |
| marker reproj px / on-table mm: 20 | 0.35 / 1.26 | 0.32 / 1.25 |
| 21 | 0.78 / 2.17 | 0.66 / 1.82 |
| 22 | 0.53 / 4.85 | 0.62 / 5.93 |
| 23 | 0.39 / 2.78 | 0.37 / 2.76 |
| held-out rms mm / centre shift mm: 20 | 2.28 / 9.3 | 1.55 / 3.5 |
| 21 | 3.35 / 7.7 | 5.09 / 12.6 |
| 22 | 5.52 / 0.8 | 6.75 / 1.5 |
| 23 | 3.23 / 0.5 | 3.57 / 1.9 |
| wrist poses used | 2, 5 | 2, 3, 5 |
| arm base source | foot + wrist | foot + wrist |
| base sigma (position / yaw) | 7.2 mm / 0.84 deg (fit) | 8.3 mm / **4.2 deg** (between-pose spread; fit says 1.3) |
| base yaw vs the base-marker line | -1.86 deg | -3.96 deg |
| wrist rms (pooled) | 11.4 px | 22.9 px |

The two overview fits describe the same marker layout: after a rigid 2-D alignment the marker centres agree to 1.2 mm
rms. The difference between the two calibrations is the arm base: the new base sits 6.8 mm / 1.2 mm (x / y of the
live world) from the live one and is rotated **-2.21 deg**. Marker centres in the new world frame (x = arm forward,
mm): 20 (469.9, -298.8), 21 (439.3, 267.7), 22 (-2.9, 229.0), 23 (35.2, -321.7). Foot front edge at overview pixel
(160.0, 84.2), measured width 89 mm (CAD 100).

### Wrist consistency (foot prior + one pose at a time, relative to the pooled base)

| pose | marker | yaw | position | rms |
|---|---|---|---|---|
| 2 | 20 | -1.4 deg | 23.7 mm | 5.3 px |
| 3 | 23 | -9.3 deg | 23.9 mm | 19.9 px |
| 5 | 21 | +5.2 deg | 17.4 mm | 22.4 px |

The wrist chain (URDF FK + range_fraction joint mapping + CAD mount) disagrees with itself by +-5-7 deg between poses,
so each pose is a +-5 deg yaw sample: the mean of three is good to ~4 deg (now reported as the yaw sigma). Pose 3 is no
more an outlier than pose 5, so the 2-pose live result is biased towards pose 5. Applying a mount pitch (0 / 3.6 /
7.5 / 9.31 deg) moves the pooled yaw by < 0.7 deg and does not shrink the spread (7.0-7.4 deg): joint offsets, not the
mount, drive it.

Independent check, the overview arm silhouette (twin arm vs dark pixels, real frame undistorted): IoU 0.642 for both,
outline twin->real 1.83 px (new) / 1.90 px (live). With the position held, the silhouette prefers +0.5 deg on the new
base and -1.5 deg on the live one, i.e. it sides with the new yaw. It constrains yaw only weakly (yaw trades against a
sideways shift because the camera looks back along the arm), so it is a check, not an input.

## Wrist mount pitch (home finding: +9.31 deg)

* Jaw outline (rigid in the camera frame, so it sees the mount alone), 7 office wrist views (rest + 6 poses):
  * with the old twin camera (centred principal point, raw frames): best **+10 deg** (per view +8..+11): the office
    confirms the home +9.31.
  * with the calibrated pinhole (principal point 7 px below centre = 2.9 deg at f = 139 px) and undistorted frames:
    best **+7.5 deg** (per view +5..+8), outline 14.2 px at CAD -> 4.8 px. Part of the home +9.31 was the principal
    point the old twin camera ignored.
* Wrist markers through FK (`camera_pose/fit_wrist_chain.py`, poses 2/3/5): pitch +3.6 +- 1.0 deg, leave-one-pose-out
  19.5 px vs CAD 19.7 px (no gain); the home "twinfix" set (+9.31 + its joint offsets) 26.3 px (worse). Home's marker fit
  gave -2.6 +- 1.4. Markers cannot separate the mount from the joint offsets.

Conclusion: the mount pitch is real and about **+7.5 +- 1 deg** for a twin/perception that models the principal point;
`sites/home/wrist_camera.json` (shared by both sites, not mine to edit) still says 9.31 and is not applied.

## Twin vs real (office twin built from the new calibration)

`compare_overview.jpg`, `compare_wrist_poseNN.jpg`, `compare_metrics.json` (`sim/compare_real.py --capture`): real
frame undistorted to the calibrated pinhole | twin | edge overlay | (overview) arm silhouette.

| view | markers | corner error twin vs real |
|---|---|---|
| overview (old fovy-only camera, live calib) | 20, 21, 22 | 7.99 px mean (centres off by -4.9, -6.6 px) |
| overview (calibrated camera, new calib) | 20, 21, 22, 23 | **0.46 px mean, 0.95 px max** |
| wrist pose 2 | 20 | 4.6 px (live calib 9.3) |
| wrist pose 3 | 23 | 30.2 px (live calib 34.7) |
| wrist pose 5 | 21 | 24.7 px (live calib 12.8) |
| wrist poses 1, 4, 6 | clipped markers | not measurable (pose 1: twin marker ~10-15 px up/right of the visible part) |

Overview arm silhouette: IoU 0.64, outline twin->real 1.8 px (real->twin 5.4 px includes the chair behind the far
table edge). The wrist residuals are the FK chain's (joint offsets), as in the consistency table above.

Distortion: not negligible at the 2 px level (k1 0.069, k2 -0.051: 2.4 px at marker 21, ~6 px in the image corners).
The twin renders an ideal pinhole; `compare_real` undistorts the real frame and corners to the same pinhole.

## Home, before / after the camera change

Same calibrated K at home (charuco, shared intrinsics), so honouring it is a pure correctness fix there too
(capture `frames/calib-20261122-022246`, home layout unchanged):

| | before (fovy only) | after (calibrated pinhole) |
|---|---|---|
| overview marker corners | 8.02 px mean (centres -4.9, -6.6 px) | 0.30 px mean, 0.68 max |
| overview arm IoU / outline twin->real | 0.36 / 3.5 px | 0.50 / 1.9 px |
| table colour error (mask alignment) | 15.5 rgb | 11.4 rgb |
| wrist poses (chain-dominated) | 11.4, 12.3, 17.8, 9.9, 12.1, 62.8 px | 14.4, 14.7, 16.4, 4.4, 7.9, 69.0 px |

`scripts_integrator/export_overview_cal.py` (legacy home mode) still writes a byte-identical
`sim/real_overview_camera.json`.
