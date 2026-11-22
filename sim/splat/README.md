# Gaussian splat of the table, in the twin's frame

A phone video of the table goes in. A 3D Gaussian splat comes out, in the same metric frame as the MuJoCo twin, so the
twin and the splat can be shown from the same camera. All data stays under `$DATA_ROOT/splat/<name>/`
(default `DATA_ROOT=/Volumes/Dell HDD/Robotics-Hackathon`). Nothing here touches the arm or Station.

```
sim/splat/prepare.sh <video> [name]
```

| stage | tool | output under `$DATA_ROOT/splat/<name>/` |
|---|---|---|
| 1 frames | ffmpeg: `SPLAT_FPS` (2) fps, at most `SPLAT_MAX_FRAMES` (300), at most `SPLAT_WIDTH` (1600) px wide | `video.<ext>`, `dataset/images/f0001.jpg …` |
| 2 poses | `sfm.py`: pycolmap SIFT, then sequential matching (exhaustive for ≤ 120 frames), then the **GLOMAP** global mapper (`pycolmap.global_mapping`; GLOMAP ships inside COLMAP 4) | `sfm/database.db`, `sfm/sparse/0` (unaligned), `sfm/sfm.json` |
| 3 align | `align_markers.py`: ArUco 20–23 in every frame, then triangulated corners, then a robust Sim3 onto the twin's corner sites | `dataset/sparse/0` (aligned), `align.json`, `twin_cameras.json` |
| 4 train | Brush (Rust/wgpu on Metal): `brush_app dataset --total-steps $SPLAT_STEPS (15000)` | `brush/export_<steps>.ply`, then `--check-ply` is run on it |

Stages whose output already exists are skipped; delete a stage's folder to redo it. Use `SPLAT_SKIP_TRAIN=1` to stop
after alignment.

## Installs (sizes measured 22 Nov)

| what | how | size | status |
|---|---|---|---|
| ffmpeg | Homebrew | already installed | ok |
| pycolmap 4.2.1 (COLMAP + GLOMAP, CPU) | `uv run --with pycolmap==4.2.1` (wheel, published 29 Sep 2026) | 20.8 MB wheel | used by prepare.sh, in the uv cache |
| Brush v0.3.0 `brush-app-aarch64-apple-darwin` | GitHub release tarball, sha256 checked | 41.4 MB download, 145 MB unpacked | installed at `$DATA_ROOT/splat/bin/brush-app-aarch64-apple-darwin/brush_app` |
| ~~`brew install colmap`~~ | Homebrew 4.2.1 + 36 missing deps (boost alone 342 MB) | **892 MB installed**, 214 MB download | **not installed** (over the 300 MB limit; pycolmap covers it) |
| ~~glomap formula~~ | no Homebrew formula any more (merged into COLMAP 4) | n/a | n/a |
| ~~OpenSplat (Metal)~~ | needs libtorch | ~1 GB | not used |

Sizes are from the Homebrew GHCR bottle manifests (`sh.brew.bottle.installed_size` for macOS 26 arm64), the PyPI JSON
API and the GitHub release assets. To reinstall Brush:

```
D="$DATA_ROOT/splat/bin"; mkdir -p "$D"; cd "$D"
curl -LO https://github.com/ArthurBrussee/brush/releases/download/v0.3.0/brush-app-aarch64-apple-darwin.tar.xz
curl -LO https://github.com/ArthurBrussee/brush/releases/download/v0.3.0/brush-app-aarch64-apple-darwin.tar.xz.sha256
shasum -a 256 -c brush-app-aarch64-apple-darwin.tar.xz.sha256 && tar xf brush-app-aarch64-apple-darwin.tar.xz
```

## Frames and alignment

- **Twin frame:** the MuJoCo world, which is the URDF base frame (+y straight ahead of the arm, z up, metres).
  - The scene-contract world maps in as `p_sim = T_sim_world @ p_world`.
  - `align.json` stores `sim_from_colmap`, `world_from_colmap` and `T_sim_world`.
- **Targets:**
  - The targets are the corner sites `aruco_<id>_c<k>` of the twin built from `sim/layouts/<SPLAT_LAYOUT>.json`. They are the same corners `sim/selftest.py` projects to under 1.5 px in the twin's overview camera, in OpenCV corner order.
  - Marker 21 is only an estimate in the layout (its id is under a cable). `residual_mm_per_marker` shows whether it disagrees.
- **Robustness:**
  - Each corner is triangulated from every view that sees it. Views off by more than 2 px are dropped, and a corner needs at least 3 views and a 3° baseline.
  - LO-RANSAC with a 1 cm threshold then removes bad corners before the Sim3.
- **Brush keeps the input frame:** the reconstruction is transformed before training and Brush exports in its input frame. So the `.ply` needs no further transform. `--check-ply` checks this.
- **Filming:**
  - Walk around the table slowly, with all four markers in view as often as possible.
  - Avoid motion blur.
  - Lock exposure if the phone allows it.
  - Keep the arm still: a splat is static.

## Self-test (synthetic capture of the twin)

```
python sim/splat/synth_video.py                     # renders $DATA_ROOT/splat/_synthetic/{video.mp4,truth.json}
SPLAT_FPS=10 SPLAT_STEPS=500 sim/splat/prepare.sh "$DATA_ROOT/splat/_synthetic/video.mp4" _synthetic_run
python sim/splat/synth_video.py --check "$DATA_ROOT/splat/_synthetic_run"
```

Run on 22 Nov with 90 frames at 640×480:
- **Poses:** all 90 frames registered, 0.86 px reprojection error. Features took 5.5 s, matching 48 s, mapping 6.7 s.
- **Alignment:** 16 corners, 0.46 mm mean residual (0.82 mm max).
- **Camera centres vs truth:** 4.6 mm mean, 6.5 mm max, at about 1 m distance.
- **Brush, 500 steps (smoke test only):** splat centres lie 0.7 mm (median) from the aligned sparse points, and 48% of them are within 1 cm of the table plane z = 0. The splat is in the twin's frame.
- **Disk:** the whole run uses 29 MB.

On exFAT, macOS writes `._*` AppleDouble files next to every file. `sfm.py` ignores them, and `prepare.sh` runs
`dot_clean -m` before Brush.

## Showing the splat next to the MuJoCo twin

The splat and the twin share one frame, so you never have to align them at view time. Use any camera pose from
`twin_cameras.json` (OpenCV convention `T_sim_cam`, plus fovy and image size) or from MuJoCo's
`data.cam_xpos`/`cam_xmat` directly.

1. **Brush viewer (works now):** `brush_app $DATA_ROOT/splat/<name>/brush/export_15000.ply`.
   - This is a free-fly view; z is up.
   - To get the twin's overview view, place the camera at `twin_cameras.json → overview_cam`.
   - For the twin, run `SIM_LAYOUT=home python3 sim/build_scene.py` (preview renders) or `mjpython -m mujoco.viewer --mjcf=sim/generated/home/workspace.xml`.
2. **Web, side by side (proposed, not built):** one static page with two panes.
   - Left pane: a WebGPU/WebGL splat renderer (Brush web, or Spark/three.js) loads the `.ply` and uses the `overview_cam` intrinsics and pose from `twin_cameras.json`.
   - Right pane: the `mirror_real.py` MJPEG stream (`/stream`) of the twin at the live joint state, from the same camera.
   - Moving the virtual camera means sending the same pose to the splat pane and to a twin render endpoint. `mirror_real` would need a `?cam=` / pose parameter for that.
3. **Composited (proposed):** splat as the photoreal background, with the MuJoCo arm drawn on top.
   - Render the splat at the MuJoCo camera's pose and intrinsics.
   - Render MuJoCo's RGB, depth and segmentation from the same camera, and keep MuJoCo pixels where the arm is in front of the splat depth.
   - This needs a splat rasterizer callable from Python on Metal (gsplat is CUDA-only), so the practical route is the web page above with the twin's arm meshes in three.js.

MuJoCo keeps the physics and collision geometry either way; the splat is visual only.
