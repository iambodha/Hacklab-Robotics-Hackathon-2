# Calibration and frames

Start with [perception's calibration guide](../../perception/README.md) and [camera-pose handoff](../../camera_pose/HANDOFF.md). `calibration/` contains camera intrinsics, the scene contract, printable ChArUco board, annotated images and wrist-pose evidence; `camera_pose/` contains CAD camera/TCP transforms and joint-mapping checks.

The scene-contract world and MuJoCo base frame differ: perception documents +x forward while the ElRobot URDF reaches along +y. Preserve explicit transforms; never infer equivalent frames from their names. Intrinsics/image resolution, marker size, base pose, joint signs/offsets and wrist mount are all inputs. Historical calibration numbers and camera IDs belong to the old setup.

Offline recomputation can use saved frames via `perception.calib_scene --frames-dir ... --dry-run`; live capture needs Station and separately authorized hardware work. The shipped images are small evidence, not a full raw calibration dataset. r6-sites/r6-office are introducing separate `home` and `office` assets; the new table needs its own verified calibration.
