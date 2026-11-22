# Perception: images to measurements

[The subsystem README](../../perception/README.md) documents modules, live/offline commands, coordinate conventions, uncertainty and the localization API. ArUco detection plus calibrated camera rays turn pixels/bounding boxes/colors into table positions; optional triangulation combines cameras. A vision model may select an object, but metric geometry supplies its coordinates.

Station owns cameras. `perception.latest.Latest` reuses one subscription per client/queue and retains the newest entry, avoiding an unbounded stream for each observation. Freshness and calibration quality remain explicit. Pose accuracy depends on camera/base calibration and resolution; historical hold-out results are not a guarantee on the office table.

For new users, follow [offline setup](../release/QUICKSTART.md) and [build a twin](../twin/README.md). The contrib OpenCV dependency includes ArUco. Avoid the historical machine-specific paths in older examples; [the path inventory](../release/LOCAL_PATHS.md) explains overrides.
