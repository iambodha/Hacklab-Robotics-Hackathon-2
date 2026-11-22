# Build a twin

The twin starts from a robot URDF/meshes, actuator and jaw/contact model, a measured table layout and calibrated cameras. MuJoCo computes motion and contact; a splat can share its frame for appearance. These remain separate representations.

1. Run [quickstart](../release/QUICKSTART.md) to obtain models and exercise an existing twin with no hardware.
2. For your own robot, implement the [connector contract](../generaloid/WRITING_A_CONNECTOR.md), including TCP, axes, gripper direction, joint order, camera roles and safety envelope.
3. Measure the table/objects and camera intrinsics, place known-size ArUco markers and solve camera/base poses with [perception](../../perception/README.md). Keep calibration and layout together for the chosen site. `home` and `office` selection is in progress in r6.
4. Inspect `sim/layout.json`, `sim/layouts/home.json` and the SO-101 bundled layout as examples. The scene builder consumes a layout name/path; set `SIM_LAYOUT` for the ElRobot server. Do not substitute home calibration for a moved table.
5. Follow [the splat pipeline](../../sim/splat/README.md) for ordinary overlapping video → SfM → metric marker alignment → training. Its marker IDs/frame must match your layout; a new marker board requires corresponding alignment targets.
6. Check rendering and marker reprojection, FK/IK and bounded motion, then independent physics outcomes over MCP. Validate hardware separately with an operator before claiming transfer.

See [original simulator details](../../sim/README.md), [measured connector checks](../generaloid/VALIDATION.md), [synthetic splat accuracy](../../sim/splat/README.md) and [physical bottle-model limitations](../learning/VALIDATION.md). The repository does not contain large trained splats or recordings, and does not yet ship the proposed composited splat/MuJoCo web viewer.
