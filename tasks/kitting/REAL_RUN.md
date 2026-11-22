# Kitting: tomorrow's operator handoff

NormaCore Station remains the hardware gateway. The runner submits one proposal per motion to the existing
executor; its real backend requires `EXECUTOR_REAL_MOTION=1` AND the console's arm-enabled switch. No real motion
was attempted in this branch. A trained simulation profile is a candidate for transfer, not a real-arm qualification.

1. Use the merged checkout and current calibrated robot profile. Confirm motor ranges, joint signs, tool/pad
   dimensions and arm-base calibration with the operator. Kitting's twin applies a 3.5 mm inward pad correction and
   a 6 mm pad-tip shift to match a nominal 10 mm table-touch TCP height. Physically verify those assumptions before
   a low grasp; do not compensate by blindly lowering the profile.
2. Generate/print the A4 landscape PDF at **100% / actual size**, with all fit-to-page scaling off. Measure the
   100 mm bar (tolerance 1 mm). Markers are DICT_4X4_50 ids 10–13; table ids 20–23 must remain distinct.
3. Place the sheet flat, bottom edge toward the arm, nominal centre `(0, 0.24)` m in the arm/base frame. Tape it
   down. Start within ±25 mm x, ±20 mm y, ±10° yaw. Clear cable and other objects from the swept area.
4. Use two 30×12 mm caps, two 30×20×15 mm boxes, and two M4×25 screws. Put them on their **printed start spots**,
   matching the outline and screw-head direction. These are fixtures, not arbitrary loose-object detection. The
   299×224 overview cannot reliably resolve a 4 mm screw. Keep placement error below ~3 mm and yaw below ~6°.
5. Capture a fresh overview image using the existing recorder/camera workflow. Do not resize/crop it. The current
   layout expects 299×224, with its calibrated intrinsics/distortion and camera-to-base transform. If the camera
   moved, recalibrate the table first. Run location and inspect the projected cell boundaries on the overlay:

```bash
export NORMA_CORE_REPO=/Users/bodha/Documents/b0dha/Robotics-Hackathon/repo/norma-core
# Use this shell function for every command below (zsh/bash).
kitpy() { uv run --no-project -q --with numpy --with scipy --with pillow --with reportlab --with mujoco \
  --with opencv-contrib-python --with 'mcp>=1.12,<2' --with httpx python "$@"; }
kitpy -m tasks.kitting.sheet --out '/Volumes/Dell HDD/Robotics-Hackathon/r5-kitsim/print'
kitpy -m tasks.kitting.locate /tmp/overview.jpg --out /tmp/kit_overlay.jpg
kitpy -m tasks.kitting.run_real --image /tmp/overview.jpg --mode dry --out /tmp/kit_commands.json
```

The offline command reads `tasks/kitting/profiles/kit_learned.json` as-is. It never connects to Station. Review the
JSON world-coordinate commands. Location must find at least two sheet markers for the real runner; any poor fit
or mismatched resolution stops it. A small residual does not rule out a systematic camera/base calibration error.

6. With a person at the table and Cancel motion visible, run the existing executor console on **8930**, leaving
   recorder/dashboard services untouched. Dry mode never closes on an object, and stays 20 mm above learned
   grasp/release heights. Review the entire path and lateral alignment before enabling in the console:

```bash
EXECUTOR_REAL_MOTION=1 kitpy -m tasks.kitting.run_real --serve --mode dry --only cap_1 \
  --port 8930 --out /tmp/kit_dry.json
```

The runner captures a fresh overview after arm enable, then fixes the sheet pose for that run. Nobody moves the
sheet or camera during it. Stop this task server between stages. Dry mode is an alignment check, not a table-height
measurement. Use the existing executor `touch_table` workflow on an empty area and verify the nominal 10 mm TCP
contact height. If calibration/pads differ, stop and update the twin/calibration with evidence before transfer.

7. After the operator passes dry alignment and table touch, test a single cap pick-and-put-back, then the same cap
   in its cell. The exact learned force, yaw, grasp height, place height and opening are used without retuning:

```bash
EXECUTOR_REAL_MOTION=1 kitpy -m tasks.kitting.run_real --serve --mode pick --only cap_1 \
  --port 8930 --out /tmp/kit_pick.json
EXECUTOR_REAL_MOTION=1 kitpy -m tasks.kitting.run_real --serve --mode full --only cap_1 \
  --port 8930 --out /tmp/kit_cap.json
```

8. Restore all objects to spots. Only after passing single-object checks, run the whole kit:

```bash
EXECUTOR_REAL_MOTION=1 kitpy -m tasks.kitting.run_real --serve --mode full --port 8930 --out /tmp/kit_full.json
```

Every PARTIAL/rejected/fault/timed-out motion stops the runner; an empty grasp stops it too. It does not blindly
retry or declare visual success. Cancel holds the arm through the executor. Results are saved next to `--out` on
successful sequence completion; individual proposals remain in the executor recorder even if the sequence stops.
Inspect every object's complete footprint in its labelled cell and measure centre error with a ruler. The overview
alone cannot certify the screw footprint. Record misses and slips as real episodes before any later tuning.

## Transfer limits

The training controller is task-local and fast; the handoff controller is the existing executor. Both read the same
plan/profile, but servo ramps, guard/contact logic and holding detection differ. The executor simulation bridge in
`executor_sim.py` checks this boundary. No claim of 'little or no tuning' is justified until the above real checks
pass. Friction, hollow caps, screws rolling, grasp-force calibration, low-resolution marker decoding, and arm-base
calibration remain the principal risks. No optional cloud model/key is needed or loaded.
