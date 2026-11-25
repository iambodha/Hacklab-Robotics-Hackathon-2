# Known issues

Read this before interpreting a successful tool response as task success. Everything here is a
measured or recorded limitation, not a guess.

## Real arm

- **Reach.** Beyond about 30 cm from the base the gripper leans 30–70°, positions from the wrist
  camera are off by several centimetres, and the shoulder motor reports overload. Keep things within
  about 25 cm.
- **Bottle caps and other small hard round things** are not picked reliably.
- **Touch while carrying a part** is weak: `place_at` needs two signals to agree, so a light touch
  can go unfelt and the part is then released at the lowest allowed height.
- **The overview camera is not calibrated.** It is used for context only.
- **`pick_thin` and the thin-sheet hold check** have held a real sheet through a lift once. That is
  all the hardware evidence there is.
- **Gripper rotation is not task rotation.** The recorded twist angle measures the gripper, which
  can slip on the cap (attempt 3: 201° of gripper motion moved the cap notch ~12°). A short 15 mm
  check lift also gave false "free" readings; only the guarded 60 mm extraction with visible
  separation is decisive.
- **A person steadies the bottle** in every attempt — the safety contract assumes it
  (`real_operator/OPERATOR_BOTTLE.md`).

## Simulation and transfer

- **The bottle skill was learned entirely on the real arm.** The MuJoCo twin's physical
  contact/extraction gap is unresolved: at real bottle height the twin fails the contact and
  extraction acceptance checks ([docs/learning/VALIDATION.md](learning/VALIDATION.md)). Do not read
  the bottle result as verified sim-to-real transfer.
- **SO-101 real calibration is unverified** — its connector checks ran against the twin with
  diagnostic images only ([docs/generaloid/VALIDATION.md](generaloid/VALIDATION.md)).
- **Inter-robot comms were exercised in simulation only.** Two distinct twin processes negotiated
  help and a local gripper move/refusal; no physical bottle hold or two-arm cooperation was verified
  ([docs/comms/evidence/](comms/evidence/README.md)).
- **Kitting results are simulation only**: held-out placements went 9/36 → 13/36 with no complete
  randomized kit; the selected settings completed 2/2 identical nominal kits. CPU projections omit
  occlusion and shadows ([docs/kitting/RESULTS.md](kitting/RESULTS.md)).
- **The Gaussian splat pipeline is a synthetic smoke test** (90/90 frames, 0.46 mm marker residual,
  4.6 mm camera-centre error), not measured accuracy on an arbitrary real table
  ([sim/splat/README.md](../sim/splat/README.md)).

## Tooling

- The learned bottle profile was derived from one series of changing conditions on one setup, not a
  controlled optimization. The slow-load free signature did not appear in the medium-speed demo take;
  human/visual confirmation still matters.
- The learned grasp depth sits 0.5 mm above the safety floor; on the taller office bottle the same
  profile is refused pre-motion (attempt 13) and needs a re-planned approach offset.
- Some commands and defaults are machine-local; see
  [docs/release/LOCAL_PATHS.md](release/LOCAL_PATHS.md).
