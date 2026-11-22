# Real run: "put the blue object in the bin"

The integrator runs this with the user at the table. Every motion goes through the executor as one proposal
(validated, recorded, cancellable). Nothing here talks to the servos directly.

Order: **0 preflight → 1 measure → 2 dry run (jaws never close) → 3 pick and put back → 4 full task.**
Do not skip a stage. Go to the next one only when the one before passed all its checks.

## The numbers this run uses

| what | value | source |
|---|---|---|
| lid | 62 mm across, 26 mm tall, wall assumed 2 mm | `measured.json` (overview silhouette, σ_d 5 mm, σ_h 7 mm) |
| bin (transparent tub) | centre (0.2545, 0.1234) m, rim Ø 109 mm, rim 78 mm high, base Ø 88 mm | `measured.json` (picked rim/base pixels, σ centre 9 mm) |
| grasp | **rim pinch**: jaws close radially (yaw = direction base→lid), the TCP over the lid's near wall, TCP z 21 mm (pad bottoms ~5 mm above the table) | `task.plan_grasp` |
| release | carry at z 0.20, over the bin at z 0.139 (lid bottom ≥ 4 cm over the rim), guarded lower to TCP z 0.084 (pads ~1 cm inside the rim), open | `task.release_point`, `task.lowered_release` |
| **park pose (safe)** | TCP (0.19, 0.00, 0.06) m world, jaws open (0.96) = profile `poses.park` | 12 cm clear of the lid, 8 cm clear of the tub |
| arm speed | ≤ 0.02 of each joint's range per 0.12 s step, ≤ 45 ticks per step (≤ ~33°/s), closed-loop settle to 6 ticks | profile `motion` |
| guarded descents | 3 mm steps, effort window 40 ticks; contact = a joint 10+ ticks short while the two most loaded arm motors sum ≥ 250 | profile `touch` |
| contact abort | any arm motor load > 550/1000 → hold, PARTIAL | profile `motion.load_abort` |
| jaw torque | closing at 50 % (`--grip-force 0.5`, torque-limit register 500/1000), opening at 100 % | `Spec.grip_force`, executor `Motion.gripper` |
| hold check | test lift of 4 cm, then the overview must see the lid's base risen ≥ 24 mm (the touch sense cannot feel a 2 mm wall) | `plan.Runner.pick` |
| attempts | at most 3; a miss opens the jaws and goes back up to the hover height before the next attempt | `Spec.max_attempts` |

Two things are known to be uncertain on the real arm: the arm base pose in the world (σ about 2 cm / 2.5°, from
`scene_calib.json`) and the signs of j1/j5/j7 (unverified, executor report). The dry run exists to catch both.

## 0. Preflight (human + integrator)

1. **Tape-measure and tell the integrator**:
   - the lid's outer diameter, height and wall thickness;
   - the tub's inner rim diameter and rim height.

   If any value differs by more than 3 mm from the table above, edit `tasks/blue_to_bin/measured.json`
   (`blue_cap.diameter_m`, `height_m`; `bin.rim_diameter_m`, `rim_height_m`) before step 2.
2. **Lid open side up.** In the 00:58 frame it looked closed side up. The rim pinch needs one pad inside the lid, so
   turn it over. If it can't be used open side up, swap in a blue object ≤ 40 mm wide. The planner then uses an
   across grasp, which was **not** trialled in sim.
3. Clear the table except for the lid, the tub, the markers and the camera stand. Keep the cable away from the tub.
4. Station is running and both cameras are live:
   `curl -s http://127.0.0.1:8900/health` shows both cameras `live: true` at about 30 fps.
5. Arm: the leader is disconnected. Run `python3 scripts/teleop.py stop`, then `python3 scripts/arm.py hold` (torque on,
   goal = present position, per `CLAUDE.md`). Check in the Station UI (:8889) that all 8 motors have torque on and no
   error flags. The executor rejects every proposal while any motor has torque off.
6. Someone stands at the table with a hand near the arm. The integrator keeps the console open on **Cancel motion**.

## 1. Measure (read only)

```bash
cd /Users/bodha/Documents/b0dha/Robotics-Hackathon/ws/r2-task     # (or the merged integration checkout)
export NORMA_CORE_REPO=/Users/bodha/Documents/b0dha/Robotics-Hackathon/repo/norma-core
UV="uv run --no-project -q --with numpy --with scipy --with pillow --with mujoco --with opencv-contrib-python --with mcp>=1.12,<2 --with httpx"
$UV python -m tasks.blue_to_bin.measure --live 10 --out /tmp/blue_measured.json
```
Open `/tmp/measured_overview.jpg` (written next to `--out`). Check two things:
- the cyan cylinder sits on the lid;
- the yellow rim ellipse and orange base ellipse sit on the tub. The magenta dots are the picked rim/base pixels.

If the tub has moved, re-pick its 7 pixels (`rim_left/right/far/near`, `base_left/right/near`, 299×224 coordinates)
into a JSON file and add `--bin-pixels that.json`. Once the overlay fits, copy `/tmp/blue_measured.json` over
`tasks/blue_to_bin/measured.json`.
**After this, nobody moves the tub.** The success check uses this measured bin position.

## 2. Dry run: the jaws never close on anything

```bash
EXECUTOR_REAL_MOTION=1 $UV python -m tasks.blue_to_bin.serve --backend real --port 8920 --run dry
```
Open http://127.0.0.1:8920 and switch **arm enabled** on. The planner waits for that switch, then runs these steps:
1. park;
2. jaw check in the air (close empty, open);
3. above the lid;
4. down to the "straddle height" (pad bottoms 5 mm above the lid's rim) and a 3 s pause;
5. up;
6. over the bin, then a guarded lower inside the rim with the jaws open, and a 3 s pause;
7. straight up, above park, park.

**The human watches and reports:**
- **At the straddle pause:** is the inner pad over the inside of the lid and the outer pad outside its near wall, both
  at least 5 mm from the wall? If not, measure the miss along the world axes (+x = away from the arm base, +y = the
  arm's left). The correction is minus that miss, in mm.
- **At the in-bin pause:** are both open pads inside the rim with ≥ 5 mm clearance, about 1 cm below the rim, without
  touching the tub?
- **Throughout:** nothing is touched, the tub does not move, the cable is not dragged, and no joint goes near an end stop.

If the dry run needed a correction, run it again with `--tcp-offset-mm dx,dy[,dz]` until both pauses look right.
For the height, run the console's `touch_table` skill at an empty spot next to the lid. If `vs_nominal_mm` is more
than 3 mm, pass the negated value as dz. Write the offset down: stages 3 and 4 use the same one.

Stop the server (Ctrl-C) after each run. The report goes to `tasks/blue_to_bin/results/real_dry_<time>.json`, and the
recording is on the recorder service (http://127.0.0.1:8900).

## 3. Pick and put back

```bash
EXECUTOR_REAL_MOTION=1 $UV python -m tasks.blue_to_bin.serve --backend real --port 8920 --run pick --tcp-offset-mm dx,dy,dz
```
The planner runs:
1. approach;
2. guarded descent;
3. close;
4. test lift of 4 cm and a camera check;
5. lift to 0.20;
6. lower back to where the lid was, open, up, park.

**The human watches:**
- The descent stops with the pads either side of the wall: no pad lands on the rim, and the lid is not pushed.
- After closing, the lid comes up with the jaws. It is expected to tilt as it hangs from one edge. Watch that it does
  not slip out halfway.
- The tub is never touched.

Pass = `stage_ok: true` in the report, and the lid back on the table within about 1 cm of where it was.

## 4. Full task

```bash
EXECUTOR_REAL_MOTION=1 $UV python -m tasks.blue_to_bin.serve --backend real --port 8920 --run full --tcp-offset-mm dx,dy,dz
```
The sequence is the pick from stage 3, then:
1. carry above the bin at 0.20;
2. down to 0.139 over the bin;
3. guarded lower inside the rim;
4. open;
5. straight up, above park, park;
6. verify.

Verify passes only if both of these hold:
- the overview sees the lid within the rim footprint with its top below the rim;
- the gripper evidence says it was held after the lift and the jaws are open now.

**The human watches** for the hanging lid swinging into the rim on the way down, the tub sliding, and the lid
staying on a pad after the jaws open. Then the human looks into the tub and states the ground truth for the report.

## Abort: press Cancel motion in the console (the executor stops issuing commands and holds), then stop the server

- Any pad or finger touches the tub, or the tub moves or tips.
- The arm presses on the table or the lid, or the lid is pushed more than about 1 cm during the descent.
- The arm heads somewhere other than the announced step, especially the wrong side of the base, which would mean a j1/j5/j7 sign problem.
- A joint approaches an end stop, a motor is hot, or a servo buzzes while holding.
- The cable or the camera stand is in the path.
- At the straddle pause a pad is over the wall instead of beside it: stop and re-run the dry run with an offset.

These stop the planner on their own: a FAULT, a contact abort (load > 550), the same step REJECTED three times,
three failed attempts, or the lid not seen by the overview.
If the arm must be dropped, kill torque only when it is low over the table, because a raised arm falls.

## Record for the report

For each stage: the report JSON, the recording id, the offset used, the tape measurements, and what the human saw
at each pause. For stages 3 and 4, also whether the lid slipped, tilted or stuck to a pad.
