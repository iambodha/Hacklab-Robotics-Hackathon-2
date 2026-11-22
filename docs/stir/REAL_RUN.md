# Trying the learned stir on the real arm: checklist

The stir profile (`learning/profiles/stir_cup.json`) was learned **in the twin only** (every entry says
`"source": "twin"`). This is the checklist for the first real attempts. They are recorded like the bottle's so the
real arm keeps learning from where the twin stopped. Follow `CLAUDE.md` and `LOG.md` for driving the arm: someone is
usually at the table, so go slowly and say what you are about to do.

## 0. What carries over and what does not

| learned in the twin | key | on the real arm |
|---|---|---|
| stir radius | `sim.stir.radius_m` | `stir(..., radius=)` (the real tool steers the tool tip in circles) |
| turns | `sim.stir.turns` | `stir(..., turns=)`. The twin needed one turn more than "3": the servos lag the streamed circle by ~20 deg, so 3 commanded turns gave 2.94 loops |
| bowl height | `sim.stir.height_m` | `stir(..., z=)`: tip height above the table top. Measure the real mug's inner bottom first (wall + base thickness) |
| step time | `sim.stir.step_dt_s` | the real `stir` streams `stir.points_per_turn` (12) points at `stir.dt` (0.06 s) from `sim/robots/elrobot.json`: compare **seconds per loop** (results table in README), not step times |
| wall margin, extra lean | `sim.stir.wall_margin_m`, `lean_extra_deg` | the real `stir` checks the shaft against the rim itself (`rim_height`, `cup_radius`, `stir.rim_margin`) before it moves |
| grasp height on the handle | `sim.stir.grasp_from_top_m` | no real tool yet grasps a spoon out of a mug (`stir_spoon` is twin-only). First real runs: the operator puts the spoon in the jaws |

So the first real runs test the **stirring motion** (radius, depth, turns, pace) with the real `stir` tool. The
twin's grasp-from-the-mug (`StirSkill`) is the follow-up: port it to `sim/real_arm_mcp.py` once the motion holds.

## 1. Before the arm moves

- [ ] Station running, recorder (:8900) recording, dashboard (:8920) up. `SITE=office`.
- [ ] Measure the real mug and spoon with a tape: outer diameter, height, wall, inner bottom height, spoon length
      bowl-centre to handle end, handle width. Put them in `learning/stir/task.json` if they differ from the twin's
      (8.0 cm, 9.5 cm, 4 mm; spoon 15.5 cm, 12 mm handle). Regenerate the twin: `python -m learning.stir.twin`.
- [ ] Re-run a short twin practice with the measured sizes (`practice --attempts 6`) and check that the profile's values
      still stir cleanly. If they don't, learn again before touching the real arm.
- [ ] Park the arm in the folded pose (`task.json rest_pose`: j2 0.5, j3 -1.3, j4 -0.9 rad, the rest 0) **before**
      the mug goes down. The office calibration pose reaches forward over the mug spot (in the twin its forearm
      pushed the spoon off the rim).
- [ ] Mug 19 cm in front of the arm base on its centre line (the twin's spot; reach map `sites/office/reach.json`).
      Water, not coffee, for the first runs. Paper towel under the mug.

## 2. Find the mug

- [ ] `look()`. Click the middle of the mug opening in the **wrist** image from a hover above it and use
      `pixel_to_table("wrist", u, v, <rim height>)`. The office overview camera is fine for a rough position only.
- [ ] Check the click against the tape position (expect +-1 cm).

## 3. Spoon in the jaws (operator)

- [ ] `gripper(1.0)`, the operator holds the spoon vertical between the jaws, bowl down, with the jaws on the handle
      at the twin's grasp height below the handle end (`sim.stir.grasp_from_top_m`); close in steps 0.7 -> 0.4 ->
      0.1 -> 0.0; `grip_state()` must say it holds.
- [ ] Measure `tool_length` = how far the bowl centre is below the fingertips.

## 4. One stir, recorded

- [ ] Start a recorder session (the dashboard, or `POST /recordings` with name `stir-real-1`).
- [ ] `stir(x, y, z, radius, turns, tool_length, clearance=0.08, rim_height=<measured>, cup_radius=<inner radius>)`
      with the profile values (z = `sim.stir.height_m` + the real mug's bottom offset against the twin's 4 mm).
      If it answers "STIR not started", read why (shaft against the rim): stir shallower or with a smaller radius, as
      it says. **Do not lower `stir.rim_margin` to make it start.**
- [ ] Afterwards: wrist and overview images. Did the mug move (overview, before/after)? Water on the table? Spoon
      still in the jaws, same height? Count the loops in the recording.
- [ ] Write the attempt down like the twin's: params, hypothesis, outcome (`stirred`, `pushed_cup`, `scraped_wall`,
      `too_few_loops`, ...), lesson. The twin store's format works: `learning/stir/store.py` with a root under
      `logs/learning/office/stir/`.

## 5. Only then

- [ ] Real values go into the profile as a new version (`history` entry, `source: "real"`, attempts = the real
      attempt numbers), the way `bottle_cap.json` was made: by a human-approved change, never automatically. A real
      skill-config change needs `python -m learning.skillcfg approve <id> --by <name>`.
- [ ] Stop and write it down if the mug moves more than 5 mm, tips, or the spoon slips: those are the twin judge's
      gates, and on hardware they are spills.
