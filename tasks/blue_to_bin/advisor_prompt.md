Task: take the blue object (a jar lid) and put it in the bin (a transparent plastic tub). Use only the executor
tools. Every motion is ONE proposal: look() (or observe()), then submit() right away with that obs_id and setup_id,
then status(execution_id, wait_s=30) until it is final. The arm is already parked and the jaws are open; a jaw check
(close empty, open) has been done.

Measured scene (world frame, metres):
- Lid: {cap_d_mm:.0f} mm across, {cap_h_mm:.0f} mm tall, wall {wall_mm:.0f} mm, lying OPEN SIDE UP. It is wider than the jaws can
  grip across (max {grip_mm:.0f} mm), so pinch its WALL: one pad inside the lid, one outside.
  Its position is in the observation's objects as "blue_cap" (xyz_world = centre, uncertainty_mm = 1 sigma; z of the
  centre is ~{cap_mid_mm:.0f} mm when it rests on the table). Prefer that estimate over your own reading of the image.
- Bin: centre [{bin_x:.4f}, {bin_y:.4f}], rim diameter {rim_d_mm:.0f} mm, rim height {rim_h_mm:.0f} mm. It is light: do not touch it.

Recipe (compute the numbers from the lid's estimate c = [cx, cy]):
1. u = c / |c| (unit vector from the arm base to the lid), yaw = atan2(u_y, u_x) in degrees (jaws close radially).
   Pinch point p = c - {pinch_mm:.0f} mm * u (the lid's near wall).
2. move_tcp to [p, {hover:.3f}] with yaw_deg = yaw.
3. descend_guarded xy_world = p, z_world = {zg:.4f}, same yaw (stops on contact; PARTIAL with contact = misaligned:
   move up, look again, re-plan).
4. gripper open_fraction 0, force {force}. The touch sense usually reports "nothing between the jaws" for a 2 mm wall:
   that is NOT proof of a miss. Decide with the camera instead:
5. move_tcp to [p, {ztest:.3f}] (test lift), look(): the lid's estimated z must have risen by >= 25 mm. If it is still on the
   table: gripper open 1, move up to [p, {hover:.3f}], look again and retry from step 1 (at most 3 attempts in total).
6. move_tcp to [p, 0.20] (lift), then carry at 0.20 to above r = bin_centre - 0.5 * (c - p), then move_tcp to
   [r, {zrel:.3f}] (the lid's bottom clears the rim + 4 cm).
7. descend_guarded xy_world = r, z_world = {zlow:.3f} (pads ~1 cm below the rim, inside it), then gripper open 1.
   Opening above the rim leaves the lid hanging on a pad.
8. move_tcp straight up to [r, 0.20], then to above the park spot at 0.20, then park.
9. look(): success = the lid's estimate is within {rim_r_mm:.0f} mm of the bin centre and its top is below the rim. Report
   DONE with the numbers you saw, or BLOCKED with the reason.
Keep yaw_deg the same from step 2 to step 8.
