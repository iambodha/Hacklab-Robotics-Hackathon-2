# Robot profiles

The real-arm server (`sim/real_arm_mcp.py`), the operator's instructions (`sim/OPERATOR_TEMPLATE.md`) and the operator page
contain no robot-specific numbers. They read one profile, chosen with `ROBOT_PROFILE` (a name in this folder or a file
path; default `elrobot`):

    ROBOT_PROFILE=so101 bash sim/run_agent_real.sh --model sonnet --effort low -p "..."
    ROBOT_PROFILE=so101 python3 sim/operator_ui.py
    ROBOT_PROFILE=so101 python3 sim/operator_prompt.py        # the instructions as that robot's agent sees them

| Profile | Status |
|---|---|
| `elrobot.json` | The arm used on 21 Nov. Behaviour is identical to the code before the profile existed (checked against a recorded reference: 48 IK cases and 339 motor commands). |
| `so101.json` | SO-101 follower, 5 arm joints + gripper. Kinematics from the public URDF (`so101/so101_new_calib.urdf`, TheRobotStudio/SO-ARM100). **Never run on hardware.** Offline, every tool works against a simulated 6-motor bus. Fill in the fields listed in its `_to_verify` first. |

## What a profile holds
- `motors`: servo ids of the arm joints and the gripper, their names, the goal register.
- `kinematics`: the model. `workspace_xml` = an existing MuJoCo XML (the ElRobot shares the simulator's model);
  `urdf` = any URDF, loaded for kinematics only (no meshes needed) with the joints by name, the TCP point and axes
  on a link (`tcp_R`: +y = out of the gripper, +x = jaw closing direction) and the wrist camera pose on a link.
  `arm_mapping`: `range_fraction` (angle = position inside the calibrated range, mapped onto the URDF limits) or
  `ticks_centered` (angle = sign x (raw tick - zero tick), for arms calibrated around a zero pose).
- `cameras`, `workspace`, `poses` (park, survey, carry height), `motion` (speeds, tolerances, contact threshold),
  `gripper`, `ik`, `pick`, `drop`, `stir`.

## Bringing up a new arm
1. Copy a profile, set `bus_serial`, `motors` and `kinematics` (URDF path, joint names, TCP link).
2. `ROBOT_PROFILE=<name> python3 sim/operator_prompt.py` must print instructions without errors.
3. With the arm on Station and torque OFF: check `joint_state` and `tcp_pose` while moving each joint by hand. Fix
   `signs` / `zero_ticks` until the reported gripper position follows the real one.
4. Measure the wrist camera pose (`T_link_camera`) and check `pixel_to_table` on an object at a known spot.
5. Only then enable torque and try `move_tcp` with small moves high above the table. Tune `workspace`, `poses`,
   `gripper` and `pick.flat_z` for this arm.
