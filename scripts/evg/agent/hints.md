Arm-specific facts (from the operator's notes; edit this file as you learn more):
- Joints are servo steps, 4096 = 360 deg. Actions use degrees.
- In the top-down grasp pose: j2 (shoulder) increase = gripper goes DOWN; j2 decrease = up. Lifting first is always safe.
- j1 (base) decrease = object moves LEFT in the gripper image (about 5 px per degree at grasp height) and the arm turns toward the white box.
- j3 (elbow) decrease = reaches further out.
- Gripper: opening 0 = closed, 1 = fully open. Holding an object leaves the gripper slightly open.
- Gripper image is 299x224; the jaws' centre is about x=150. Centre the object between the jaws before descending.
- Moves are small steps (a few degrees); look at the result in the next image before continuing.
