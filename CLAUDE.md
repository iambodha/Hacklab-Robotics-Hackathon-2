# NormaCore robot arm — hackathon project

Read `LOG.md` first. It has the hardware IDs, every script, what worked and what didn't,
the grasp recipe, and the arm's state when the last session ended.

Rules for driving the arm:
- Station must be running (`scripts/run-station.sh`, TCP API on localhost:8888).
- Run `python3 scripts/teleop.py stop` before any script moves the follower, then `python3 scripts/arm.py hold`.
- Move with `scripts/grasp.py step|goto`, small steps (a few degrees), and look at the cameras after each move (`python3 scripts/camera.py`, then read `frames/e091eeda.jpg` for the gripper camera and `frames/5cd55831.jpg` for the overview).
- Don't use `arm.py move` / `arm.py fwd` or trust `arm.py pose`'s tip xyz: the URDF model doesn't match this arm.
- Never `arm.py relax` with the arm raised; it drops.
- Someone is usually standing at the table. Go slowly and say what you're about to do.
