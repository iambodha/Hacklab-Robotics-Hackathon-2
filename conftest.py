"""Repo-wide test guard: tests never talk to a live NormaCore Station unless asked to.

22 Nov: a full test run on the dev machine opened dozens of Station connections per second while an operator agent
was driving the real arm, and the arm server's motor reads stalled mid-task. Every Station client in this repo reads
STATION_HOST, so pointing it at a name that cannot resolve keeps tests (and the subprocesses they start) off the
live Station. Opt in with ROBOT_LIVE_TESTS=1 when the arm is idle and you mean it.
"""
import os

if os.environ.get("ROBOT_LIVE_TESTS") != "1":
    os.environ["STATION_HOST"] = "station-disabled-in-tests.invalid"
