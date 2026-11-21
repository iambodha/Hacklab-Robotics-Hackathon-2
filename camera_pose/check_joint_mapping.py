"""Which normalized-joint -> URDF-radian convention is right? Evidence from the taught grasp grid.

All nine poses in robot_mcp/data/grid_grasp.json touch the same flat sheet, so the correct
convention puts all nine fingertip centres (TCP) at one height. Ranks every convention
(direct = lower + n*span, per-joint flips of j2..j6) by the spread of TCP height.
j1, j5, j7 barely change TCP height, so this test CANNOT determine their sign.

usage: python3 camera_pose/check_joint_mapping.py [--urdf FILE]
"""
import argparse
import itertools
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
URDF = Path(os.environ.get("NORMA_CORE_REPO", "/mnt/d/Robotics-Hackathon/repos/norma-core")) / \
    "hardware/elrobot/simulation/elrobot_follower.urdf"
TCP = np.r_[json.loads((HERE / "wrist_camera.json").read_text())["tcp_m"], 1.0]


class Urdf:
    """Minimal serial-chain FK for elrobot_follower.urdf (all joint rpy are 0)."""

    def __init__(self, path, tip="Gripper_Base_v1_1"):
        root = ET.parse(path).getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        self.chain, link = [], tip
        while link in by_child:
            self.chain.insert(0, by_child[link])
            link = by_child[link].find("parent").get("link")
        self.limits = {j.get("name"): tuple(float(j.find("limit").get(k)) for k in ("lower", "upper"))
                       for j in root.findall("joint") if j.find("limit") is not None}

    def fk(self, q):
        """q: {joint_name: radians} -> 4x4 base_link -> tip link."""
        T = np.eye(4)
        for j in self.chain:
            o = np.eye(4)
            o[:3, 3] = list(map(float, j.find("origin").get("xyz").split()))
            T = T @ o
            if j.get("type") == "revolute":
                a = np.array(list(map(float, j.find("axis").get("xyz").split())))
                a /= np.linalg.norm(a)
                K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
                th = q.get(j.get("name"), 0.0)
                R = np.eye(4)
                R[:3, :3] = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
                T = T @ R
        return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", type=Path, default=URDF)
    robot = Urdf(ap.parse_args().urdf)
    grid = json.loads((HERE.parent / "robot_mcp/data/grid_grasp.json").read_text())
    poses = [v["grasp_pose"] for v in grid.values()]
    rows = []
    for flips in itertools.product([False, True], repeat=5):
        z, down = [], []
        for p in poses:
            q = {}
            for i in range(7):
                name = f"rev_motor_0{i + 1}"
                lo, hi = robot.limits[name]
                n = p[f"j{i + 1}"]
                if 1 <= i <= 5 and flips[i - 1]:
                    n = 1 - n
                q[name] = lo + n * (hi - lo)
            T = robot.fk(q)
            z.append((T @ TCP)[2])
            down.append((T[:3, :3] @ [0, 1, 0])[2])
        rows.append((float(np.std(z)), float(np.mean(z)), float(np.mean(down)),
                     "".join("F" if f else "." for f in flips)))
    rows.sort()
    print("std_z_m  mean_z_m  gripper_axis_z  flips(j2 j3 j4 j5 j6; F = 1-n)")
    for r in rows[:6]:
        print("%.4f   %.3f     %.2f            %s" % r)


if __name__ == "__main__":
    main()
