"""Derive the ElRobot wrist-camera pose from the official CAD, in the URDF gripper frame.

The URDF (hardware/elrobot/simulation/elrobot_follower.urdf) has no camera link. The
assembly STEP (hardware/elrobot/STEP/elrobot_follower_parts/Gripper.stp) contains the
standard UVC_Camera on CameraMount_square32mm. This script:

  1. parses the STEP assembly tree (no CAD kernel needed) and places every part,
  2. registers the STEP Gripper_Base onto the URDF Gripper_Base_v1_1 mesh (rigid ICP),
  3. expresses the camera lens (optical centre + axis) and the jaw fingertip centre (TCP)
     in the Gripper_Base_v1_1 link frame, and writes camera_pose/wrist_camera.json.

usage: python3 camera_pose/extract_wrist_camera.py [--hardware DIR] [--out FILE]
  DIR = norma-core/hardware/elrobot (needs the full clone; the shallow one lacks hardware/)
needs numpy + scipy.
"""
import argparse
import itertools
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


class Step:
    def __init__(self, path):
        txt = open(path, encoding="latin-1").read()
        start = txt.index("DATA;") + 5
        data = txt[start:txt.index("ENDSEC;", start)]
        self.e = {int(m.group(1)): " ".join(m.group(2).split())
                  for m in re.finditer(r"#(\d+)\s*=\s*(.*?);\s*(?=#\d+\s*=|\Z)", data, re.S)}
        self.by_type = defaultdict(list)
        for k in self.e:
            self.by_type[self.typ(k)].append(k)

    def typ(self, k):
        s = self.e[k]
        return "COMPLEX" if s.startswith("(") else s[:s.index("(")].strip()

    def args(self, k):
        s = self.e[k]
        return s[s.index("(") + 1:s.rindex(")")]

    @staticmethod
    def refs(s):
        return [int(x) for x in re.findall(r"#(\d+)", s)]

    def vec(self, k):
        a = self.args(k)
        return np.array([float(x) for x in re.findall(NUM, a[a.index("(") + 1:a.rindex(")")])])

    def axis(self, k):
        r = self.refs(self.args(k))
        p = self.vec(r[0])
        z = self.vec(r[1]) if len(r) > 1 else np.array([0, 0, 1.0])
        x = self.vec(r[2]) if len(r) > 2 else np.array([1, 0, 0.0])
        z = z / np.linalg.norm(z)
        x = x - z * (x @ z)
        x /= np.linalg.norm(x)
        T = np.eye(4)
        T[:3, :3] = np.c_[x, np.cross(z, x), z]
        T[:3, 3] = p
        return T

    def instances(self):
        """[(product_name, product_definition, T_root_part)] for every placed part."""
        pname = {}
        for pd in self.by_type["PRODUCT_DEFINITION"]:
            prod = self.refs(self.args(self.refs(self.args(pd))[0]))[0]
            pname[pd] = re.match(r"\s*'([^']*)'", self.args(prod)).group(1)
        pds = {p: self.refs(self.args(p))[0] for p in self.by_type["PRODUCT_DEFINITION_SHAPE"] if self.refs(self.args(p))}
        self.rep_of = {}
        for sdr in self.by_type["SHAPE_DEFINITION_REPRESENTATION"]:
            a, b = self.refs(self.args(sdr))[:2]
            if pds.get(a) in pname:
                self.rep_of[pds[a]] = b
        children = defaultdict(list)
        for c in self.by_type["CONTEXT_DEPENDENT_SHAPE_REPRESENTATION"]:
            rr, pdsh = self.refs(self.args(c))[:2]
            nauo = pds[pdsh]
            idt = self.refs(re.search(r"REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION\(([^)]*)\)", self.e[rr]).group(1))[0]
            a1, a2 = self.refs(self.args(idt))[:2]
            parent, child = self.refs(self.args(nauo))[:2]
            children[parent].append((child, self.axis(a2) @ np.linalg.inv(self.axis(a1))))
        out = []

        def walk(pd, T):
            out.append((pname[pd], pd, T))
            for c, Tc in children[pd]:
                walk(c, T @ Tc)

        for root in set(pname) - {c for v in children.values() for c, _ in v}:
            walk(root, np.eye(4))
        return out

    def solid_entities(self, pd):
        rep = self.rep_of[pd]
        a = self.args(rep)
        items = self.refs(a[a.index("("):a.index(")") + 1])
        seen, stack = set(), [i for i in items if self.typ(i) in ("MANIFOLD_SOLID_BREP", "BREP_WITH_VOIDS")]
        while stack:
            k = stack.pop()
            if k in seen or k not in self.e:
                continue
            seen.add(k)
            stack.extend(self.refs(self.e[k]))
        return seen

    def vertices(self, pd, T):
        P = np.array([self.vec(self.refs(self.args(k))[0]) for k in self.solid_entities(pd) if self.typ(k) == "VERTEX_POINT"])
        return P @ T[:3, :3].T + T[:3, 3]


def load_stl(path):
    b = open(path, "rb").read()
    n = np.frombuffer(b[80:84], "<u4")[0]
    d = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return np.unique(d["v"].reshape(-1, 3).astype(float).round(4), axis=0)


def register(src, dst):
    """Rigid ICP from src to dst, trying the 24 axis-aligned starts. Returns (T, median_mm, p95_mm)."""
    tree = cKDTree(dst)
    best = None
    for perm in itertools.permutations(range(3)):
        for s in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3))
            R[range(3), perm] = s
            if np.linalg.det(R) < 0:
                continue
            t = dst.mean(0) - (src @ R.T).mean(0)
            for _ in range(40):
                _, i = tree.query(src @ R.T + t)
                m = dst[i]
                U, _, Vt = np.linalg.svd((src - src.mean(0)).T @ (m - m.mean(0)))
                D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
                R = Vt.T @ D @ U.T
                t = m.mean(0) - R @ src.mean(0)
            d, _ = tree.query(src @ R.T + t)
            if best is None or np.median(d) < best[1]:
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = R, t
                best = (T, float(np.median(d)), float(np.percentile(d, 95)))
    return best


def main():
    ap = argparse.ArgumentParser()
    default_hw = Path(os.environ.get("NORMA_CORE_REPO", "/mnt/d/Robotics-Hackathon/repos/norma-core")) / "hardware/elrobot"
    ap.add_argument("--hardware", type=Path, default=default_hw)
    ap.add_argument("--out", type=Path, default=HERE / "wrist_camera.json")
    a = ap.parse_args()
    hw = a.hardware
    step = Step(hw / "STEP/elrobot_follower_parts/Gripper.stp")
    inst = step.instances()
    part = lambda name: next((pd, T) for n, pd, T in inst if n == name)

    # URDF Gripper_Base_v1_1: meshes are in assembly mm; the link origin is minus the visual origin.
    import xml.etree.ElementTree as ET
    urdf = ET.parse(hw / "simulation/elrobot_follower.urdf").getroot()
    link = urdf.find("link[@name='Gripper_Base_v1_1']")
    link0 = -1000 * np.array(list(map(float, link.find("visual/origin").get("xyz").split())))
    stl = load_stl(hw / "simulation/assets/Gripper_Base_v1_1.stl")
    T_reg, med, p95 = register(step.vertices(*part("Gripper_Base")), stl)
    to_link = lambda P: P @ T_reg[:3, :3].T + T_reg[:3, 3] - link0

    cam_pd, cam_T = part("UVC_Camera")
    cam = to_link(step.vertices(cam_pd, cam_T))
    lens = []
    for k in step.solid_entities(cam_pd):
        if step.typ(k) == "CYLINDRICAL_SURFACE":
            rad = float(re.findall(NUM, step.args(k).split(",")[-1])[0])
            if rad >= 3.0:  # lens barrel; smaller cylinders are screw holes / pins
                A = cam_T @ step.axis(step.refs(step.args(k))[0])
                lens.append((rad, to_link(A[:3, 3][None])[0], T_reg[:3, :3] @ A[:3, 2]))
    axis = np.mean([l[2] for l in lens], 0)
    axis /= np.linalg.norm(axis)
    if axis[1] < 0:
        axis = -axis  # look toward the fingertips (+y)
    base = min(lens, key=lambda l: l[0])[1]
    along = (cam - base) @ axis
    radial = np.linalg.norm((cam - base) - np.outer(along, axis), axis=1)
    optical_centre = base + axis * along[radial < 5.5].max()  # front of the lens barrel

    # Image axes (OpenCV: x right, y down, z forward). On the live wrist image the jaws are at
    # the bottom, i.e. image-down points from the camera toward the jaw plane (-z of the link).
    z_cv = axis
    y_cv = np.array([0, 0, -1.0]) - z_cv * (-z_cv[2])
    y_cv /= np.linalg.norm(y_cv)
    x_cv = np.cross(y_cv, z_cv)

    # TCP: centre between the two fingertip pads of the URDF jaws at q=0 (fully open).
    tips = []
    for name in ("Gripper_Jaw_01_v1_1", "Gripper_Jaw_02_v1_1"):
        v = load_stl(hw / f"simulation/assets/{name}.stl") - link0
        tips.append(v[v[:, 1] > v[:, 1].max() - 15])
    tcp = np.array([np.mean([t[:, 0].mean() for t in tips]), np.mean([t[:, 1].max() for t in tips]) - 10,
                    np.mean([t[:, 2].mean() for t in tips])])

    T = np.eye(4)
    T[:3, :3] = np.c_[x_cv, y_cv, z_cv]
    T[:3, 3] = optical_centre / 1000
    out = {
        "frame": "Gripper_Base_v1_1 link frame of elrobot_follower.urdf (metres); child of rev_motor_07",
        "source": {"step": "hardware/elrobot/STEP/elrobot_follower_parts/Gripper.stp",
                   "mount": "CameraMount_square32mm + UVC_Camera (standard follower assembly)",
                   "registration_median_mm": round(med, 4), "registration_p95_mm": round(p95, 3)},
        "T_link_camera": T.round(6).tolist(),
        "camera_convention": "OpenCV: x right, y down, z = optical axis",
        "optical_centre_m": (optical_centre / 1000).round(5).tolist(),
        "optical_axis": axis.round(5).tolist(),
        "tilt_below_gripper_axis_deg": round(float(np.degrees(np.arcsin(-axis[2]))), 2),
        "tcp_m": (tcp / 1000).round(5).tolist(),
        "tcp_note": "midpoint between fingertip pads, 10 mm behind the tips; gripper points along +y",
        "jaw_travel": "each jaw moves inward by 0.0255*(1 - j8_norm) m along -/+x (j8 norm 1 = open)",
        "intrinsics": None,
        "intrinsics_note": "UNKNOWN - calibrate with a checkerboard (see HANDOFF.md). Store normalised fx/W, fy/H, cx/W, cy/H.",
    }
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
