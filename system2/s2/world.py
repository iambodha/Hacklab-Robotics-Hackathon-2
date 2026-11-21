"""Shared world state for both systems: tracked entities, the gripper, and how sure we are.

System 2 names things (`track`), the tracker follows them through detections from both
cameras, and every belief carries its evidence and age. Nothing here reads simulator state.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from .arm import Frame
from .geometry import wrap_deg
from .perception import Detection, detect_overview, detect_wrist

STALL_GAP = float(os.environ.get('S2_STALL_GAP', 0.012))    # jaw opening left short of the command
STALL_LOAD = int(os.environ.get('S2_STALL_LOAD', 200))      # j8 load (0..1000) while stalled
# Real arm: its gripper eases the squeeze after a stall (load drops), but j8 stays above ~0.045 when it holds
# something (real_arm_mcp CLOSED_NORM). Set S2_GRIP_MIN_OPEN=0.045 there; the twin's cube stops near 0.03.
GRIP_MIN_OPEN = float(os.environ.get('S2_GRIP_MIN_OPEN', 1.0))
RELEASE_MAX_LEVEL = 30.0   # deg: steeper closing axis at release = drop not guaranteed


@dataclass
class Entity:
    name: str
    kind: str                     # object | container | point
    height: float                 # metres, top above the table (0 for points on the table)
    xyz: np.ndarray               # belief: centre (objects/containers), the point itself (points)
    size: np.ndarray = field(default_factory=lambda: np.zeros(2))  # footprint [long, short] m
    yaw: float | None = None      # deg of the long side (objects seen from the wrist camera)
    last_seen: float = 0.0
    seen_by: str = ''
    visible: bool = False
    attached: bool = False        # held by the gripper: position follows the TCP
    grasp_xy: np.ndarray | None = None   # where it was picked from (to catch a missed grasp)
    note: str = ''
    rgb: tuple = (0, 0, 0)
    depth: float | None = None    # containers: extent away from the camera, if System 2 knows it
    release_doubtful: bool = False  # let go with a steep jaw axis: it may not have dropped

    def age(self, now: float) -> float:
        return now - self.last_seen

    def summary(self, now: float) -> dict:
        d = {'kind': self.kind, 'xyz_m': [round(float(v), 3) for v in self.xyz]}
        if self.kind == 'point':
            return d
        d.update(height_m=self.height, footprint_m=[round(float(v), 3) for v in self.size],
                 visible=self.visible, seen_by=self.seen_by or None, last_seen_s_ago=round(self.age(now), 1))
        if self.yaw is not None:
            d['yaw_deg'] = round(self.yaw, 1)
        if self.attached:
            d['held'] = True
        if self.note:
            d['note'] = self.note
        return d


@dataclass
class Gripper:
    tcp: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tilt_deg: float = 0.0
    jaw_yaw_deg: float = 0.0
    jaw: float = 1.0              # measured opening, 1 = open
    jaw_cmd: float | None = None  # last commanded opening (None: not commanded by us yet)
    jaw_load: int = 0
    jaw_level_deg: float | None = None
    faults: list[str] = field(default_factory=list)

    @property
    def stalled(self) -> bool:
        """Jaws stopped short of a closing command while pushing: something is between them. (In the
        twin a 2 cm cube stops them at about 0.03 with load 1000; closing on nothing reaches the target.)"""
        return (self.jaw_cmd is not None and self.jaw_cmd < 0.5 and self.jaw - self.jaw_cmd >= STALL_GAP
                and (self.jaw_load >= STALL_LOAD or self.jaw >= GRIP_MIN_OPEN))

    def summary(self) -> dict:
        return {'tcp_xyz_m': [round(float(v), 3) for v in self.tcp], 'jaws_from_vertical_deg': round(self.tilt_deg),
                'jaw_axis_yaw_deg': round(self.jaw_yaw_deg), 'jaw_axis_from_level_deg': self.jaw_level_deg,
                'jaw_open': round(self.jaw, 3),
                'jaw_commanded': None if self.jaw_cmd is None else round(self.jaw_cmd, 3),
                'something_between_jaws': self.stalled, 'faults': self.faults}


class World:
    def __init__(self, workspace=(-0.42, 0.42, -0.08, 0.5)):
        self.workspace = workspace
        self.entities: dict[str, Entity] = {}
        self.gripper = Gripper()
        self.frame: Frame | None = None
        self.detections: dict[str, Detection] = {}
        self.t = 0.0

    # --- observation -------------------------------------------------------
    def update(self, frame: Frame):
        self.frame, self.t = frame, frame.t
        g = self.gripper
        g.tcp, g.tilt_deg, g.jaw_yaw_deg, g.jaw_level_deg = frame.tcp, frame.tilt_deg, frame.jaw_yaw_deg, frame.jaw_level_deg
        j8 = frame.joints['j8']
        g.jaw, g.jaw_load = j8.norm, j8.load
        g.faults = [f'{n} status {hex(j.status)}' for n, j in frame.joints.items() if j.status] + \
                   [f'{n} torque off' for n, j in frame.joints.items() if not j.torque]
        dets = []
        if 'overview' in frame.images and 'overview' in frame.cameras:
            arm_pts = [[0, 0, z] for z in (0.01, 0.05, 0.1)] + [frame.cameras['wrist'].t.tolist(), frame.tcp.tolist(),
                                                               ((frame.cameras['wrist'].t + frame.tcp) / 2).tolist()]
            dets += detect_overview(frame.images['overview'], frame.cameras['overview'], arm_pts, self.workspace)
        if 'wrist' in frame.images and 'wrist' in frame.cameras and g.tilt_deg < 60:
            dets += detect_wrist(frame.images['wrist'])
        self.detections = {d.id: d for d in dets}
        self._update_grasp()
        self._associate(dets)

    def _update_grasp(self):
        g = self.gripper
        held = [e for e in self.entities.values() if e.attached]
        if g.stalled and not held:
            near = [e for e in self.entities.values() if e.kind == 'object' and
                    np.hypot(*(e.xyz[:2] - g.tcp[:2])) < 0.025 and abs(e.xyz[2] - g.tcp[2]) < 0.035]
            if near:
                e = min(near, key=lambda e: np.hypot(*(e.xyz[:2] - g.tcp[:2])))
                e.attached, e.grasp_xy, e.note = True, e.xyz[:2].copy(), 'held: jaws stalled on it'
        elif held and self._let_go():
            level = self.frame.jaw_level_deg if self.frame else None
            for e in held:
                e.attached, e.visible, e.note = False, False, f'released at {np.round(g.tcp, 3).tolist()}'
                e.xyz = np.array([g.tcp[0], g.tcp[1], max(e.height / 2, g.tcp[2] - 0.01)])
                e.last_seen = self.t
                # With the closing axis steep, the lower jaw is under the object: it may not have dropped.
                e.release_doubtful = level is not None and level > RELEASE_MAX_LEVEL
                if e.release_doubtful:
                    e.note += f' with the jaw axis {level:.0f} deg from level: it may still rest on the lower jaw'
        elif not held and g.jaw_cmd is not None and g.jaw_cmd <= 0.05 and g.jaw <= 0.05:
            # Closed fully on nothing: whatever we believed was between the jaws is not there.
            for e in self.entities.values():
                if e.kind == 'object' and np.hypot(*(e.xyz[:2] - g.tcp[:2])) < 0.015 and abs(e.xyz[2] - g.tcp[2]) < 0.03:
                    e.last_seen, e.note = -1e9, 'missed: the jaws closed on nothing where it was believed to be'
        for e in self.entities.values():
            if e.attached:
                e.xyz = g.tcp.copy() + np.array([0, 0, -0.005])
                e.last_seen, e.seen_by, e.visible = self.t, 'gripper', True

    def _let_go(self) -> bool:
        """Hysteresis on a grip: load fluctuates while carrying, so only the jaws opening, or closing
        onto nothing (the object slipped out), ends it."""
        g = self.gripper
        return g.jaw_cmd is None or g.jaw_cmd >= 0.5 or g.jaw - g.jaw_cmd < STALL_GAP / 2

    def _estimate(self, e: Entity, d: Detection):
        cam = self.frame.cameras[d.camera]
        if e.kind == 'container' and d.camera == 'overview':
            box = d.container_footprint(cam, e.depth)
            if box is None:
                return None
            return np.array([box[0][0], box[0][1], e.height / 2]), (box[0], box[1], 0.0)
        # Seen from above (wrist), a container's outline is its rim; an object's centroid is about mid-height.
        p = d.at_height(cam, e.height if e.kind == 'container' else e.height / 2)
        if p is None:
            return None
        p[2] = e.height / 2
        fp = d.footprint(cam, e.height)  # outline of the top face (wrist view looks down onto it)
        return p, fp

    def _associate(self, dets: list[Detection]):
        used = set()
        objs = [e for e in self.entities.values() if e.kind != 'point']
        for e in objs:
            e.visible = e.attached
        # A "held" object seen back at its pick-up spot after the gripper rose means the grasp missed.
        for e in objs:
            if not (e.attached and self.gripper.tcp[2] > e.height + 0.03):
                continue
            for d in dets:
                est = self._estimate(e, d)
                if est and d.id not in used and np.hypot(*(est[0][:2] - e.grasp_xy)) < 0.02 and self._size_ok(e, est[1]):
                    e.attached, e.note = False, 'grasp missed: it is still on the table'
                    used.add(d.id)
                    self._accept(e, d, *est)
                    break
        # Wrist detections are precise but only exist up close; prefer them. Something not seen for a
        # while may have been pushed: search wider for it (the size check still has to pass).
        for cam_name, gate in (('wrist', 0.035), ('overview', 0.05)):
            for e in objs:
                if e.visible:
                    continue
                # Containers are furniture: only small, size-consistent refinements (an arm passing over a box
                # merges with it in the image and would drag the estimate away).
                radius = 0.02 if e.kind == 'container' else gate if self.t - e.last_seen < 2.0 else 3 * gate
                best = None
                for d in dets:
                    if d.id in used or d.camera != cam_name:
                        continue
                    est = self._estimate(e, d)
                    if est is None:
                        continue
                    dist = float(np.hypot(*(est[0][:2] - e.xyz[:2])))
                    if dist <= radius and self._size_ok(e, est[1]) and (best is None or dist < best[0]):
                        best = (dist, d, est)
                if best:
                    used.add(best[1].id)
                    self._accept(e, best[1], *best[2])

    def _size_ok(self, e: Entity, fp) -> bool:
        if fp is None or not e.size.any():
            return True
        lo, hi = (0.85, 1.18) if e.kind == 'container' else (0.4, 2.5)
        return lo < fp[1][0] / max(e.size[0], 1e-3) < hi

    def _accept(self, e: Entity, d: Detection, p, fp):
        e.xyz = p
        e.visible, e.last_seen, e.seen_by = True, self.t, d.camera
        if fp is not None and (not e.size.any() or (d.camera == 'wrist' and e.kind == 'object')):
            e.size = fp[1]
            if d.camera == 'wrist' and e.kind == 'object':
                e.yaw = fp[2]
        if not e.attached and e.note.startswith('released'):
            e.note, e.release_doubtful = '', False

    # --- System 2 bookkeeping -----------------------------------------------
    def track(self, name: str, detection_id: str, height: float, kind: str, depth: float | None = None) -> Entity:
        if kind not in ('object', 'container'):
            raise ValueError("kind must be 'object' or 'container' (use mark_point for points)")
        d = self.detections.get(detection_id)
        if d is None:
            raise ValueError(f'no detection {detection_id!r} in the latest observation; call look() or scene() first')
        e = Entity(name, kind, float(height), np.zeros(3), rgb=d.rgb, depth=depth)
        est = self._estimate(e, d)
        if est is None:
            raise ValueError(f'{detection_id} does not project onto the table')
        self._accept(e, d, *est)
        if est[1] is not None:
            e.size = est[1][1]
        self.entities[name] = e
        return e

    def mark_point(self, name: str, xyz) -> Entity:
        e = Entity(name, 'point', 0.0, np.asarray(xyz, float), last_seen=time.monotonic(), visible=True)
        self.entities[name] = e
        return e

    def pixel_to_table(self, camera: str, u: float, v: float, z: float) -> np.ndarray:
        if self.frame is None or camera not in self.frame.cameras:
            raise ValueError(f'no calibration for camera {camera!r}')
        p = self.frame.cameras[camera].to_plane(u, v, z)
        if p is None:
            raise ValueError(f'pixel ({u}, {v}) of {camera} does not reach the plane z = {z}')
        return p

    def summary(self) -> dict:
        now = self.t
        return {'gripper': self.gripper.summary(),
                'entities': {n: e.summary(now) for n, e in self.entities.items()},
                'untracked_detections': [self.describe(d) for d in self.detections.values()]}

    def describe(self, d: Detection) -> dict:
        out = {'id': d.id, 'camera': d.camera, 'pixel': [round(d.u), round(d.v)], 'area_px': d.area, 'color': d.color}
        cam = self.frame.cameras.get(d.camera) if self.frame else None
        if cam is not None:
            p = d.at_height(cam, 0.01)
            if p is not None:
                out['table_xy_if_2cm_tall'] = [round(float(p[0]), 3), round(float(p[1]), 3)]
        return out


def jaw_yaw_error(jaw_yaw: float, obj_yaw: float) -> float:
    """Jaws close along their axis; a square-ish object is gripped on parallel faces every 90 deg."""
    return abs(wrap_deg(jaw_yaw - obj_yaw, 90.0))
