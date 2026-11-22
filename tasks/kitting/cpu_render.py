"""CPU camera projection for headless hosts without a GL connection.

Physics stays MuJoCo. This renders the printed plane and object top silhouettes from
model/data camera transforms. It has no arm shadows or arm occlusion; it is a
perception smoke-test renderer, not a photorealistic substitute for the GL twin.
"""
import cv2
import numpy as np
from perception.geometry import project
from .locate import CameraModel
from .sheet import draw, footprint_mm
from .spec import Pose2D


def render(sim, cam_name):
    m, d = sim.m, sim.d
    cid = m.camera(cam_name).id
    W, H = sim.layout['image_size']
    f = H / 2 / np.tan(np.radians(m.cam_fovy[cid]) / 2)
    from perception.geometry import intr_from_K
    K = np.array([[f, 0, (W-1)/2], [0, f, (H-1)/2], [0, 0, 1.]])
    T = np.eye(4)
    T[:3, :3] = d.cam_xmat[cid].reshape(3, 3) @ np.diag([1., -1., -1.])
    T[:3, 3] = d.cam_xpos[cid]
    cam = CameraModel(T, intr_from_K(K, [0]*5, W, H), W, H)
    def pixels(xyz):
        return project(cam.T, cam.intr, W, H, xyz)[0].astype(np.float32)
    S = sim.spec
    tex = np.asarray(draw(S, 4)).copy()
    w, h = S.paper_mm
    corners = np.array([[-w/2,h/2],[w/2,h/2],[w/2,-h/2],[-w/2,-h/2]]) / 1000
    body = d.body('kit_sheet')
    xyz = np.c_[corners, np.full(4, .0004)] @ body.xmat.reshape(3,3).T + body.xpos
    src = np.float32([[0,0],[tex.shape[1]-1,0],[tex.shape[1]-1,tex.shape[0]-1],[0,tex.shape[0]-1]])
    Hm = cv2.getPerspectiveTransform(src, pixels(xyz))
    gray = cv2.warpPerspective(tex, Hm, (W,H), flags=cv2.INTER_AREA, borderValue=140)
    img = np.repeat(gray[:,:,None], 3, axis=2)
    for name, obj in sim.obj.items():
        p = sim.obj_pose(name)
        poly = footprint_mm(S, obj['class'], Pose2D(0,0,0)) / 1000
        pts = np.c_[poly, np.full(len(poly), S.class_dims(obj['class'])['height']/2)]
        b = d.body(name)
        uv = pixels(pts @ b.xmat.reshape(3,3).T + b.xpos)
        colour = tuple(int(255*v) for v in S.classes()[obj['class']]['rgba'][:3])
        cv2.fillConvexPoly(img, np.round(uv).astype(np.int32), colour, cv2.LINE_AA)
    brightness = np.clip(m.vis.headlight.ambient[0] / .75, .6, 1.1)
    return np.clip(img.astype(float)*brightness, 0, 255).astype(np.uint8)
