"""What the console draws over the overview camera: the world grid, axes, reach circle, ArUco markers, perception's
estimates, the bin, the TCP and the planned TCP path of the running (or last accepted) execution.

Everything is projected with the same camera model perception uses (executor.camera_info(): the twin's camera in sim,
scene_calib.json otherwise), pinhole without distortion, and returned as normalised image coordinates (0..1) so the
page can draw it over a frame of any size. A point behind the camera is None.
"""
from __future__ import annotations

import numpy as np

BIN_WORDS = ("bin", "bucket", "box", "basket", "tray")
GRID_STEP = 0.05
DEFAULT_EXTENT = ((-0.05, -0.40), (0.55, 0.40))


class Camera:
    def __init__(self, info: dict):
        self.R = np.array(info["R_base_cam"], float)      # camera (OpenCV axes) -> world rotation
        self.t = np.array(info["t_base_cam"], float)      # camera centre, world
        self.w, self.h = float(info["width"]), float(info["height"])
        self.fx, self.fy, self.cx, self.cy = (float(info[k]) for k in ("fx", "fy", "cx", "cy"))
        self.source = info.get("source", "")

    def uv(self, p) -> list[float] | None:
        c = self.R.T @ (np.asarray(p, float) - self.t)
        if c[2] <= 0.02:
            return None
        u, v = self.fx * c[0] / c[2] + self.cx, self.fy * c[1] / c[2] + self.cy
        if not (-0.5 * self.w < u < 1.5 * self.w and -0.5 * self.h < v < 1.5 * self.h):
            return None
        return [round(u / self.w, 4), round(v / self.h, 4)]

    def line(self, pts) -> list:
        return [self.uv(p) for p in pts]

    def radius(self, p, r_m: float) -> float | None:
        """Approximate projected radius (normalised by width) of a horizontal circle of r_m around p."""
        a = self.uv(p)
        rs = [self.uv(np.asarray(p, float) + d) for d in ([r_m, 0, 0], [0, r_m, 0])]
        rs = [np.hypot((b[0] - a[0]) * self.w, (b[1] - a[1]) * self.h) / self.w for b in rs if a and b]
        return round(float(max(rs)), 4) if rs else None


def tcp_path(kin, qs) -> list[list[float]]:
    return [kin.fk(np.asarray(q, float))[0].tolist() for q in qs or []]


def build(console) -> dict:
    ex = console.ex
    info = (ex.camera_info() or {}).get("overview")
    if not info:
        return {"camera": None, "why": "no overview camera model (scene_calib.json missing)"}
    cam = Camera(info)
    out: dict = {"camera": {"role": "overview", "source": cam.source, "size": [cam.w, cam.h]}}

    (x0, y0), (x1, y1) = DEFAULT_EXTENT
    if cam.source == "scene_calib.json" and ex.scene and ex.scene.get("table", {}).get("extent_xy"):
        (x0, y0), (x1, y1) = ex.scene["table"]["extent_xy"]
    xs = np.arange(np.ceil(x0 / GRID_STEP) * GRID_STEP, x1 + 1e-9, GRID_STEP)
    ys = np.arange(np.ceil(y0 / GRID_STEP) * GRID_STEP, y1 + 1e-9, GRID_STEP)
    def gl(axis, v, pts):
        major = bool(abs(round(v / 0.1) * 0.1 - v) < 1e-6)
        return {"major": major, "label": f"{axis} {v + 0:.1f}" if major else None, "pts": cam.line(pts)}
    out["grid"] = ([gl("x", float(x), [[x, y, 0] for y in np.linspace(y0, y1, 12)]) for x in xs]
                   + [gl("y", float(y), [[x, y, 0] for x in np.linspace(x0, x1, 12)]) for y in ys])
    out["axes"] = {"o": cam.uv([0, 0, 0]), "x": cam.uv([0.1, 0, 0]), "y": cam.uv([0, 0.1, 0]), "z": cam.uv([0, 0, 0.1])}
    R = float(ex.kin.P.workspace.max_radius)
    out["reach"] = cam.line([[R * np.cos(a), R * np.sin(a), 0] for a in np.linspace(-np.pi / 2, np.pi / 2, 40)])

    markers = []
    if cam.source == "scene_calib.json" and ex.scene:
        s = ex.scene.get("markers", {}).get("size_m", 0.1) / 2
        for m in ex.scene.get("markers", {}).get("items", []):
            T = np.array(m["T_world_marker"], float)
            corners = [T[:3, :3] @ np.array([dx, dy, 0]) + T[:3, 3] for dx, dy in ((-s, -s), (s, -s), (s, s), (-s, s))]
            markers.append({"id": m["id"], "pts": cam.line(corners), "c": cam.uv(T[:3, 3])})
    out["markers"] = markers

    objs = []
    for o in console.estimates():
        name = o.name.lower()
        kind = "bin" if any(w in name for w in BIN_WORDS) else "target" if "blue" in name else "other"
        p = np.array(o.xyz_world, float)
        ring = None
        if o.size and len(o.size) >= 2 and kind == "bin":
            sx, sy = float(o.size[0]) / 2, float(o.size[1]) / 2
            top = float(o.size[2]) if len(o.size) >= 3 else 0.0
            ring = [cam.line([p + [dx, dy, z - p[2]] for dx, dy in ((-sx, -sy), (sx, -sy), (sx, sy), (-sx, sy))]) for z in (0.0, top)]
        objs.append({"name": o.name, "kind": kind, "xyz": np.round(p, 3).tolist(), "uv": cam.uv(p), "visible": o.visible,
                     "unc_mm": o.uncertainty_mm, "r": cam.radius(p, (o.uncertainty_mm or 20) / 1000), "box": ring, "source": o.source,
                     "size": [float(v) for v in o.size] if o.size else None})
    out["objects"] = objs

    bin_fixture = None
    layout = getattr(getattr(ex.backend, "arm", None), "layout", None)
    for b in (layout or {}).get("bins") or []:
        c = ex.kin.to_world([b["pos"][0], b["pos"][1], 0.0])
        sx, sy, sz = (float(v) for v in b.get("size", [0.1, 0.1, 0.09]))
        yaw = np.radians(float(b.get("yaw", 0))) + np.radians(ex.kin.yaw_offset)
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        ring = [cam.line([c + Rz @ np.array([dx, dy, z]) for dx, dy in ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2))])
                for z in (0.0, sz)]
        bin_fixture = {"name": b.get("name", "bin"), "xyz": np.round(c, 3).tolist(), "uv": cam.uv(c), "box": ring, "source": "twin layout",
                       "size": [sx, sy, sz], "yaw_deg": round(float(np.degrees(yaw)), 1)}
        break
    out["bin_fixture"] = bin_fixture

    q = ex.motion.q_now()
    tcp = ex.kin.fk(q)[0]
    out["tcp"] = {"xyz": np.round(tcp, 3).tolist(), "uv": cam.uv(tcp), "shadow": cam.uv([tcp[0], tcp[1], 0.0])}

    eid = ex.current
    state = "running"
    if eid is None:
        accepted = [k for k, r in ex.results.items() if k in ex.plans]
        eid, state = (accepted[-1], "last") if accepted else (None, None)
    path = None
    if eid is not None and eid in ex.plans:
        plan, p = ex.plans[eid], ex.proposals.get(eid)
        segs = []
        for label, qs in plan.paths.items():
            pts = tcp_path(ex.kin, qs)
            if pts:
                segs.append({"label": label, "pts": cam.line(pts), "xyz": np.round(pts, 3).tolist(), "xyz_end": np.round(pts[-1], 3).tolist()})
        tgt = plan.target.get("xyz_world") or (plan.target.get("xy") and [*plan.target["xy"], plan.target.get("goal_z", 0.0)])
        path = {"execution_id": eid, "skill": p.skill if p else "", "state": state,
                "status": ex.results[eid].status.value if eid in ex.results else "", "segments": segs,
                "target": cam.uv(tgt) if tgt else None, "target_xyz": np.round(tgt, 3).tolist() if tgt else None}
    out["path"] = path
    return out
