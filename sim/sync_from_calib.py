"""Copy the scene calibration (prompts/scene-contract.md format) into the home layout of the twin.

  python3 sim/sync_from_calib.py <scene_calib.json> [--layout sim/layouts/home.json] [--src scene_calib] [--force]

Frames: scene_calib uses the contract world (origin = follower base on the table, +x = arm forward at j1 = 0, z up).
The twin keeps the MuJoCo world = URDF base frame (+y = arm forward, as every sim tool assumes), so
p_sim = T_sim_world @ p_world with T_sim_world = Rz(+90 deg) . If the calibration's arm.T_world_base carries a rotation,
T_sim_world = inv(T_world_base) is used instead (T_world_base = the URDF base_link pose in the contract world).

Every value written gets "src": <--src>. A value whose src is already "scene_calib" is not overwritten by an
"estimate" unless --force, so running the camera estimator again never clobbers the real calibration.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RZ90 = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
PRIORITY = {"fixed": 0, "estimate": 1, "scene_calib": 2}
POSE_FILE = "camera_pose/wrist_camera.json"


def r(x, n=5):
    return np.round(np.asarray(x, float), n).tolist()


def dumps(obj, indent: int = 0) -> str:
    """JSON with dicts one key per line and lists of numbers (also 4x4 matrices) kept on one line."""
    pad, inner = " " * indent, " " * (indent + 2)
    flat = lambda v: not isinstance(v, (dict, list)) or (isinstance(v, list) and all(not isinstance(x, (dict, list)) or flat(x) for x in v))
    if isinstance(obj, dict) and obj:
        return "{\n" + ",\n".join(f"{inner}{json.dumps(k)}: {dumps(v, indent + 2)}" for k, v in obj.items()) + f"\n{pad}}}"
    if isinstance(obj, list) and obj and (not flat(obj) or (len(json.dumps(obj)) > 110 and any(isinstance(x, str) for x in obj))):
        return "[\n" + ",\n".join(inner + dumps(v, indent + 2) for v in obj) + f"\n{pad}]"
    return json.dumps(obj)


def t_sim_world(calib: dict) -> np.ndarray:
    T = np.asarray(calib.get("arm", {}).get("T_world_base", np.eye(4)), float)
    if np.allclose(T[:3, :3], np.eye(3), atol=1e-6):
        return RZ90 @ np.linalg.inv(T)        # translation only (or identity): the contract convention, x forward
    return np.linalg.inv(T)


def rigid2d(src_xy, dst_xy) -> np.ndarray:
    """4x4 rotation about z + translation (least squares, Kabsch in 2D) taking src points onto dst points."""
    a, b = np.asarray(src_xy, float), np.asarray(dst_xy, float)
    ca, cb = a.mean(0), b.mean(0)
    h = (a - ca).T @ (b - cb)
    ang = math.atan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
    M = np.eye(4)
    M[:2, :2] = [[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]]
    M[:2, 3] = cb - M[:2, :2] @ ca
    return M


def covers(table: dict, c_xy, size, yaw_deg) -> bool:
    """Does the layout table contain the rectangle (centre, size, yaw)?"""
    a, b = math.radians(yaw_deg), math.radians(table.get("yaw", 0.0))
    corners = [np.asarray(c_xy) + np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]) @ [sx * size[0] / 2, sy * size[1] / 2]
               for sx in (-1, 1) for sy in (-1, 1)]
    Rb = np.array([[math.cos(b), math.sin(b)], [-math.sin(b), math.cos(b)]])
    return all((np.abs(Rb @ (p - table["center"])) <= np.asarray(table["size"]) / 2 + 1e-3).all() for p in corners)


def fovy_of(cam: dict) -> float | None:
    if cam.get("fovy_deg"):
        return float(cam["fovy_deg"])
    k = cam.get("K_normalized")
    return math.degrees(2 * math.atan(0.5 / k["fy"])) if k and k.get("fy") else None


def apply_calib(calib: dict, layout: dict, src: str = "scene_calib", force: bool = False) -> tuple[dict, list]:
    changes = []

    def allowed(old: dict | None) -> bool:
        return force or old is None or PRIORITY.get(src, 1) >= PRIORITY.get(old.get("src", "estimate"), 1)

    T_sw = t_sim_world(calib)
    layout["T_sim_world"] = {"value": r(T_sw), "src": src,
                             "_about": "p_sim = T_sim_world @ p_world (contract world: x forward; sim/URDF: y forward)"}

    tbl = calib.get("table")
    if tbl and allowed(layout.get("table")):
        if "center_xy" in tbl:   # extension: a table that is not axis-aligned with the arm
            c_w, size, yaw_w = tbl["center_xy"], tbl["size_xy"], tbl.get("yaw_deg", 0.0)
        else:
            (x0, y0), (x1, y1) = tbl["extent_xy"]
            c_w, size, yaw_w = [(x0 + x1) / 2, (y0 + y1) / 2], [x1 - x0, y1 - y0], 0.0
        c_s = T_sw @ [c_w[0], c_w[1], 0, 1]
        yaw_s = yaw_w + math.degrees(math.atan2(T_sw[1, 0], T_sw[0, 0]))
        if "not the real table edge" in str(tbl.get("extent_source", "")) and "size" in layout["table"]:
            # only a lower bound (marker bounding box): keep the measured edges, note the check
            inside = covers(layout["table"], c_s[:2], size, yaw_s)
            layout["table"]["calib_extent_inside"] = inside
            changes.append(f"table: scene_calib extent is the marker bounding box; kept the edge estimate (covers it: {inside})")
        else:
            layout["table"].update(center=r(c_s[:2], 4), size=r(size, 4), yaw=round(yaw_s, 3), src=src)
            changes.append(f"table: centre {r(c_s[:2], 3)} size {r(size, 3)} yaw {yaw_s:.1f} deg")

    mk = calib.get("markers")
    if mk and mk.get("items"):
        old = {it["id"]: it for it in layout.get("markers", {}).get("items", [])}
        items = []
        for it in mk["items"]:
            if not allowed(old.get(it["id"])):
                items.append(old[it["id"]]); continue
            T = T_sw @ np.asarray(it["T_world_marker"], float)
            items.append({"id": it["id"], "role": it.get("role", "other"), "T": r(T), "src": src,
                          **({"rms_mm": it["rms_mm"]} if "rms_mm" in it else {})})
            changes.append(f"marker {it['id']}: {r(T[:3, 3], 3)}")
        # Markers only the estimate has (e.g. one whose id a cable hides): carry them with the rigid 2D motion that
        # takes the estimate's other markers onto the calibrated ones.
        new_ids = {it["id"] for it in items}
        common = [k for k in new_ids if k in old and old[k].get("src") != src and "T" in old[k]]
        extra = [v for k, v in old.items() if k not in new_ids]
        if extra and len(common) >= 2:
            M = rigid2d([np.asarray(old[k]["T"])[:2, 3] for k in common],
                        [np.asarray(next(it for it in items if it["id"] == k)["T"])[:2, 3] for k in common])
            for v in extra:
                v = dict(v, T=r(M @ np.asarray(v["T"], float)), note=(v.get("note", "") + f" moved with markers {sorted(common)} onto {src}").strip())
                items.append(v)
                changes.append(f"marker {v['id']}: carried along ({v['src']}) -> {r(np.asarray(v['T'])[:3, 3], 3)}")
        else:
            items += extra
        layout["markers"] = {**layout.get("markers", {}), "dictionary": mk["dictionary"], "size_m": mk["size_m"],
                             "items": sorted(items, key=lambda it: it["id"])}

    cams = calib.get("cameras", {})
    ov = cams.get("overview")
    if ov and ov.get("T_world_cam"):
        entry = next((c for c in layout["cameras"] if c["name"] == "overview_cam"), None)
        if entry is None:
            entry = {"name": "overview_cam"}; layout["cameras"].insert(0, entry)
        if allowed(entry if "src" in entry else None):
            T = T_sw @ np.asarray(ov["T_world_cam"], float)
            for k in ("pos", "lookat"):
                entry.pop(k, None)
            entry.update(T_cv=r(T), fovy=round(fovy_of(ov) or entry.get("fovy", 70), 3), src=src,
                         queue=ov.get("queue"), K_normalized=ov.get("K_normalized"), dist=ov.get("dist"),
                         reproj_px=ov.get("reproj_px"))
            changes.append(f"overview_cam: pos {r(T[:3, 3], 3)} fovy {entry['fovy']}")
            for fx in layout.get("fixed", []):      # its stand rises just behind the camera, out of its own view
                if fx["name"] == "cam_stand_pole" and fx.get("src") != "fixed":
                    c = T[:3, 3]; away = c[:2] / np.linalg.norm(c[:2])
                    fx.update(pos=r([*(c[:2] + 0.03 * away), (c[2] - 0.1) / 2], 4), src=src,
                              size=[fx["size"][0], round((c[2] + 0.1) / 2, 4)])
    wr = cams.get("wrist")
    if wr and allowed(layout.get("wrist_camera")):
        w = layout["wrist_camera"]
        if wr.get("T_link_cam") and POSE_FILE in str(wr.get("source", "")):
            w.pop("T_link_cam", None)             # the mount's single source: build_scene reads the pose file itself
            w.update(pose_file=POSE_FILE, link=wr.get("link", "Gripper_Base_v1_1"))
            w.pop("pitch_offset_deg", None)
        elif wr.get("T_link_cam"):
            w["T_link_cam"] = r(wr["T_link_cam"], 6)
            w["link"] = wr.get("link", "Gripper_Base_v1_1")
            w.pop("pitch_offset_deg", None)       # a calibrated pose replaces the jaw-silhouette tilt estimate
        elif wr.get("pitch_offset_deg") is not None:
            w["pitch_offset_deg"] = wr["pitch_offset_deg"]
        if fovy_of(wr):
            w["fovy"] = round(fovy_of(wr), 3)
        w.update(src=src, K_normalized=wr.get("K_normalized"), dist=wr.get("dist"), queue=wr.get("queue"),
                 pose_source=wr.get("source"))
        changes.append(f"wrist_camera: fovy {w['fovy']} pose from {wr.get('source', 'CAD') if wr.get('T_link_cam') else 'CAD'}")
    return layout, changes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("calib", type=Path)
    p.add_argument("--layout", type=Path, default=HERE / "layouts" / "home.json")
    p.add_argument("--src", default="scene_calib", choices=list(PRIORITY))
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    layout = json.loads(a.layout.read_text())
    layout, changes = apply_calib(json.loads(a.calib.read_text()), layout, a.src, a.force)
    layout.setdefault("_synced_from", []).append(str(a.calib))
    layout["_synced_from"] = layout["_synced_from"][-3:]
    a.layout.write_text(dumps(layout) + "\n")
    print("\n".join(changes) or "nothing changed")
    print(f"wrote {a.layout}; rebuild: SIM_LAYOUT={a.layout.stem} python3 sim/build_scene.py")


if __name__ == "__main__":
    main()
