"""Twin preflight: the WHOLE planned joint path is replayed kinematically in the MuJoCo twin (default layout `home`,
EXECUTOR_TWIN_LAYOUT or SIM_LAYOUT override) and checked for collisions with the table, the floor, the camera stand,
the bin(s) and table spots under the ArUco markers, plus reachability (joint limits of the twin, TCP agreement between
the twin and the planner's kinematics).

    pf = TwinPreflight(kin)
    pf(proposal, plan, obs) -> {"result": "pass" | "fail" | "skipped", "why", "seconds", "samples", "min_clearance_mm", ...}

* Interpolated in joint space between consecutive waypoints (the executor's creeping moves are linear in joint space),
  at most `step_rad` apart, starting from the plan's seed joints.
* Contact is expected on guarded-descent / stroke paths (plan.data["contact_paths"], default ["descent"]): there the
  fingertip pads may touch the table, nothing else may.
* Static geometry only fails the check. Loose objects in the layout are at their layout positions, not where they are
  now, so touching them is reported (object_contacts) but does not fail.
* Bins: those in the layout, plus one from the proposal (`params.bin` = {xyz_world, size [w, d, h], yaw_deg}) or from
  an observation object whose name mentions a bin/bucket/box (its size; height default 8 cm).
* Time-boxed: the model loads once per process (in the background at construction); a check that would run past
  `budget_s` stops and reports `skipped` with how far it got. No MuJoCo / no twin model -> `skipped` with the reason.
"""
from __future__ import annotations

import math
import os
import threading
import time
from pathlib import Path

import numpy as np

BIN_WORDS = ("bin", "bucket", "box", "basket", "bowl")
_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


def _load(layout: str):
    """(model, info) for a layout, built if missing or older than the layout file; cached per process."""
    with _CACHE_LOCK:
        if layout in _CACHE:
            return _CACHE[layout]
        import mujoco
        import build_scene                                  # sim/ is on sys.path (executor.kinematics)
        lp = build_scene.layout_path(layout)
        out = build_scene.generated_dir(lp)
        xml = out / "workspace.xml"
        srcs = [lp, Path(build_scene.__file__)]
        if not xml.exists() or xml.stat().st_mtime < max(p.stat().st_mtime for p in srcs):
            build_scene.build(lp, out, report=False)
        spec = mujoco.MjSpec.from_file(str(xml))
        bin_body = spec.worldbody.add_body(name="preflight_bin", mocap=True, pos=[0, 0, -10])
        for k in range(5):                                  # bottom + 4 walls, sized per check
            bin_body.add_geom(name=f"preflight_bin_{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.01, 0.01],
                              contype=2, conaffinity=1)
        m = spec.compile()
        import json
        info = {"layout": str(lp), "name": lp.stem, "markers": json.loads(lp.read_text()).get("markers")}
        _CACHE[layout] = (m, info)
        return _CACHE[layout]


class TwinPreflight:
    def __init__(self, kin, layout: str | None = None, budget_s: float = 1.5, step_rad: float = 0.03,
                 margin_m: float = 0.02, tol_m: float = 0.001, warm: bool = True):
        self.kin = kin
        self.layout = layout or os.environ.get("EXECUTOR_TWIN_LAYOUT") or os.environ.get("SIM_LAYOUT") or "home"
        self.budget_s, self.step, self.margin, self.tol = budget_s, step_rad, margin_m, tol_m
        self.error: str | None = None
        self._ready = threading.Event()
        self.lock = threading.Lock()
        self.m = self.d = None
        if warm:
            threading.Thread(target=self._prepare, daemon=True, name="twin-preflight-load").start()
        else:
            self._prepare()

    # --- model --------------------------------------------------------------------------------
    def _prepare(self):
        try:
            import mujoco
            m, self.info = _load(self.layout)
            self.mj = mujoco
            self.m = m
            self.d = mujoco.MjData(m)
            act_j = m.actuator_trnid[:, 0]
            self.qadr = m.jnt_qposadr[act_j]
            self.jrange = m.jnt_range[act_j].copy()
            self.jaws = [(m.jnt_qposadr[m.joint(n).id], s) for n, s in (("rev_motor_08_1", -0.0115), ("rev_motor_08_2", 0.0115))
                         if self._has_joint(n)]
            self.site = m.site(self.kin.P.kinematics.tcp_site).id
            self.bin_geoms = [m.geom(f"preflight_bin_{k}").id for k in range(5)]
            self.bin_body = m.body("preflight_bin").mocapid[0]
            names = [m.geom(i).name for i in range(m.ngeom)]
            self.names = names
            body = m.geom_bodyid
            self.arm = np.array([(m.geom_contype[i] & 5) != 0 and m.body_weldid[body[i]] != 0 for i in range(m.ngeom)])
            self.pads = np.array([n.endswith("_pad") for n in names])
            self.free = np.array([self._on_free_body(body[i]) for i in range(m.ngeom)])
            static = (m.geom_contype & 2).astype(bool) & ~self.free & (m.body_weldid[body] == 0)
            static[self.bin_geoms] = True
            self.static = static
            m.geom_margin[static] = self.margin
            m.geom_margin[self.arm] = 0.0
            self.table = np.array([n == "table_top" for n in names])
            mk = self.info.get("markers") or {}
            import build_scene
            self.marker_xy = [(it["id"], build_scene.marker_pose(it)[:2, 3], mk.get("size_m", 0.1) / 2 + mk.get("paper_margin_m", 0.012))
                              for it in mk.get("items", [])]
        except Exception as e:  # noqa: BLE001
            self.error = f"twin not available: {type(e).__name__}: {e}"
        finally:
            self._ready.set()

    def _has_joint(self, name):
        try:
            self.m.joint(name)
            return True
        except KeyError:
            return False

    def _on_free_body(self, b):
        m = self.m
        while b > 0:
            j0, n = m.body_jntadr[b], m.body_jntnum[b]
            if n and any(m.jnt_type[j] == self.mj.mjtJoint.mjJNT_FREE for j in range(j0, j0 + n)):
                return True
            b = m.body_parentid[b]
        return False

    def describe(self) -> dict:
        return {"layout": self.layout, "ready": self._ready.is_set(), "error": self.error, "budget_s": self.budget_s}

    # --- bins ---------------------------------------------------------------------------------
    def bin_from(self, p, obs) -> dict | None:
        b = (p.params or {}).get("bin") if p is not None else None
        if isinstance(b, dict) and b.get("xyz_world"):
            return {"xyz_world": list(b["xyz_world"]), "size": list(b.get("size") or [0.2, 0.15, 0.08]),
                    "yaw_deg": float(b.get("yaw_deg") or 0.0), "wall": float(b.get("wall") or 0.004), "from": "proposal"}
        for o in (obs.objects if obs is not None else []):
            if o.visible and any(w in o.name.lower() for w in BIN_WORDS):
                s = list(o.size or [0.2, 0.15])
                return {"xyz_world": list(o.xyz_world), "size": [s[0], s[1], s[2] if len(s) > 2 else 0.08],
                        "yaw_deg": math.degrees(o.yaw or 0.0), "wall": 0.004, "from": f"observation ({o.source})"}
        return None

    def _place_bin(self, b):
        m, d = self.m, self.d
        if b is None:
            d.mocap_pos[self.bin_body] = [0, 0, -10]
            return
        w, dp, h = b["size"]
        t = b["wall"]
        parts = [([0, 0, t / 2], [w / 2, dp / 2, t / 2]), ([0, -(dp - t) / 2, h / 2], [w / 2, t / 2, h / 2]),
                 ([0, (dp - t) / 2, h / 2], [w / 2, t / 2, h / 2]), ([-(w - t) / 2, 0, h / 2], [t / 2, dp / 2, h / 2]),
                 ([(w - t) / 2, 0, h / 2], [t / 2, dp / 2, h / 2])]
        for g, (pos, size) in zip(self.bin_geoms, parts):
            m.geom_pos[g], m.geom_size[g] = pos, size
            m.geom_rbound[g] = float(np.linalg.norm(size))
            m.geom_aabb[g] = np.r_[[0, 0, 0], size]
        c = self.kin.to_base(np.r_[b["xyz_world"][:2], 0.0])
        yaw_b = math.radians(b["yaw_deg"] - self.kin.yaw_offset)
        d.mocap_pos[self.bin_body] = c
        d.mocap_quat[self.bin_body] = [math.cos(yaw_b / 2), 0, 0, math.sin(yaw_b / 2)]

    # --- the check ----------------------------------------------------------------------------
    def segments(self, plan, q0):
        """[(label, q_from, q_to, contact_ok)] in execution order. The first segment that leaves the end of a contact
        path may touch too: the planned bottom of a guarded descent lies below the table (contact stops it earlier)."""
        contact = set(plan.data.get("contact_paths", ["descent"]))
        start_from = plan.data.get("preflight_from", {})
        ends, out, prev, prev_label = {}, [], np.asarray(q0, float), None
        for label, path in plan.paths.items():
            src = start_from.get(label, prev_label)
            cur = ends.get(src, prev) if label in start_from else prev
            leaving = src in contact
            for q in path:
                q = np.asarray(q, float)
                out.append((label, cur, q, label in contact or leaving))
                cur, leaving = q, False
            if path:
                ends[label] = cur
                if label not in start_from:
                    prev, prev_label = cur, label
        return out

    def _set(self, q):
        d = self.d
        n = min(len(q), len(self.qadr))
        d.qpos[self.qadr[:n]] = q[:n]
        if len(q) >= len(self.qadr):
            for adr, s in self.jaws:
                d.qpos[adr] = s * q[len(self.qadr) - 1]
        self.mj.mj_kinematics(self.m, self.d)
        self.mj.mj_collision(self.m, self.d)

    def _on_marker(self, xyz_base):
        for mid, c, half in self.marker_xy:
            if abs(xyz_base[0] - c[0]) <= half and abs(xyz_base[1] - c[1]) <= half:
                return mid
        return None

    def __call__(self, p, plan, obs=None, q0=None) -> dict:
        t0 = time.monotonic()
        if not self._ready.wait(timeout=max(0.0, self.budget_s - 0.2)):
            return {"result": "skipped", "why": f"twin model ({self.layout}) still loading", "seconds": round(time.monotonic() - t0, 3)}
        if self.error:
            return {"result": "skipped", "why": self.error, "seconds": 0.0}
        if plan is None or not plan.paths or not any(plan.paths.values()):
            return {"result": "skipped", "why": "no motion in the plan", "seconds": 0.0}
        with self.lock:
            return self._check(p, plan, obs, q0, t0)

    def _check(self, p, plan, obs, q0, t0):
        b = self.bin_from(p, obs)
        self._place_bin(b)
        if q0 is None:
            q0 = next(np.asarray(path[0], float) for path in plan.paths.values() if path)
        segs = self.segments(plan, q0)
        na = self.kin.na
        reach = {"max_tcp_mismatch_mm": 0.0, "joint_limit_violations": []}
        for label, _, q, _ in segs:                               # waypoints: limits and twin TCP vs planner FK
            lo, hi = self.jrange[:na, 0], self.jrange[:na, 1]
            if np.any(q[:na] < lo - 1e-6) or np.any(q[:na] > hi + 1e-6):
                reach["joint_limit_violations"].append(label)
            self._set(q)
            err = float(np.linalg.norm(self.d.site_xpos[self.site] - self.kin.to_base(self.kin.fk(q)[0]))) * 1000
            reach["max_tcp_mismatch_mm"] = round(max(reach["max_tcp_mismatch_mm"], err), 2)
        samples, hits, objects, expected = 0, [], set(), 0
        best = (self.margin, None)
        total = sum(max(1, math.ceil(float(np.max(np.abs(b_ - a)[:na])) / self.step)) for _, a, b_, _ in segs)
        done_all = True
        for label, a, q, contact_ok in segs:
            n = max(1, math.ceil(float(np.max(np.abs(q - a)[:na])) / self.step))
            for k in range(1, n + 1):
                if time.monotonic() - t0 > self.budget_s:
                    done_all = False
                    break
                self._set(a + (q - a) * k / n)
                samples += 1
                for c in self.d.contact[:self.d.ncon]:
                    g1, g2 = c.geom1, c.geom2
                    if self.arm[g2]:
                        g1, g2 = g2, g1
                    if not self.arm[g1] or self.arm[g2]:
                        continue
                    if self.free[g2]:
                        if c.dist < 0:
                            objects.add(self.names[g2])
                        continue
                    if not self.static[g2]:
                        continue
                    arm = self.names[g1] or self.m.body(self.m.geom_bodyid[g1]).name
                    if contact_ok and self.table[g2] and (self.pads[g1] or "Gripper" in arm):
                        expected += 1
                        continue
                    if c.dist < best[0]:
                        best = (float(c.dist), (arm, self.names[g2], label))
                    if c.dist < -self.tol:
                        what = self.names[g2]
                        if self.table[g2]:
                            mid = self._on_marker(c.pos)
                            what = f"table (marker {mid})" if mid is not None else "table"
                        hits.append({"path": label, "sample": k, "of": n, "arm": arm,
                                     "with": what, "depth_mm": round(-c.dist * 1000, 1),
                                     "at_world": np.round(self.kin.to_world(c.pos), 3).tolist()})
                if len(hits) >= 5:
                    break
            if not done_all or len(hits) >= 5:
                break
        secs = round(time.monotonic() - t0, 3)
        out = {"seconds": secs, "layout": self.info["name"], "samples": samples, "of_samples": total,
               "waypoints": len(segs), "min_clearance_mm": round(best[0] * 1000, 1),
               "closest": None if best[1] is None else {"arm": best[1][0], "with": best[1][1], "path": best[1][2]},
               "expected_table_contacts": expected, "object_contacts": sorted(objects), "bin": b, "reach": reach}
        if hits:
            h = hits[0]
            return {"result": "fail", "why": f"{h['arm']} hits {h['with']} on the {h['path']} path ({h['depth_mm']} mm deep, "
                                              f"near {h['at_world']})", "collisions": hits, **out}
        if reach["joint_limit_violations"]:
            return {"result": "fail", "why": f"twin joint limits exceeded on {reach['joint_limit_violations']}", **out}
        if reach["max_tcp_mismatch_mm"] > 5.0:
            return {"result": "fail", "why": f"twin and planner disagree on the TCP by {reach['max_tcp_mismatch_mm']} mm", **out}
        if not done_all:
            return {"result": "skipped", "why": f"time budget {self.budget_s} s used up after {samples}/{total} samples "
                                                f"(no collision in the checked part)", **out}
        return {"result": "pass", **out}
