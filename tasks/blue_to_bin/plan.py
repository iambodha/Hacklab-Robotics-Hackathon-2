"""The scripted planner for "blue object -> bin": every motion is ONE executor proposal (fresh observation,
preconditions, expiry); it never commands the arm itself.

Steps (per attempt, at most spec.max_attempts):
  observe -> locate blue (sigma) -> approach above -> [re-observe: moved? re-plan] -> descend (guarded) ->
  close -> test lift (verify hold: the overview sees the lid rise; touch alone cannot feel a 2 mm wall) -> lift ->
  above bin centre (rim + margin) -> lower into the bin (guarded, open jaws checked to fit inside the rim) ->
  release -> retreat straight up -> park -> verify (overview + gripper evidence, task.judge)
Before the first attempt: park, then a jaw check in the air (close empty, open) that gives the touch sense its
empty-close reference (thin objects such as a 2 mm lid wall are only recognised against it).

    runner = Runner(ex, Spec.load()); report = runner.run()
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from executor.contracts import ActionProposal, Status, to_dict

from .locate import NAME
from .task import GOAL, Grasp, Spec, judge, lowered_release, plan_grasp, release_point


@dataclass
class StepLog:
    step: str
    skill: str | None
    params: dict
    preconditions: dict
    status: str
    termination: str
    seconds: float
    attempt: int
    execution_id: str | None = None
    obs_id: str | None = None
    extra: dict = field(default_factory=dict)


class TaskFailed(Exception):
    pass


class Runner:
    def __init__(self, ex, spec: Spec | None = None, source: str = "script:blue_to_bin", ttl_s: float = 20.0,
                 on_step=None, wait_s: float = 120.0):
        self.ex, self.spec = ex, spec or Spec.load()
        self.source, self.ttl_s, self.wait_s = source, ttl_s, wait_s
        self.on_step = on_step or (lambda s: None)
        self.steps: list[StepLog] = []
        self.attempt = 0
        self.t0 = time.monotonic()
        self.G = ex.kin.P.gripper

    # ---- primitives -------------------------------------------------------------------------------------------
    def log(self, step, skill, params, pre, status, term, t0, eid=None, obs=None, **extra):
        s = StepLog(step, skill, to_dict(params), pre, status, term, round(time.monotonic() - t0, 2), self.attempt, eid, obs, extra)
        self.steps.append(s)
        self.on_step(s)
        return s

    def observe(self, step="observe"):
        t0 = time.monotonic()
        obs = self.ex.observe()
        stale = [r for r, f in obs.frames.items() if f.stale]
        self.log(step, None, {}, {}, "STALE" if stale else "OK", f"frames {[(r, f.seq) for r, f in obs.frames.items()]}"
                 + (f", stale {stale}" if stale else ""), t0, obs=obs.obs_id, objects=[to_dict(o) for o in obs.objects])
        return obs

    def blue(self, obs):
        return next((o for o in obs.objects if o.name == NAME and o.visible), None)

    def propose(self, step, skill, params, pre: dict, ttl_s=None):
        """Check the preconditions (bool values must be True; other values are recorded as information), then observe
        fresh, submit, wait. Returns the ExecutionResult."""
        t0 = time.monotonic()
        failed = [k for k, v in pre.items() if v is False]
        if failed:
            self.log(step, skill, params, pre, "NOT_SUBMITTED", f"preconditions false: {failed}", t0)
            raise TaskFailed(f"{step}: preconditions false: {failed}")
        for _ in range(3):     # a rejection for freshness only (slow observation) is re-observed and resubmitted
            obs = self.ex.observe()
            p = ActionProposal(ActionProposal.new_id(), self.source, obs.obs_id, obs.setup_id, skill, "", params,
                               [f"{k}: {v}" for k, v in pre.items()], self.ex.now() + (ttl_s or self.ttl_s),
                               f"{GOAL} | attempt {self.attempt} | step {step}")
            eid = self.ex.submit(p)
            r = self.ex.wait(eid, self.wait_s)
            val = r.validation if r is not None else None
            failed_checks = [c for c, x in (val.checks if val else {}).items() if x.get("result") == "fail"]
            if not (r is not None and r.status == Status.REJECTED and failed_checks == ["freshness"]):
                break
        if r is None or not r.status.final:
            self.ex.cancel(eid)
            r = self.ex.wait(eid, 10.0)
        self.log(step, skill, params, pre, r.status.value, r.termination, t0, eid, obs.obs_id,
                 residuals=r.residuals, achieved=r.achieved,
                 rejected=(r.validation.reasons if r.status == Status.REJECTED and r.validation else None))
        return r

    def gripper_state(self):
        st = self.ex.motion.read()[self.ex.kin.grip_id]
        return {"norm": round(float(st["norm"]), 3), "load": int(st["load"])}

    def tcp(self):
        return self.ex.kin.fk(self.ex.motion.q_now())[0]

    # ---- the task -------------------------------------------------------------------------------------------------
    def prepare(self):
        self.ex.observe()                      # warm-up (first render / perception import)
        r = self.propose("park", "park", {}, {"no execution running": not self.ex.busy()})
        if r.status != Status.SUCCEEDED:
            raise TaskFailed(f"park: {r.status.value} {r.termination}")
        for step, f in (("jaw check: close empty", 0.0), ("jaw check: open", 1.0)):
            r = self.propose(step, "gripper", {"open_fraction": f}, {"parked, nothing in the jaws": True})
            if r.status != Status.SUCCEEDED or (f == 0.0 and r.achieved.get("holding")):
                raise TaskFailed(f"{step}: {r.status.value} {r.termination} {r.achieved}")

    def locate(self, tries=3):
        for k in range(tries):
            obs = self.observe("observe" if k == 0 else f"observe (retry {k})")
            est = self.blue(obs)
            t0 = time.monotonic()
            if est is not None and est.xyz_world[2] - self.spec.cap_height / 2 > 0.012:
                self.log("locate blue", None, {}, {}, "NOT_RESTING", f"lid base {1000 * (est.xyz_world[2] - self.spec.cap_height / 2):.0f} mm "
                         f"above the table (held, tilted or on something): not a grasp target", t0, obs=obs.obs_id)
                self.ex.backend.sleep(1.0)
                continue
            if est is not None and (est.uncertainty_mm or 99) <= self.spec.sigma_max_mm:
                self.log("locate blue", None, {}, {}, "OK", f"{est.xyz_world} sigma {est.uncertainty_mm} mm ({est.source})", t0,
                         obs=obs.obs_id, xyz=est.xyz_world, sigma_mm=est.uncertainty_mm)
                return est
            self.log("locate blue", None, {}, {}, "NOT_FOUND", "no blue lid in the overview" if est is None else
                     f"sigma {est.uncertainty_mm} mm > {self.spec.sigma_max_mm}", t0, obs=obs.obs_id)
        return None

    def back_up(self, g: Grasp, why: str):
        self.propose(f"recover: open ({why})", "gripper", {"open_fraction": 1.0}, {"arm stopped": not self.ex.busy()})
        p = self.tcp()
        self.propose(f"recover: up ({why})", "move_tcp", {"xyz_world": [p[0], p[1], max(g.z_hover, p[2])], "yaw_deg": g.yaw_deg},
                     {"arm stopped": not self.ex.busy()})

    def pick(self, est) -> tuple[bool, Grasp, str]:
        S = self.spec
        if S.grasp_mode() == "across" and S.cap_diameter > S.max_grip_width:
            raise TaskFailed(f"object {S.cap_diameter * 1000:.0f} mm wide > grip width {S.max_grip_width * 1000:.0f} mm: no across grasp")
        g = plan_grasp(S, est.xyz_world)
        for replan in range(3):
            r = self.propose("approach above", "move_tcp", {"xyz_world": [*g.tcp_xy, g.z_hover], "yaw_deg": g.yaw_deg},
                             {"blue located": True, "sigma_mm<=max": (est.uncertainty_mm or 99) <= S.sigma_max_mm,
                              "jaws open": self.gripper_state()["norm"] >= self.G.open_enough, "grasp": g.describe()})
            if r.status not in (Status.SUCCEEDED, Status.PARTIAL):
                return False, g, f"approach {r.status.value}: {r.termination}"
            again = self.blue(self.observe("re-observe above"))
            if again is None or again.xyz_world[2] > S.cap_height:     # hidden under the gripper: keep the estimate
                break
            moved = float(np.linalg.norm(np.subtract(again.xyz_world[:2], est.xyz_world[:2])))
            if moved <= S.moved_tol:
                break
            self.log("replan", None, {}, {}, "MOVED", f"object moved {moved * 1000:.0f} mm: new approach", time.monotonic())
            est, g = again, plan_grasp(S, again.xyz_world)
        r = self.propose("descend (guarded)", "descend_guarded", {"xy_world": list(g.tcp_xy), "z_world": g.z_grasp, "yaw_deg": g.yaw_deg},
                         {"above the grasp point": bool(np.hypot(*(self.tcp()[:2] - g.tcp_xy)) <= 0.015),
                          "jaws open": self.gripper_state()["norm"] >= self.G.open_enough})
        short = (r.residuals or {}).get("tcp_mm")
        if r.status == Status.PARTIAL and (r.achieved or {}).get("contact_z") is None and short is not None and short <= 8:
            pass                                               # stopped a few mm short without touching: close anyway
        elif r.status != Status.SUCCEEDED:
            self.back_up(g, "descent")
            return False, g, f"descent {r.status.value}: {r.termination}"
        r = self.propose("close (verify hold)", "gripper", {"open_fraction": 0.0, "force": S.grip_force},
                         {"at grasp height": bool(self.tcp()[2] <= g.z_grasp + 0.01)})
        if r.status != Status.SUCCEEDED:
            self.back_up(g, "close failed")
            return False, g, f"close: {r.status.value} {r.termination}"
        touch = bool(r.achieved.get("holding"))
        # Independent hold check: lift a little and look. Touch sense alone cannot see a 2 mm lid wall (the jaw
        # motor reaches 'closed' through the linkage's compliance), so a thin rim grasp needs the camera.
        z_test = g.z_grasp + S.test_lift
        r = self.propose("test lift", "move_tcp", {"xyz_world": [*g.tcp_xy, z_test], "yaw_deg": g.yaw_deg},
                         {"jaws closed": self.gripper_state()["norm"] < self.G.open_enough,
                          "touch": "holding" if touch else "nothing felt (thin wall?): the camera decides"})
        lifted = self.lifted_seen(est, "verify hold (test lift)", touch, min_rise=0.6 * S.test_lift)
        if r.status not in (Status.SUCCEEDED, Status.PARTIAL) or not (lifted or (touch and lifted is None)):
            self.back_up(g, "missed")
            return False, g, f"hold not confirmed: touch {touch}, overview sees it lifted: {lifted}"
        r = self.propose("lift", "move_tcp", {"xyz_world": [*g.tcp_xy, S.carry_z], "yaw_deg": g.yaw_deg},
                         {"hold confirmed": True})
        lifted = self.lifted_seen(est, "verify lifted", touch, min_rise=0.04)
        if r.status not in (Status.SUCCEEDED, Status.PARTIAL) or lifted is False:
            self.back_up(g, "slipped")
            return False, g, f"lift {r.status.value}: overview sees it lifted: {lifted}"
        return True, g, "holding"

    def lifted_seen(self, est, step, touch, min_rise) -> bool | None:
        """True = the overview sees the lid's base risen by >= min_rise; False = it is still on the table near where it
        was; None = not measurable (hidden by the gripper, silhouette not a clean lid)."""
        seen = self.blue(self.observe(step + ": observe"))
        rise = None if seen is None else float(seen.xyz_world[2] - self.spec.cap_height / 2)
        lifted = None if rise is None else bool(rise >= min_rise)
        self.log(step, None, {}, {}, {True: "LIFTED", False: "ON_TABLE", None: "NOT_SEEN"}[lifted],
                 f"touch {touch}, jaws {self.gripper_state()}, lid base z {None if rise is None else round(rise * 1000)} mm",
                 time.monotonic(), seen=to_dict(seen) if seen else None)
        return lifted

    def place(self, g: Grasp) -> str | None:
        S = self.spec
        rel = release_point(S, g)
        r = self.propose("move above bin", "move_tcp", {"xyz_world": rel.tolist(), "yaw_deg": g.yaw_deg},
                         {"hold confirmed after the lift": True, "jaws closed": self.gripper_state()["norm"] < self.G.open_enough,
                          "release clears rim + margin": bool(rel[2] - g.z_grasp >= S.bin_rim_height + S.rim_margin - 1e-6)})
        if r.status not in (Status.SUCCEEDED, Status.PARTIAL) or (r.residuals or {}).get("tcp_mm", 0) > 10:
            return f"above bin {r.status.value}: {r.termination}"
        low, fit = lowered_release(S, g)
        r = self.propose("lower into bin (guarded)", "descend_guarded", {"xy_world": low[:2].tolist(), "z_world": float(low[2]),
                                                                          "yaw_deg": g.yaw_deg},
                         {"open jaws fit inside the rim": fit["fits"], "jaw fit": fit,
                          "above the release point": bool(np.hypot(*(self.tcp()[:2] - low[:2])) <= 0.015)})
        if r.status not in (Status.SUCCEEDED, Status.PARTIAL):
            return f"lower into bin {r.status.value}: {r.termination}"
        self.propose("release", "gripper", {"open_fraction": 1.0}, {"TCP inside the rim": bool(self.tcp()[2] < S.bin_rim_height + 0.02)})
        p = self.tcp()
        self.propose("retreat (straight up)", "move_tcp", {"xyz_world": [p[0], p[1], max(S.retreat_z, p[2])], "yaw_deg": g.yaw_deg},
                     {"jaws open": self.gripper_state()["norm"] >= self.G.open_enough})
        self.propose("park", "park", {}, {"above the rim": bool(self.tcp()[2] > S.bin_rim_height + 0.05)})
        return None

    def verify(self, held_after_lift: bool) -> dict:
        obs = self.observe("verify: observe")
        est = self.blue(obs)
        grip = self.gripper_state()
        v = judge(self.spec, est, {"held_after_lift": held_after_lift, "open_after_release": grip["norm"] >= self.G.open_enough,
                                   "empty_after_release": True})
        self.log("verify", None, {}, {}, {True: "SUCCESS", False: "FAIL", None: "UNKNOWN"}[v["success"]],
                 str(v["overview"]), time.monotonic(), obs=obs.obs_id, verdict=v)
        return v

    def run(self) -> dict:
        S = self.spec
        verdict, why = None, ""
        try:
            self.prepare()
            for self.attempt in range(1, S.max_attempts + 1):
                est = self.locate()
                if est is None:
                    why = "blue object not found"
                    continue
                pre = judge(S, est, {"held_after_lift": True, "open_after_release": True})
                if pre["overview"]["result"]:
                    verdict, why = self.verify(held_after_lift=False), "already in the bin"
                    break
                ok, g, why = self.pick(est)
                if not ok:
                    continue
                err = self.place(g)
                if err:
                    why = err
                    continue
                verdict = self.verify(held_after_lift=True)
                if verdict["success"]:
                    why = "in the bin"
                    break
                why = f"verify: {verdict['overview'].get('why') or verdict['overview']}"
        except TaskFailed as e:
            why = str(e)
        return {"goal": GOAL, "success": bool(verdict and verdict["success"]), "verdict": verdict, "why": why,
                "attempts": self.attempt, "seconds": round(time.monotonic() - self.t0, 1), "grasp_mode": S.grasp_mode(),
                "proposals": sum(1 for s in self.steps if s.skill), "steps": [vars(s) for s in self.steps]}
