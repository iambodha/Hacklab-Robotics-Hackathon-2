"""The executor: the ONE owner of arm motion. Advisors (Claude, Gemini ER, policies, scripts, the console) only
observe() and submit() proposals; they never send motor commands and never block motion.

    ex = Executor(make_backend("fake", kin=kin), kin)
    obs = ex.observe()
    eid = ex.submit(ActionProposal(..., based_on_obs=obs.obs_id, setup_id=obs.setup_id, skill="move_tcp", ...))
    ex.wait(eid); ex.status(eid); ex.cancel(eid); ex.evaluate(eid)
"""
from __future__ import annotations

import os
import threading
import time
import traceback
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .backends.base import Backend, MotionDisabled
from .contracts import (ActionProposal, ExecutionResult, FrameRef, JointState, Observation, Outcome, SetupId, Status,
                        ValidationResult, to_dict)
from .kinematics import Kinematics, load_scene_calib, scene_calib_path
from .motion import Cancelled, Motion, MotionResult
from .perception import Locator, camera_info_from_scene, make_locator
from .recording import RecorderLink, TableMemory
from .skills import SKILLS, Verdict
from .validation import validate, validate_in_sim
from learning import skillcfg as SC  # noqa: E402  (kinematics put the repo on sys.path)
from sites import get_site  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
WIPE_SKILLS = ("wipe", "wipe_region")


class EventLog:
    """The timeline the console shows: everything the robot logic did, in order."""

    def __init__(self, keep: int = 3000):
        self.items: deque = deque(maxlen=keep)
        self.seq = 0
        self.lock = threading.Lock()
        self.listeners: list = []

    def add(self, kind: str, **data) -> dict:
        with self.lock:
            self.seq += 1
            ev = {"seq": self.seq, "t": round(time.time(), 3), "kind": kind, **to_dict(data)}
            self.items.append(ev)
        for fn in list(self.listeners):
            try:
                fn(ev)
            except Exception:  # noqa: BLE001
                pass
        return ev

    def since(self, seq: int = 0, limit: int = 500) -> list[dict]:
        with self.lock:
            return [e for e in self.items if e["seq"] > seq][:limit]


@dataclass
class Run:
    """Mutable record of one execution while it runs."""
    execution_id: str
    proposal: ActionProposal
    phases: list[dict] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    markers: list[str] = field(default_factory=list)
    ex: "Executor" = None

    def note(self, text: str) -> None:
        if self.phases:
            self.phases[-1].setdefault("notes", []).append(text)
        self.ex.events.add("note", execution_id=self.execution_id, text=text)

    def phase_start(self, name: str) -> dict:
        ph = {"name": name, "start": round(self.ex.now(), 3), "end": None, "status": "running", "notes": []}
        self.phases.append(ph)
        self.mark(f"phase {name} start", {"execution_id": self.execution_id})
        self.ex.events.add("phase", execution_id=self.execution_id, name=name, status="running")
        return ph

    def phase_end(self, ph: dict, status: str, **extra) -> None:
        ph.update(end=round(self.ex.now(), 3), status=status, **extra)
        self.mark(f"phase {ph['name']} {status}", {"execution_id": self.execution_id, **extra})
        self.ex.events.add("phase", execution_id=self.execution_id, name=ph["name"], status=status, **extra)

    def mark(self, label, data=None):
        m = self.ex.rec.marker(label, data)
        if m:
            self.markers.append(m)


class Ctx:
    """What a skill sees: kinematics, motion, profile, seed joints, table memory for THIS setup, and the observation
    the proposal was based on (with its frames and object estimates) for skills that localise from it."""

    def __init__(self, ex: "Executor", q0, obs: Observation | None = None):
        self.ex, self.kin, self.motion, self.P = ex, ex.kin, ex.motion, ex.kin.P
        self.q0 = np.asarray(q0, float)
        self.setup_id = ex.setup_id()
        self.min_z = ex.min_z
        self.obs = obs
        self.locator = ex.locator

    def frame(self, role: str = "overview") -> bytes | None:
        return self.ex.frame_jpeg(self.obs.obs_id, role) if self.obs else None

    def camera_info(self) -> dict:
        return self.ex.camera_info()

    def memory_hit(self, x, y):
        return self.ex.memory.lookup(self.setup_id, x, y)

    def table_z(self, x, y) -> float:
        felt = self.memory_hit(x, y)
        return self.P.touch.nominal_table_tcp_z if felt is None else felt

    def remember_touch(self, x, y, z) -> bool:
        ok = self.ex.memory.remember(self.setup_id, x, y, z)
        self.ex.events.add("table_memory", setup_id=self.setup_id, xy=[x, y], z=z, remembered=ok)
        return ok


class Executor:
    def __init__(self, backend: Backend, kin: Kinematics | None = None, recorder: RecorderLink | None = None,
                 locator: Locator | None = None, layout: str = "default", tool: str = "none",
                 table_memory_path: Path | None = None, max_frame_age: float = 1.0, events: EventLog | None = None,
                 twin: bool | str | None = None):
        self.backend = backend
        self.kin = kin or Kinematics()
        self.skills = dict(SKILLS)
        self.events = events or EventLog()
        self.rec = recorder or RecorderLink("local")
        self.scene = load_scene_calib()
        self.scene_version = (f"{self.scene.get('version')}:{self.scene.get('created')}" if self.scene else None)
        self.locator = locator if locator is not None else make_locator(self.scene)
        self.layout, self.tool = layout, tool
        self.max_frame_age = max_frame_age
        self.min_z = float(self.kin.P.workspace.min_tcp_z)
        self.memory = TableMemory(table_memory_path, self.kin.P.touch)
        self.preflight = validate_in_sim          # (proposal, plan, executor, obs, q0) -> check dict
        self.twin = None
        if twin is None:
            twin = os.environ.get("EXECUTOR_TWIN_PREFLIGHT", "1") != "0"
        if twin:
            try:
                from .twin import TwinPreflight
                self.twin = TwinPreflight(self.kin, None if twin is True else str(twin))
            except ImportError:
                self.twin = None
        self.kin.cfg_target = "real" if backend.name == "real" else "twin"     # skill config the IK and skills read
        self.cancel_ev = threading.Event()
        self.motion = Motion(backend, self.kin, self.cancel_ev, log=lambda *a, **k: self.events.add("motion_log", text=" ".join(map(str, a)), **k))
        self.lock = threading.Lock()
        self.current: str | None = None
        self.results: OrderedDict[str, ExecutionResult] = OrderedDict()
        self.proposals: dict[str, ActionProposal] = {}
        self.plans: dict[str, object] = {}
        self.wipe_plans: dict[str, dict] = {}                  # plan_id -> before image + stroke executions
        self.threads: dict[str, threading.Thread] = {}
        self._obs: OrderedDict[str, Observation] = OrderedDict()
        self._frames: OrderedDict[str, dict] = OrderedDict()   # obs_id -> {role: jpeg}
        self._seq_seen: dict[str, tuple[int, float]] = {}      # role -> (seq, first time seen)
        self._last_objects: list = []
        self._setup_cache: tuple | None = None

    # --- clock / setup ----------------------------------------------------------------------
    def now(self) -> float:
        return self.backend.now()

    def setup_id(self, st: dict | None = None) -> str:
        if st is None:
            st = self.backend.read_state().motors
        ranges = {m: (s["lo"], s["hi"]) for m, s in st.items()}
        key = (self.scene_version, self.kin.P.name, tuple(sorted(ranges.items())), self.tool, self.layout)
        if self._setup_cache is None or self._setup_cache[0] != key:
            sid = SetupId.compute(self.scene_version, self.kin.P.name, ranges, self.tool, self.layout)
            if self._setup_cache is not None:
                self.events.add("setup_changed", old=self._setup_cache[1], new=sid, tool=self.tool, layout=self.layout)
            self._setup_cache = (key, sid)
        return self._setup_cache[1]

    def set_tool(self, tool: str) -> str:
        self.tool = tool or "none"
        return self.setup_id()

    def ctx(self, q0=None, obs: Observation | None = None) -> Ctx:
        return Ctx(self, self.motion.q_now() if q0 is None else q0, obs)

    def camera_info(self) -> dict:
        if hasattr(self.backend, "camera_info"):
            return self.backend.camera_info(self.kin)
        return camera_info_from_scene(self.scene)

    # --- observe ----------------------------------------------------------------------------
    def observe(self) -> Observation:
        b = self.backend
        st = b.read_state()
        frames = b.frames()
        now, wall = b.now(), time.time()
        obs_id = "obs-" + uuid.uuid4().hex[:10]
        refs, fresh = {}, {}
        for role, f in frames.items():
            seen = self._seq_seen.get(role)
            if seen is None or seen[0] != f.seq:
                seen = self._seq_seen[role] = (f.seq, now)
            age = now - f.t_capture
            held = now - seen[1]                 # how long this same sequence number has been the newest frame
            stale = age > self.max_frame_age or held > self.max_frame_age
            refs[role] = FrameRef(f.queue, f.seq, f.t_capture, f"/api/frame/{obs_id}/{role}.jpg", round(age, 3), stale,
                                  f.joints_norm)
            if not stale:
                fresh[role] = f
        sid = self.setup_id(st.motors)
        m = st.motors
        ids = self.kin.ids
        q = self.motion.q_now(m)
        tcp = self.kin.fk(q)[0]
        joints = JointState(names=self.kin.names, raw=[m[i]["raw"] for i in ids], rad=np.round(q, 5).tolist(),
                            norm=[m[i]["norm"] for i in ids], goal=[m[i]["goal"] for i in ids], load=[m[i]["load"] for i in ids],
                            current=None, temp=[m[i]["temp"] for i in ids], torque=[m[i]["torque"] for i in ids],
                            status=[m[i]["status"] for i in ids], t_mono=st.t_mono, tcp_world=np.round(tcp, 4).tolist(),
                            gripper_open=m[self.kin.grip_id]["norm"])
        notes = []
        if "overview" in fresh:
            try:
                objects = self.locator.locate(fresh, {i: m[i]["norm"] for i in ids}, self.kin, self.camera_info())
                self._last_objects = objects
            except Exception as e:  # noqa: BLE001
                objects, notes = [], [f"perception failed: {e}"]
        else:
            import dataclasses
            objects = [dataclasses.replace(o, visible=False, source=o.source + " (memory: overview not fresh)") for o in self._last_objects]
            if "overview" in frames:
                notes.append("overview camera stale: object positions unknown (last estimates marked visible=False)")
        oldest = min([st.t_mono] + [f.t_capture for f in frames.values()])
        obs = Observation(obs_id, now, wall, sid, refs, joints, objects, round(now - oldest, 3), self.rec.id, notes)
        self._obs[obs_id] = obs
        self._frames[obs_id] = {r: f.jpeg for r, f in frames.items()}
        while len(self._obs) > 40:
            k, _ = self._obs.popitem(last=False)
            self._frames.pop(k, None)
        self.events.add("observation", obs_id=obs_id, setup_id=sid, freshness_s=obs.freshness_s,
                        frames={r: {"seq": f.seq, "age_s": f.age_s, "stale": f.stale} for r, f in refs.items()},
                        objects=len(objects), tcp_world=joints.tcp_world, notes=notes)
        return obs

    def observation(self, obs_id: str) -> Observation | None:
        return self._obs.get(obs_id)

    def frame_jpeg(self, obs_id: str, role: str) -> bytes | None:
        return self._frames.get(obs_id, {}).get(role)

    # --- submit / status / cancel ------------------------------------------------------------
    def submit(self, p: ActionProposal) -> str:
        """Validate (fast checks, then the whole path with the IK); start it if valid. Returns the execution id."""
        eid = "exec-" + uuid.uuid4().hex[:10]
        self.events.add("proposal", execution_id=eid, proposal=p)
        if not self.rec.active:
            self.rec.start(f"exec-{p.skill}", note=f"standalone proposal from {p.source}")
        marks = [m for m in (self.rec.marker("proposal", {"execution_id": eid, **to_dict(p)}),) if m]
        with self.lock:
            busy = self.current is not None
            v, plan, secs = validate(self, p, busy)
            v.checks["validation_s"] = {"result": "pass", "seconds": round(secs, 3)}
            v.checks["skill_config"] = {"result": "pass", **SC.stamp(self.kin.cfg_target)}   # which version planned it
            self.proposals[eid] = p
            m = self.rec.marker("validation", {"execution_id": eid, "ok": v.ok, "reasons": v.reasons, "checks": v.checks})
            marks += [m] if m else []
            self.events.add("validation", execution_id=eid, ok=v.ok, reasons=v.reasons, checks=v.checks)
            if not v.ok:
                r = ExecutionResult(eid, p.proposal_id, Status.REJECTED, {}, {}, "rejected: " + "; ".join(v.reasons),
                                    [{"name": "validate", "status": "fail", "seconds": round(secs, 3), "reasons": v.reasons}],
                                    self.rec.id, marks, v, p.skill)
                self.results[eid] = r
                self._result_event(r)
                return eid
            if p.skill in WIPE_SKILLS:
                self._register_wipe(eid, p)
            self.current = eid
            self.cancel_ev.clear()
            self.results[eid] = ExecutionResult(eid, p.proposal_id, Status.RUNNING, {}, {}, "", [
                {"name": "plan", "status": "pass", "seconds": round(secs, 3), "notes": [str(plan.checks)]},
                {"name": "validate", "status": "pass", "seconds": round(secs, 3)}], self.rec.id, marks, v, p.skill)
            self.plans[eid] = plan
            th = threading.Thread(target=self._run, args=(eid, p, plan), daemon=True, name=eid)
            self.threads[eid] = th
            th.start()
        return eid

    def _register_wipe(self, eid: str, p: ActionProposal) -> None:
        """Strokes of one coverage plan (params.plan_id, else this execution alone) share the overview image of the
        first stroke's observation as the evaluator's BEFORE image (evaluator._wipe_outcome)."""
        pid = p.params.get("plan_id") or eid
        o = self.observation(p.based_on_obs)
        wp = self.wipe_plans.setdefault(pid, {
            "before": self.frame_jpeg(p.based_on_obs, "overview"), "before_obs": p.based_on_obs,
            "before_stale": bool(o and o.frames.get("overview") and o.frames["overview"].stale), "eids": [],
            "board_world_xy": p.params.get("board_world_xy") or p.params.get("region_world_xy")})
        wp["eids"].append(eid)
        while len(self.wipe_plans) > 20:
            self.wipe_plans.pop(next(iter(self.wipe_plans)))

    def _run(self, eid: str, p: ActionProposal, plan) -> None:
        skill = self.skills[p.skill]
        res0 = self.results[eid]
        run = Run(eid, p, list(res0.phases), markers=list(res0.markers), ex=self)
        self.backend.tag = eid
        ctx = self.ctx(obs=self.observation(p.based_on_obs))
        ph = run.phase_start("execute")
        try:
            res = skill.execute(ctx, plan, run)
        except Cancelled:
            res = MotionResult(Status.CANCELLED, "cancelled: no further commands, holding position")
        except MotionDisabled as e:
            res = MotionResult(Status.BLOCKED, f"motion disabled: {e}")
        except Exception as e:  # noqa: BLE001
            try:
                self.backend.hold()
            except Exception:  # noqa: BLE001
                pass
            res = MotionResult(Status.FAULT, f"fault: {type(e).__name__}: {e}")
            run.data["traceback"] = traceback.format_exc()[-1500:]
        run.phase_end(ph, res.status.value, termination=res.termination)
        ph = run.phase_start("verify")
        try:
            verdict = skill.verify(ctx, plan, run, res)
        except Exception as e:  # noqa: BLE001
            verdict = Verdict(Status.FAULT if res.status == Status.SUCCEEDED else res.status, f"verify failed: {e}", {}, {})
        if res.status == Status.CANCELLED:
            verdict = Verdict(Status.CANCELLED, res.termination, verdict.achieved, verdict.residuals)
        elif res.status != Status.SUCCEEDED and res.termination not in verdict.termination:
            verdict.termination = f"{res.termination}; {verdict.termination}"
        if res.residual_ticks and verdict.status != Status.SUCCEEDED:
            verdict.residuals.setdefault("joints_ticks", res.residual_ticks)
        run.phase_end(ph, verdict.status.value, termination=verdict.termination, residuals=verdict.residuals)
        if verdict.status not in (Status.SUCCEEDED, Status.CANCELLED) and not self.cancel_ev.is_set():
            ph = run.phase_start("recover")
            try:
                note = skill.recover(ctx, plan, run, res)
            except Cancelled:
                note = "cancelled during recovery"
            except Exception as e:  # noqa: BLE001
                note = f"recovery failed: {e}"
            run.phase_end(ph, "done" if note else "skipped", note=note or "nothing to recover")
        else:
            run.phases.append({"name": "recover", "status": "skipped", "start": None, "end": None, "notes": []})
        r = ExecutionResult(eid, p.proposal_id, verdict.status, verdict.achieved, verdict.residuals, verdict.termination,
                            run.phases, self.rec.id, run.markers, res0.validation, p.skill)
        run.mark("result", {"execution_id": eid, "status": r.status.value, "termination": r.termination, "residuals": r.residuals})
        r.markers = run.markers
        with self.lock:
            self.results[eid] = r
            self.current = None
            self.backend.tag = None
        self._result_event(r)

    def _result_event(self, r: ExecutionResult) -> None:
        self.events.add("result", execution_id=r.execution_id, status=r.status.value, termination=r.termination,
                        residuals=r.residuals, achieved=r.achieved, phases=r.phases, skill=r.skill)
        while len(self.results) > 300:
            self.results.popitem(last=False)

    def status(self, eid: str) -> ExecutionResult | None:
        return self.results.get(eid)

    def wait(self, eid: str, timeout: float = 60.0) -> ExecutionResult | None:
        th = self.threads.get(eid)
        if th:
            th.join(timeout)
        return self.results.get(eid)

    def cancel(self, eid: str | None = None) -> bool:
        """Stop issuing commands immediately and hold. eid None = whatever runs."""
        with self.lock:
            cur = self.current
        if cur is None or (eid not in (None, cur)):
            return False
        self.cancel_ev.set()
        self.events.add("cancel", execution_id=cur)
        self.rec.marker("cancel", {"execution_id": cur})
        return True

    def busy(self) -> bool:
        return self.current is not None

    def validate_in_sim(self, p: ActionProposal) -> dict:
        """Twin preflight of a proposal's whole planned path, without submitting it (nothing moves)."""
        return validate_in_sim(p, ex=self)

    # --- skill config (learning/skillcfg.py): the same layer as the MCP servers ----------------------------------
    def skill_config(self) -> dict:
        """Active config of this executor's target, the registry (envelopes, classes) and the pending proposals."""
        t = self.kin.cfg_target
        return {**SC.active(t), "registry": {k: {**s, "envelope": SC.envelope_text(s)} for k, s in SC.registry().items()},
                "pending": SC.proposals("pending", t), "text": SC.config_text(t)}

    def propose_skill_change(self, changes: dict, reason: str, evidence=(), source: str = "operator",
                             twin_check: bool = False) -> dict:
        """Skills and advisors request a strategy change here. Validated, checked offline (skillcheck), applied at once
        on a twin target when no check fails; the real target waits for `python -m learning.skillcfg approve`."""
        from learning import skillcheck
        t = self.kin.cfg_target
        r = SC.propose(changes, reason, evidence, source, t)
        if not r["ok"]:
            self.events.add("skill_config", state="refused", errors=r["errors"], changes=changes)
            return r
        p = skillcheck.run_checks(r["proposal"]["id"], "async" if twin_check else False)
        out = {"ok": True, "proposal": p, "applied": False}
        if t == "twin" and not SC.checks_failed(p):
            a = SC.apply(p["id"], by=f"auto (executor twin, {source})")
            out.update(applied=a["ok"], version=a.get("version"), errors=a.get("errors", []))
        self.events.add("skill_config", state="applied" if out["applied"] else "pending", proposal=p["id"], target=t,
                        changes=p["changes"], version=SC.version(t))
        return out

    # --- evaluate ------------------------------------------------------------------------------
    def evaluate(self, eid: str) -> Outcome:
        from .evaluator import evaluate
        o = evaluate(self, eid)
        self.events.add("outcome", **to_dict(o))
        self.rec.marker("outcome", to_dict(o))
        return o

    # --- tasks (recording sessions) ----------------------------------------------------------------
    def start_task(self, name: str, prompt: str = "", advisor: str = "none") -> str:
        self.rec.stop()
        rid = self.rec.start(f"task-{name}"[:40], note=prompt[:500])
        self.rec.marker("task start", {"prompt": prompt, "advisor": advisor, "backend": self.backend.name, "setup_id": self.setup_id()})
        self.events.add("task", state="started", prompt=prompt, advisor=advisor, recording_id=rid,
                        viewer=self.rec.viewer_url(rid))
        return rid

    def end_task(self, why: str = "stopped") -> None:
        if self.rec.active:
            rid = self.rec.id
            self.rec.marker("task end", {"why": why})
            self.rec.stop()
            self.events.add("task", state="ended", why=why, recording_id=rid)

    def describe(self) -> dict:
        return {"backend": self.backend.describe(), "skills": {k: s.describe() for k, s in self.skills.items()},
                "locator": self.locator.name, "perception": self.locator.describe(), "site": get_site().name,
                "scene_calib": str(scene_calib_path()) if scene_calib_path() else None,
                "tool": self.tool, "layout": self.layout, "profile": self.kin.P.name,
                "twin_preflight": self.twin.describe() if self.twin else None,
                "skill_config": SC.stamp(self.kin.cfg_target)}
