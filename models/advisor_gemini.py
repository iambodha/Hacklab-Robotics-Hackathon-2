"""Gemini Robotics ER as an advisor: Observation + goal -> ActionProposal(s); camera evidence -> Outcome.

    adv = GeminiAdvisor()                                      # GeminiER() + perception locator if importable
    props = adv.propose(obs, "wipe the coffee stain off the table", history=["wipe_region done"])
    out = adv.residual(obs, execution_id, what="coffee stain") # Outcome(metric="residual_stain_area", cm2)
    out = adv.judge(obs, execution_id, "pick the purple wrapper") # Outcome(metric="gemini_task_done")

Gemini only *chooses* from a small vocabulary (gemini_er.VOCABULARY); the executor validates and runs the
skill. Pixels become world metres through perception.locate (overview camera; wrist camera when the
calibrated joint ranges are given). When perception isn't available the proposal carries the pixel, the
camera and the frame ref, with `needs_localisation: true` and a matching precondition, so the executor
can localise it or reject it.

expires_at is on the executor's monotonic clock: obs.t_mono + the skill's TTL (the decision is only as
good as the observation it was based on). `to_api(proposal, now)` adds `ttl_s` for the executor's HTTP
API (contracts.proposal_from_dict), since an advisor in another process can't read that clock.
"""
from __future__ import annotations

import json
import math
import os
import urllib.request
from pathlib import Path

from . import perception_bridge as pb
from ._contracts import ActionProposal, Observation, Outcome, to_dict
from .gemini_er import Box, GeminiER, GeminiError, Image, ParseError, Step

VERSION = "0.1"
SOURCE = "gemini-er"
# vocabulary action -> executor skill name (the integrator renames here if the executor's skills differ)
SKILLS = {"inspect": "inspect", "pick": "pick_at", "retry_grasp": "pick_at", "place": "drop_at",
          "wipe_region": "wipe_region", "stop": "stop"}
# seconds a proposal stays valid after the observation it was based on
TTL_S = {"inspect": 30.0, "pick_at": 12.0, "drop_at": 12.0, "wipe_region": 20.0, "stop": 5.0}
WIPE_MAX_LINE_M = 0.25            # sim/robots/elrobot.json wipe.max_length
WIPE_MIN_LINE_M = 0.03
EXECUTOR_URL = os.environ.get("EXECUTOR_URL", "http://127.0.0.1:8770")
RECORDING_DIRS = [Path(p) for p in os.environ.get("RECORDER_DIRS", "").split(os.pathsep) if p] + \
    [Path(__file__).resolve().parents[2] / "recorder" / "recordings"]


# ---- frames -----------------------------------------------------------------------------------------------
def load_frame(ref: str, role: str) -> bytes:
    """JPEG bytes for a FrameRef.path_or_recording_ref: "recording:<id>@<t>", "/api/..." (executor), URL or path."""
    if ref.startswith("recording:"):
        from recorder.replay import Recording
        rid, _, t = ref.removeprefix("recording:").rpartition("@")
        path = next((d / rid for d in RECORDING_DIRS if (d / rid / "session.json").exists()), rid)
        hit = Recording(path).frame_at(role, float(t), causal=True)
        if hit is None:
            raise FileNotFoundError(f"no {role} frame in {rid} at {t}")
        return hit[1]
    if ref.startswith("/api/") or ref.startswith("http://") or ref.startswith("https://"):
        url = ref if ref.startswith("http") else EXECUTOR_URL.rstrip("/") + ref
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.read()
    return Path(ref).read_bytes()


# ---- geometry helpers --------------------------------------------------------------------------------------
def polygon_area_m2(pts) -> float:
    return abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))) / 2


def wipe_lines(region_xy, spacing_m: float, max_len: float = WIPE_MAX_LINE_M) -> list[list[float]]:
    """Parallel wipe strokes [x0, y0, x1, y1] covering an axis-aligned world rectangle [[xmin, ymin], [xmax, ymax]],
    along its longer side, `spacing_m` apart, each within the wipe skill's length limits."""
    (x0, y0), (x1, y1) = region_xy
    along_x = (x1 - x0) >= (y1 - y0) - 1e-9
    a0, a1, c0, c1 = (x0, x1, y0, y1) if along_x else (y0, y1, x0, x1)
    if a1 - a0 < WIPE_MIN_LINE_M:
        mid = (a0 + a1) / 2
        a0, a1 = mid - WIPE_MIN_LINE_M / 2, mid + WIPE_MIN_LINE_M / 2
    n_seg = max(1, math.ceil((a1 - a0) / max_len))
    seg = (a1 - a0) / n_seg
    n_lines = max(1, math.ceil((c1 - c0) / max(spacing_m, 1e-3)) + 1) if c1 - c0 > 1e-6 else 1
    cs = [c0 + (c1 - c0) * k / (n_lines - 1) for k in range(n_lines)] if n_lines > 1 else [(c0 + c1) / 2]
    out = []
    for c in cs:
        for s in range(n_seg):
            p, q = a0 + s * seg, a0 + (s + 1) * seg
            out.append([round(p, 4), round(c, 4), round(q, 4), round(c, 4)] if along_x else
                       [round(c, 4), round(p, 4), round(c, 4), round(q, 4)])
    return out


def adapt_spacing(spacing_m: float, residual_fraction: float | None, lo: float = 0.01, hi: float = 0.05) -> float:
    """Tighter strokes when stain is left behind, looser when the last wipe left nothing."""
    if residual_fraction is None:
        return spacing_m
    f = 0.6 if residual_fraction > 0.25 else 0.8 if residual_fraction > 0.05 else 1.15
    return round(min(hi, max(lo, spacing_m * f)), 4)


# ---- advisor ------------------------------------------------------------------------------------------------
class GeminiAdvisor:
    source = SOURCE

    def __init__(self, er: GeminiER | None = None, frame_loader=load_frame, ranges=None,
                 cameras: tuple[str, ...] = ("overview", "wrist"), stroke_spacing_m: float = 0.03,
                 min_confidence: float = 0.6, use_perception: bool = True):
        self.er = er or GeminiER(source_tag="advisor")
        self.frame_loader, self.ranges, self.cameras = frame_loader, ranges, cameras
        self.stroke_spacing_m, self.min_confidence = stroke_spacing_m, min_confidence
        self.use_perception = use_perception
        self.last_error: str | None = None
        self.last_step: Step | None = None

    # -- inputs --
    def images(self, obs: Observation) -> list[Image]:
        out = []
        for role in self.cameras:
            fr = obs.frames.get(role)
            if fr is None or fr.stale:
                continue
            try:
                out.append(Image(self.frame_loader(fr.path_or_recording_ref, role), role=role, ref=fr.path_or_recording_ref))
            except Exception as e:
                self.last_error = f"{role} frame {fr.path_or_recording_ref}: {type(e).__name__}: {e}"
        return out

    def _joints(self, obs: Observation):
        if not self.ranges or not obs.joints or not obs.joints.raw:
            return None
        return pb.joints_for_locate(obs.joints.raw, self.ranges)

    def _to_world(self, cam: str, u: float, v: float, obs: Observation, z: float = 0.0) -> dict:
        if not self.use_perception:
            return {"ok": False, "reason": "perception disabled"}
        return pb.pixel_to_world(cam, u, v, z, self._joints(obs) if cam == "wrist" else None)

    # -- proposals --
    def propose(self, obs: Observation, goal: str, history: list[str] | None = None) -> list[ActionProposal]:
        """0 or 1 proposals. Empty list = nothing usable (no fresh frames, model error, unparseable answer):
        the reason is in self.last_error."""
        self.last_error = None
        imgs = self.images(obs)
        if not imgs:
            self.last_error = self.last_error or "no fresh camera frames in the observation"
            return []
        try:
            step = self.er.propose_step(imgs, goal, history or [])
        except (GeminiError, ParseError) as e:
            self.last_error = str(e)
            return []
        self.last_step = step
        return [self.to_proposal(step, obs, imgs)]

    def to_proposal(self, step: Step, obs: Observation, imgs: list[Image]) -> ActionProposal:
        skill = SKILLS[step.action]
        a = step.args
        refs = {i.role: i.ref for i in imgs}
        params: dict = {"vocabulary_action": step.action}
        pre = [f"setup_id == {obs.setup_id}"]
        if step.action == "inspect":
            params["camera"] = a.get("camera", "overview")
        elif step.action == "stop":
            params["reason"] = str(a.get("reason") or step.rationale)
        elif step.action in ("pick", "retry_grasp", "place"):
            name = a.get("object") or a.get("target")
            params.update({"object" if step.action != "place" else "target": name, "camera": a["camera"],
                           "pixel": [a["u"], a["v"]], "frame_ref": refs.get(a["camera"])})
            w = self._to_world(a["camera"], a["u"], a["v"], obs)
            if w.get("ok"):
                params.update(xyz_world=w["xyz"], xyz_frame="world (scene-contract), metres, z = table plane",
                              sigma_mm=w.get("sigma_mm"))
                pre.append(f"{name!r} still within {max(20, 3 * (w.get('sigma_mm') or 10)):.0f} mm of xyz_world")
            else:
                params.update(needs_localisation=True, localisation_error=w.get("reason"))
                pre.append("executor localises pixel (camera, frame_ref) to world before running")
            if step.action == "retry_grasp":
                params["retry"] = True
                pre.append("gripper is empty")
            if step.action == "place":
                pre.append("gripper is holding the object")
        elif step.action == "wipe_region":
            box: Box = a["box"]
            params.update(camera=box.camera, pixel_box=[box.u0, box.v0, box.u1, box.v1], frame_ref=refs.get(box.camera),
                          stroke_spacing_m=self.stroke_spacing_m)
            ws = [self._to_world(box.camera, u, v, obs) for u, v in box.corners()]
            if all(w.get("ok") for w in ws):
                poly = [w["xyz"][:2] for w in ws]
                xs, ys = [p[0] for p in poly], [p[1] for p in poly]
                region = [[min(xs), min(ys)], [max(xs), max(ys)]]
                params.update(polygon_world_xy=poly, region_world_xy=region, area_cm2=round(polygon_area_m2(poly) * 1e4, 1),
                              lines_world_xy=wipe_lines(region, self.stroke_spacing_m),
                              xy_frame="world (scene-contract), metres")
                pre.append("region clear of objects other than the stain")
            else:
                params.update(needs_localisation=True,
                              localisation_error=next(w.get("reason") for w in ws if not w.get("ok")))
                pre.append("executor localises pixel_box to a world region before running")
            pre.append("napkin held in the jaws")
        info = self.er.last
        tail = f" [{self.er.model}, {info.latency_s:.1f} s, ${info.cost_usd or 0:.4f}]" if info else ""
        conf = f" (confidence {step.confidence:.2f})" if step.confidence is not None else ""
        return ActionProposal(proposal_id=ActionProposal.new_id(), source=self.source, based_on_obs=obs.obs_id,
                              setup_id=obs.setup_id, skill=skill, skill_version=f"gemini-advisor/{VERSION}",
                              params=params, preconditions=pre, expires_at=obs.t_mono + TTL_S.get(skill, 10.0),
                              rationale=(step.rationale or step.action) + conf + tail)

    # -- evaluation (Outcome, never the executor's own claim) --
    def judge(self, obs: Observation, execution_id: str, task: str, criteria: str | None = None) -> Outcome:
        imgs = self.images(obs)
        ev = [f"{i.role}:{i.ref}" for i in imgs]
        if not imgs:
            return Outcome(execution_id, "gemini_task_done", float("nan"), "probability", None, ["no fresh frames"])
        try:
            j = self.er.judge(imgs, task, criteria)
        except (GeminiError, ParseError) as e:
            return Outcome(execution_id, "gemini_task_done", float("nan"), "probability", None, ev + [f"error: {e}"])
        conf = j.confidence if j.confidence is not None else 0.0
        success = j.done if (j.done is not None and conf >= self.min_confidence) else None
        p_done = conf if j.done else (1 - conf) if j.done is False else 0.5
        return Outcome(execution_id, "gemini_task_done", round(p_done, 3), "probability", success,
                       ev + [f"gemini:{self.er.last.call_id}" if self.er.last else "gemini",
                             f"evidence: {j.evidence}"] + ([f"failure: {j.failure}"] if j.failure else []))

    def residual(self, obs: Observation, execution_id: str, what: str = "stain", camera: str = "overview",
                 max_area_cm2: float = 2.0) -> Outcome:
        """Remaining `what` on the table, as world area (cm2) from Gemini's boxes on the overview frame."""
        imgs = [i for i in self.images(obs) if i.role == camera]
        if not imgs:
            return Outcome(execution_id, "residual_stain_area", float("nan"), "cm2", None, [f"no fresh {camera} frame"])
        try:
            boxes = self.er.boxes(imgs[0], [f"every remaining patch of {what} on the table"])
        except (GeminiError, ParseError) as e:
            return Outcome(execution_id, "residual_stain_area", float("nan"), "cm2", None, [f"error: {e}"])
        total, ev = 0.0, [f"{camera}:{imgs[0].ref}", f"gemini:{self.er.last.call_id}"]
        for b in boxes:
            ws = [self._to_world(camera, u, v, obs) for u, v in b.corners()]
            if not all(w.get("ok") for w in ws):
                return Outcome(execution_id, "residual_stain_area", float("nan"), "cm2", None,
                               ev + [f"box {b.label} not localised: {next(w.get('reason') for w in ws if not w.get('ok'))}"])
            a = polygon_area_m2([w["xyz"][:2] for w in ws]) * 1e4
            total += a
            ev.append(f"box px {[b.u0, b.v0, b.u1, b.v1]} -> {a:.1f} cm2")
        return Outcome(execution_id, "residual_stain_area", round(total, 2), "cm2", total <= max_area_cm2, ev)

    def update_spacing(self, residual_cm2: float | None, wiped_cm2: float | None) -> float:
        frac = None if residual_cm2 is None or not wiped_cm2 else residual_cm2 / wiped_cm2
        self.stroke_spacing_m = adapt_spacing(self.stroke_spacing_m, frac)
        return self.stroke_spacing_m


def to_api(p: ActionProposal, now_mono: float | None = None) -> dict:
    """JSON for the executor's POST /proposals. With now_mono, also `ttl_s` (seconds left)."""
    d = to_dict(p)
    if now_mono is not None:
        d["ttl_s"] = round(p.expires_at - now_mono, 3)
    return json.loads(json.dumps(d, default=str))
