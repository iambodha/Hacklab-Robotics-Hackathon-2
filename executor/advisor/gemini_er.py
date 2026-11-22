"""Gemini Robotics ER as a background advisor, same interface as the Claude advisor (make_advisor("gemini", ...)).

All model work is delegated to models/advisor_gemini.GeminiAdvisor:
  propose(obs, task, history) -> 0/1 ActionProposal (pixels -> world via perception; skill names pick_at / drop_at /
                                 wipe_region / inspect / stop, all executor skills)
  judge(obs, execution_id, task) -> Outcome (independent: "is the task done?", None when unsure)
Loop: observe -> propose -> submit to the executor (it validates; the proposal's expiry is on the executor clock) ->
wait for the result -> judge after every SUCCEEDED step -> stop when the judge says done, on `stop`, or after
EXECUTOR_GEMINI_STEPS steps. Model: EXECUTOR_GEMINI_MODEL (default models.gemini_er.DEFAULT_MODEL).

Without a key, or with a key the API rejects, the advisor says so in the console timeline and ends; nothing crashes
and nothing moves.
"""
from __future__ import annotations

import json
import os
import time

from .base import Advisor

AUTH_WORDS = ("api key", "api_key", "permission_denied", "unauthenticated", "401", "403", "invalid_argument")


def enabled() -> bool:
    try:
        from models._secrets import has_gemini_key
    except ImportError:
        return bool(os.environ.get("GEMINI_API_KEY"))
    return has_gemini_key()


def key_problem(text: str | None) -> bool:
    t = (text or "").lower()
    return any(w in t for w in AUTH_WORDS) and ("key" in t or "auth" in t or "permission" in t)


class GeminiERAdvisor(Advisor):
    name = "gemini"

    def __init__(self, *a, executor=None, advisor=None, steps: int | None = None, **kw):
        super().__init__(*a, **kw)
        self.ex = executor
        self.adv = advisor                     # models.advisor_gemini.GeminiAdvisor (built in run() if None)
        self.steps = steps or int(os.environ.get("EXECUTOR_GEMINI_STEPS", "20"))
        self.model = os.environ.get("EXECUTOR_GEMINI_MODEL")
        self.last_error: str | None = None

    def describe(self) -> dict:
        return {**super().describe(), "model": getattr(getattr(self.adv, "er", None), "model", self.model),
                "key_present": enabled() if self.adv is None else True, "last_error": self.last_error}

    def _frame(self, ref: str, role: str) -> bytes:
        parts = ref.strip("/").split("/")           # api/frame/<obs_id>/<role>.jpg
        jpeg = self.ex.frame_jpeg(parts[2], role) if len(parts) >= 4 and parts[:2] == ["api", "frame"] else None
        if jpeg is None:
            raise FileNotFoundError(f"no {role} frame for {ref}")
        return jpeg

    def _ranges(self) -> dict:
        st = self.ex.backend.read_state().motors
        return {f"j{i + 1}": [st[m]["lo"], st[m]["hi"]] for i, m in enumerate(self.ex.kin.ids)}

    def _make(self):
        from models.advisor_gemini import GeminiAdvisor
        from models.gemini_er import GeminiER
        er = GeminiER(model=self.model, source_tag="executor") if self.model else GeminiER(source_tag="executor")
        return GeminiAdvisor(er=er, frame_loader=self._frame, ranges=self._ranges())

    def _disabled(self, text: str, why: str) -> str:
        self.last_error = text
        self.log("message", text=text)
        self.log("error", text=text)
        return why

    def run(self):
        if self.adv is None:
            if not enabled():
                return self._disabled("Gemini ER advisor disabled: no GEMINI_API_KEY (set it in the executor's environment or "
                                      "secrets.env). Nothing was proposed.", "disabled (no GEMINI_API_KEY)")
            try:
                self.adv = self._make()
            except Exception as e:  # noqa: BLE001  (models package / SDK missing)
                return self._disabled(f"Gemini ER advisor unavailable: {type(e).__name__}: {e}", "disabled (models unavailable)")
        from ..contracts import Status, to_dict
        history: list[str] = []
        failures = 0
        for step in range(self.steps):
            if self._stop.is_set():
                return "stopped by the operator"
            obs = self.ex.observe()
            self.log("tool_call", tool="gemini.propose", input={"step": step, "obs_id": obs.obs_id})
            props = self.adv.propose(obs, self.task, history)
            if not props:
                err = self.adv.last_error or "no proposal"
                if key_problem(err):
                    return self._disabled(f"Gemini rejected the API key: {err[:300]}", "disabled (key rejected)")
                failures += 1
                history.append(f"step {step}: no usable answer ({err[:200]})")
                self.last_error = err
                self.log("tool_result", text=history[-1], error=True)
                if failures >= 3:
                    return f"gave up after 3 failed calls: {err[:200]}"
                continue
            failures = 0
            p = props[0]
            self.log("message", text=f"{p.skill} {json.dumps(p.params, default=str)[:400]} — {p.rationale[:300]}")
            if p.skill == "stop":
                return "stop: " + str(p.params.get("reason", ""))[:300]
            eid = self.ex.submit(p)
            r = self.ex.wait(eid, 180)
            history.append(f"step {step}: {p.skill} {json.dumps({k: v for k, v in p.params.items() if k != 'frame_ref'}, default=str)[:300]}"
                           f" -> {r.status.value}: {r.termination} residuals {json.dumps(r.residuals, default=str)[:200]}")
            self.log("tool_result", text=history[-1], execution_id=eid)
            if r.status == Status.SUCCEEDED and p.skill != "inspect":
                o = self.adv.judge(self.ex.observe(), eid, self.task)
                self.log("outcome", outcome=to_dict(o))
                history.append(f"judge after step {step}: {o.metric}={o.value} success={o.success}")
                if o.success is True:
                    return f"done: judged complete ({o.metric}={o.value})"
            time.sleep(0.2)
        return f"step budget ({self.steps}) used up"
