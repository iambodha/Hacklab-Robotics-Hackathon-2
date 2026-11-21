"""Gemini Robotics ER as a background advisor (same interface as the Claude advisor). Enabled only when
GEMINI_API_KEY is set; otherwise start() logs why and ends at once.

Loop: observe (with both images) -> ask the model for ONE proposal as JSON -> submit -> wait for the result ->
feed the result back -> repeat, at most EXECUTOR_GEMINI_STEPS times. Model: EXECUTOR_GEMINI_MODEL
(default gemini-robotics-er-1.5-preview), REST generateContent, stdlib only.
NOT yet run against the real API in this workstream (no key here): treat the request format as unverified.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.request

from .base import Advisor

URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
ANSWER = """Reply with JSON only, one of:
{"proposal": {"skill": "...", "params": {...}, "rationale": "..."}}
{"done": true, "summary": "..."}
{"blocked": true, "reason": "..."}"""


def enabled() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


class GeminiERAdvisor(Advisor):
    name = "gemini"

    def __init__(self, *a, executor=None, **kw):
        super().__init__(*a, **kw)
        self.ex = executor
        self.model = os.environ.get("EXECUTOR_GEMINI_MODEL", "gemini-robotics-er-1.5-preview")
        self.steps = int(os.environ.get("EXECUTOR_GEMINI_STEPS", "20"))

    def ask(self, parts: list) -> str:
        body = {"system_instruction": {"parts": [{"text": self.system + "\n\n" + ANSWER}]},
                "contents": [{"role": "user", "parts": parts}], "generationConfig": {"temperature": 0.2}}
        req = urllib.request.Request(URL.format(model=self.model), data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]})
        with urllib.request.urlopen(req, timeout=90) as r:
            d = json.loads(r.read())
        return "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])

    def run(self):
        if not enabled():
            self.log("message", text="Gemini ER advisor disabled: set GEMINI_API_KEY in the executor's environment.")
            return "disabled (no GEMINI_API_KEY)"
        from ..contracts import proposal_from_dict, to_dict
        history: list[str] = []
        for step in range(self.steps):
            if self._stop.is_set():
                return "stopped by the operator"
            obs = self.ex.observe()
            o = to_dict(obs)
            parts = [{"text": f"Task: {self.task}\n\nHistory:\n" + "\n".join(history[-8:]) +
                      f"\n\nObservation:\n{json.dumps({k: o[k] for k in ('obs_id', 'setup_id', 'freshness_s', 'joints', 'objects', 'notes')})[:6000]}"}]
            for role in ("wrist", "overview"):
                jpeg = self.ex.frame_jpeg(obs.obs_id, role)
                if jpeg:
                    parts += [{"text": f"{role} camera:"}, {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(jpeg).decode()}}]
            self.log("tool_call", tool="generateContent", input={"step": step, "obs_id": obs.obs_id})
            text = self.ask(parts)
            self.log("message", text=text[:2000])
            m = re.search(r"\{.*\}", text, re.S)
            try:
                ans = json.loads(m.group(0)) if m else {}
            except ValueError:
                ans = {}
            if ans.get("done"):
                return "done: " + str(ans.get("summary", ""))[:300]
            if ans.get("blocked"):
                return "blocked: " + str(ans.get("reason", ""))[:300]
            prop = ans.get("proposal")
            if not prop:
                history.append(f"step {step}: unreadable answer")
                continue
            p = proposal_from_dict({**prop, "based_on_obs": obs.obs_id, "setup_id": obs.setup_id, "source": "gemini-er", "ttl_s": 20},
                                   self.ex.now())
            eid = self.ex.submit(p)
            r = self.ex.wait(eid, 120)
            history.append(f"step {step}: {p.skill} {json.dumps(p.params)} -> {r.status.value}: {r.termination} "
                           f"residuals {json.dumps(r.residuals)}")
            self.log("tool_result", text=history[-1])
            time.sleep(0.2)
        return f"step budget ({self.steps}) used up"
