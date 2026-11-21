"""Advisors run in the background and talk to the executor ONLY through its API (observe / submit / status /
cancel). They never send motor commands and nothing waits for them: a late answer is simply rejected by
validation (expired, stale observation, setup changed)."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOG_DIR = REPO / "executor" / "logs"

FRAME_TEXT = ("World frame (scene-contract.md): metres, origin = the arm base on the table top, +z up, +x = straight "
              "ahead of the arm (away from its base, towards the far side of the table), +y = the arm's left. Yaw angles in "
              "degrees about +z, 0 = along +x.")


def system_prompt(skills: dict, scene: dict) -> str:
    lines = [f"- {s['name']} (v{s['version']}, needs an observation <= {s['freshness_s']} s old): {s['doc']}\n    params: "
             + json.dumps(s["params"]) for s in skills.values()]
    return f"""You advise a robot arm (ElRobot follower, 7 joints + gripper) on a table. You do NOT move it yourself:
a local executor owns all motion. You look, then PROPOSE one skill at a time; the executor validates every
proposal (freshness of your observation, expiry, setup, joint limits, whole-path reachability) and runs it, or
rejects it with the reasons. Nothing you send is a motor command.

{FRAME_TEXT}
Workspace: TCP (the point between the fingertip pads) z from {scene.get('min_tcp_z')} to {scene.get('max_tcp_z')} m, at most
{scene.get('max_radius')} m from the base. The table top is z = 0; fingertips touch it at TCP z ~ {scene.get('table_tcp_z')}.

Loop:
1. look() for fresh images + the observation (obs_id, setup_id, joints, TCP, object estimates). Object estimates come
   from a simple detector and can be wrong or missing; visible=false means unknown, not absent. Prefer what you see.
2. submit(skill, params, obs_id, setup_id, rationale) with ONE skill. Say in the rationale what you expect to see after.
   Submit right after looking: proposals based on an observation older than the skill's limit are rejected.
3. status(execution_id, wait_s=20) until it is final. Read status, termination and residuals:
   SUCCEEDED = measured within tolerance; PARTIAL = stopped short (residuals say how far); BLOCKED = could not proceed
   (e.g. nothing held); FAULT; REJECTED (read the reasons, fix, re-observe, resubmit); CANCELLED (the operator stopped it).
4. Verify in fresh images before the next step. Never assume a step worked because it was accepted.

Skills:
{chr(10).join(lines)}

Rules: one proposal in flight at a time. If something is unclear or unsafe, stop and report BLOCKED with the reason.
When the task is finished, reply with DONE and a one-paragraph summary of what you verified."""


class Advisor:
    name = "none"

    def __init__(self, api_url: str, events, skills: dict, scene: dict, on_done=None):
        self.api, self.events, self.skills, self.scene = api_url, events, skills, scene
        self.on_done = on_done or (lambda why: None)
        self.system = system_prompt(skills, scene)
        self.task = ""
        self.running = False
        self.log_path: Path | None = None
        self._stop = threading.Event()

    def describe(self) -> dict:
        return {"name": self.name, "running": self.running, "system_prompt": self.system, "task_prompt": self.task,
                "log": str(self.log_path) if self.log_path else None}

    def log(self, kind: str, **data) -> None:
        self.events.add("advisor_" + kind, advisor=self.name, **data)
        if self.log_path:
            with self.log_path.open("a") as f:
                f.write(json.dumps({"t": time.time(), "kind": kind, **data}, default=str) + "\n")

    def start(self, task: str) -> None:
        self.task, self.running = task, True
        self._stop.clear()
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOG_DIR / f"advisor_{self.name}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        self.log("prompt", system_prompt=self.system, task_prompt=task)
        threading.Thread(target=self._main, daemon=True, name=f"advisor-{self.name}").start()

    def _main(self):
        why = "finished"
        try:
            why = self.run() or why
        except Exception as e:  # noqa: BLE001
            why = f"advisor error: {type(e).__name__}: {e}"
            self.log("error", text=why)
        finally:
            self.running = False
            self.log("end", why=why)
            self.on_done(why)

    def run(self) -> str | None:
        return "no advisor: submit proposals from the console"

    def stop(self) -> None:
        self._stop.set()
