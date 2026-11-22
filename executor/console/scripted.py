"""Scripted advisor: a deterministic planner for "put the <colour> object in the bin", same interface as the Claude
and Gemini advisors. It only observes and submits proposals (never moves the arm), and logs every step to the
timeline as tool calls, so the console shows its logic the same way it shows an LLM's.

Plan: look -> locate the target (colour from the prompt, default blue) and the bin -> pick (retry up to 3 times,
re-observing each time) -> look -> place above the bin centre (rim + clearance) -> park -> independent outcome.

Where the bin comes from, in order: a perception estimate named bin/bucket/box; EXECUTOR_BIN_XYZ="x,y,rim_z" (world,
metres); the twin layout's fixed bin (sim only: a fixture of the scene, not object ground truth).
"""
from __future__ import annotations

import os
import re
import time

import numpy as np

from ..advisor.base import Advisor
from ..contracts import ActionProposal, Status

COLOURS = ("blue", "red", "green", "yellow", "orange", "purple", "black", "white", "grey", "cyan", "magenta", "pink")
BIN_WORDS = ("bin", "bucket", "box", "basket", "tray")
MAX_ATTEMPTS = 3
PLACE_CLEARANCE = 0.07            # release this far above the rim (the place skill wants >= 6 cm)
DEFAULT_RIM = 0.09


def target_colour(prompt: str) -> str:
    words = re.findall(r"[a-z]+", prompt.lower())
    return next((w for w in words if w in COLOURS), "blue")


class ScriptedAdvisor(Advisor):
    name = "scripted"

    def __init__(self, api_url, events, skills, scene, executor=None, on_done=None):
        super().__init__(api_url, events, skills, scene, on_done=on_done)
        self.ex = executor
        self.system = ("Scripted planner (deterministic, no model). It proposes one skill at a time through the "
                       "executor, which validates and runs each proposal.\n"
                       "1. look: fresh observation (both cameras, joints, perception estimates)\n"
                       "2. locate the target by the colour named in the task (default blue) and the bin\n"
                       f"3. pick at the target (up to {MAX_ATTEMPTS} attempts, re-observing before each)\n"
                       f"4. look, then place above the bin centre at rim + {PLACE_CLEARANCE * 100:.0f} cm\n"
                       "5. park, then ask the executor's independent evaluator for the outcome of pick and place\n"
                       "Bin source: perception estimate named bin/bucket/box, else EXECUTOR_BIN_XYZ, else the twin "
                       "layout's fixed bin (sim only).")

    # --- steps (each one is logged as a tool call + result) ------------------------------------
    def look(self):
        self.log("tool_call", tool="observe", input={})
        o = self.ex.observe()
        objs = [{"name": x.name, "xyz_world": x.xyz_world, "visible": x.visible, "uncertainty_mm": x.uncertainty_mm,
                 "source": x.source} for x in o.objects]
        self.log("tool_result", text=f"{o.obs_id}: freshness {o.freshness_s} s, {len(objs)} object estimate(s)"
                 + "".join(f"\n  {d['name']} @ {np.round(d['xyz_world'], 3).tolist()}" + ("" if d["visible"] else " (not visible)")
                           for d in objs), obs_id=o.obs_id)
        return o

    def propose(self, skill: str, params: dict, obs, rationale: str):
        p = ActionProposal(ActionProposal.new_id(), "script", obs.obs_id, obs.setup_id, skill, "", params, [],
                           self.ex.now() + 15.0, rationale)
        self.log("tool_call", tool="submit", input={"skill": skill, "params": params, "rationale": rationale})
        eid = self.ex.submit(p)
        while not self._stop.is_set():
            r = self.ex.status(eid)
            if r.status.final:
                break
            time.sleep(0.1)
        r = self.ex.status(eid)
        self.log("tool_result", text=f"{skill} {r.status.value}: {r.termination}", execution_id=eid, error=r.status != Status.SUCCEEDED)
        return eid, r

    def outcome(self, eid: str):
        self.log("tool_call", tool="evaluate", input={"execution_id": eid})
        o = self.ex.evaluate(eid)
        self.log("tool_result", text=f"{o.metric} = {o.value} {o.unit} -> "
                 + ("unknown" if o.success is None else "success" if o.success else "failure"), execution_id=eid)
        return o

    # --- target finding -----------------------------------------------------------------------
    def find_target(self, obs, colour: str):
        cands = [o for o in obs.objects if o.visible and colour in o.name.lower() and not any(w in o.name.lower() for w in BIN_WORDS)]
        if not cands:
            return None
        return min(cands, key=lambda o: (o.uncertainty_mm or 50, float(np.hypot(*o.xyz_world[:2]))))

    def find_bin(self, obs) -> dict | None:
        """{xy, rim, half (inner half-extent, m), src} of the bin, or None."""
        for o in obs.objects:
            if o.visible and any(w in o.name.lower() for w in BIN_WORDS):
                rim = float(o.size[2]) if o.size and len(o.size) >= 3 else float(os.environ.get("EXECUTOR_BIN_RIM_M", DEFAULT_RIM))
                half = min(o.size[:2]) / 2 if o.size and len(o.size) >= 2 else 0.04
                return {"xy": [float(o.xyz_world[0]), float(o.xyz_world[1])], "rim": rim, "half": half,
                        "src": f"perception: {o.name} ({o.source})"}
        env = os.environ.get("EXECUTOR_BIN_XYZ")
        if env:
            x, y, rim = (float(v) for v in env.split(","))
            return {"xy": [x, y], "rim": rim, "half": 0.04, "src": "EXECUTOR_BIN_XYZ"}
        layout = getattr(getattr(self.ex.backend, "arm", None), "layout", None)
        bins = (layout or {}).get("bins") or []
        if bins:
            b = bins[0]
            w = self.ex.kin.to_world([b["pos"][0], b["pos"][1], 0.0])
            size = b.get("size", [0.1, 0.1, DEFAULT_RIM])
            return {"xy": [round(float(w[0]), 4), round(float(w[1]), 4)], "rim": float(size[2]), "half": min(size[:2]) / 2,
                    "src": f"twin layout fixture {b.get('name')}"}
        return None

    @staticmethod
    def in_bin(target, b) -> bool:
        return b is not None and all(abs(target.xyz_world[i] - b["xy"][i]) < b["half"] for i in (0, 1))

    # --- the plan -------------------------------------------------------------------------------
    def run(self):
        colour = target_colour(self.task)
        self.log("message", text=f"Task: move the {colour} object into the bin. Looking first.")
        pick_eid = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if self._stop.is_set():
                return "stopped by the operator"
            obs = self.look()
            target = self.find_target(obs, colour)
            if target is None:
                self.log("message", text=f"BLOCKED: no visible {colour} object in observation {obs.obs_id}.")
                return f"BLOCKED: no {colour} object seen"
            if attempt == 1 and self.in_bin(target, self.find_bin(obs)):
                self.log("message", text=f"DONE: the {colour} object at {np.round(target.xyz_world, 3).tolist()} is already inside the bin.")
                return "DONE (already in the bin)"
            xyz = [round(float(v), 4) for v in target.xyz_world]
            h = float(target.size[2]) if target.size and len(target.size) >= 3 else None
            params = {"xyz_world": xyz} | ({"object_height": round(h, 3)} if h else {})
            self.log("message", text=f"Attempt {attempt}: {target.name} at {xyz} (±{target.uncertainty_mm or '?'} mm, "
                                     f"{target.source}). Proposing a pick.")
            pick_eid, r = self.propose("pick", params, obs, f"pick the {colour} object seen at {xyz}; expect it held after the lift")
            if r.status == Status.SUCCEEDED:
                o_pick = self.outcome(pick_eid)
                if o_pick.success is not False:
                    break
                self.log("message", text="The executor reports a hold but the independent check disagrees. Retrying.")
                continue
            if r.status == Status.CANCELLED or self._stop.is_set():
                return "cancelled"
            self.log("message", text=f"Pick {r.status.value} ({r.termination}). Re-observing and retrying.")
        else:
            return f"BLOCKED: pick failed {MAX_ATTEMPTS} times"
        if self._stop.is_set():
            return "stopped by the operator"
        obs = self.look()
        found = self.find_bin(obs)
        if found is None:
            self.log("message", text="BLOCKED: holding the object but no bin position (no estimate, no EXECUTOR_BIN_XYZ).")
            return "BLOCKED: bin not found"
        xy, rim, src = found["xy"], found["rim"], found["src"]
        z = round(max(rim + PLACE_CLEARANCE, 0.15), 3)
        self.log("message", text=f"Bin at {xy} (rim {rim:.3f} m, {src}). Releasing at z {z} m.")
        place_eid, r = self.propose("place", {"xyz_world": [*xy, z]}, obs, f"release above the bin centre {xy}, {z * 100:.0f} cm up")
        if r.status not in (Status.SUCCEEDED, Status.PARTIAL):
            return f"BLOCKED: place {r.status.value}"
        o_place = self.outcome(place_eid)
        obs = self.look()
        self.propose("park", {}, obs, "park clear of the bin and the cameras")
        verdict = "success" if o_place.success else "unknown" if o_place.success is None else "failure"
        self.log("message", text=f"DONE. pick: {o_pick.metric}={o_pick.value}; place: {o_place.metric}={o_place.value} ({verdict}).")
        return f"DONE ({verdict})"

    def describe(self) -> dict:
        return super().describe() | {"model": "deterministic script"}


__all__ = ["ScriptedAdvisor", "target_colour"]
