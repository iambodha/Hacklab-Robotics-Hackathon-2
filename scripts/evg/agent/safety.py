"""Validate and clamp what the VLM asks for. The model cannot bypass this; it only sees the notes.

Limits, in order: calibrated range (minus arm.MARGIN) -> per-call step cap -> demo envelope + slack -> floor.
Arm-specific facts (LOG.md): j2 increase = gripper goes DOWN in the top-down grasp pose, so j2 has a hard ceiling.
"""
ARM_JOINTS = range(1, 8)
MAX_STEP_DEG = 6.0        # per joint per action
ENVELOPE_SLACK = 300      # steps (~26 deg) beyond what the demos covered
FLOOR_MARGIN = 60         # steps past the lowest j2 seen in the demos
STEPS_PER_DEG = 4096 / 360


class Invalid(Exception):
    pass


def _num(x, what):
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise Invalid(f"{what} must be a number, got {x!r}")
    if v != v or abs(v) > 1e6:
        raise Invalid(f"{what} is not a sane number")
    return v


def bounds(j, m, pkg, margin):
    lo, hi = m["mn"] + margin, m["mx"] - margin
    e = (pkg or {}).get("envelope", {}).get(j)
    if e:
        lo, hi = max(lo, e[0] - ENVELOPE_SLACK), min(hi, e[1] + ENVELOPE_SLACK)
    if j == 2 and pkg and pkg.get("floor_j2"):
        hi = min(hi, pkg["floor_j2"] + FLOOR_MARGIN)
    return lo, hi


def clamp_targets(targets, st, pkg, margin):
    """targets {j: steps} -> (clamped, notes)."""
    out, notes = {}, []
    for j, t in targets.items():
        lo, hi = bounds(j, st[j], pkg, margin)
        c = int(min(max(t, lo), hi))
        if c != int(t):
            notes.append(f"j{j} limited to {c} (allowed {lo}..{hi})")
        out[j] = c
    return out, notes


def plan(action, st, pkg, margin):
    """Returns (kind, payload, notes). kind in step|goto|grip|wait|done. Raises Invalid."""
    if not isinstance(action, dict) or "action" not in action:
        raise Invalid('reply must contain {"action": {"action": ...}}')
    kind, notes = str(action["action"]), []
    if kind == "wait":
        return "wait", {"seconds": min(max(_num(action.get("seconds", 1), "seconds"), 0), 5)}, notes
    if kind == "done":
        return "done", {"reason": str(action.get("reason", ""))[:300]}, notes
    if kind == "grip":
        o = _num(action.get("opening"), "opening")
        if not 0 <= o <= 1:
            raise Invalid("opening must be 0..1")
        return "grip", {"opening": o}, notes
    if kind == "step":
        joints = action.get("joints")
        if not isinstance(joints, dict) or not joints:
            raise Invalid('step needs "joints": {"1": deg, ...}')
        targets = {}
        for k, d in joints.items():
            try: j = int(k)
            except ValueError: raise Invalid(f"bad joint {k!r}")
            if j not in ARM_JOINTS:
                raise Invalid(f"joint {j} not allowed (1-7; use grip for the gripper)")
            d = _num(d, f"joint {j} delta")
            if abs(d) > MAX_STEP_DEG:
                notes.append(f"j{j} delta capped to {MAX_STEP_DEG:+.0f} deg")
                d = max(-MAX_STEP_DEG, min(MAX_STEP_DEG, d))
            targets[j] = st[j]["pos"] + round(d * STEPS_PER_DEG)
        t, n = clamp_targets(targets, st, pkg, margin)
        return "step", {"targets": t}, notes + n
    if kind == "goto":
        name = action.get("waypoint")
        w = (pkg or {}).get("waypoints", {}).get(name)
        if not w:
            raise Invalid(f"unknown waypoint {name!r}; known: {', '.join((pkg or {}).get('waypoints', {}))}")
        t, n = clamp_targets({j: v for j, v in w["joints"].items() if j in ARM_JOINTS}, st, pkg, margin)
        return "goto", {"targets": t, "waypoint": name}, n
    raise Invalid(f"unknown action {kind!r}")


def reply_schema(chunk=1, waypoints=()):
    """JSON schema of a valid reply. Local servers (Ollama, llama.cpp, vLLM) can enforce it while generating, so the model
    cannot produce joint 8, an unknown waypoint or a missing wrapper, and generation stops when the object is complete."""
    num = {"type": "number"}
    step = {"type": "object", "properties": {"action": {"type": "string", "enum": ["step"]},
            "joints": {"type": "object", "properties": {str(j): num for j in ARM_JOINTS}, "additionalProperties": False}},
            "required": ["action", "joints"], "additionalProperties": False}
    goto = {"type": "object", "properties": {"action": {"type": "string", "enum": ["goto"]},
            "waypoint": {"type": "string", "enum": list(waypoints) or ["none"]}}, "required": ["action", "waypoint"], "additionalProperties": False}
    grip = {"type": "object", "properties": {"action": {"type": "string", "enum": ["grip"]}, "opening": num},
            "required": ["action", "opening"], "additionalProperties": False}
    wait = {"type": "object", "properties": {"action": {"type": "string", "enum": ["wait"]}, "seconds": num},
            "required": ["action", "seconds"], "additionalProperties": False}
    done = {"type": "object", "properties": {"action": {"type": "string", "enum": ["done"]}, "reason": {"type": "string"}},
            "required": ["action", "reason"], "additionalProperties": False}
    one = {"anyOf": [step, goto, grip, wait, done]}
    if chunk > 1:
        return {"type": "object", "properties": {"thought": {"type": "string"}, "actions": {"type": "array", "items": one, "minItems": 1, "maxItems": chunk}},
                "required": ["thought", "actions"], "additionalProperties": False}
    return {"type": "object", "properties": {"thought": {"type": "string"}, "action": one}, "required": ["thought", "action"], "additionalProperties": False}
