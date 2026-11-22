"""A twin layout for the bottle demo: a copy of sim/layouts/home.json plus a bottle and a screw cap.

Only tests and sim runs use it (sim/layouts/home.json itself is untouched). The bottle is static (the user's hand
holds it); the cap sits on a screw-joint approximation that sim/sim_arm_mcp.py adds to the generated scene:

    cap_hinge  hinge about the cap axis (+z), frictionloss = the torque needed to turn it (N m), cannot tighten past 0
    cap_slide  slide along +z, coupled to the hinge: slide = pitch / (2 pi) * hinge (the thread lifts it)
    free       once the hinge passes turns_to_free turns the coupling and the friction go away: the cap lifts off

Geometry (sim frame = base frame): body cylinder, a narrow neck (so the jaws can close on the cap without touching
the shoulder), cap on top. Sizes in metres.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HOME = REPO / "sim" / "layouts" / "home.json"

BOTTLE = {"xy": [0.0, 0.20], "body_r": 0.035, "body_h": 0.09, "neck_r": 0.015, "neck_top": 0.0995,
          "cap_r": 0.02, "cap_h": 0.02, "cap_base": 0.10}
SCREW = {"pitch_mm": 3.0, "turns_to_free": 1.25, "friction_torque": 0.30, "mass": 0.02}


def cap_top_z(b: dict = BOTTLE) -> float:
    return b["cap_base"] + b["cap_h"]


def make_layout(path: Path, bottle: dict | None = None, screw: dict | None = None) -> Path:
    b, s = {**BOTTLE, **(bottle or {})}, {**SCREW, **(screw or {})}
    lay = json.loads(HOME.read_text())
    x, y = b["xy"]
    black = [0.04, 0.04, 0.045, 1]
    lay["fixed"] = list(lay.get("fixed", [])) + [
        {"name": "bottle_body", "type": "cylinder", "pos": [x, y, b["body_h"] / 2], "size": [b["body_r"], b["body_h"] / 2],
         "yaw": 0, "rgba": black},
        {"name": "bottle_neck", "type": "cylinder", "pos": [x, y, (b["body_h"] + b["neck_top"]) / 2],
         "size": [b["neck_r"], (b["neck_top"] - b["body_h"]) / 2], "yaw": 0, "rgba": black}]
    lay["screw_cap"] = {"name": "screw_cap", "pos": [x, y, b["cap_base"]], "radius": b["cap_r"],
                        "half_height": b["cap_h"] / 2, "rgba": [0.12, 0.12, 0.14, 1], **s}
    lay["_notes"] = list(lay.get("_notes", [])) + ["TEST COPY of home.json with a bottle + screw cap (learning/bottle_twin.py)."]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lay, indent=1))
    return path
