"""A twin layout for the bottle demo: a copy of the site's table layout (sites.get_site().layout: SITE, default home =
sim/layouts/home.json) plus a bottle and a screw cap.

Only tests and sim runs use it (the site layout itself is untouched). The bottle is static (the user's hand
holds it); the cap sits on a screw-joint approximation that sim/sim_arm_mcp.py adds to the generated scene:

    cap_hinge  hinge about the cap axis (+z), frictionloss = the torque needed to turn it (N m), cannot tighten past 0
    cap_slide  slide along +z, coupled to the hinge: slide = pitch / (2 pi) * hinge (the thread lifts it)
    free       once the hinge passes turns_to_free turns the coupling and the friction go away: the cap lifts off

Geometry (sim frame = base frame): body cylinder, a narrow neck (so the jaws can close on the cap without touching
the shoulder), cap on top. Sizes in metres. Positions are in the arm base frame, so the same xy gives the same arm
pose (and wrist lean) on every table; a site may move the demo bottle in sites/<site>/tasks.json "bottle".
"""
from __future__ import annotations

import json
from pathlib import Path

from sites import get_site

REPO = Path(__file__).resolve().parents[1]
HOME = REPO / "sim" / "layouts" / "home.json"   # the home site's layout (sites/home/site.json points at an equal copy)

BOTTLE = {"xy": [0.0, 0.20], "body_r": 0.035, "body_h": 0.09, "neck_r": 0.015, "neck_top": 0.0995,
          "cap_r": 0.02, "cap_h": 0.02, "cap_base": 0.10}
# The bottle on the real table: ~25 cm tall, cap ~4.4 cm, its base ~0.28-0.30 m in front of the arm base (overview
# camera calibration, 22 Nov 02:40; the contract's "10 cm" is the gap to the base's front edge).
REAL_BOTTLE = {"xy": [0.0, 0.28], "body_r": 0.036, "body_h": 0.215, "neck_r": 0.016, "neck_top": 0.2245,
               "cap_r": 0.022, "cap_h": 0.025, "cap_base": 0.225}
SCREW = {"pitch_mm": 3.0, "turns_to_free": 1.25, "friction_torque": 0.30, "mass": 0.02}


def cap_top_z(b: dict = BOTTLE) -> float:
    return b["cap_base"] + b["cap_h"]


def site_bottle(site: str | None = None) -> dict:
    """The site's demo bottle placement (sites/<site>/tasks.json "bottle"), {} when the site keeps the profile's."""
    path = get_site(site).directory / "tasks.json"
    return json.loads(path.read_text()).get("bottle", {}) if path.is_file() else {}


def make_layout(path: Path, bottle: dict | None = None, screw: dict | None = None, site: str | None = None) -> Path:
    b, s = {**BOTTLE, **(bottle or {})}, {**SCREW, **(screw or {})}
    st = get_site(site)
    lay = json.loads(st.layout.read_text())
    x, y = b["xy"]
    black = [0.04, 0.04, 0.045, 1]
    lay["fixed"] = list(lay.get("fixed", [])) + [
        {"name": "bottle_body", "type": "cylinder", "pos": [x, y, b["body_h"] / 2], "size": [b["body_r"], b["body_h"] / 2],
         "yaw": 0, "rgba": black},
        {"name": "bottle_neck", "type": "cylinder", "pos": [x, y, (b["body_h"] + b["neck_top"]) / 2],
         "size": [b["neck_r"], (b["neck_top"] - b["body_h"]) / 2], "yaw": 0, "rgba": black}]
    lay["screw_cap"] = {"name": "screw_cap", "pos": [x, y, b["cap_base"]], "radius": b["cap_r"],
                        "half_height": b["cap_h"] / 2, "rgba": [0.12, 0.12, 0.14, 1], **s}
    lay["_notes"] = list(lay.get("_notes", [])) + [
        f"TEST COPY of the {st.name} site layout ({st.layout.name}) with a bottle + screw cap (learning/bottle_twin.py)."]
    lay["bottle_twin"] = {"site": st.name, "base_layout": str(st.layout.relative_to(REPO))}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lay, indent=1))
    return path


def learned_layout(path: Path, site: str | None = None) -> Path:
    """The demo bottle from the same versioned profile used by episode_begin.

    The historical BOTTLE/REAL_BOTTLE fixtures remain for r3 regression tests.
    Dimensions are measured evidence, not a tunable safety envelope. The cap xy is the profile's unless the site
    places the bottle elsewhere (site_bottle "cap_xy_m").
    """
    from . import skillcfg
    profile = next(p for p in skillcfg.profiles() if p.get("profile") == "bottle_cap")
    obj = profile["object"]
    height = obj["cap_height_m"]
    base = obj["cap_top_z_m"] - height
    xy = site_bottle(site).get("cap_xy_m", obj["cap_xy_m"])
    b = {**REAL_BOTTLE, "xy": xy, "cap_h": height,
         "cap_r": obj["cap_diameter_m"] / 2, "cap_base": base,
         "neck_top": base - 0.0005, "body_h": base - 0.01}
    # 0.5 mm neck gap is the scene's contact clearance; 10 mm shoulder clearance
    # preserves the old model's narrow neck. Thread angle comes from real attempts.
    return make_layout(path, b, {"turns_to_free": obj["thread_free_deg"] / 360}, site)
