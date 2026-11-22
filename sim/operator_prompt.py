"""Render the operator's instructions for the selected robot profile.

  python3 sim/operator_prompt.py            # print the instructions (used by the run scripts and the operator page)
  python3 sim/operator_prompt.py --write    # also write sim/OPERATOR_REAL.md (the rendered copy kept in git)

OPERATOR_TEMPLATE.md holds the text with <<NAME>> placeholders; the values come from the robot profile
(sim/robots/<name>.json, chosen with ROBOT_PROFILE). Stdlib only, so the shell scripts can call it with plain python3.
"""
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def profile() -> dict:
    choice = os.environ.get("ROBOT_PROFILE") or "elrobot"
    p = Path(choice)
    if not p.exists():
        p = HERE / "robots" / (choice if choice.endswith(".json") else choice + ".json")
    return json.loads(p.read_text())


def _n(v) -> str:
    """Numbers the way a person writes them: 0.2 -> 0.20 for metres, whole numbers without decimals."""
    if isinstance(v, int):
        return str(v)
    s = f"{v:.3f}".rstrip("0")
    return s + "0" * (2 - len(s.split(".")[1])) if len(s.split(".")[1]) < 2 else s


def values(p: dict | None = None) -> dict:
    p = p or profile()
    cam, ws, po, gr, pk, dr, st = p["cameras"], p["workspace"], p["poses"], p["gripper"], p["pick"], p["drop"], p["stir"]
    park, hand = po["park"], po["handover"]
    names = p["motors"]["names"]
    return {
        "ROBOT_NAME": p["name"].replace("_", " "), "SERVER": p["mcp_server_name"],
        "DOF": len(p["motors"]["arm"]) + 1, "GRIPPER_JOINT": names[str(p["motors"]["gripper"])], "FIRST_JOINT": names[str(p["motors"]["arm"][0])],
        "FRAME": p["frame_text"].split("Base frame, metres: ", 1)[-1],
        "IMG": f"{cam['width'] * cam['scale']}x{cam['height'] * cam['scale']}", "GRID_STEP": cam["grid_step"],
        "SURVEY_NAMES": ", ".join(f'"{s[0]}"' for s in po["survey"]), "SURVEY_VIEWS": " / ".join(f'"{s[0]}"' for s in po["survey"]),
        "SURVEY_COUNT": {2: "two", 3: "three", 4: "four"}.get(len(po["survey"]), str(len(po["survey"]))),
        "MIN_Z": _n(ws["min_tcp_z"]), "MAX_RADIUS": _n(ws["max_radius"]), "COMFORT_RADIUS": _n(ws["comfortable_radius"]),
        "TRAVEL_H": _n(ws["travel_height"]), "DIRECT_CM": round(ws["direct_move_within"] * 100),
        "CARRY_Z": _n(po["carry_z"]), "BOX_Z": _n(dr["box_z"]), "BOX_Z_LOW": _n(dr["box_z"] - 0.02), "MIN_Z_OVER_BOX": _n(dr["min_z_over_box"]),
        "DROP_Z": _n(dr["default_z"]), "BOX_H": _n(dr["default_box_height"]),
        "PARK": f"{_n(park[0])[:-1] if _n(park[0]) == '0.00' else _n(park[0])}, {_n(park[1])}, {_n(park[2])}",
        "PARK_XY": f"{_n(park[0])[:-1] if _n(park[0]) == '0.00' else _n(park[0])}, {_n(park[1])}",
        "HANDOVER": f"{_n(hand[0])[:-1] if _n(hand[0]) == '0.00' else _n(hand[0])}, {_n(hand[1])}, {_n(hand[2])}",
        "CLOSE_Z": _n(pk["close_view_z"]), "FLAT_BELOW": _n(pk["flat_below"]), "PICK_H": _n(pk["default_height"]),
        "MAX_GRIP_CM": round(gr["max_grip_width"] * 100), "JAW_CM": round(gr["jaw_opening"] * 100 + 0.01), "CLOSED": _n(round(gr["closed_norm"] - 0.015, 2)),
        "WIPE_MAX_CM": round(p.get("wipe", {}).get("max_length", 0.25) * 100), "NAPKIN_FORCE": _n(p.get("wipe", {}).get("napkin_force", 0.5)),
        "GRIP_FORCE": _n(gr.get("force_default", 0.35)),
        "PLACE_Z_FROM": _n(p.get("place", {}).get("z_from", 0.09)),
        "STIR_MAX_R": _n(st["max_radius"]), "STIR_CLEAR": _n(st["default_clearance"] - 0.01),
    }


def fill(text: str, vals: dict | None = None) -> str:
    vals = vals or values()
    def one(m):
        if m.group(1) not in vals:
            raise KeyError(f"unknown placeholder <<{m.group(1)}>>")
        return str(vals[m.group(1)])
    return re.sub(r"<<([A-Z_]+)>>", one, text)


def render() -> str:
    return fill((HERE / "OPERATOR_TEMPLATE.md").read_text())


if __name__ == "__main__":
    text = render()
    if "--write" in sys.argv:
        (HERE / "OPERATOR_REAL.md").write_text(text)
    sys.stdout.write(text)
