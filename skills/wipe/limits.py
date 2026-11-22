"""FIXED wipe settings: contact height and force/torque limits of the existing, hardware-validated `wipe` tool
(sim/real_arm_mcp.py, values in sim/robots/<profile>.json "wipe" + "touch"). They are read from the profile, never
adapted, never part of a skill version, and the improvement loop may not edit this file or the profile.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def profile_path(name: str | None = None) -> Path:
    choice = name or os.environ.get("ROBOT_PROFILE") or "elrobot"
    p = Path(choice)
    return p if p.exists() else ROOT / "sim" / "robots" / (choice if choice.endswith(".json") else choice + ".json")


def load(name: str | None = None) -> dict:
    """Contact/limit settings the executor must apply to every wipe stroke (passed inside each proposal)."""
    P = json.loads(profile_path(name).read_text())
    WP, TS, WS = P["wipe"], P["touch"], P["workspace"]
    return {
        "contact": "guarded_descent",                       # go down until the fingertips feel the table, back off
        "nominal_table_tcp_z_m": TS["nominal_table_tcp_z"],
        "max_below_nominal_m": TS["max_below"],
        "backoff_m": TS["backoff"],                         # light pressure: back on the surface after contact
        "hard_load_sum": WP["hard_load_sum"],               # stop when the two most loaded arm motors exceed this
        "hard_reads": WP["hard_reads"],
        "napkin_force": WP["napkin_force"],
        "start_height_m": WP["start_height"],
        "lift_after_m": WP["lift_after"],
        "min_stroke_m": WP["min_length"],
        "max_stroke_m": WP["max_length"],
        "max_passes": WP["max_strokes"],
        "max_radius_m": WS["max_radius"],
        "min_tcp_z_m": WS["min_tcp_z"],
    }
