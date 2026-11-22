"""Import legacy real tools without silently reusing another robot's global profile.

The legacy servers register tools and capture their profile at import time. Run one
connector per process; switching an environment variable cannot change those tools.
"""
import importlib
import os
import sys

from .paths import ensure_sim_on_path


def real_tools(profile_name: str):
    ensure_sim_on_path()
    import robot_profile

    expected = robot_profile.load(profile_name).name
    loaded = sys.modules.get("real_arm_mcp")
    if loaded is not None and loaded.P.name != expected:
        raise RuntimeError("a different real connector is already mounted; use a fresh server process")
    os.environ["ROBOT_PROFILE"] = profile_name
    return importlib.import_module("real_arm_mcp")
