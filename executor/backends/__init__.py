"""Backends behind one interface; pick one with make_backend(name)."""
from __future__ import annotations

from .base import Backend, FrameSample, MotionDisabled, MotorRead
from .fake import FakeBackend

__all__ = ["Backend", "FakeBackend", "FrameSample", "MotionDisabled", "MotorRead", "make_backend"]


def make_backend(name: str, kin=None, **kw) -> Backend:
    """fake | sim | real | replay:<recording>"""
    if name == "fake":
        start = kw.pop("start_norm", None)
        if start is None and kin is not None:
            start = park_norm(kin)
        return FakeBackend(start_norm=start, kin=kin, **kw)
    if name == "sim":
        from .sim import SimBackend
        return SimBackend(**kw)
    if name == "real":
        from .station import StationBackend
        return StationBackend(**kw)
    if name.startswith("replay:"):
        from .replay import ReplayBackend
        return ReplayBackend(name.removeprefix("replay:"), kin=kin, **kw)
    raise ValueError(f"unknown backend {name!r} (fake | sim | real | replay:<recording>)")


def park_norm(kin) -> dict[int, float]:
    """Motor norms of the profile's park pose (jaws open), used as the fake arm's start."""
    import numpy as np
    park_b = np.array(kin.P.poses.park, float)
    q0 = np.zeros(len(kin.ids))
    q0[:kin.na] = (kin.low[:kin.na] + kin.high[:kin.na]) / 2
    q, _ = kin.solve(kin.to_world(park_b), q0, yaw_deg=None)
    norm = kin.norm_from_q(q if q is not None else q0)
    norm[kin.grip_id] = 0.9
    return norm
