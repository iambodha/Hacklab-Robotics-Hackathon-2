"""Record everything the robot does (motor state, cameras, commands, events) and replay it.

Live capture needs norma-core (recorder.live); reading recordings (recorder.replay) does not.
"""
from .replay import Recording, ReplaySource

__all__ = ["Recording", "ReplaySource"]
