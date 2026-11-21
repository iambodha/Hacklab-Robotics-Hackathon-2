"""Replay mode: camera frames (and the starting joints) come from a recording via recorder.ReplaySource, motion
goes to a FakeBackend. The replay clock is the fake's virtual clock, so the recording advances exactly as the
executor sleeps. This runs the whole stack (observe -> proposal -> validation -> skill -> markers) with no arm.
"""
from __future__ import annotations

from .base import FrameSample, MotorRead
from .fake import FakeBackend


class ReplayBackend(FakeBackend):
    name = "replay"

    def __init__(self, recording, start: float = 0.0, speed: float = 1.0, **fake_kw):
        from recorder.replay import Recording, ReplaySource
        rec = recording if isinstance(recording, Recording) else Recording(recording)
        super().__init__(**fake_kw)
        self.src = ReplaySource(rec, speed=speed, start=start, clock=self.now)
        try:
            st = self.src.read_state_sync()
            ranges = {m: (s["lo"], s["hi"]) for m, s in st.items() if s["hi"] > s["lo"]}
            self.ranges.update(ranges)
            self.pos.update({m: float(s["norm"]) for m, s in st.items() if s["norm"] is not None})
            self.goal = dict(self.pos)
        except RuntimeError:
            pass   # recording without full motor dumps: keep the fake's start pose
        self.recording_id = rec.id

    def frames(self) -> dict[str, FrameSample]:
        out, src, rec = {}, self.src, self.src.rec
        t = src.now()
        for h8, c in rec.cams.items():
            row = rec.frame_index(h8, t, causal=True)
            if row is None or row["t"] > t:
                continue
            role = c["info"].get("role") or h8
            data = rec.frame_bytes(h8, row["n"])
            t_cap = src._c0 + (row["t"] - src._base) / max(src.speed, 1e-9)
            seq = row.get("index") if row.get("index") is not None else row["n"]
            st = rec.state_at(row["t"], causal=True) or {}
            out[role] = FrameSample(role, f"recording:{rec.id}/{h8}", int(seq), t_cap, data, st.get("norm"))
        return out

    def read_state(self) -> MotorRead:
        return super().read_state()
