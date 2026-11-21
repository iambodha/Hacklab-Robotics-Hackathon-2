"""Robot adapters.

StationAdapter  — the real NormaCore/ElRobot arm over the Station normfs API.
MockAdapter     — a software arm with the same joint space, for developing and
                  testing the MCP + skills without hardware (ROBOT_MCP_ENABLE_MOCK=1).
"""
from __future__ import annotations

import asyncio
import logging

from . import config
from .core import JointState, Observation, RobotAdapter

_log = logging.getLogger("robot_mcp.adapters")


class StationAdapter(RobotAdapter):
    """NormaCore/ElRobot 8-DOF arm via the Station TCP API (normfs :8888).

    Safety envelope is enforced HERE, in this process — not by the model:
      * normalized targets clamped to [0,1]
      * per-call motion clamped to MAX_STEP from the current position
      * refuses to move uncalibrated joints (range_min == range_max)
      * unspecified joints hold their position (sync-write covers all motors)
    """

    robot_id = "elrobot"
    backend = "normacore-station"

    def __init__(self):
        super().__init__()
        from . import station_lib as sl
        self._sl = sl
        self.joint_names = list(sl.JOINT_NAMES)
        self.gripper_joint = sl.GRIPPER_JOINT
        self._cli = None
        self._bus: str | None = None
        self._conn_lock = asyncio.Lock()

    async def _ensure(self):
        """Connect (and reconnect) to the station; cache client + bus serial.
        Drops a dead handle so a station restart self-heals."""
        async with self._conn_lock:
            if self._cli is not None and not getattr(self._cli, "connected", False):
                self._cli = None
            if self._cli is None:
                self._cli = await self._sl.connect()
                self._bus = await self._sl.discover_bus_serial(self._cli)
        return self._cli, self._bus

    async def _retry(self, fn):
        """Run an async station op; on a connection drop, reconnect once."""
        try:
            return await fn()
        except Exception as e:
            if "NotConnected" in type(e).__name__ or "onnect" in str(e):
                self._cli = None
                await self._ensure()
                return await fn()
            raise

    def _obs(self, joints: dict, ranges, raws, images=None, src="none") -> Observation:
        sl = self._sl
        return Observation(
            joints=[JointState(name=n, pos_norm=joints[n], raw=raws.get(n),
                               range_min=r[0], range_max=r[1],
                               role=sl.JOINT_ROLES.get(n, ""))
                    for n, r in zip(sl.JOINT_NAMES, ranges)],
            images=images or [], image_source=src)

    async def joint_state(self) -> Observation:
        async def go():
            cli, bus = await self._ensure()
            joints, ranges, raws = await self._sl.current_st3215(cli, bus)
            return self._obs(joints, ranges, raws)
        return await self._retry(go)

    async def observe(self, images: bool = True) -> Observation:
        async def go():
            cli, bus = await self._ensure()
            joints, ranges, raws = await self._sl.current_st3215(cli, bus)
            imgs, src = ([], "none")
            if images:
                imgs, src = await self._sl.camera_jpegs(cli)
            return self._obs(joints, ranges, raws, imgs, src)
        return await self._retry(go)

    async def move_norm(self, targets: dict[str, float], max_step: float) -> Observation:
        sl = self._sl
        unknown = [k for k in targets if k not in self.joint_names]
        if unknown:
            raise ValueError(f"unknown joints {unknown}; valid: {self.joint_names}")
        async def go():
            cli, bus = await self._ensure()
            joints, ranges, raws = await sl.current_st3215(cli, bus)
            goals = []
            for i, name in enumerate(sl.JOINT_NAMES):
                cur = joints[name]
                if name in targets:
                    rmin, rmax = ranges[i]
                    if rmin == rmax:
                        raise ValueError(f"joint {name} is not calibrated — calibrate first")
                    want = max(0.0, min(1.0, float(targets[name])))
                    want = max(cur - max_step, min(cur + max_step, want))
                else:
                    want = cur                                # hold
                goals.append(want)
            await sl.move_norm(cli, bus, goals, ranges)
            return cli, bus
        cli, bus = await self._retry(go)
        await asyncio.sleep(0.4)                              # let it move before reading back
        joints, ranges, raws = await sl.current_st3215(cli, bus)
        return self._obs(joints, ranges, raws)

    async def hold(self) -> None:
        async def go():
            cli, bus = await self._ensure()
            joints, ranges, _ = await self._sl.current_st3215(cli, bus)
            await self._sl.move_norm(cli, bus, [joints[n] for n in self._sl.JOINT_NAMES], ranges)
        await self._retry(go)

    def capabilities(self) -> dict:
        caps = super().capabilities()
        caps["bus_serial"] = self._bus or config.BUS_SERIAL
        caps["joint_roles"] = dict(self._sl.JOINT_ROLES)
        caps["safety"] = {"max_step_per_call": config.MAX_STEP,
                          "uncalibrated_joints_refused": True,
                          "units_clamped_to": [0, 1]}
        return caps


class MockAdapter(RobotAdapter):
    """Software stand-in for the ElRobot: same joint space and calibration
    ranges as this year's follower, instant motion, no cameras. Lets the whole
    MCP + skill stack run without hardware."""

    robot_id = "mock"
    backend = "mock"

    # This year's follower calibration (calibration/calibration-2026-11-21.txt).
    _RANGES = [(1066, 3030), (982, 3114), (885, 3211), (882, 3214),
               (128, 3968), (1006, 3090), (109, 3987), (1249, 2847)]

    def __init__(self):
        super().__init__()
        from . import station_lib as sl
        self.joint_names = list(sl.JOINT_NAMES)
        self.gripper_joint = sl.GRIPPER_JOINT
        self._sl = sl
        self._pos = {n: 0.5 for n in self.joint_names}
        # Simulates an object sitting between the jaws: they cannot close past
        # this point while it is present. Set False to exercise the miss path.
        self.object_present = True
        self._object_floor = 0.45

    def has_cameras(self) -> bool:
        return False

    def _obs(self) -> Observation:
        return Observation(
            joints=[JointState(name=n, pos_norm=self._pos[n],
                               raw=int(r[0] + self._pos[n] * (r[1] - r[0])),
                               range_min=r[0], range_max=r[1],
                               role=self._sl.JOINT_ROLES.get(n, ""))
                    for n, r in zip(self.joint_names, self._RANGES)],
            images=[], image_source="none")

    async def joint_state(self) -> Observation:
        return self._obs()

    async def observe(self, images: bool = True) -> Observation:
        return self._obs()

    async def move_norm(self, targets: dict[str, float], max_step: float) -> Observation:
        for k in targets:
            if k not in self._pos:
                raise ValueError(f"unknown joint {k!r}")
        for k, v in targets.items():
            cur = self._pos[k]
            want = max(0.0, min(1.0, float(v)))
            if k == self.gripper_joint and self.object_present:
                want = max(want, self._object_floor)     # jaws stall on the object
            self._pos[k] = max(cur - max_step, min(cur + max_step, want))
        return self._obs()

    async def hold(self) -> None:
        pass
