"""MCP client for the arm server (the MuJoCo twin `sim/sim_arm_mcp.py`, or a real-arm server with
the same tools). System 2 never talks to the arm directly; only this client does.

Required arm tools: look, joint_state, gripper, tcp_pose, move_tcp, camera_info.
"""
from __future__ import annotations

import base64
import io
import json
import re
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

import numpy as np
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from PIL import Image

from .geometry import Camera

REQUIRED_TOOLS = {'look', 'joint_state', 'gripper', 'tcp_pose', 'move_tcp', 'camera_info'}
_JOINT_ROW = re.compile(r'^\s*(j\d)\s+(-?[\d.]+)\s+(-?\d+)\s+(-?\d+)\s+\[\s*(-?\d+),\s*(-?\d+)\]\s+(\S+)\s+(\d)\s+(0x[0-9a-f]+)\s+(\d+)\s+(\d+)')
_TCP = re.compile(r'tcp xyz = \[(-?[\d.]+), (-?[\d.]+), (-?[\d.]+)\] m; jaws point (-?\d+) deg from straight down; '
                  r'jaw closing axis yaw = (-?\d+) deg')
_LEVEL = re.compile(r'\((\d+) deg from level\)')
_CAMERA = re.compile(r'camera ([0-9a-f]{8})')
_AGE = re.compile(r'age=([\d.]+)s')


class ArmError(RuntimeError):
    pass


@dataclass
class Joint:
    norm: float
    raw: int
    goal: int
    torque: bool
    status: int
    load: int


@dataclass
class Frame:
    """One fresh observation: images, joints, gripper pose and camera calibration, all taken together."""
    t: float
    images: dict[str, np.ndarray]           # role -> RGB uint8
    image_age: dict[str, float]
    joints: dict[str, Joint]
    tcp: np.ndarray
    tilt_deg: float
    jaw_yaw_deg: float
    cameras: dict[str, Camera]
    notes: list[str] = field(default_factory=list)
    jaw_level_deg: float | None = None      # angle of the jaw closing axis from horizontal


def parse_joints(text: str) -> dict[str, Joint]:
    out = {}
    for line in text.splitlines():
        m = _JOINT_ROW.match(line)
        if m:
            out[m[1]] = Joint(float(m[2]), int(m[3]), int(m[4]), m[8] == '1', int(m[9], 16), int(m[10]))
    return out


def parse_tcp(text: str) -> tuple[np.ndarray, float, float, float | None]:
    """-> tcp xyz, jaws-from-vertical deg, jaw axis yaw deg, jaw axis from level deg (None if not reported)."""
    m = _TCP.search(text)
    if not m:
        raise ArmError(f'cannot parse tcp_pose output: {text[:200]!r}')
    level = _LEVEL.search(text)
    return (np.array([float(m[1]), float(m[2]), float(m[3])]), float(m[4]), float(m[5]),
            float(level[1]) if level else None)


class ArmClient:
    def __init__(self, command: str, args: list[str], env: dict[str, str] | None, camera_roles: dict[str, str]):
        self.params = StdioServerParameters(command=command, args=args, env=env)
        self.camera_roles = camera_roles  # camera hash prefix -> 'wrist' | 'overview'
        self.stack = AsyncExitStack()
        self.session: ClientSession | None = None
        self.tools: set[str] = set()
        self.calls = 0

    async def __aenter__(self) -> 'ArmClient':
        read, write = await self.stack.enter_async_context(stdio_client(self.params))
        self.session = await self.stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        self.tools = {t.name for t in (await self.session.list_tools()).tools}
        missing = REQUIRED_TOOLS - self.tools
        if missing:
            await self.stack.aclose()
            raise ArmError(f'arm server lacks required tools {sorted(missing)}; has {sorted(self.tools)}')
        return self

    async def __aexit__(self, *exc):
        await self.stack.aclose()

    async def raw(self, tool: str, **args) -> list:
        """Result content in order: str for text, bytes for images."""
        if self.session is None:
            raise ArmError('arm client is not connected')
        self.calls += 1
        result = await self.session.call_tool(tool, args)
        items = [c.text if c.type == 'text' else base64.b64decode(c.data) for c in result.content
                 if c.type in ('text', 'image')]
        if result.isError:
            raise ArmError(f'{tool} failed: {" ".join(i for i in items if isinstance(i, str))[:500]}')
        return items

    async def call(self, tool: str, **args) -> tuple[list[str], list[bytes]]:
        items = await self.raw(tool, **args)
        return [i for i in items if isinstance(i, str)], [i for i in items if isinstance(i, bytes)]

    async def text(self, tool: str, **args) -> str:
        return '\n'.join((await self.call(tool, **args))[0])

    async def frame(self) -> Frame:
        """look() + tcp_pose() + camera_info(), back to back, as one observation."""
        items = await self.raw('look')
        tcp, tilt, yaw, level = parse_tcp(await self.text('tcp_pose'))
        cams = {k: Camera.from_info(v) for k, v in json.loads(await self.text('camera_info')).items()}
        t = time.monotonic()
        frames, ages, notes = {}, {}, []
        # Each camera label is followed by its image, or by nothing when the camera is stale or not live.
        for label, nxt in zip(items, items[1:] + [None]):
            m = _CAMERA.search(label) if isinstance(label, str) else None
            role = self.camera_roles.get(m[1][:8]) if m else None
            if role is None:
                continue
            if not isinstance(nxt, bytes) or 'STALE' in label:
                notes.append(f'{role} camera unavailable: {label[:120]}')
                continue
            frames[role] = np.asarray(Image.open(io.BytesIO(nxt)).convert('RGB'))
            age = _AGE.search(label)
            ages[role] = float(age[1]) if age else 0.0
        texts = [i for i in items if isinstance(i, str)]
        joints = parse_joints(texts[-1] if texts else '')
        if len(joints) < 8:
            raise ArmError('look() did not return a complete joint table')
        return Frame(t, frames, ages, joints, tcp, tilt, yaw, cams, notes, level)

    async def move_tcp(self, xyz, top_down: bool, yaw_deg: float) -> str:
        x, y, z = (round(float(v), 4) for v in xyz)
        return await self.text('move_tcp', x=x, y=y, z=z, top_down=top_down, yaw_deg=round(float(yaw_deg), 1))

    async def gripper(self, open_fraction: float) -> str:
        return await self.text('gripper', open_fraction=round(float(open_fraction), 3))

    async def reset_scene(self, randomize: bool) -> str:
        if 'operator_reset_scene' not in self.tools:
            raise ArmError('this arm server has no operator_reset_scene (real hardware?)')
        return await self.text('operator_reset_scene', randomize=randomize)
