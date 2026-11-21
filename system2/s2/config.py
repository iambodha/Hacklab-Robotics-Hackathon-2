"""All settings come from the environment, so the same code runs against the twin or a real arm."""
from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # system2/
REPO = HERE.parent


def _f(name, default):
    return float(os.environ.get(name, default))


@dataclass
class Config:
    arm_command: list[str] = field(default_factory=lambda: shlex.split(
        os.environ.get('S2_ARM_CMD', f'bash {REPO / "sim" / "run_server.sh"}')))
    policy: str = os.environ.get('S2_POLICY', 'geometric')          # geometric | jev | shadow
    jev_url: str = os.environ.get('S2_JEV_URL', 'http://127.0.0.1:8791')
    jev_timeout: float = _f('S2_JEV_TIMEOUT', 20)
    jev_fallback: str = os.environ.get('S2_JEV_FALLBACK', 'geometric')  # geometric | fail
    max_decision_age: float = _f('S2_MAX_DECISION_AGE', 6.0)        # s from observation to executing
    belief_ttl: float = _f('S2_BELIEF_TTL', 20.0)                   # s an unseen object's position is trusted
    stall_steps: int = int(_f('S2_STALL_STEPS', 6))
    unknown_steps: int = int(_f('S2_UNKNOWN_STEPS', 4))
    trace_dir: Path = Path(os.environ.get('S2_TRACE_DIR', HERE / 'traces'))
    # Camera hash prefixes in look() labels. Defaults: the twin's; on the real arm the elrobot-real server's own
    # ROBOT_WRIST_CAM / ROBOT_OVERVIEW_CAM are used, so both always agree.
    camera_roles: dict = field(default_factory=lambda: {
        os.environ.get('S2_WRIST_CAM', os.environ.get('ROBOT_WRIST_CAM', '322ab551')): 'wrist',
        os.environ.get('S2_OVERVIEW_CAM', os.environ.get('ROBOT_OVERVIEW_CAM', '8c697f6e')): 'overview'})

    def arm_env(self) -> dict:
        """Environment for the arm server: clean images (no grid, native size) for perception."""
        env = dict(os.environ)
        for k in ('ROBOT_SIM_GRID', 'ROBOT_GRID', 'ROBOT_DROP_LINE'):   # twin / real-arm server names
            env.setdefault(k, '0')
        for k in ('ROBOT_SIM_IMG_SCALE', 'ROBOT_IMG_SCALE'):
            env.setdefault(k, '1')
        return env

    def validate(self):
        if self.policy not in ('geometric', 'jev', 'shadow'):
            raise ValueError(f'S2_POLICY must be geometric, jev or shadow, not {self.policy!r}')
        if self.jev_fallback not in ('geometric', 'fail'):
            raise ValueError('S2_JEV_FALLBACK must be geometric or fail')
