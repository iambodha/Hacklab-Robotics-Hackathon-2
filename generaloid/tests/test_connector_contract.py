"""The connector contract: every connector (generaloid/connectors/*) passes the same suite. A new robot's connector
is done when this file passes for it unchanged."""
import json

import numpy as np
import pytest

from generaloid import connectors
from generaloid.connectors.base import KNOWN_SKILLS, MOTION_SKILLS, Connector

from .conftest import run_json

NAMES = connectors.available()
IK_POS_TOL = 0.003          # m: fk(ik(p)) must land this close (twin IK converges to 2 mm)
IK_MAX_LEAN = 15.0          # deg: the probe point is chosen so the jaws reach close to straight down

# Skills whose tool names are the same on every connector (connector-specific skills map to their own tools).
SKILL_TOOLS = {s: [s] for s in MOTION_SKILLS} | {
    "twist_cap": ["twist_cap", "check_cap"], "episodes": ["episode_begin", "episode_history", "episode_end"],
    "skill_config": ["skill_config"], "operator_reset_scene": ["operator_reset_scene"]}


@pytest.fixture(params=NAMES)
def conn(request) -> Connector:
    return connectors.get(request.param)


def test_registry():
    assert {"elrobot", "so101"} <= set(NAMES)
    for n in NAMES:
        c = connectors.get(n)
        assert c.name == n and connectors.get(f"generaloid.connectors.{n}:CONNECTOR") is c
    with pytest.raises(KeyError):
        connectors.get("no_such_robot")


def test_so101_mode_frames_are_explicit():
    conn = connectors.get("so101")
    assert "+y = straight ahead" in conn.frame_for("sim")
    assert "+x = straight ahead" in conn.frame_for("real")


def test_joints_and_gripper(conn):
    js = conn.joints
    assert len({j.name for j in js}) == len(js) and len({j.motor_id for j in js}) == len(js)
    assert [j.role for j in js] == ["arm"] * (len(js) - 1) + ["gripper"]       # arm joints first, gripper last
    assert len(conn.arm_joints) >= 5
    g = conn.gripper
    assert g.joint == js[-1].name and 0 < g.max_width_m < 0.2 and 0 < g.torque_nm < 3


def test_cameras_and_safety(conn):
    roles = {c.role for c in conn.cameras}
    assert {"wrist", "overview"} <= roles
    assert all(c.width > 0 and c.height > 0 for c in conn.cameras)
    s = conn.safety
    assert 0 <= s.min_tcp_z < s.travel_height < s.max_tcp_z <= 1.0 and 0 < s.max_radius <= 1.0
    assert 0 < s.max_joint_delta <= 0.5 and s.track_tol_ticks > 0 and 0 < s.load_abort <= 1000
    assert conn.frame_text.startswith("Base frame")


def test_skills_and_twin(conn):
    assert conn.skills <= KNOWN_SKILLS
    for mode in conn.modes:
        assert MOTION_SKILLS <= conn.skills_for(mode) <= conn.skills, mode
    t = conn.twin
    assert t.kind in ("elrobot-twin", "generic") and t.layout.exists()
    assert t.kind != "generic" or t.build_scene is not None
    json.dumps(conn.describe())


def test_kinematics_round_trip(conn):
    pytest.importorskip("mujoco")
    k = conn.kinematics()
    p = np.array(conn.reach_probe())
    q0 = np.zeros(len(conn.joints))
    q, lean = k.solve_least_lean(p, q0, 0.0)
    assert q is not None, f"{conn.name}: probe {p} unreachable jaws-down"
    assert np.all(q >= k.low - 1e-6) and np.all(q <= k.high + 1e-6)
    xyz, R = k.fk(q)
    assert np.linalg.norm(xyz - p) < IK_POS_TOL
    tilt = np.degrees(np.arccos(np.clip(-R[2, 1], -1, 1)))              # R[:,1] = approach axis, down = -z
    assert abs(lean) <= IK_MAX_LEAN and tilt <= IK_MAX_LEAN + 2


@pytest.mark.parametrize("name", NAMES)
def test_sim_mount_exposes_declared_skills(name, needs_twin):
    """Mounted in the twin through the real CLI, the tool surface covers every skill the connector declares."""
    c = connectors.get(name)
    out = run_json(["-m", "generaloid", "tools", "--connector", name, "--mode", "sim", "--no-addons"])
    want = {t for s in c.skills_for("sim") for t in SKILL_TOOLS.get(s, [])}
    assert want <= set(out["tools"]), want - set(out["tools"])
    assert out["server"] == f"generaloid-{name}-sim" and out["addons"] == {"comms": "disabled"}
