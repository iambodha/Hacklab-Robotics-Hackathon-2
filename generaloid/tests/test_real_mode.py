"""Real mode against a fake NormaCore Station (generaloid.testing.FakeStation): the connector mounts the real-arm
server, every bus read and write goes through the Station layer, and nothing reaches a real Station."""
import pytest

from generaloid import connectors

from .conftest import run_json

pytestmark = pytest.mark.usefixtures("needs_twin")


@pytest.mark.parametrize("name", connectors.available())
def test_real_mode_on_a_fake_station(name, tmp_path):
    c = connectors.get(name)
    out = run_json(["-m", "generaloid.tests.run_fake_real", name], {"ROBOT_REAL_LOG_DIR": str(tmp_path)})
    for j in c.joints:                                    # the joint table is the connector's joints, from the fake bus
        assert any(line.split()[0] == j.name for line in out["joint_state"].splitlines()[1:]), j.name
    assert out["torque_writes"] == [0x2A, 0x28]           # goal = present first, then torque on (nothing jumps)
    assert out["moved"] == {str(c.arm_joints[0].motor_id): pytest.approx(52, abs=3)}   # 0.524 -> 0.55 of 2000 ticks
    assert out["other_writes"] == []
    assert out["fake_connects"] == 1
    assert out["real_connect"].startswith("refused")
