"""Tool parity: `python -m generaloid serve --connector elrobot` offers exactly the tools of the existing servers,
with the same descriptions and input schemas (checked over MCP stdio, as a client sees them)."""
import pytest

from .conftest import run_json

pytestmark = pytest.mark.usefixtures("needs_twin")


def test_elrobot_sim_matches_sim_arm_mcp_over_stdio():
    direct = run_json(["-m", "generaloid.tests.run_tool_specs", "sim_arm_mcp"])
    served = run_json(["-m", "generaloid.checks", "tools", "--connector", "elrobot"])
    assert sorted(direct) == served["tools"]
    assert len(direct) >= 18
    for name, spec in direct.items():
        assert served["specs"][name] == spec, name


def test_elrobot_real_matches_real_arm_mcp_import_only(tmp_path):
    """Real mode is checked by import only: neither process opens a Station connection (STATION_HOST unresolvable,
    and the generaloid guard refuses under pytest)."""
    env = {"ROBOT_REAL_LOG_DIR": str(tmp_path)}
    direct = run_json(["-m", "generaloid.tests.run_tool_specs", "real_arm_mcp"], env)
    served = run_json(["-m", "generaloid", "tools", "--connector", "elrobot", "--mode", "real", "--no-addons"], env)
    assert sorted(direct) == served["tools"]
    assert {"pick_at", "drop_at", "twist_cap", "check_cap", "stir", "wipe"} <= set(served["tools"])
