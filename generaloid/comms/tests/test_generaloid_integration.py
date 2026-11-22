"""Through the real Generaloid server (r5-generaloid's `python -m generaloid serve`), once both branches are merged:
an SO-101 twin and an ElRobot twin, each its own Generaloid MCP process with the comms add-on registered by
generaloid.core.hooks. The SO-101 asks the ElRobot to hold the bottle; the ElRobot's agent accepts and moves its
own twin's gripper with its own tool.

Skipped until generaloid.core.server exists in this checkout (it lives on branch r5-generaloid). Verified before the
merge on a scratch copy that combined both branches (docs/comms/JOURNAL.md, 08:40).
"""
import asyncio
import contextlib
import importlib.util
import os

import pytest

pytest.importorskip("mcp")
from generaloid.comms.demo import help_scenario, robot, twin_gripper_action  # noqa: E402

TOKEN = "integration-token-0123456789abcdef"


def _ready():
    try:
        has_server = importlib.util.find_spec("generaloid.core.server") is not None
    except ModuleNotFoundError:
        has_server = False
    return has_server and importlib.util.find_spec("mujoco") is not None and bool(os.environ.get("NORMA_CORE_REPO"))


@pytest.mark.skipif(not _ready(), reason="needs generaloid.core (branch r5-generaloid), mujoco and NORMA_CORE_REPO")
def test_so101_asks_elrobot_through_generaloid_servers():
    async def main():
        async with contextlib.AsyncExitStack() as st:
            el = await st.enter_async_context(robot(
                st, ["serve", "--connector", "elrobot", "--mode", "sim"], "ElRobot",
                {"GENERALOID_ROBOT_ID": "elrobot-twin"}, module="generaloid"))
            names = {t.name for t in (await el.s.list_tools()).tools}
            assert {"gripper", "comms_host", "comms_reply"} <= names
            so = await st.enter_async_context(robot(
                st, ["serve", "--connector", "so101", "--mode", "sim"], "SO-101",
                {"GENERALOID_ROBOT_ID": "so101-twin"}, module="generaloid"))
            assert "comms_join" in {t.name for t in (await so.s.list_tools()).tools}
            await help_scenario(el, so, twin_gripper_action, TOKEN)
            peers = (await so.j("comms_peers"))["peers"]
            assert peers[0]["robot_id"] == "elrobot-twin" and peers[0]["connector"] == "elrobot"
            await so.j("comms_leave")
            await el.j("comms_leave")
    asyncio.run(asyncio.wait_for(main(), 300))
