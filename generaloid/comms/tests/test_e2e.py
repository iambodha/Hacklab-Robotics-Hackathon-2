"""End to end, no hardware: two separate robot MCP server processes on localhost, each driven over stdio by a
scripted agent (this test plays both LLMs). Robot A hosts, robot B joins, B asks A for help, A accepts, does a twin
action, reports progress and the result. Plus auth failure, timeout and reconnect across processes.

Light version: `python -m generaloid.comms mcp` (comms tools only). Twin version: `python -m generaloid.comms twin`
(the MuJoCo ElRobot twin of sim/sim_arm_mcp.py with the comms tools added), where A's "twin action" is real twin
motion through the twin's own tools.
"""
import asyncio
import contextlib
import json
import os

import pytest

pytest.importorskip("mcp")
from generaloid.comms.demo import help_scenario, no_twin, robot, twin_gripper_action  # noqa: E402

TOKEN = "e2e-token-0123456789abcdef"


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 240))


def test_two_processes_help_request_auth_timeout():
    async def main():
        async with contextlib.AsyncExitStack() as st:
            async with robot(st, ["mcp", "--robot-id", "elrobot-a", "--connector", "elrobot"], "A") as a, \
                    robot(st, ["mcp", "--robot-id", "so101-b", "--connector", "so101"], "B") as b, \
                    robot(st, ["mcp", "--robot-id", "intruder-c"], "C") as c:
                url, host = await help_scenario(a, b, no_twin, TOKEN)

                # auth failure: a third process with the wrong token gets nothing, and A sees no new peer
                bad = json.loads((await c.call("comms_join", url=url, token="guessed-token"))[0])
                assert bad["ok"] is False and "401" in bad["error"]
                assert [p["robot_id"] for p in (await a.j("comms_peers"))["peers"]] == ["so101-b"]

                # timeout: B asks, A never answers; the hub expires it and tells both
                rid = (await b.j("comms_send", peer="elrobot-a", kind="request",
                                 payload={"task": "hand_over_cap", "ttl_s": 1.5}))["id"]
                exp_b = await b.wait_inbox(lambda m: m["kind"] == "expired" and m["re"] == rid)
                await a.wait_inbox(lambda m: m["kind"] == "expired" and m["re"] == rid)
                assert exp_b["from"] == "hub"
                late = json.loads((await a.call("comms_reply", request_id=rid, kind="accept"))[0])
                assert late["ok"] is False and "409" in late["error"]
                assert (await a.j("comms_leave"))["hub_stopped"]
                await b.j("comms_leave")
    run(main())


def test_reconnect_when_the_host_process_restarts():
    """A's process dies (hub gone). A new A process hosts again on the same port with the same token. B never calls
    comms_join again: its client backs off, rejoins on its own, and the conversation continues."""
    async def main():
        async with contextlib.AsyncExitStack() as outer:
            b = await outer.enter_async_context(robot(outer, ["mcp", "--robot-id", "so101-b"], "B"))
            async with contextlib.AsyncExitStack() as st1:
                a1 = await st1.enter_async_context(robot(st1, ["mcp", "--robot-id", "elrobot-a"], "A1"))
                host = await a1.j("comms_host", port=0, token=TOKEN)
                await b.j("comms_join", url=host["hub_url"], token=TOKEN)
            # A1's process has exited here (stdio closed): the hub is gone.
            port = int(host["hub_url"].rsplit(":", 1)[1])
            await asyncio.sleep(1.0)
            down = (await b.j("comms_peers"))
            assert down.get("ok") is False or down["session"]["connected"] is False
            async with contextlib.AsyncExitStack() as st2:
                a2 = await st2.enter_async_context(robot(st2, ["mcp", "--robot-id", "elrobot-a"], "A2"))
                await a2.j("comms_host", port=port, token=TOKEN)
                loop = asyncio.get_running_loop()
                end = loop.time() + 30
                while loop.time() < end:
                    if [p["robot_id"] for p in (await a2.j("comms_peers"))["peers"]] == ["so101-b"]:
                        break
                    await asyncio.sleep(0.5)
                else:
                    raise AssertionError("B did not reconnect")
                await b.j("comms_send", peer="elrobot-a", kind="message", payload={"text": "back online"})
                m = await a2.wait_inbox(lambda m: m["kind"] == "message")
                assert m["payload"]["text"] == "back online"
                assert (await b.j("comms_peers"))["session"]["reconnects"] >= 1
                await a2.j("comms_leave")
            await b.j("comms_leave")
    run(main())


def _twin_available():
    try:
        import mujoco  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("NORMA_CORE_REPO"))


@pytest.mark.skipif(not _twin_available(), reason="needs mujoco and NORMA_CORE_REPO for the twin scene")
def test_two_mujoco_twins_help_with_real_twin_motion():
    """Both robots are full twin MCP servers (sim/sim_arm_mcp.py + comms). A really moves its twin's gripper
    between its progress reports, using its own tool, after its agent accepted."""
    async def main():
        async with contextlib.AsyncExitStack() as st:
            a = await st.enter_async_context(robot(st, ["twin", "--robot-id", "elrobot-twin-a"], "A"))
            names = {t.name for t in (await a.s.list_tools()).tools}
            assert {"gripper", "twist_cap", "comms_host", "comms_inbox", "comms_reply"} <= names
            b = await st.enter_async_context(robot(st, ["twin", "--robot-id", "so101-twin-b"], "B"))
            await help_scenario(a, b, twin_gripper_action, TOKEN)
            peers = (await b.j("comms_peers"))["peers"]
            assert peers[0]["connector"] == "elrobot" and "twist_cap" in peers[0]["skills"]
            await b.j("comms_leave")
            await a.j("comms_leave")
    run(main())
