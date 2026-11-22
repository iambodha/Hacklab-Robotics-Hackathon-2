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
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")
from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
TOKEN = "e2e-token-0123456789abcdef"


class Agent:
    """A scripted stand-in for an LLM agent connected to one robot's MCP server."""

    def __init__(self, session: ClientSession, name: str):
        self.s, self.name = session, name

    async def call(self, tool: str, **args):
        res = await self.s.call_tool(tool, args)
        texts = [c.text for c in res.content if getattr(c, "type", "") == "text"]
        if res.isError:
            raise AssertionError(f"{self.name}.{tool} failed: {texts}")
        return texts

    async def j(self, tool: str, **args) -> dict:
        return json.loads((await self.call(tool, **args))[0])

    async def wait_inbox(self, pred, timeout=15.0) -> dict:
        """Poll comms_inbox (as an agent would) until a message matching pred shows up."""
        loop = asyncio.get_running_loop()
        end = loop.time() + timeout
        while loop.time() < end:
            out = await self.j("comms_inbox", wait_s=1.0)
            for m in out["messages"]:
                if pred(m):
                    return m
        raise AssertionError(f"{self.name}: no matching message within {timeout} s")


@contextlib.asynccontextmanager
async def robot(stack: contextlib.AsyncExitStack, args: list[str], name: str):
    env = {**os.environ, "STATION_HOST": "station-disabled-in-tests.invalid", "ROBOT_SIM_RECORD": "0",
           "PYTHONPATH": str(ROOT), "GENERALOID_COMMS_TOKEN": ""}
    env.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "osmesa")
    params = StdioServerParameters(command=sys.executable, args=["-m", "generaloid.comms", *args], cwd=str(ROOT), env=env)
    r, w = await stack.enter_async_context(stdio_client(params))
    s = await stack.enter_async_context(ClientSession(r, w))
    await s.initialize()
    yield Agent(s, name)


async def help_scenario(a: Agent, b: Agent, twin_action):
    """The SO-101 <-> ElRobot exchange, with A = the robot that helps and B = the robot that asks."""
    host = await a.j("comms_host", port=0, token=TOKEN)
    url = host["hub_url"]
    joined = await b.j("comms_join", url=url, token=TOKEN)
    assert [p["robot_id"] for p in joined["peers"]] == [host["you"]]
    peers = await b.j("comms_peers")
    assert peers["peers"][0]["robot_id"] == host["you"]

    # B is stuck and asks A for help, with an observation of where the bottle is.
    sent = await b.j("comms_send", peer=host["you"], kind="help", payload={
        "problem": "cap turns but the bottle rotates with it",
        "ask": "hold the bottle body still while I unscrew the cap",
        "context": {"task": "open_bottle", "strokes_done": 6, "bottle_world_xyz": [0.21, 0.02, 0.0]},
        "ttl_s": 120})
    rid = sent["id"]
    assert sent["request"]["state"] == "open"
    await b.j("comms_send", peer=host["you"], kind="observation", payload={
        "poses": [{"name": "bottle", "frame": "world", "xyz": [0.21, 0.02, 0.0]}], "note": "bottle base, table frame"})

    # A's agent reads its inbox, decides, accepts, acts with its own tools, reports.
    req = await a.wait_inbox(lambda m: m["kind"] == "help")
    assert req["id"] == rid and req["from"] == joined["joined"]
    inbox = await a.j("comms_inbox")
    assert {"id": rid, "from": joined["joined"], "kind": "help"} in inbox["requests_waiting_for_you"]
    acc = await a.j("comms_reply", request_id=rid, kind="accept", payload={"eta_s": 30, "note": "on my way"})
    assert acc["request"]["state"] == "accepted"
    await a.j("comms_reply", request_id=rid, kind="progress", payload={"fraction": 0.3, "stage": "approach"})
    evidence = await twin_action(a)
    await a.j("comms_reply", request_id=rid, kind="progress", payload={"fraction": 0.8, "stage": "holding"})
    res = await a.j("comms_reply", request_id=rid, kind="result",
                    payload={"ok": True, "summary": "holding the bottle; go ahead", "data": evidence})
    assert res["request"]["state"] == "done"

    # B sees accept, progress x2 and the result, in order.
    got = []
    while not got or got[-1]["kind"] != "result":
        got += [m for m in (await b.j("comms_inbox", wait_s=1.0))["messages"] if m.get("re") == rid]
    assert [m["kind"] for m in got] == ["accept", "progress", "progress", "result"]
    assert got[-1]["payload"]["data"] == evidence
    states = await b.j("comms_requests")
    assert [r["state"] for r in states["requests"] if r["id"] == rid] == ["done"]
    return url, host


async def no_twin(a: Agent) -> dict:
    return {"twin": "none (comms-only server)"}


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 240))


def test_two_processes_help_request_auth_timeout():
    async def main():
        async with contextlib.AsyncExitStack() as st:
            async with robot(st, ["mcp", "--robot-id", "elrobot-a", "--connector", "elrobot"], "A") as a, \
                    robot(st, ["mcp", "--robot-id", "so101-b", "--connector", "so101"], "B") as b, \
                    robot(st, ["mcp", "--robot-id", "intruder-c"], "C") as c:
                url, host = await help_scenario(a, b, no_twin)

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
    async def twin_action(a: Agent) -> dict:
        before = (await a.call("joint_state"))[0]
        await a.call("gripper", open_fraction=0.9)
        # A's own safety envelope still applies while it helps: the twin refuses a jump > ROBOT_MAX_DELTA (0.35)...
        refused = await a.s.call_tool("gripper", {"open_fraction": 0.2})
        assert refused.isError and "max delta" in refused.content[0].text
        await a.call("gripper", open_fraction=0.6)                # ...so the agent closes in waypoints
        await a.call("gripper", open_fraction=0.3)
        after = (await a.call("joint_state"))[0]
        assert before != after
        return {"twin_tool_calls": ["joint_state", "gripper(0.9)", "gripper(0.2) refused: max delta",
                                    "gripper(0.6)", "gripper(0.3)", "joint_state"],
                "j8_after": [l for l in after.splitlines() if l.strip().startswith("j8")][:1]}

    async def main():
        async with contextlib.AsyncExitStack() as st:
            a = await st.enter_async_context(robot(st, ["twin", "--robot-id", "elrobot-twin-a"], "A"))
            names = {t.name for t in (await a.s.list_tools()).tools}
            assert {"gripper", "twist_cap", "comms_host", "comms_inbox", "comms_reply"} <= names
            b = await st.enter_async_context(robot(st, ["twin", "--robot-id", "so101-twin-b"], "B"))
            await help_scenario(a, b, twin_action)
            peers = (await b.j("comms_peers"))["peers"]
            assert peers[0]["connector"] == "elrobot" and "twist_cap" in peers[0]["skills"]
            await b.j("comms_leave")
            await a.j("comms_leave")
    run(main())
