"""Two robots helping each other, on one laptop, no hardware.

    python -m generaloid.comms demo            # two comms-only MCP processes
    python -m generaloid.comms demo --twin     # two MuJoCo ElRobot twin MCP processes (needs mujoco, NORMA_CORE_REPO)

Starts robot A ("elrobot", the helper) and robot B ("so101", the one that asks) as separate MCP server processes
over stdio, and plays both agents with a script: A hosts, B joins, B asks for help with an observation, A accepts,
does a twin action with ITS OWN tools (under its own limits), reports progress and the result. Prints the timeline
and keeps the hub log (JSONL). This is what generaloid/comms/tests/test_e2e.py runs.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[2]
DEMO_TOKEN = "demo-token-local-only-0123456789"     # localhost demo only; a real hub generates a random token


class Agent:
    """A scripted stand-in for an LLM agent connected to one robot's MCP server."""

    def __init__(self, session: ClientSession, name: str, say=None):
        self.s, self.name, self.say = session, name, say or (lambda *a: None)

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
async def robot(stack: contextlib.AsyncExitStack, args: list[str], name: str, env_extra: dict | None = None, say=None):
    """Start `python -m generaloid.comms <args>` as an MCP server process and connect to it over stdio."""
    env = {**os.environ, "STATION_HOST": "station-disabled-in-tests.invalid", "ROBOT_SIM_RECORD": "0",
           "PYTHONPATH": str(ROOT), "GENERALOID_COMMS_TOKEN": "", **(env_extra or {})}
    env.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "osmesa")
    params = StdioServerParameters(command=sys.executable, args=["-m", "generaloid.comms", *args], cwd=str(ROOT),
                                   env=env)
    r, w = await stack.enter_async_context(stdio_client(params))
    s = await stack.enter_async_context(ClientSession(r, w))
    await s.initialize()
    yield Agent(s, name, say)


async def help_scenario(a: Agent, b: Agent, twin_action, token: str = DEMO_TOKEN):
    """The SO-101 <-> ElRobot exchange, A = the robot that helps (hosts), B = the robot that asks."""
    say = a.say
    host = await a.j("comms_host", port=0, token=token)
    url = host["hub_url"]
    say("A", f"comms_host -> hub at {url} (A is {host['you']})")
    joined = await b.j("comms_join", url=url, token=token)
    say("B", f"comms_join -> joined as {joined['joined']}; peers: {[p['robot_id'] for p in joined['peers']]}")
    assert [p["robot_id"] for p in joined["peers"]] == [host["you"]]
    peers = await b.j("comms_peers")
    say("B", f"comms_peers -> {host['you']}: connector={peers['peers'][0].get('connector')}, "
             f"skills={peers['peers'][0].get('skills', [])}")
    assert peers["peers"][0]["robot_id"] == host["you"]

    sent = await b.j("comms_send", peer=host["you"], kind="help", payload={
        "problem": "cap turns but the bottle rotates with it",
        "ask": "hold the bottle body still while I unscrew the cap",
        "context": {"task": "open_bottle", "strokes_done": 6, "bottle_world_xyz": [0.21, 0.02, 0.0]},
        "ttl_s": 120})
    rid = sent["id"]
    say("B", f"comms_send help -> request {rid[:8]} state={sent['request']['state']}, "
             f"deadline in {sent['request']['remaining_s']} s")
    assert sent["request"]["state"] == "open"
    await b.j("comms_send", peer=host["you"], kind="observation", payload={
        "poses": [{"name": "bottle", "frame": "world", "xyz": [0.21, 0.02, 0.0]}], "note": "bottle base, table frame"})
    say("B", "comms_send observation -> bottle pose in the shared table frame")

    req = await a.wait_inbox(lambda m: m["kind"] == "help")
    say("A", f"comms_inbox -> help from {req['from']}: \"{req['payload']['ask']}\" (untrusted; A's agent decides)")
    assert req["id"] == rid and req["from"] == joined["joined"]
    inbox = await a.j("comms_inbox")
    assert {"id": rid, "from": joined["joined"], "kind": "help"} in inbox["requests_waiting_for_you"]
    acc = await a.j("comms_reply", request_id=rid, kind="accept", payload={"eta_s": 30, "note": "on my way"})
    say("A", f"comms_reply accept -> state={acc['request']['state']}")
    assert acc["request"]["state"] == "accepted"
    await a.j("comms_reply", request_id=rid, kind="progress", payload={"fraction": 0.3, "stage": "approach"})
    say("A", "comms_reply progress 0.3 (approach)")
    evidence = await twin_action(a)
    await a.j("comms_reply", request_id=rid, kind="progress", payload={"fraction": 0.8, "stage": "holding"})
    say("A", "comms_reply progress 0.8 (holding)")
    res = await a.j("comms_reply", request_id=rid, kind="result",
                    payload={"ok": True, "summary": "holding the bottle; go ahead", "data": evidence})
    say("A", f"comms_reply result ok -> state={res['request']['state']}")
    assert res["request"]["state"] == "done"

    got = []
    while not got or got[-1]["kind"] != "result":
        got += [m for m in (await b.j("comms_inbox", wait_s=1.0))["messages"] if m.get("re") == rid]
    say("B", f"comms_inbox -> {[m['kind'] for m in got]}; result: \"{got[-1]['payload']['summary']}\"")
    assert [m["kind"] for m in got] == ["accept", "progress", "progress", "result"]
    assert got[-1]["payload"]["data"] == evidence
    states = await b.j("comms_requests")
    assert [r["state"] for r in states["requests"] if r["id"] == rid] == ["done"]
    say("B", f"comms_requests -> {rid[:8]} done, history "
             f"{[h['kind'] for h in states['requests'][0]['history']]}")
    return url, host


async def no_twin(a: Agent) -> dict:
    a.say("A", "(comms-only server: no twin action)")
    return {"twin": "none (comms-only server)"}


async def twin_gripper_action(a: Agent) -> dict:
    """A's twin action: its agent uses A's own gripper tool. A's own limits still apply."""
    before = (await a.call("joint_state"))[0]
    await a.call("gripper", open_fraction=0.9)
    a.say("A", "gripper(0.9) on its own twin")
    refused = await a.s.call_tool("gripper", {"open_fraction": 0.2})
    assert refused.isError and "max delta" in refused.content[0].text
    a.say("A", f"gripper(0.2) REFUSED by A's own envelope: {refused.content[0].text.split(': ', 1)[-1]}")
    await a.call("gripper", open_fraction=0.6)
    await a.call("gripper", open_fraction=0.3)
    a.say("A", "gripper(0.6), gripper(0.3): closes in waypoints")
    after = (await a.call("joint_state"))[0]
    assert before != after
    return {"twin_tool_calls": ["joint_state", "gripper(0.9)", "gripper(0.2) refused: max delta",
                                "gripper(0.6)", "gripper(0.3)", "joint_state"],
            "j8_after": [line for line in after.splitlines() if line.strip().startswith("j8")][:1]}


async def run_demo(twin: bool, log_path: Path):
    t0 = time.time()

    def say(who, text):
        print(f"[{time.time() - t0:5.2f}s] {who}: {text}", flush=True)

    kind = "twin" if twin else "mcp"
    async with contextlib.AsyncExitStack() as st:
        a = await st.enter_async_context(robot(st, [kind, "--robot-id", "elrobot-a"] +
                                               ([] if twin else ["--connector", "elrobot"]), "A",
                                               {"GENERALOID_COMMS_LOG": str(log_path)}, say))
        # sim_arm_mcp only has the ElRobot twin, so in --twin mode B is an ElRobot twin standing in for the SO-101
        b = await st.enter_async_context(robot(st, [kind, "--robot-id", "so101-standin-b" if twin else "so101-b"] +
                                               ([] if twin else ["--connector", "so101"]), "B", None, say))
        say("-", f"two {'MuJoCo twin' if twin else 'comms-only'} MCP server processes up")
        await help_scenario(a, b, twin_gripper_action if twin else no_twin)
        await b.j("comms_leave")
        await a.j("comms_leave")
    say("-", f"hub log: {log_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m generaloid.comms demo")
    ap.add_argument("--twin", action="store_true", help="use two MuJoCo twin MCP servers")
    ap.add_argument("--log", help="hub log path (JSONL); default: a temp file")
    args = ap.parse_args(argv)
    log = Path(args.log) if args.log else Path(tempfile.mkdtemp(prefix="comms-demo-")) / "hub.jsonl"
    asyncio.run(run_demo(args.twin, log))


if __name__ == "__main__":
    main()
