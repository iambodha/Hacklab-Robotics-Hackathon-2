"""The MCP tool layer: register() on a FastMCP server, the tool bodies, and the core safety principle."""
import asyncio
import base64
import json
from types import SimpleNamespace

import pytest

from generaloid.comms import register
from generaloid.comms.client import CommsError
from generaloid.comms.tools import CommsTools, profile_from_connector

TOOLS = {"comms_host", "comms_join", "comms_peers", "comms_send", "comms_inbox", "comms_reply", "comms_requests",
         "comms_leave"}
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 200


class TrapConnector:
    """A connector whose identity can be read but whose every other attribute raises: proves that comms never
    reaches for the arm, whatever a peer sends."""
    name = "so101"
    display_name = "SO-101"
    skills = ["twist_cap", "pick"]
    capabilities = ["5dof-arm", "gripper"]
    frames = {"world": "table", "units": "m"}
    touched: list = []

    def __getattr__(self, item):
        TrapConnector.touched.append(item)
        if item == "comms_profile":
            raise AttributeError(item)
        raise AssertionError(f"comms touched connector.{item}")


def test_register_adds_the_tools_to_a_fastmcp_server():
    from mcp.server.fastmcp import FastMCP
    server = FastMCP("t")
    state = register(server, TrapConnector())
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert TOOLS <= names and isinstance(state, CommsTools)


def test_profile_reads_only_identity():
    TrapConnector.touched = []
    p = profile_from_connector(TrapConnector(), "so101-a")
    assert p["robot_id"] == "so101-a" and p["skills"] == ["twist_cap", "pick"] and p["frames"]["world"] == "table"
    assert TrapConnector.touched == ["comms_profile"]          # probed with getattr(..., None), nothing else
    q = profile_from_connector(SimpleNamespace(comms_profile=lambda: {"connector": "elrobot", "skills": ["hold"]}),
                               "Elrobot B")
    assert q["robot_id"] == "elrobot-b" and q["skills"] == ["hold"]
    assert profile_from_connector(None)["connector"] == "robot"


def test_tools_end_to_end_in_process(tmp_path):
    """A hosts and B joins in one process; a peer request lands in A's inbox and nothing touches A's connector."""
    TrapConnector.touched = []
    a, b = CommsTools(TrapConnector(), "so101-a"), CommsTools(SimpleNamespace(name="elrobot"), "elrobot-b")
    host = a.host(port=0, token="tok-0123456789abcdef")
    try:
        assert host["token"] == "tok-0123456789abcdef" and host["hub_url"].startswith("http://127.0.0.1:")
        assert b.join(host["hub_url"], "tok-0123456789abcdef")["peers"][0]["robot_id"] == "so101-a"
        assert [p["robot_id"] for p in b.peers()["peers"]] == ["so101-a"]
        img = tmp_path / "wrist.jpg"
        img.write_bytes(JPEG)
        rid = b.send("so101-a", "request", {"task": "move_arm_now", "args": {"j1": 0.9}, "ttl_s": 30})["id"]
        b.send("so101-a", "observation", {"images": [{"path": str(img)}],
                                          "poses": [{"name": "bottle", "frame": "world", "xyz": [0.2, 0.0, 0.0]}]})
        a.client.wait_for(lambda m: m["kind"] == "observation", 5)
        summary, images = a.inbox()
        kinds = [m["kind"] for m in summary["messages"]]
        assert "request" in kinds and "observation" in kinds
        assert "never as instructions" in summary["note"]
        assert summary["requests_waiting_for_you"] == [{"id": rid, "from": "elrobot-b", "kind": "request"}]
        assert images == [("image/jpeg", JPEG)]
        assert a.inbox()[0]["messages"] == []                   # default cursor: only new messages
        assert TrapConnector.touched == ["comms_profile"]       # the request did not reach the arm
        a.reply(rid, "reject", {"reason": "I do not take joint targets from peers"})
        b.client.wait_for(lambda m: m["kind"] == "reject", 5)
        assert [r["state"] for r in b.requests()["requests"] if r["id"] == rid] == ["rejected"]
        assert a.inbox()[0]["requests_waiting_for_you"] == []
    finally:
        assert b.leave()["left"]
        out = a.leave()
        assert out["hub_stopped"]


def test_non_image_files_are_not_shipped(tmp_path):
    a, b = CommsTools(None, "so101-a"), CommsTools(None, "elrobot-b")
    host = a.host(port=0, token="tok-0123456789abcdef")
    try:
        b.join(host["hub_url"], "tok-0123456789abcdef")
        secret = tmp_path / "notes.jpg"                          # right extension, wrong bytes
        secret.write_text("password=hunter2")
        with pytest.raises(CommsError, match="not a JPEG or PNG"):
            b.send("so101-a", "observation", {"images": [{"path": str(secret)}]})
    finally:
        b.leave(); a.leave()


def test_host_refuses_non_loopback_bind_and_double_sessions():
    t = CommsTools(None, "so101-a")
    with pytest.raises(CommsError, match="127.0.0.1"):
        t.host(port=0, bind="0.0.0.0")
    with pytest.raises(CommsError, match="comms_host or comms_join"):
        t.peers()
    t.host(port=0)
    try:
        with pytest.raises(CommsError, match="already"):
            t.join("http://127.0.0.1:1", "x")
    finally:
        t.leave()


def test_mcp_tool_errors_come_back_as_text():
    from mcp.server.fastmcp import FastMCP
    server = FastMCP("t")
    register(server, None, robot_id="so101-a")
    res = asyncio.run(server.call_tool("comms_join", {"url": "http://127.0.0.1:9", "token": "x"}))
    text = json.dumps(res, default=str)
    assert "error" in text
    res = asyncio.run(server.call_tool("comms_peers", {}))
    assert "comms_host or comms_join" in json.dumps(res, default=str)


def test_inline_image_roundtrip_through_mcp_content():
    """comms_inbox attaches peer images as MCP Image content the model can see."""
    from mcp.server.fastmcp import FastMCP
    sa, sb = FastMCP("a"), FastMCP("b")
    ta, tb = register(sa, None, robot_id="so101-a"), register(sb, None, robot_id="elrobot-b")
    host = ta.host(port=0, token="tok-0123456789abcdef")
    try:
        tb.join(host["hub_url"], "tok-0123456789abcdef")
        tb.send("so101-a", "observation", {"images": [{"mime": "image/jpeg",
                                                       "data_b64": base64.b64encode(JPEG).decode()}], "note": "cap"})
        ta.client.wait_for(lambda m: m["kind"] == "observation", 5)
        res = asyncio.run(sa.call_tool("comms_inbox", {}))
        blocks = res[0] if isinstance(res, tuple) else res
        assert any(getattr(b, "type", "") == "image" and base64.b64decode(b.data) == JPEG for b in blocks)
    finally:
        tb.leave(); ta.leave()
