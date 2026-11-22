"""The comms add-on hook: generaloid.comms.register(server, connector) if the module imports, a quiet skip if it is
absent, and a robot server that still starts when the add-on is broken. Absence is simulated with
sys.modules[name] = None, so these tests hold whether or not r5-comms' package is installed."""
import sys
import types

import pytest

pytest.importorskip("mcp")
from mcp.server.fastmcp import FastMCP  # noqa: E402

from generaloid.connectors.base import (Camera, Connector, Gripper, Joint, MOTION_SKILLS, Mount,  # noqa: E402
                                        SafetyEnvelope, TwinSpec)
from generaloid.core import hooks  # noqa: E402
from generaloid.core.server import build_server, tool_names  # noqa: E402


class TinyConnector(Connector):
    """A connector with one tool and no robot behind it: the server assembly is all that is under test."""
    name, title, modes = "tiny", "Tiny test arm", ("sim",)
    joints = [Joint("a1", 1, "arm"), Joint("jaws", 2, "gripper")]
    gripper = Gripper("jaws", True, 0.03, 0.3)
    cameras = [Camera("wrist", "wrist_cam", None, 299, 224)]
    safety = SafetyEnvelope(0.01, 0.3, 0.3, 0.35, 250, 550, 0.08)
    twin = TwinSpec("generic", hooks.__file__)
    skills = MOTION_SKILLS
    frame_text = "test frame"

    def kinematics(self):
        raise NotImplementedError

    def mount(self, mode, station):
        tools = FastMCP("tiny")

        @tools.tool()
        async def ping() -> str:
            return "pong"
        return Mount(self.name, mode, tools)


def test_absent_module_is_skipped_quietly(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, hooks.COMMS_MODULE, None)
    server, mount = build_server(TinyConnector(), "sim")
    assert mount.addons == {"comms": "absent"}
    assert tool_names(server) == ["ping"]
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""


def test_present_module_is_called_with_server_and_connector(monkeypatch):
    seen = []

    def register(server, connector):
        seen.append((server, connector.name))

        @server.tool()
        async def send_to_robot(robot: str, message: str) -> str:
            return f"sent to {robot}"
    monkeypatch.setitem(sys.modules, hooks.COMMS_MODULE, types.SimpleNamespace(register=register))
    conn = TinyConnector()
    server, mount = build_server(conn, "sim")
    assert mount.addons["comms"] == "registered"
    assert seen == [(server, "tiny")]
    assert sorted(tool_names(server)) == ["ping", "send_to_robot"]


def test_broken_addon_never_stops_the_robot(monkeypatch, capsys):
    def register(server, connector):
        raise RuntimeError("comms bus down")
    monkeypatch.setitem(sys.modules, hooks.COMMS_MODULE, types.SimpleNamespace(register=register))
    server, mount = build_server(TinyConnector(), "sim")
    assert mount.addons["comms"].startswith("register failed: RuntimeError")
    assert tool_names(server) == ["ping"]
    assert capsys.readouterr().out == ""                  # stdout is the MCP channel: nothing there


def test_addon_without_register_and_import_errors(monkeypatch):
    monkeypatch.setitem(sys.modules, hooks.COMMS_MODULE, types.SimpleNamespace())
    assert hooks.register_comms(FastMCP("x"), TinyConnector()) == "no register()"
    # a missing dependency *inside* the add-on is reported, not mistaken for "absent"
    monkeypatch.setattr(hooks, "_load", lambda m: (None, "import failed: ModuleNotFoundError: No module named 'zmq'"))
    assert hooks.register_comms(FastMCP("x"), TinyConnector()).startswith("import failed")


def test_addons_can_be_disabled(monkeypatch):
    monkeypatch.setitem(sys.modules, hooks.COMMS_MODULE, types.SimpleNamespace(register=lambda s, c: 1 / 0))
    _, mount = build_server(TinyConnector(), "sim", addons=False)
    assert mount.addons == {"comms": "disabled"}


def test_load_tells_absent_from_broken(tmp_path, monkeypatch):
    pkg = tmp_path / "gen_hook_probe"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("import gen_hook_missing_dependency\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert hooks._load("gen_hook_nowhere")[1] == "absent"
    mod, status = hooks._load("gen_hook_probe")
    assert mod is None and status.startswith("import failed") and "gen_hook_missing_dependency" in status


def test_build_server_rejects_unknown_modes():
    with pytest.raises(ValueError):
        build_server(TinyConnector(), "real")
    with pytest.raises(ValueError):
        build_server(TinyConnector(), "hardware")
