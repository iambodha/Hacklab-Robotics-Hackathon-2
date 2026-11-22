"""The one shared Station connect helper refuses to connect under pytest unless ROBOT_LIVE_TESTS=1, and reads
STATION_HOST at call time."""
import asyncio
import sys
import types

import pytest

from generaloid.core import station as S


def test_refuses_under_pytest(monkeypatch):
    monkeypatch.delenv("ROBOT_LIVE_TESTS", raising=False)
    assert S.under_pytest() and not S.live_allowed()
    with pytest.raises(S.StationRefused) as e:
        asyncio.run(S.connect())
    assert isinstance(e.value, ConnectionError)          # existing "Station unreachable" handling still applies
    with pytest.raises(S.StationRefused):
        asyncio.run(S.Station("localhost").connect())


def test_live_opt_in_uses_station_host(monkeypatch):
    calls = []

    async def fake_client(host, log):
        calls.append(host)
        return "client"
    monkeypatch.setenv("ROBOT_LIVE_TESTS", "1")
    monkeypatch.setenv("STATION_HOST", "station-under-test.invalid")
    monkeypatch.setattr(S, "_station_py", lambda: types.SimpleNamespace(new_station_client=fake_client))
    assert asyncio.run(S.connect()) == "client"
    assert asyncio.run(S.Station().connect()) == "client"
    assert asyncio.run(S.Station("other.invalid").connect()) == "client"
    assert calls == ["station-under-test.invalid", "station-under-test.invalid", "other.invalid"]


def test_station_host_default(monkeypatch):
    monkeypatch.delenv("STATION_HOST", raising=False)
    assert S.station_host() == "localhost"
    monkeypatch.setenv("STATION_HOST", "rig.local")
    assert S.station_host() == "rig.local" and S.Station().host == "rig.local"


def test_scripts_read_station_host():
    """The five standalone scripts connect to STATION_HOST, never a hard-coded localhost."""
    root = S.REPO / "scripts"
    for name in ("scene_cam", "teleop", "arm_state", "camera", "calibrate"):
        src = (root / f"{name}.py").read_text()
        assert 'new_station_client("localhost"' not in src, name
        assert 'new_station_client(os.environ.get("STATION_HOST", "localhost")' in src, name


def test_core_import_is_light():
    """generaloid.core must not pull MuJoCo, MCP or a model SDK: the Station layer is usable on its own."""
    import subprocess
    code = ("import sys, generaloid.core; bad = [m for m in ('mujoco', 'mcp', 'google.genai', 'numpy') "
            "if m in sys.modules]; print(bad)")
    out = subprocess.run([sys.executable, "-c", code], cwd=S.REPO, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out
