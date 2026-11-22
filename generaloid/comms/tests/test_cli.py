"""CLI credentials must not have a world-readable creation window."""
import stat
from types import SimpleNamespace

import pytest

from generaloid.comms.__main__ import _token


def test_token_file_created_private_and_reused(tmp_path, monkeypatch):
    monkeypatch.delenv("GENERALOID_COMMS_TOKEN", raising=False)
    path = tmp_path / "token"
    args = SimpleNamespace(token_file=str(path))
    token = _token(args, create=True)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert _token(args, create=False) == token and len(token) >= 32


def test_empty_token_file_is_an_explicit_error(tmp_path, monkeypatch):
    monkeypatch.delenv("GENERALOID_COMMS_TOKEN", raising=False)
    path = tmp_path / "empty"
    path.touch()
    with pytest.raises(SystemExit, match="empty"):
        _token(SimpleNamespace(token_file=str(path)), create=True)


def test_standalone_hub_and_client_processes(tmp_path):
    import json
    import os
    import socket
    import subprocess
    import sys
    import time
    import httpx

    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    token_file = tmp_path / "hub-token"
    env = {**os.environ, "STATION_HOST": "station-disabled-in-tests.invalid", "GENERALOID_COMMS_TOKEN": ""}
    command = [sys.executable, "-m", "generaloid.comms"]
    with (tmp_path / "hub-output.txt").open("w") as output:
        hub = subprocess.Popen(command + ["hub", "--port", str(port), "--token-file", str(token_file)],
                               env=env, stdout=output, stderr=output)
        try:
            url = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                assert hub.poll() is None, "standalone hub exited before ready"
                try:
                    if httpx.get(url + "/health", timeout=0.5).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.05)
            else:
                raise AssertionError("standalone hub did not become ready")
            client = subprocess.run(command + ["client", "--url", url, "--robot-id", "human-ops",
                                    "--token-file", str(token_file), "peers"], env=env,
                                    capture_output=True, text=True, timeout=10)
            assert client.returncode == 0, client.stderr
            assert [p["robot_id"] for p in json.loads(client.stdout)] == ["human-ops"]
            assert httpx.get(url + "/health").json()["peers_online"] == 0  # one-shot CLI leaves
        finally:
            hub.terminate()
            try:
                hub.wait(timeout=5)
            except subprocess.TimeoutExpired:
                hub.kill()
                hub.wait(timeout=5)
