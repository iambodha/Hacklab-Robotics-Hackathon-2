"""Site selection is offline: never import a real MCP server or Station client."""
import io
import json
from pathlib import Path

import pytest

from sites import DATA_ROOT, ROOT, get_site
from sites.__main__ import discover, shell_defaults


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for key in ("SITE", "ROBOT_OVERVIEW_CAL", "ROBOT_REAL_LOG_DIR", "BOTTLE_LEARN_DIR", "RECORDER_CAM_ROLES",
                "PERCEPTION_OVERVIEW_QUEUE", "PERCEPTION_WRIST_QUEUE", "TWIN_OVERVIEW_QUEUE", "TWIN_WRIST_QUEUE",
                "ROBOT_OVERVIEW_CAM", "ROBOT_WRIST_CAM", "RECORDER_DIR"):
        monkeypatch.delenv(key, raising=False)


def test_default_home_assets_are_the_frozen_originals():
    site = get_site()
    assert site.name == "home"
    assert site.scene_calib == ROOT / "sites/home/scene_calib.json"
    for key, original in [("scene_calib", "calibration/scene_calib.json"), ("layout", "sim/layouts/home.json"),
                          ("overview_camera", "sim/real_overview_camera.json"),
                          ("wrist_mount", "camera_pose/wrist_camera.json"),
                          ("intrinsics_overview", "calibration/intrinsics_overview.json"),
                          ("intrinsics_wrist", "calibration/intrinsics_wrist.json")]:
        assert json.loads(site.asset(key).read_text()) == json.loads((ROOT / original).read_text())
    assert site.robot_camera("overview") == "9ea4cacd"
    assert site.robot_camera("wrist") == "28ff0ca8"
    assert site.queue("overview") == "usbvideo/9ea4cacd5be94751e1eea7be90bcf628"
    assert site.queue("wrist") == "usbvideo/28ff0ca87978e8f993c2ac64a84d188a"
    assert site.recorder_roles() == "28ff0ca8=wrist,9ea4cacd=overview"
    assert site.action_log_dir == DATA_ROOT / "logs/real_sessions_home"
    assert site.episode_dir == DATA_ROOT / "logs/learning/bottle"
    assert site.record_dir == ROOT / "recordings"


def test_office_selection_and_separate_records(monkeypatch):
    monkeypatch.setenv("SITE", "office")
    site = get_site()
    assert site.camera_hash("overview") == "ce4e355ece1334a6c4c77e01a0db2dcc"
    assert site.camera_hash("wrist", "TWIN") == "9b061799a7c5f6fa2f9fbd0e1a46d4b8"
    assert site.robot_camera("wrist") == "9b061799"
    assert site.scene_calib == ROOT / "sites/office/scene_calib.json"
    assert site.layout == ROOT / "sites/office/layout.json"
    assert site.overview_camera == ROOT / "sites/office/real_overview_camera.json"
    assert site.episode_dir == DATA_ROOT / "logs/learning/office/bottle"
    assert site.action_log_dir == DATA_ROOT / "logs/real_sessions_office"
    assert site.record_dir != get_site("home").record_dir


@pytest.mark.parametrize("consumer", ["PERCEPTION", "TWIN"])
@pytest.mark.parametrize("role", ["overview", "wrist"])
def test_queue_override_wins(monkeypatch, consumer, role):
    monkeypatch.setenv("SITE", "office")
    monkeypatch.setenv(f"{consumer}_{role.upper()}_QUEUE", "usbvideo/custom")
    assert get_site().queue(role, consumer) == "usbvideo/custom"
    assert shell_defaults()[f"{consumer}_{role.upper()}_QUEUE"] == "usbvideo/custom"


def test_all_other_overrides_win(monkeypatch, tmp_path):
    monkeypatch.setenv("SITE", "office")
    for role in ("wrist", "overview"):
        monkeypatch.setenv(f"ROBOT_{role.upper()}_CAM", "custom-" + role)
        assert get_site().robot_camera(role) == "custom-" + role
    for key, attr in [("ROBOT_OVERVIEW_CAL", "overview_camera"), ("ROBOT_REAL_LOG_DIR", "action_log_dir"),
                      ("BOTTLE_LEARN_DIR", "episode_dir"), ("RECORDER_DIR", "record_dir")]:
        monkeypatch.setenv(key, str(tmp_path / key))
        assert getattr(get_site(), attr) == tmp_path / key
    monkeypatch.setenv("RECORDER_CAM_ROLES", "abc=wrist")
    from recorder.fmt import cam_roles
    assert cam_roles() == {"abc": "wrist"}


@pytest.mark.parametrize("name", ["../home", "", "missing-site"])
def test_invalid_or_unknown_site_fails_clearly(name):
    with pytest.raises(ValueError, match="SITE"):
        get_site(name)


def test_layout_precedence_and_generated_isolation(monkeypatch, tmp_path):
    import sys
    monkeypatch.syspath_prepend(str(ROOT / "sim"))
    import build_scene
    monkeypatch.delenv("SIM_LAYOUT", raising=False)
    monkeypatch.delenv("ROBOT_SIM_LAYOUT", raising=False)
    assert build_scene.layout_path() == get_site("home").layout
    monkeypatch.setenv("SITE", "office")
    assert build_scene.layout_path() == get_site().layout  # resolves even before calibration is installed
    assert build_scene.generated_dir(get_site().layout) != build_scene.generated_dir(get_site("home").layout)
    assert build_scene.layout_path("venue") == ROOT / "sim/layout.json"
    path = tmp_path / "custom.json"
    path.write_text("{}")
    monkeypatch.setenv("ROBOT_SIM_LAYOUT", str(path))
    assert build_scene.layout_path() == path
    monkeypatch.setenv("SIM_LAYOUT", "home")
    assert build_scene.layout_path() == get_site("home").layout
    assert build_scene.layout_path(str(path)) == path


def test_discovery_uses_only_read_only_relay_requests():
    calls = []
    h = get_site("office").camera_hash("wrist")
    def opener(url, timeout):
        calls.append(url)
        assert url == "http://relay/api/cameras"
        return io.BytesIO(json.dumps([{"hash": h, "live": True}, {"hash": "old", "live": False}]).encode())
    result = discover("http://relay/", opener=opener)
    assert len(result) == 1
    assert result[0]["queue"] == "usbvideo/" + h
    assert "wrist" in result[0]["looks_like"]
    assert "jaws" in result[0]["looks_like"]
    assert calls == ["http://relay/api/cameras"]


def test_discovery_unknown_camera_snapshot_failure_is_not_a_guess():
    def opener(url, timeout):
        if url.endswith("/api/cameras"):
            return io.BytesIO(b'[{"hash":"new-camera","live":true}]')
        assert url == "http://relay/snapshot/new-camera"
        raise OSError("snapshot unavailable")
    result = discover("http://relay", opener=opener)
    assert result[0]["looks_like"].startswith("unknown")


def test_office_consumers_start_with_one_setting(monkeypatch):
    import os
    import subprocess
    import sys
    monkeypatch.setenv("SITE", "office")
    code = """
from sites import get_site
from perception import scene, station_io
from sim import station_io as twin_io
from recorder.fmt import cam_roles
from dashboard import data, server
from learning import episodes
from generaloid.connectors.elrobot import CONNECTOR
site = get_site()
assert scene.SCENE_PATH == site.scene_calib
assert scene.intrinsics_file('wrist') == site.intrinsics('wrist')
assert station_io.QUEUES['overview'] == site.queue('overview')
assert twin_io.QUEUES['wrist'] == site.queue('wrist', 'TWIN')
assert cam_roles() == {'9b061799': 'wrist', 'ce4e355e': 'overview'}
assert data.LEARN_DIR == episodes.DEFAULT_DIR == site.episode_dir
assert data.ACTIONS_LOG.parent == site.action_log_dir
assert server.CAMERAS[0]['hash'] == site.camera_hash('wrist')
assert CONNECTOR.cameras[0].station_id == site.robot_camera('wrist')
assert CONNECTOR.twin.layout == site.layout
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=os.environ.copy(), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
