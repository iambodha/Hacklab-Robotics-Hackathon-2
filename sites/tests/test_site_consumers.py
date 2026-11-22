"""SITE=office reaches every runtime consumer: scene calibration, twin layout, overview camera, camera queues and
record dirs. Import- and function-level only, each consumer group in a fresh process (most resolve at import);
no Station or camera relay connection is opened (the repo conftest points STATION_HOST at an invalid name)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sites import ROOT, get_site

OVERRIDES = ("ROBOT_OVERVIEW_CAL", "ROBOT_REAL_LOG_DIR", "BOTTLE_LEARN_DIR", "RECORDER_CAM_ROLES", "RECORDER_DIR",
             "PERCEPTION_OVERVIEW_QUEUE", "PERCEPTION_WRIST_QUEUE", "TWIN_OVERVIEW_QUEUE", "TWIN_WRIST_QUEUE",
             "ROBOT_OVERVIEW_CAM", "ROBOT_WRIST_CAM", "EXECUTOR_SCENE_CALIB", "EXECUTOR_RECORDINGS", "EXECUTOR_TWIN_LAYOUT",
             "SIM_LAYOUT", "ROBOT_SIM_LAYOUT", "PERCEPTION_ROOT")
OFFICE = get_site("office")
needs_office = pytest.mark.skipif(not all(p.is_file() for p in (OFFICE.scene_calib, OFFICE.layout, OFFICE.overview_camera)),
                                  reason="office calibration/layout/overview camera not produced yet (sites/office)")
needs_norma = pytest.mark.skipif(not any(p and (Path(p) / "software/station/shared/station_py").exists() for p in
                                         (os.environ.get("NORMA_CORE_REPO"), ROOT / "norma-core", ROOT.parent / "norma-core")),
                                 reason="arm_mcp/real_arm_mcp import the Station client: set NORMA_CORE_REPO")


def run(code: str, site: str = "office") -> str:
    env = {k: v for k, v in os.environ.items() if k not in OVERRIDES}
    env.update(SITE=site, MUJOCO_GL=env.get("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl"))
    r = subprocess.run([sys.executable, "-c", "import warnings\nwarnings.simplefilter('ignore')\n" + code], cwd=ROOT,
                       env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout


PRELUDE = """
import json
from pathlib import Path
from sites import get_site
site = get_site()
assert site.name == {site!r}
created = json.loads(site.scene_calib.read_text())['created']
"""


@needs_office
@needs_norma
def test_perception_and_executor_use_the_office_calibration():
    run(PRELUDE.format(site="office") + """
import numpy as np
from perception import locate, scene
assert scene.SCENE_PATH == site.scene_calib and locate.scene_file_exists()
assert locate.reload().d['created'] == created
from executor.kinematics import Kinematics, load_scene_calib, scene_calib_path
assert scene_calib_path() == site.scene_calib
sc = load_scene_calib()
assert sc['created'] == created
assert np.allclose(Kinematics().T, sc['arm']['T_world_base'])
from executor.perception import PerceptionLocator
d = PerceptionLocator(sc).describe()
assert d['scene_calib'] == created
assert d['board_intrinsics_file'] == str(site.intrinsics('overview'))
from models import perception_bridge as pb
assert pb.PERCEPTION_ROOT == Path(pb.__file__).resolve().parents[1]
assert pb.locator() is not None, pb.last_error()
assert pb.locator().sc.d['created'] == created
from arm_mcp import server
assert server._locator().sc.d['created'] == created
""")


@needs_office
def test_task_locator_and_measurement_follow_the_site():
    out = run(PRELUDE.format(site="office") + """
import warnings
from tasks.blue_to_bin import measure, task
assert task.measured_path() == measure.OUT == Path(task.__file__).with_name('measured_office.json')
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    spec = task.Spec.load()
print(json.dumps({'exists': task.measured_path().exists(), 'source': spec.source, 'warned': [str(x.message) for x in w]}))
from tasks.blue_to_bin.locate import BlueLocator
assert BlueLocator(spec).loc.sc.d['created'] == created
""")
    rep = json.loads(out.strip().splitlines()[-1])
    if rep["exists"]:
        assert rep["source"]["file"].endswith("measured_office.json") and "bin_from_site" not in rep["source"]
    else:
        assert rep["source"]["bin_from_site"] == "home" and "not measured yet" in rep["warned"][0]


@needs_office
@needs_norma
def test_real_arm_mcp_uses_office_overview_camera_ids_and_log():
    run(PRELUDE.format(site="office") + """
import sys
sys.path.insert(0, 'sim')
import real_arm_mcp as r
assert r.OVERVIEW_CAL == site.overview_camera and r.OVERVIEW_CAL.is_file()
assert (r.WRIST_CAM, r.OVERVIEW_CAM) == (site.robot_camera('wrist'), site.robot_camera('overview')) == ('9b061799', 'ce4e355e')
assert r.LOG_DIR == site.action_log_dir and 'office' in str(r.LOG_DIR)
from camera_pose import elrobot_camera
assert elrobot_camera.CAMERA_JSON == site.wrist_mount
""")


@needs_office
def test_twin_queues_and_records_follow_the_site():
    run(PRELUDE.format(site="office") + """
import sys
from sim import build_scene, station_io as twin_io
from perception import station_io
assert build_scene.layout_path() == site.layout
assert station_io.QUEUES == {'overview': 'usbvideo/ce4e355ece1334a6c4c77e01a0db2dcc', 'wrist': 'usbvideo/9b061799a7c5f6fa2f9fbd0e1a46d4b8'}
assert twin_io.QUEUES == station_io.QUEUES
from executor import recording
assert recording.LOCAL_ROOT == Path(recording.__file__).resolve().parents[1] / 'recordings' / 'office'
from recorder import store
from recorder.fmt import cam_roles
assert store.ROOT == site.record_dir and cam_roles() == {'9b061799': 'wrist', 'ce4e355e': 'overview'}
from skills.loop import bundle
assert bundle.OVERVIEW_HASH == 'ce4e355ece1334a6c4c77e01a0db2dcc'
from dashboard import data, server
assert data.LEARN_DIR == site.episode_dir and data.ACTIONS_LOG.parent == site.action_log_dir
assert [c['hash'] for c in server.CAMERAS] == ['9b061799a7c5f6fa2f9fbd0e1a46d4b8', 'ce4e355ece1334a6c4c77e01a0db2dcc']
""")


@needs_office
def test_sim_backend_reports_the_site_layout():
    run(PRELUDE.format(site="office") + """
from executor.backends import sim
assert sim.layout_label(site.layout) == 'office'
assert sim.layout_label(Path('sim/layout.json')) == 'layout'
""")


def test_home_consumers_unchanged():
    run(PRELUDE.format(site="home") + """
from perception import scene
from executor import recording
from executor.kinematics import scene_calib_path
from tasks.blue_to_bin import task
assert scene.SCENE_PATH == scene_calib_path() == site.scene_calib
assert recording.LOCAL_ROOT == Path(recording.__file__).resolve().parents[1] / 'recordings'
assert task.measured_path() == task.MEASURED == Path(task.__file__).with_name('measured.json')
assert task.Spec.load().source['file'] == str(task.MEASURED)
""", site="home")


@pytest.mark.parametrize("site", ["home", "office"])
def test_sites_check_passes_in_a_clean_environment(site):
    if site == "office" and not OFFICE.scene_calib.is_file():
        pytest.skip("office calibration not produced yet")
    out = run("import runpy, sys\nsys.argv = ['sites', 'check']\ntry:\n    runpy.run_module('sites', run_name='__main__')\n"
              "except SystemExit as e:\n    print('EXIT', e.code)", site=site)
    assert out.strip().splitlines()[-1] in ("EXIT 0", "EXIT None"), out
    assert f"sites/{site}/scene_calib.json" in out


def test_sites_check_flags_home_leftovers_under_office(monkeypatch):
    from sites.__main__ import check
    for k in OVERRIDES:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SITE", "office")
    monkeypatch.setenv("SIM_LAYOUT", "home")
    monkeypatch.setenv("PERCEPTION_WRIST_QUEUE", get_site("home").queue("wrist"))
    monkeypatch.setenv("ROBOT_OVERVIEW_CAL", str(get_site("home").overview_camera))
    _, problems = check()
    assert len(problems) == 3 and all("home" in p for p in problems)


def test_stack_rejects_an_unknown_site():
    r = subprocess.run(["bash", str(ROOT / "scripts/stack.sh"), "status", "--site", "nowhere"], cwd=ROOT,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "unknown SITE 'nowhere'" in r.stderr
