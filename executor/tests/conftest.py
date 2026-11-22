import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for cand in (os.environ.get("NORMA_CORE_REPO"), ROOT / "norma-core", ROOT.parent.parent / "repo" / "norma-core"):
    if cand and (Path(cand) / "hardware/elrobot/simulation/elrobot_follower.urdf").exists():
        os.environ["NORMA_CORE_REPO"] = str(cand)
        break
os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
if "SKILLCFG_DIR" not in os.environ or "skillcfg-tests-" not in os.environ["SKILLCFG_DIR"]:
    import tempfile
    os.environ["SKILLCFG_DIR"] = tempfile.mkdtemp(prefix="skillcfg-tests-")   # never the live logs/skillcfg

mujoco = pytest.importorskip("mujoco")

from executor.backends import FakeBackend, park_norm  # noqa: E402
from executor.contracts import ActionProposal  # noqa: E402
from executor.core import Executor  # noqa: E402
from executor.kinematics import Kinematics  # noqa: E402
from executor.perception import Locator  # noqa: E402
from executor.recording import RecorderLink  # noqa: E402


@pytest.fixture(scope="session")
def kin():
    if not (ROOT / "sim/generated/workspace.xml").exists() and "NORMA_CORE_REPO" not in os.environ:
        pytest.skip("needs sim/generated/workspace.xml or NORMA_CORE_REPO (ElRobot URDF) for the kinematic model")
    return Kinematics()


@pytest.fixture(scope="session")
def park(kin):
    return park_norm(kin)


@pytest.fixture
def make_ex(kin, park, tmp_path):
    made = []

    def _make(**fake_kw):
        fake = FakeBackend(start_norm=dict(park), kin=kin, **fake_kw)
        ex = Executor(fake, kin, recorder=RecorderLink("local", root=tmp_path / "rec"), locator=Locator(),
                      table_memory_path=tmp_path / "table.json")
        made.append(ex)
        return ex, fake
    yield _make
    for ex in made:
        ex.cancel()
        ex.end_task("test end")


def proposal(ex, obs, skill, params, ttl=30.0, **kw):
    return ActionProposal(ActionProposal.new_id(), kw.pop("source", "script"), obs.obs_id, kw.pop("setup_id", obs.setup_id), skill,
                          kw.pop("skill_version", ""), params, [], kw.pop("expires_at", ex.now() + ttl), kw.pop("rationale", "test"))


def tcp(ex):
    return ex.kin.fk(ex.motion.q_now())[0]
