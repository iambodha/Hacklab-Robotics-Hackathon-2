import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for cand in (os.environ.get("NORMA_CORE_REPO"), ROOT / "norma-core", ROOT.parent.parent / "repo" / "norma-core"):
    if cand and (Path(cand) / "hardware/elrobot/simulation/elrobot_follower.urdf").exists():
        os.environ["NORMA_CORE_REPO"] = str(cand)
        break
os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")

DATA = Path(__file__).resolve().parent / "data"
