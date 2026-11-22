import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Never read or write the live skill config (logs/skillcfg) from tests, also not from module-scoped fixtures and
# their subprocesses (they start before the per-test fixture below).
os.environ["SKILLCFG_DIR"] = tempfile.mkdtemp(prefix="skillcfg-tests-")
os.environ.pop("SKILLCFG_TARGET", None)
# Legacy r3/r4 tests exercise the unchanged baseline; learned-profile tests opt in explicitly.
os.environ["SKILL_PROFILES"] = "none"


@pytest.fixture(autouse=True)
def _skillcfg_dir(tmp_path, monkeypatch):
    from learning import skillcfg
    monkeypatch.setenv("SKILLCFG_DIR", str(tmp_path / "skillcfg"))
    monkeypatch.setattr(skillcfg, "_target", None)
    monkeypatch.setenv("SKILL_PROFILES", "none")
    yield
