import sys
import types

import pytest

from generaloid.core.legacy import real_tools


def test_cached_real_server_cannot_be_reused_for_another_robot(monkeypatch):
    from generaloid.connectors import get

    loaded = types.SimpleNamespace(P=types.SimpleNamespace(name=get("elrobot").profile.name))
    monkeypatch.setitem(sys.modules, "real_arm_mcp", loaded)
    assert real_tools("elrobot") is loaded
    with pytest.raises(RuntimeError, match="fresh server process"):
        real_tools("so101")
