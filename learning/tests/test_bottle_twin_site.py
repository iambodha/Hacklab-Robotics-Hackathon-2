"""The bottle twin copies the selected site's table layout (SITE, default home); geometry stays in the base frame."""
import json

import pytest

from learning import bottle_twin
from sites import get_site


def _markers(lay):
    return {m["id"]: m["T"] for m in lay["markers"]["items"]}


@pytest.mark.parametrize("site", ["home", "office"])
def test_make_layout_copies_the_site_table(tmp_path, monkeypatch, site):
    monkeypatch.setenv("SITE", site)
    lay = json.loads(bottle_twin.make_layout(tmp_path / "twin.json").read_text())
    base = json.loads(get_site(site).layout.read_text())
    assert _markers(lay) == _markers(base) and lay["table"] == base["table"]
    assert lay["screw_cap"]["pos"][:2] == bottle_twin.BOTTLE["xy"]
    assert lay["bottle_twin"]["site"] == site


def test_default_is_home_and_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("SITE", raising=False)
    lay = json.loads(bottle_twin.make_layout(tmp_path / "twin.json").read_text())
    home = json.loads(bottle_twin.HOME.read_text())
    assert lay["table"] == home["table"] and _markers(lay) == _markers(home)
    assert {k: v for k, v in lay.items() if k not in ("fixed", "screw_cap", "_notes", "bottle_twin")} == \
        {k: v for k, v in home.items() if k not in ("fixed", "_notes")}


def test_learned_layout_uses_the_site_bottle_placement(tmp_path, monkeypatch):
    monkeypatch.setenv("SKILL_PROFILES", "bottle_cap")
    from learning import skillcfg
    obj = next(p for p in skillcfg.profiles() if p.get("profile") == "bottle_cap")["object"]
    home = json.loads(bottle_twin.learned_layout(tmp_path / "home.json", site="home").read_text())
    assert home["screw_cap"]["pos"][:2] == obj["cap_xy_m"]
    office = json.loads(bottle_twin.learned_layout(tmp_path / "office.json", site="office").read_text())
    assert office["screw_cap"]["pos"][:2] == bottle_twin.site_bottle("office").get("cap_xy_m", obj["cap_xy_m"])
    assert _markers(office) == _markers(json.loads(get_site("office").layout.read_text()))
