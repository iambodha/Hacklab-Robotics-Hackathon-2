"""Coverage planner, proposals and level-1 adaptation."""
import math

import pytest

from skills._contracts import ActionProposal
from skills.wipe import Region, WidthMeasurement, WipeSkill, plan_coverage, update_width
from skills.wipe import limits as L

LIM = L.load()


def test_limits_come_from_profile():
    assert LIM["hard_load_sum"] == 520 and LIM["max_stroke_m"] == 0.25 and LIM["contact"] == "guarded_descent"


def test_boustrophedon_lanes_and_spacing():
    p = plan_coverage(Region(0.2, -0.05, 0.32, 0.05), 0.03, 0.8, 10, LIM)
    assert p.axis == "x" and p.lanes == p.lanes_needed == 5 and not p.capped
    ys = [s.start[1] for s in p.strokes]
    assert math.isclose(ys[0], -0.05 + 0.012) and math.isclose(ys[-1], 0.05 - 0.012)     # inset beta * w_hat / 2
    assert p.spacing <= 0.8 * 0.03 + 1e-9
    assert [s.start[0] < s.end[0] for s in p.strokes] == [True, False, True, False, True]  # alternating direction
    assert all(math.isclose(s.length, 0.12 + 0.02) for s in p.strokes)


def test_budget_caps_lanes():
    p = plan_coverage(Region(0.2, -0.05, 0.32, 0.05), 0.02, 0.8, 3, LIM)
    assert p.capped and p.lanes == 3 and p.lanes_needed > 3 and len(p.strokes) <= 3


def test_long_region_split_into_segments():
    p = plan_coverage(Region(0.1, -0.2, 0.2, 0.2), 0.04, 0.8, 50, LIM)       # 40 cm along y > 25 cm max stroke
    assert p.axis == "x" and p.segments == 1                                 # so strokes run along the short side
    p = plan_coverage(Region(0.0, -0.2, 0.3, 0.2), 0.04, 0.8, 50, LIM)       # neither side fits in one stroke
    assert p.axis == "y" and p.segments == 2 and all(s.length <= LIM["max_stroke_m"] + 1e-9 for s in p.strokes)


def test_validate_and_proposals():
    sk = WipeSkill()
    v = sk.seed_version(w_hat=0.03)
    plan = sk.plan(v, {"region": (0.22, -0.07, 0.34, 0.03), "budget": 6})
    assert sk.validate(plan).ok
    props = sk.execute(plan, based_on_obs="obs-1", setup_id="setup-x", now=100.0, version=v.version)
    assert all(isinstance(p, ActionProposal) for p in props)
    assert [p.skill for p in props] == ["wipe"] * len(plan.strokes) + ["park"]
    w = props[0]
    assert w.setup_id == "setup-x" and w.based_on_obs == "obs-1" and w.expires_at == 130.0 and w.skill_version == v.version
    assert w.params["limits"]["hard_load_sum"] == 520 and len(w.params["from_xy_world"]) == 2


def test_validate_rejects_unreachable():
    sk = WipeSkill()
    plan = sk.plan(sk.seed_version(), {"region": (0.40, 0.0, 0.52, 0.1), "budget": 6})
    r = sk.validate(plan)
    assert not r.ok and r.checks["reach"]["result"] == "fail"


def test_applicability():
    sk = WipeSkill()
    assert sk.applicable({"tool": "tissue"}) == []
    assert sk.applicable({"tool": "spoon"}) and sk.applicable({"tool": "tissue", "objects_in_region": ["cup"]})


def test_versions_bounded_and_traceable():
    sk = WipeSkill()
    v = sk.seed_version(w_hat=0.5)                             # clamped to the bound
    assert v.params["w_hat"] == sk.spec["w_hat"].hi and sk.check_version(v) == []
    c = v.child({"w_hat": 0.03}, "human")
    assert c.parent == v.version and c.version != v.version
    assert sk.check_version(v.child({"w_hat": 0.2}, "x"))      # out of bounds is reported


def test_adaptation_converges_and_stays_in_bounds():
    sk = WipeSkill()
    spec = sk.spec["w_hat"]
    v = sk.seed_version(w_hat=0.07)
    for _ in range(12):
        n, rec = update_width(v, spec, WidthMeasurement(0.03, True), visible=True)
        v = n or v
        assert spec.lo <= v.params["w_hat"] <= spec.hi
    assert abs(v.params["w_hat"] - 0.03) < 0.0005
    for obs in (0.5, -1.0, 0.0):                               # absurd measurements cannot leave the bounds
        n, rec = update_width(v, spec, WidthMeasurement(obs, True), visible=True, alpha=1.0)
        assert spec.lo <= (n or v).params["w_hat"] <= spec.hi


def test_adaptation_ignores_invalid_or_invisible():
    sk = WipeSkill()
    v = sk.seed_version()
    assert update_width(v, sk.spec["w_hat"], WidthMeasurement(0.02, False), visible=True)[0] is None
    n, rec = update_width(v, sk.spec["w_hat"], WidthMeasurement(0.02, True), visible=False)
    assert n is None and not rec["accepted"] and "visible" in rec["reason"]
    with pytest.raises(ValueError):
        update_width(v, sk.spec["w_hat"], WidthMeasurement(0.02, True), visible=True, alpha=0)
