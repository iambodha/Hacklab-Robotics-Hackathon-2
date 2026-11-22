"""Learned defaults, real-record replay, guarded extraction and resumable budgets."""
import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from learning import capfree, episodes as E, rules, skillcfg as SC, tools
from learning.twist import Twister, rz

DATA = Path(__file__).resolve().parents[2] / 'docs/learning/data/attempts.json'


@pytest.fixture
def learned(monkeypatch):
    monkeypatch.setenv('SKILL_PROFILES', 'bottle_cap')
    return SC.defaults()


def test_profile_defaults_and_immutable_safety(learned):
    d = E.defaults()
    assert (d['grip_torque_limit'], d['strokes'], d['approach_z_offset_mm']) == (.5, 12, -6.5)
    assert (d['yaw_start_deg'], d['twist_deg_per_stroke'], d['regrasp_backoff_deg']) == (-12, 32, 31)
    assert learned['extract.lift_m'] == .06
    assert 0 < learned['twist.time_budget_s'] < SC.MCP_CLIENT_TIMEOUT_S
    assert SC.validate({'safety.max_grip_force': .9})[1]
    for profile in SC.profiles():
        for entry in profile['entries'].values():
            assert entry['attempts'] and entry['date'] and entry['evidence']


def strokes(attempt):
    return [s for p in attempt['phases'] for s in p['measurements'].get('strokes', [])]


def test_real_record_free_signal_replay(learned):
    attempts = json.loads(DATA.read_text())
    assert all(not capfree.free_signal(strokes(a), learned)['free'] for a in attempts[:8])
    sig = capfree.free_signal(strokes(attempts[9]), learned)
    assert (sig['began_at'], sig['confirmed_at']) == (11, 12)
    assert not capfree.free_signal(strokes(attempts[11]), learned)['free']  # medium speed caveat


def test_slip_or_invalid_stroke_breaks_confirmation(learned):
    def s(i, load, **kw):
        return dict(stroke=i, mean_j7_load=load, achieved_deg=100, commanded_deg=100, **kw)
    seq = [s(1, 100), s(2, 100), s(3, 50), s(4, 50, slip=True), s(5, 50)]
    assert not capfree.free_signal(seq, learned)['free']
    seq.append(s(6, 50))
    assert capfree.free_signal(seq, learned)['confirmed_at'] == 6


def test_replay_learning_rules(learned):
    attempts = json.loads(DATA.read_text())
    # Replay on r4 baseline to see what learning would have proposed before adopting the profile.
    cfg = SC.baseline()
    one = {r['rule']: r for r in rules.suggest(attempts[0], cfg)}
    assert one['more_strokes']['changes']['params.default.strokes'] == 8
    assert one['longer_lift']['changes']['extract.lift_m'] >= .04
    eight = {r['rule']: r for r in rules.suggest(attempts[7], cfg)}
    assert eight['deeper_grasp']['changes']['params.default.approach_z_offset_mm'] == -6.5
    assert eight['top_approach']['changes']['twist.approach'] == 'top'
    five = {r['rule']: r for r in rules.suggest(attempts[4], cfg)}
    assert five['gripper_overload']['changes']['params.default.grip_torque_limit'] == .5
    ten = {r['rule']: r for r in rules.suggest(attempts[9], cfg)}
    assert ten['fit_the_call']['changes']['twist.time_budget_s'] == 110
    assert ten['trim_strokes']['changes']['params.default.strokes'] == 12


def test_proposal_is_pending_and_next_attempt_inherits_applied_default(tmp_path, monkeypatch):
    from learning.tests.test_episodes import FakeRecorder
    st = E.EpisodeStore(tmp_path / 'learn', recorder=FakeRecorder())
    assert st.begin({'strokes': 4}, 'baseline')['ok']
    st.end('loosened', 'still attached after short lift')
    SC.set_target('real')
    out = rules.propose_after(st, 1)
    assert out and all(r['proposal']['status'] == 'pending' for r in out)
    assert SC.version() == 0
    SC.set_target('twin')
    p = SC.propose({'params.default.strokes': 8}, 'more turning', ['attempt 1'], t='twin')
    assert SC.apply(p['proposal']['id'], 'test')['ok']
    r = st.begin({}, 'use applied learning')
    assert r['ok'] and r['params']['strokes'] == 8


class LiftRig:
    twist_sign = 1

    def __init__(self, resisted=False, release_error=False):
        self.pos = np.array([0., .19, .19])
        self.resisted, self.release_error = resisted, release_error
        self.held = True
        self.moves = 0
        self.retreats = 0

    async def q(self): return np.r_[self.pos, np.zeros(5)]
    def fk(self, q): return np.asarray(q[:3]), np.eye(3)
    def solve(self, xyz, R, seed): return np.r_[xyz, np.zeros(5)]
    async def holding(self): return self.held
    async def state(self):
        return {f'j{i}': dict(raw=100, goal=100, load=10, norm=.5, status=0) for i in range(1, 9)}
    async def go(self, q, dt, contact):
        self.moves += 1
        if not self.resisted: self.pos = np.asarray(q[:3])
        return 'ok'
    async def grip(self, fraction, force):
        if self.release_error: raise RuntimeError('release failed')
        self.held = False
        return 'open', False
    async def move_down(self, *a):
        self.retreats += 1
        return 'up'


def test_extraction_rises_six_cm_and_resistance_stops_early(learned):
    async def run():
        rig = LiftRig()
        r = await Twister(rig, E.defaults(), lambda *a: None).extract_cap()
        assert r['cap_free'] and r['cap_lift_mm'] == pytest.approx(60)
        rig = LiftRig(resisted=True)
        tw = Twister(rig, E.defaults(), lambda *a: None)
        r = await tw.extract_cap()
        assert r['resisted'] and not r['cap_free'] and rig.moves < 20
        await tw._retreat('resisted')
        assert not rig.held and rig.retreats == 1
    asyncio.run(run())


def test_failed_release_never_lifts_attached_cap(learned):
    rig = LiftRig(resisted=True, release_error=True)
    msg = asyncio.run(Twister(rig, E.defaults(), lambda *a: None)._retreat('resisted'))
    assert 'refusing to lift' in msg and rig.retreats == 0


def test_budget_includes_approach_and_resume_backs_off(learned):
    class Clock:
        t = 0.
        def __call__(self): return self.t
    class BudgetTwister(Twister):
        async def _holding_at(self, *a): return bool(self.prior)
        async def _pose(self): return np.zeros(8), np.array([0., .19, .19]), rz(0)
        async def _approach_down(self, *a): clock.t += 104; return True
        async def _grasp(self, *a): return True
        async def _release_backoff_regrasp(self, *a): self.backed = True; return True
        async def _stroke(self, n, *a):
            clock.t += 10
            return dict(stroke=n, achieved_deg=32, commanded_deg=32, mean_j7_load=100, abort_reason=None)
    rig = LiftRig()
    async def reachable(*a): return 35
    rig.reachable = reachable
    clock = Clock()
    tw = BudgetTwister(rig, E.defaults(), lambda *a: None, clock=clock)
    r = asyncio.run(tw.twist_cap(0, .19, .205, .022))
    assert r['stopped_by'] == 'time budget' and not r['strokes'] and clock.t == 104
    clock.t = 0
    tw = BudgetTwister(rig, E.defaults(), lambda *a: None, clock=clock,
                        prior_strokes=[dict(stroke=1, achieved_deg=32, commanded_deg=32, mean_j7_load=100)])
    r = asyncio.run(tw.twist_cap(0, .19, .205, .022))
    assert r['resumed'] and tw.backed and r['elapsed_s'] <= 110


def test_actual_deadline_halts_a_slow_approach(learned):
    class SlowTwister(Twister):
        async def _twist_cap(self, *a): await asyncio.sleep(1)
    rig = LiftRig()
    rig.halted = False
    async def halt(): rig.halted = True
    rig.halt = halt
    tw = SlowTwister(rig, E.defaults(), lambda *a: None)
    tw.c = {**tw.c, 'twist.time_budget_s': .01}
    r = asyncio.run(tw.twist_cap(0, .19, .205))
    assert r['stopped_by'] == 'time budget' and rig.halted and not r['strokes']
    assert r['elapsed_s'] < 1


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_tunables_refused(bad):
    assert SC.validate({'params.default.strokes': bad})[1]
    assert E.validate({'strokes': bad})[2]


class KinematicBottleTwin(LiftRig):
    """Deterministic screw/contact contract, without MuJoCo friction or IK.

    This isolates the shared skill from the known physical twin contact mismatch.
    It is not a substitute for the MuJoCo acceptance probe documented in VALIDATION.md.
    """
    def __init__(self, profile):
        super().__init__()
        self.obj = profile['object']
        self.pos = np.array([*self.obj['cap_xy_m'], self.obj['cap_top_z_m'] + .05])
        self.yaw = 0.
        self.held = False
        self.norm = .96
        self.turned = 0.
        self.start_z = None

    @property
    def free(self): return self.turned >= self.obj['thread_free_deg']
    async def q(self): return np.r_[self.pos, np.radians(self.yaw), np.zeros(4)]
    def fk(self, q): return np.asarray(q[:3]), rz(np.degrees(q[3])) @ np.diag([1., -1., -1.])
    def solve(self, xyz, R, seed):
        return np.r_[xyz, np.arctan2(R[1, 0], R[0, 0]), np.zeros(4)]
    async def reachable(self, *a): return 35.
    async def move_down(self, x, y, z, yaw):
        self.pos = np.array([x, y, z]); self.yaw = yaw
        return 'reached'
    async def descend(self, x, y, z, yaw, top):
        self.pos = np.array([x, y, z]); self.yaw = yaw; self.start_z = z
        return 'reached', None
    async def grip(self, opening, force):
        self.held = opening < .6
        self.norm = .6 if self.held else opening
        return ('holding' if self.held else 'open'), self.held
    async def state(self):
        state = await super().state()
        state['j8'].update(norm=self.norm, raw=2500)
        state['j7']['load'] = (self.obj['j7_mean_load']['free_slow'][0] if self.free else
                              self.obj['j7_mean_load']['closed_slow'])
        return state
    async def go(self, q, dt, contact):
        from learning.twist import wrap
        if self.held:
            self.turned += max(0., wrap(np.degrees(q[3]) - self.yaw))
        target = np.asarray(q[:3]).copy()
        if self.held and not self.free:
            target[2] = min(target[2], self.start_z + .003)
        self.pos = target; self.yaw = np.degrees(q[3])
        return 'reached'


def test_closed_cap_kinematic_twin_opens_in_profile_budget_and_extracts_6cm(learned):
    async def run():
        rig = KinematicBottleTwin(SC.profiles()[0])
        tw = Twister(rig, E.defaults(), lambda *a: None)
        x, y = rig.obj['cap_xy_m']
        r = await tw.twist_cap(x, y, rig.obj['cap_top_z_m'])
        assert r['ok'] and rig.free
        assert r['strokes_total'] <= E.defaults()['strokes']
        z = rig.pos[2]
        lift = await tw.extract_cap()
        assert lift['cap_free'] and lift['cap_lift_mm'] == pytest.approx(60)
        assert rig.pos[2] - z == pytest.approx(.06)
    asyncio.run(run())


def test_lift_never_ignores_immutable_hard_loads(learned):
    class HotRig(LiftRig):
        async def state(self):
            st = await super().state()
            st['j2']['load'] = E.SAFETY['arm_load_hard'] + 1
            return st
    r = asyncio.run(Twister(HotRig(), E.defaults(), lambda *a: None).extract_cap())
    assert r['resisted'] and not r['cap_free'] and r['lift_steps'] == 1


def test_late_profile_change_replaces_inherited_defaults(tmp_path, monkeypatch):
    from learning.tests.test_episodes import FakeRecorder
    st = E.EpisodeStore(tmp_path/'learn', recorder=FakeRecorder())
    assert st.begin({}, 'baseline before profile')['ok']
    st.end('no_progress', 'before adopting the learned profile')
    # A focused profile change is within the three-parameter attribution budget.
    profile = tmp_path/'profile.json'
    profile.write_text(json.dumps({'profile':'test', 'version':1, 'entries':{
        'params.default.strokes': {'value':9}}}))
    monkeypatch.setenv('SKILL_PROFILES', str(profile))
    r = st.begin({}, 'profile supplies next default')
    assert r['ok'] and r['params']['strokes'] == 9
    assert r['from_config']['strokes'] == {'before':3, 'after':9}


def test_old_override_conflicting_with_new_profile_is_ignored(tmp_path, monkeypatch):
    SC.set_target('twin')
    p = SC.propose({'params.bounds.strokes.hi':5}, 'old baseline bound', ['baseline'], t='twin')
    assert SC.apply(p['proposal']['id'], 'test')['ok']
    monkeypatch.setenv('SKILL_PROFILES', 'bottle_cap')
    active = SC.active()
    assert 'params.bounds.strokes.hi' in active['ignored']
    assert active['values']['params.bounds.strokes.hi'] >= E.defaults()['strokes']


def test_shared_strategy_changes_reach_executor_and_wipe():
    from executor.skills import SKILLS
    from skills.config import get as wipe_strategy
    SC.set_target('twin')
    p = SC.propose({'executor.freshness.pick_s':2.5, 'wipe.adapt.alpha':.2},
                   'offline strategy trial', ['synthetic wipe and pick'], t='twin')
    assert SC.apply(p['proposal']['id'], 'test')['ok']
    assert SKILLS['pick'].freshness_s == 2.5
    assert wipe_strategy('wipe.adapt.alpha') == .2


def test_twist_stops_early_on_confirmed_free_load(learned):
    import copy
    async def run():
        profile = copy.deepcopy(SC.profiles()[0])
        # A partly loosened cap frees earlier; the detector must stop before the full ceiling.
        profile['object']['thread_free_deg'] = 250
        rig = KinematicBottleTwin(profile)
        x, y = rig.obj['cap_xy_m']
        result = await Twister(rig, E.defaults(), lambda *a: None).twist_cap(x, y, rig.obj['cap_top_z_m'])
        assert result['ok'] and result['stopped_by'] == 'free'
        assert result['strokes_total'] < E.defaults()['strokes']
        assert result['strokes_needed'] == result['free_signal']['began_at']
        assert result['holding'] and rig.free
    asyncio.run(run())
