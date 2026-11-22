"""Fault-injection tests on the fake backend (no hardware, no Station)."""
import threading
import time

import numpy as np

from executor.contracts import Status
from executor.recording import TableMemory
from executor.tests.conftest import proposal, tcp


def _move(ex, obs, dxyz, **kw):
    return proposal(ex, obs, "move_tcp", {"xyz_world": (np.array(obs.joints.tcp_world) + dxyz).tolist()}, **kw)


def test_move_succeeds_and_is_recorded(make_ex):
    ex, fake = make_ex()
    obs = ex.observe()
    eid = ex.submit(_move(ex, obs, [0.03, 0.02, 0.03]))
    r = ex.wait(eid)
    assert r.status == Status.SUCCEEDED, r.termination
    assert r.residuals["tcp_mm"] <= 5
    assert [p["name"] for p in r.phases] == ["plan", "validate", "execute", "verify", "recover"]
    labels = [m.split(":", 1)[1] for m in r.markers]
    assert labels[:2] == ["proposal", "validation"] and "phase execute start" in labels and labels[-1] == "result"


def test_stalled_joint_is_partial_not_succeeded(make_ex, park):
    ex, fake = make_ex(stall={2: park[2]})          # j2 cannot move at all
    obs = ex.observe()
    r = ex.wait(ex.submit(_move(ex, obs, [0.06, 0.0, 0.05])))
    assert r.status == Status.PARTIAL, (r.status, r.termination)
    assert "stall" in r.termination or "short" in r.termination or "PARTIAL" in r.termination
    assert r.residuals, "a partial move must report residuals"
    assert fake.motion_writes(), "it did try to move"


def test_frozen_camera_rejects_with_zero_motion(make_ex):
    ex, fake = make_ex(frozen={"overview"})
    ex.observe()
    fake.advance(2.0)
    obs = ex.observe()
    assert obs.frames["overview"].stale and not obs.frames["wrist"].stale
    r = ex.status(ex.submit(_move(ex, obs, [0.02, 0, 0])))
    assert r.status == Status.REJECTED
    assert r.validation.checks["freshness"]["result"] == "fail"
    assert "overview" in r.validation.checks["freshness"]["stale_cameras"]
    assert fake.motion_writes() == []


def test_advisor_reply_after_expiry_is_rejected(make_ex):
    ex, fake = make_ex()
    obs = ex.observe()
    p = _move(ex, obs, [0.02, 0, 0], expires_at=ex.now() + 0.5)
    fake.advance(1.0)                                   # the advisor took too long
    r = ex.status(ex.submit(p))
    assert r.status == Status.REJECTED and any("expired" in x for x in r.validation.reasons)
    assert fake.motion_writes() == []


def test_two_simultaneous_submits_never_interleave(make_ex):
    ex, fake = make_ex(step_delay=0.002)
    obs = ex.observe()
    props = [_move(ex, obs, [0.05, 0.0, 0.04]), _move(ex, obs, [0.0, 0.05, 0.04])]
    ids, gate = [None, None], threading.Barrier(2)

    def go(i):
        gate.wait()
        ids[i] = ex.submit(props[i])
    ts = [threading.Thread(target=go, args=(i,)) for i in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    first = [ex.status(i) for i in ids]
    rejected = [r for r in first if r.status == Status.REJECTED]
    accepted = [r for r in first if r.status != Status.REJECTED]
    assert len(rejected) == 1 and len(accepted) == 1
    assert rejected[0].validation.checks["busy"]["result"] == "fail"
    done = ex.wait(accepted[0].execution_id)
    assert done.status == Status.SUCCEEDED
    tags = {e[3] for e in fake.motion_writes()}
    assert tags == {accepted[0].execution_id}


def test_cancel_mid_skill_stops_commands(make_ex):
    ex, fake = make_ex(step_delay=0.003)
    obs = ex.observe()
    eid = ex.submit(_move(ex, obs, [0.08, 0.05, 0.06]))
    deadline = time.monotonic() + 30
    while len(fake.motion_writes(eid)) < 5:               # sleep-poll: a bare spin starves the motion thread under load
        assert time.monotonic() < deadline and ex.busy(), "motion never got going"
        time.sleep(0.0005)
    marks, set_ev = [], ex.cancel_ev.set
    ex.cancel_ev.set = lambda: (marks.append(len(fake.log)), set_ev())[1]   # log position at the instant cancel lands
    assert ex.cancel(eid)
    r = ex.wait(eid)
    assert r.status == Status.CANCELLED
    after = fake.log[marks[0]:]
    kinds = [e[1] for e in after]
    assert kinds.count("goals") <= 1, after              # at most the write already in flight
    assert "hold" in kinds and kinds[-1] == "hold"        # then hold, then nothing
    assert [p for p in r.phases if p["name"] == "recover"][0]["status"] == "skipped"


def test_changed_setup_is_rejected(make_ex):
    ex, fake = make_ex()
    obs = ex.observe()
    ex.set_tool("spoon")
    r = ex.status(ex.submit(_move(ex, obs, [0.02, 0, 0])))
    assert r.status == Status.REJECTED
    assert r.validation.checks["setup"]["result"] == "fail"
    assert fake.motion_writes() == []


STIR = {"xyz_world": [0.24, 0.0, 0.06], "radius": 0.015, "turns": 2, "clearance": 0.06}


def test_unreachable_final_stir_waypoint_means_zero_motion(make_ex, kin, monkeypatch):
    ex, fake = make_ex()
    calls = []
    orig = kin.solve_tip

    def counting(*a, **k):
        calls.append(a[0])
        return orig(*a, **k)
    monkeypatch.setattr(kin, "solve_tip", counting)
    ex.skills["stir"].plan(ex.ctx(), STIR)                 # dry run: how many tip solves the plan makes
    last = len(calls)
    calls.clear()

    def last_fails(*a, **k):
        calls.append(a[0])
        return None if len(calls) == last else orig(*a, **k)
    monkeypatch.setattr(kin, "solve_tip", last_fails)
    obs = ex.observe()
    r = ex.status(ex.submit(proposal(ex, obs, "stir", STIR)))
    assert r.status == Status.REJECTED
    reach = r.validation.checks["reachability"]
    assert reach["result"] == "fail" and reach["index"] == reach["of"], reach
    assert fake.motion_writes() == []


def test_stir_abort_lifts_straight_up(make_ex, kin):
    ex, fake = make_ex()
    def stall_in_cup(ev):
        if ev["kind"] == "note" and ev["text"].startswith("lower 1/1") and 1 not in fake.stall:
            fake.stall[1] = fake.pos[1]                    # j1 jams once the tip is down in the cup
    ex.events.listeners.append(stall_in_cup)
    obs = ex.observe()
    eid = ex.submit(proposal(ex, obs, "stir", STIR))
    r = ex.wait(eid, 120)
    assert r.status == Status.PARTIAL, (r.status, r.termination)
    rec = [p for p in r.phases if p["name"] == "recover"][0]
    assert rec["status"] == "done" and "straight up" in rec["note"], rec
    t_abort = [p for p in r.phases if p["name"] == "execute"][0]["end"]
    zs = [kin.fk(kin.q_from_norm(pos))[0][2] for t, pos in fake.trace if t >= t_abort]
    assert min(zs) >= zs[0] - 0.002, "recovery must not descend"
    assert zs[-1] >= STIR["xyz_world"][2] + STIR["clearance"] - 0.01


def test_table_memory_is_keyed_by_setup(make_ex, kin, tmp_path):
    ex, fake = make_ex(table_z=0.008)
    obs = ex.observe()
    r = ex.wait(ex.submit(proposal(ex, obs, "touch_table", {"xy_world": [0.22, 0.03]})), 120)
    assert r.status == Status.SUCCEEDED, r.termination
    z = round(r.achieved["contact_tcp_z"], 4)
    assert abs(z - 0.008) < 0.003
    sid = ex.setup_id()
    assert ex.memory.lookup(sid, 0.22, 0.03) == z
    ex.set_tool("tissue")
    assert ex.memory.lookup(ex.setup_id(), 0.22, 0.03) is None
    assert ex.ctx().table_z(0.22, 0.03) == kin.P.touch.nominal_table_tcp_z
    again = TableMemory(tmp_path / "table.json", kin.P.touch)     # persisted, still keyed
    assert again.lookup(sid, 0.22, 0.03) == z


def test_pick_holds_object(make_ex):
    ex, fake = make_ex(object_norm=0.35)
    obs = ex.observe()
    r = ex.wait(ex.submit(proposal(ex, obs, "pick", {"xyz_world": [0.22, -0.04, 0.0], "object_height": 0.03})), 120)
    assert r.status == Status.SUCCEEDED, (r.termination, r.phases)
    assert r.achieved["holding"] is True
    o = ex.evaluate(r.execution_id)
    assert o.metric == "held" and o.success is True


def test_pick_on_empty_table_is_blocked_and_backs_off(make_ex):
    ex, fake = make_ex()
    obs = ex.observe()
    r = ex.wait(ex.submit(proposal(ex, obs, "pick", {"xyz_world": [0.22, -0.04, 0.0], "object_height": 0.03})), 120)
    assert r.status == Status.BLOCKED, r.termination
    assert [p for p in r.phases if p["name"] == "recover"][0]["status"] == "done"
    assert tcp(ex)[2] > 0.07
