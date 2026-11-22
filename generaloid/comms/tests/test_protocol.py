"""Schema validation and the request state machine (pure, no network)."""
import pytest

from generaloid.comms import protocol as P


def env(kind, to="elrobot-b", payload=None, re=None, **kw):
    return {"v": 1, "id": P.new_id(), "kind": kind, "to": to, "re": re, "payload": payload or {}, **kw}


def test_valid_envelopes():
    P.validate_envelope(env("request", payload={"task": "hold_bottle", "args": {"x": 0.2}, "ttl_s": 60}))
    P.validate_envelope(env("help", to="*", payload={"problem": "cap stuck", "context": {"strokes": 12}}))
    P.validate_envelope(env("observation", payload={"poses": [{"name": "bottle", "frame": "world", "xyz": [0.2, 0, 0.1]}]}))
    P.validate_envelope(env("accept", re=P.new_id()))
    P.validate_envelope(env("result", re=P.new_id(), payload={"ok": False, "summary": "slipped"}))
    P.validate_envelope(env("cancel", to="*", re=P.new_id()))


@pytest.mark.parametrize("msg, needle", [
    (env("request", payload={}), "task"),
    (env("request", to="*", payload={"task": "x"}), "one robot"),
    (env("accept"), "re"),
    (env("result", re=P.new_id(), payload={}), "ok"),
    (env("progress", re=P.new_id(), payload={"fraction": 1.5}), "fraction"),
    (env("request", payload={"task": "x", "ttl_s": 99999}), "ttl_s"),
    (env("observation", payload={}), "observation needs"),
    (env("observation", payload={"poses": [{"name": "b", "frame": "world", "xyz": [1, 2]}]}), "xyz"),
    (env("observation", payload={"images": [{"data_b64": "xx", "mime": "text/html"}]}), "mime"),
    (env("teleport", payload={}), "kind"),
    (env("message", to="Robot B!", payload={"text": "hi"}), "to must"),
    ({**env("message", payload={"text": "hi"}), "id": "short"}, "id must"),
    ({**env("message", payload={"text": "hi"}), "v": 2}, "version"),
    (env("message", payload={"text": "x" * (P.MAX_TEXT + 1)}), "longer"),
])
def test_invalid_envelopes(msg, needle):
    with pytest.raises(P.ProtocolError, match=needle):
        P.validate_envelope(msg)


def test_version_mismatch_is_426():
    with pytest.raises(P.ProtocolError) as e:
        P.validate_hello({"robot_id": "so101-a", "protocol_versions": [7]})
    assert e.value.code == 426


def test_hello_normalises_and_checks_ids():
    prof = P.validate_hello({"robot_id": "so101-a", "connector": "so101", "skills": ["twist_cap"],
                             "frames": {"world": "table", "base_in_world": {"frame": "world", "xyz": [0, 0.3, 0]}}})
    assert prof["robot_id"] == "so101-a" and prof["skills"] == ["twist_cap"] and prof["capabilities"] == []
    for bad in ("hub", "A", "x", "so 101", ""):
        with pytest.raises(P.ProtocolError):
            P.validate_hello({"robot_id": bad})


def _open(t, kind="help", to="elrobot-b", ttl=10):
    m = P.validate_envelope(env(kind, to=to, payload={"problem": "p", "ttl_s": ttl} if kind == "help"
                                else {"task": "t", "ttl_s": ttl}))
    return t.open(m, "so101-a", 100.0)


def _reply(t, r, kind, sender, payload=None, to="so101-a", now=101.0):
    return t.apply(P.validate_envelope(env(kind, to=to, re=r.id, payload=payload or ({"ok": True} if kind == "result"
                                                                                  else {"reason": "no"} if kind == "reject" else {}))),
                   sender, now)


def test_state_machine_happy_path():
    t = P.RequestTracker()
    r = _open(t)
    assert r.deadline == 110.0 and r.state == P.OPEN
    _reply(t, r, "accept", "elrobot-b")
    _reply(t, r, "progress", "elrobot-b", {"fraction": 0.4})
    assert r.progress == 0.4
    _reply(t, r, "result", "elrobot-b")
    assert r.state == P.DONE and [h[2] for h in r.history] == ["help", "accept", "progress", "result"]


def test_state_machine_refusals():
    t = P.RequestTracker()
    r = _open(t)
    with pytest.raises(P.ProtocolError) as e:
        _reply(t, r, "progress", "elrobot-b")              # before accept
    assert e.value.code == 409
    with pytest.raises(P.ProtocolError) as e:
        _reply(t, r, "accept", "intruder-c")               # not the addressee
    assert e.value.code == 403
    with pytest.raises(P.ProtocolError):
        _reply(t, r, "accept", "elrobot-b", to="intruder-c")  # reply not addressed to the requester
    _reply(t, r, "accept", "elrobot-b")
    with pytest.raises(P.ProtocolError) as e:
        _reply(t, r, "accept", "elrobot-b")                # twice
    assert e.value.code == 409
    with pytest.raises(P.ProtocolError) as e:              # only the requester cancels
        t.apply(P.validate_envelope(env("cancel", to="so101-a", re=r.id)), "elrobot-b", 102)
    assert e.value.code == 403
    t.apply(P.validate_envelope(env("cancel", to="elrobot-b", re=r.id)), "so101-a", 102)
    assert r.state == P.CANCELLED
    with pytest.raises(P.ProtocolError) as e:              # terminal stays terminal
        _reply(t, r, "result", "elrobot-b")
    assert e.value.code == 409


def test_broadcast_help_first_acceptor_wins_and_rejects_keep_it_open():
    t = P.RequestTracker()
    r = _open(t, to="*")
    _reply(t, r, "reject", "robot-c")
    assert r.state == P.OPEN
    _reply(t, r, "accept", "elrobot-b")
    assert r.assignee == "elrobot-b"
    with pytest.raises(P.ProtocolError, match="already accepted"):
        _reply(t, r, "accept", "robot-c")
    with pytest.raises(P.ProtocolError):
        _reply(t, r, "progress", "robot-c")


def test_expiry_and_prune():
    t = P.RequestTracker()
    r = _open(t, ttl=5)
    assert t.expire(104.9) == []
    assert t.expire(105.0) == [r] and r.state == P.EXPIRED
    assert t.expire(200.0) == []                           # only once
    t.prune(105.0 + 601, keep_s=600)
    assert r.id not in t.requests


@pytest.mark.parametrize("msg", [
    {**env("accept", re=P.new_id()), "kind": []},
    {**env("accept", re=P.new_id()), "v": True},
    {**env("accept", re=P.new_id()), "payload": []},
    env("observation", payload={"images": [{"mime": "image/png", "data_b64": "%%%"}]}),
    env("observation", payload={"poses": [{"name": "b", "frame": "world", "xyz": [float("inf"), 0, 0]}]}),
    env("request", payload={"task": "t", "args": {"x": float("nan")}}),
])
def test_malformed_shapes_and_nonfinite_values_are_protocol_errors(msg):
    with pytest.raises(P.ProtocolError):
        P.validate_envelope(msg)


def test_hello_rejects_malformed_transform():
    with pytest.raises(P.ProtocolError, match="base_in_world"):
        P.validate_hello({"robot_id": "so101-a", "frames": {"base_in_world": [1, 2, 3]}})


def test_oversized_integer_pose_and_reach_are_rejected_cleanly():
    huge = 10 ** 400
    with pytest.raises(P.ProtocolError):
        P.validate_pose({"name": "b", "frame": "world", "xyz": [huge, 0, 0]})
    with pytest.raises(P.ProtocolError):
        P.validate_hello({"robot_id": "robot-a", "reach": {"max_radius": huge}})
