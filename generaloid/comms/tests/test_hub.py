"""The hub over real HTTP on localhost: auth, impersonation, idempotency, rate limits, timeouts, reconnects."""
import json
import time

import httpx
import pytest

from generaloid.comms import protocol as P
from generaloid.comms.client import CommsClient, CommsError
from generaloid.comms.hub import Hub, HubConfig, HubServer

TOKEN = "test-token-0123456789abcdef"


def fast_config(**kw):
    return HubConfig(sweep_s=0.05, max_wait_s=2.0, **kw)


@pytest.fixture
def hub(tmp_path):
    hs = HubServer(Hub(token=TOKEN, log_path=tmp_path / "hub.jsonl", config=fast_config())).start()
    yield hs
    hs.stop()


@pytest.fixture
def pair(hub):
    a = CommsClient(hub.url, TOKEN, {"robot_id": "so101-a", "connector": "so101", "skills": ["twist_cap"]}, poll_wait=1)
    b = CommsClient(hub.url, TOKEN, {"robot_id": "elrobot-b", "connector": "elrobot"}, poll_wait=1)
    a.join()
    b.join()
    yield a, b
    for c in (a, b):
        c.leave()


def raw(hub, method, path, token=None, **kw):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return httpx.request(method, hub.url + path, headers=headers, timeout=5, **kw)


def test_health_needs_no_auth_and_leaks_no_names(hub, pair):
    r = raw(hub, "GET", "/health")
    assert r.status_code == 200 and r.json()["peers_online"] == 2 and "so101-a" not in r.text


def test_join_with_wrong_or_missing_token_is_refused(hub):
    hello = {"hello": {"robot_id": "intruder-x"}}
    assert raw(hub, "POST", "/v1/join", json=hello).status_code == 401
    assert raw(hub, "POST", "/v1/join", token="wrong", json=hello).status_code == 401
    with pytest.raises(CommsError) as e:
        CommsClient(hub.url, "wrong", {"robot_id": "intruder-x"}).join()
    assert e.value.code == 401
    assert "intruder-x" not in [p["robot_id"] for p in hub.hub.roster(time.time())]


def test_every_control_endpoint_needs_a_session(hub, pair):
    for method, path in [("POST", "/v1/send"), ("GET", "/v1/inbox"), ("GET", "/v1/peers"), ("POST", "/v1/leave"),
                         ("GET", "/v1/requests"), ("GET", "/v1/log"), ("POST", "/v1/heartbeat"), ("GET", "/v1/stream")]:
        # the join token is not a session; after 5 failures the address is locked out (429)
        assert raw(hub, method, path, token=TOKEN).status_code in (401, 429), path


def test_auth_failures_lock_the_address_out(hub):
    for _ in range(hub.hub.config.auth_fail_max):
        assert raw(hub, "POST", "/v1/join", token="bad", json={"hello": {"robot_id": "x-1"}}).status_code == 401
    r = raw(hub, "POST", "/v1/join", token=TOKEN, json={"hello": {"robot_id": "x-1"}})
    assert r.status_code == 429                                 # even the right token, for a minute
    assert sum(1 for e in hub.hub.log if e["event"] == "auth_fail") == hub.hub.config.auth_fail_max


def test_lockout_does_not_cut_off_robots_already_in_session(hub, pair):
    a, b = pair
    for _ in range(hub.hub.config.auth_fail_max + 2):
        raw(hub, "POST", "/v1/join", token="bad", json={"hello": {"robot_id": "x-1"}})
    a.send("elrobot-b", "message", {"text": "still talking"})
    assert b.wait_for(lambda m: m["kind"] == "message", 5)["payload"]["text"] == "still talking"


def test_per_peer_keys(tmp_path):
    keys = {"so101-a": "key-a-0123456789abcdef", "elrobot-b": "key-b-0123456789abcdef"}
    hs = HubServer(Hub(peer_keys=keys, config=fast_config())).start()
    try:
        assert hs.hub.token is None
        CommsClient(hs.url, keys["so101-a"], {"robot_id": "so101-a"}).join(start_polling=False)
        with pytest.raises(CommsError) as e:                     # B's key cannot claim A... or an unknown id
            CommsClient(hs.url, keys["elrobot-b"], {"robot_id": "so101-x"}).join(start_polling=False)
        assert e.value.code == 401
        with pytest.raises(CommsError):
            CommsClient(hs.url, keys["so101-a"], {"robot_id": "elrobot-b"}).join(start_polling=False)
    finally:
        hs.stop()


def test_hub_stamps_from_so_peers_cannot_impersonate(hub, pair):
    a, b = pair
    env = {"v": 1, "id": P.new_id(), "kind": "message", "to": "elrobot-b", "from": "hub", "payload": {"text": "hi"}}
    r = raw(hub, "POST", "/v1/send", token=a.session, json=env)
    assert r.status_code == 200
    m = b.wait_for(lambda m: m["kind"] == "message", 5)
    assert m["from"] == "so101-a"


def test_duplicate_robot_id_is_refused_while_online(hub, pair):
    with pytest.raises(CommsError) as e:
        CommsClient(hub.url, TOKEN, {"robot_id": "so101-a"}).join(start_polling=False)
    assert e.value.code == 409


def test_sends_are_idempotent(pair):
    a, b = pair
    mid = P.new_id()
    first = a.send("elrobot-b", "message", {"text": "once"}, msg_id=mid)
    again = a.send("elrobot-b", "message", {"text": "once"}, msg_id=mid)
    assert again["duplicate"] and again["id"] == first["id"]
    time.sleep(1.5)
    assert len(b.inbox(0, kinds={"message"})) == 1


def test_send_rate_limit(tmp_path):
    hs = HubServer(Hub(token=TOKEN, config=fast_config(send_rate=1.0, send_burst=3))).start()
    try:
        a = CommsClient(hs.url, TOKEN, {"robot_id": "so101-a"}, poll_wait=1)
        b = CommsClient(hs.url, TOKEN, {"robot_id": "elrobot-b"}, poll_wait=1)
        a.join(); b.join()
        for i in range(3):
            a.send("elrobot-b", "message", {"text": str(i)})
        with pytest.raises(CommsError) as e:
            a.send("elrobot-b", "message", {"text": "too many"})
        assert e.value.code == 429
        time.sleep(1.1)
        a.send("elrobot-b", "message", {"text": "refilled"})
        a.leave(); b.leave()
    finally:
        hs.stop()


def test_body_size_cap(hub, pair):
    a, _ = pair
    big = {"v": 1, "id": P.new_id(), "kind": "message", "to": "elrobot-b", "payload": {"text": "x" * 600_000}}
    assert raw(hub, "POST", "/v1/send", token=a.session, content=json.dumps(big)).status_code == 413


def test_unknown_peer_and_self_message(pair):
    a, _ = pair
    with pytest.raises(CommsError) as e:
        a.send("nobody-here", "message", {"text": "hi"})
    assert e.value.code == 404
    with pytest.raises(CommsError):
        a.send("so101-a", "message", {"text": "me"})


def test_help_request_full_lifecycle_and_log(hub, pair, tmp_path):
    a, b = pair
    ack = a.help("elrobot-b", "cap does not turn", ask="hold the bottle", context={"strokes": 11}, ttl_s=30)
    rid = ack["id"]
    req = b.wait_for(lambda m: m["kind"] == "help", 5)
    assert req["id"] == rid and req["payload"]["context"] == {"strokes": 11}
    assert [m["id"] for m in b.waiting_for_me()] == [rid]
    with pytest.raises(CommsError):                              # A cannot accept its own request
        a.send("elrobot-b", "accept", {}, re=rid)
    b.reply(rid, "accept", {"eta_s": 5})
    b.reply(rid, "progress", {"fraction": 0.5, "stage": "holding"})
    done = b.reply(rid, "result", {"ok": True, "summary": "held for 20 s"})
    assert done["request"]["state"] == "done" and b.waiting_for_me() == []
    kinds = [m["kind"] for m in a.inbox(0, wait=0)]
    a.wait_for(lambda m: m["kind"] == "result", 5)
    assert [k for k in [m["kind"] for m in a.inbox(0)] if k in ("accept", "progress", "result")] == \
        ["accept", "progress", "result"], kinds
    assert [r["state"] for r in a.requests() if r["id"] == rid] == ["done"]
    log = [json.loads(line) for line in (tmp_path / "hub.jsonl").read_text().splitlines()]
    assert [e["kind"] for e in log if e.get("re") == rid or e.get("id") == rid] == ["help", "accept", "progress", "result"]
    assert any(e.get("kind") == "help" for e in a.log())


def test_request_timeout_notifies_both_sides_and_blocks_late_results(pair):
    a, b = pair
    rid = a.request("elrobot-b", "hold_bottle", ttl_s=1.0)["id"]
    b.wait_for(lambda m: m["kind"] == "request", 5)
    b.reply(rid, "accept")
    ea = a.wait_for(lambda m: m["kind"] == "expired" and m["re"] == rid, 5)
    b.wait_for(lambda m: m["kind"] == "expired" and m["re"] == rid, 5)
    assert ea["from"] == "hub" and ea["payload"]["request"]["state"] == "expired"
    with pytest.raises(CommsError) as e:
        b.reply(rid, "result", {"ok": True})
    assert e.value.code == 409
    assert b.waiting_for_me() == []


def test_cancel_reaches_the_assignee(pair):
    a, b = pair
    rid = a.request("elrobot-b", "hold_bottle", ttl_s=30)["id"]
    b.wait_for(lambda m: m["kind"] == "request", 5)
    b.reply(rid, "accept")
    a.cancel(rid, "cap came off by itself")
    m = b.wait_for(lambda m: m["kind"] == "cancel", 5)
    assert m["re"] == rid and m["to"] == "elrobot-b" and b.waiting_for_me() == []


def test_broadcast_help_first_acceptor(hub, pair):
    a, b = pair
    c = CommsClient(hub.url, TOKEN, {"robot_id": "robot-c"}, poll_wait=1)
    c.join()
    try:
        rid = a.help("*", "anyone free to hold a bottle?", ttl_s=30)["id"]
        b.wait_for(lambda m: m["id"] == rid, 5)
        c.wait_for(lambda m: m["id"] == rid, 5)
        b.reply(rid, "accept")
        with pytest.raises(CommsError) as e:
            c.reply(rid, "accept")
        assert e.value.code == 409
    finally:
        c.leave()


def test_observation_with_image_and_pose(pair):
    import base64
    a, b = pair
    jpeg = b"\xff\xd8\xff\xe0" + b"0" * 100
    a.send("elrobot-b", "observation", {"images": [{"mime": "image/jpeg", "data_b64": base64.b64encode(jpeg).decode(),
                                                    "camera": "wrist"}],
                                        "poses": [{"name": "bottle", "frame": "world", "xyz": [0.21, 0.02, 0.0]}]})
    m = b.wait_for(lambda m: m["kind"] == "observation", 5)
    assert base64.b64decode(m["payload"]["images"][0]["data_b64"]) == jpeg
    assert m["payload"]["poses"][0]["xyz"] == [0.21, 0.02, 0.0]


def test_peer_join_and_leave_events(hub, pair):
    a, b = pair
    c = CommsClient(hub.url, TOKEN, {"robot_id": "robot-c", "skills": ["pour"]}, poll_wait=1)
    c.join()
    j = a.wait_for(lambda m: m["kind"] == "peer_joined" and m["payload"]["peer"]["robot_id"] == "robot-c", 5)
    assert j["payload"]["peer"]["skills"] == ["pour"]
    c.leave()
    a.wait_for(lambda m: m["kind"] == "peer_left" and m["payload"]["robot_id"] == "robot-c", 5)
    assert "robot-c" not in [p["robot_id"] for p in a.peers()]


def test_reconnect_after_session_loss_resumes_the_mailbox(hub, pair):
    """The hub forgets B's session (as after a network split long enough for a proxy to drop it). Messages sent to B
    meanwhile wait in its mailbox; B's poller gets 401, rejoins with resume and receives them."""
    a, b = pair
    old = b.session
    hub.hub.sessions.clear()                                     # every session invalid, mailboxes kept
    a_session = a.session
    rid = None
    deadline = time.time() + 10
    while rid is None and time.time() < deadline:                # A's poller may also have to rejoin first
        try:
            rid = a.help("elrobot-b", "still there?", ttl_s=30)["id"]
        except CommsError:
            time.sleep(0.2)
    m = b.wait_for(lambda m: m["id"] == rid, 10)
    assert m["kind"] == "help" and b.session != old and b.reconnects >= 1 and a.session != a_session
    assert [e["resumed"] for e in hub.hub.log if e["event"] == "join" and e["robot_id"] == "elrobot-b"][-1] is True


def test_reconnect_after_hub_restart(tmp_path):
    """The hub process dies and comes back on the same port with empty state: clients notice, back off, rejoin
    fresh (not resumed) and carry on."""
    hs = HubServer(Hub(token=TOKEN, config=fast_config())).start()
    port = hs.port
    a = CommsClient(hs.url, TOKEN, {"robot_id": "so101-a"}, poll_wait=1)
    b = CommsClient(hs.url, TOKEN, {"robot_id": "elrobot-b"}, poll_wait=1)
    a.join(); b.join()
    hs.stop()
    time.sleep(1.0)
    hs2 = HubServer(Hub(token=TOKEN, config=fast_config()), port=port).start()
    try:
        deadline = time.time() + 20
        while time.time() < deadline and len(hs2.hub.peers) < 2:
            time.sleep(0.1)
        assert set(hs2.hub.peers) == {"so101-a", "elrobot-b"}
        a.send("elrobot-b", "message", {"text": "back"})
        assert b.wait_for(lambda m: m["kind"] == "message", 5)["payload"]["text"] == "back"
        assert a.reconnects >= 1 and b.reconnects >= 1
    finally:
        a.leave(); b.leave(); hs2.stop()


def test_stale_peer_is_dropped_and_its_id_freed(tmp_path):
    hs = HubServer(Hub(token=TOKEN, config=fast_config(peer_stale_s=0.3, peer_drop_s=0.6))).start()
    # A long-polls for 1 s at a time, longer than peer_drop_s: an open poll counts as alive, so only B drops.
    try:
        a = CommsClient(hs.url, TOKEN, {"robot_id": "so101-a"}, poll_wait=1)
        a.join()
        crashed = CommsClient(hs.url, TOKEN, {"robot_id": "elrobot-b"})
        crashed.join(start_polling=False)                        # joins, then never calls again
        a.wait_for(lambda m: m["kind"] == "peer_left" and m["payload"]["reason"] == "timeout", 5)
        CommsClient(hs.url, TOKEN, {"robot_id": "elrobot-b"}).join(start_polling=False)   # id is free again
        a.leave()
    finally:
        hs.stop()


def test_sse_stream_delivers_the_mailbox(hub, pair):
    a, b = pair
    probe = CommsClient(hub.url, TOKEN, {"robot_id": "dash-1"})
    probe.join(start_polling=False)
    a.send("dash-1", "status", {"state": "twisting", "note": "stroke 4/12"})
    with httpx.stream("GET", hub.url + "/v1/stream", headers={"Authorization": f"Bearer {probe.session}"},
                      timeout=5) as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                m = json.loads(line[6:])
                if m["kind"] == "status":
                    assert m["payload"]["state"] == "twisting" and m["from"] == "so101-a"
                    break
    probe.leave()
