"""Deterministic regressions for responses racing with a session replacement."""
from generaloid.comms.client import CommsClient


def test_stale_poll_cannot_advance_a_new_hub_cursor():
    c = CommsClient("http://127.0.0.1:1", "test-token", {"robot_id": "robot-a"})
    c.session, c.cursor = "old-session", 100
    calls = []

    def call(method, path, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            # A simultaneous send/rejoin reset the session while this old poll was in flight.
            with c.cond:
                c.session, c.cursor = "new-session", 0
                c.generation += 1
            return {"messages": [{"id": "old-message", "kind": "message", "payload": {}, "from": "robot-b"}],
                    "cursor": 101}
        assert kwargs["params"]["since"] == 0 and kwargs["auth"] == "new-session"
        c.stop_ev.set()
        return {"messages": [{"id": "new-message", "kind": "message", "payload": {}, "from": "robot-b"}],
                "cursor": 1}

    c._call = call
    try:
        c._poll_loop()
        assert c.cursor == 1
        assert [m["id"] for m in c.inbox()] == ["new-message"]
    finally:
        c.http.close()


def test_duplicate_401_does_not_rotate_an_already_replaced_session():
    c = CommsClient("http://127.0.0.1:1", "test-token", {"robot_id": "robot-a"})
    c.session = "new-session"
    def unexpected_join(*args, **kwargs):
        raise AssertionError("stale 401 must not trigger another join")
    c.join = unexpected_join
    try:
        c._rejoin("old-session")
        assert c.reconnects == 0
    finally:
        c.http.close()
