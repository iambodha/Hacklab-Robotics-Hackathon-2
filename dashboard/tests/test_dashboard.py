import json
import threading
import urllib.error
import urllib.request

import pytest

from dashboard.data import AgentStore, LearnStore, build_attempts, trace_columns
from dashboard.fixtures import generate
from dashboard.server import App, make_server


def get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=5) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


@pytest.fixture
def served(tmp_path):
    learn = tmp_path / "learn"
    actions = tmp_path / "actions.jsonl"
    oplog = tmp_path / "op.md"
    app = App(LearnStore(learn), AgentStore(actions, str(oplog)), recorder_url="http://127.0.0.1:9", relay_url="http://x")
    srv = make_server(app, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", learn, actions, oplog
    srv.shutdown()


def test_empty_everything(served):
    base, *_ = served
    code, _, body = get(base, "/api/state")
    st = json.loads(body)
    assert code == 200 and st["attempts"] == [] and st["exists"] is False and st["summary"] is False
    st = json.loads(get(base, "/api/agent")[2])
    assert st["actions"] == [] and st["operator"] is None
    assert get(base, "/api/live")[0] == 503
    code, hdr, html = get(base, "/")
    assert code == 200 and b"Learning across attempts" in html
    assert get(base, "/static/app.js")[0] == 200


def test_ten_attempts_and_versioning(served):
    base, learn, *_ = served
    generate(learn, 10)
    st = json.loads(get(base, "/api/state")[2])
    A = st["attempts"]
    assert [a["attempt"] for a in A] == list(range(1, 11)) and all(a["done"] for a in A)
    assert A[1]["changed"] == ["approach_z_offset_mm"] and A[0]["changed"] == []
    assert A[-1]["outcome"] == "opened" and A[-1]["peak_j7_load"] > 0 and A[1]["slip_events"] == 1
    assert "before_wrist.jpg" in A[0]["files"] and st["summary"] and st["reports"] == list(range(1, 11))
    assert json.loads(get(base, "/api/state?v=" + st["version"])[2]) == {"unchanged": True, "version": st["version"]}
    with open(learn / "episodes.jsonl", "a") as f:
        f.write(json.dumps({"type": "episode_begin", "attempt": 11, "t": 1, "params": {"strokes": 8}}) + "\n")
    st2 = json.loads(get(base, "/api/state?v=" + st["version"])[2])
    assert st2["attempts"][-1]["attempt"] == 11 and not st2["attempts"][-1]["done"]
    tr = json.loads(get(base, "/api/trace/3")[2])
    assert {"t", "j7_load", "j7_pos"} <= set(tr["columns"])
    assert get(base, "/api/trace/11")[0] == 404
    code, hdr, body = get(base, "/files/SUMMARY.md?download=1")
    assert code == 200 and "attachment" in hdr["Content-Disposition"] and body.startswith(b"# Bottle")
    assert get(base, "/files/attempt_2/report.md")[0] == 200


def test_partial_file_and_path_traversal(served):
    base, learn, *_ = served
    generate(learn, 4, partial=True)
    A = json.loads(get(base, "/api/state")[2])["attempts"]
    assert len(A) == 4 and not A[-1]["done"] and A[-1]["phase"]
    for p in ("/files/../actions.jsonl", "/files/%2e%2e/actions.jsonl", "/files/attempt_1/../../op.md"):
        assert get(base, p)[0] == 404


def test_agent_stream(served):
    base, _, actions, oplog = served
    actions.write_text("\n".join(json.dumps({"t": f"01:0{i}:00", "tool": "look", "args": {}, "s": 0.1, "error": None,
                                             "result": "x" * 5000}) for i in range(3)) + "\n{\"t\": \"tor")
    oplog.write_text("## User\nopen it\n\n## Assistant\nFirst idea.\n\n## User\ngo\n\n## Assistant\nI will lower the jaws by 8 mm.\n")
    st = json.loads(get(base, "/api/agent")[2])
    assert len(st["actions"]) == 3 and len(st["actions"][0]["result"]) < 2000
    assert st["operator"]["latest"] == "I will lower the jaws by 8 mm."


def test_trace_shapes():
    assert trace_columns({"samples": [{"t": 0, "a": 1}, {"t": 1, "b": 2}]}) == {"t": [0, 1], "a": [1, None], "b": [None, 2]}
    assert trace_columns({"t": [0, 1], "j7_load": [3, 4], "meta": "x"}) == {"t": [0, 1], "j7_load": [3, 4]}
    assert trace_columns(None) == {}


def test_build_attempts_ignores_junk():
    A = build_attempts([{"type": "phase", "attempt": "x"}, {"type": "phase", "attempt": 2, "phase": "twist", "status": "ok"}])
    assert len(A) == 1 and A[0]["attempt"] == 2 and A[0]["phase"] == "twist" and A[0]["params"] == {}
