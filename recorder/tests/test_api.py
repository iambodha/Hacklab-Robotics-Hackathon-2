import json
import threading
import urllib.error
import urllib.request

import pytest

from recorder.server import Service, make_server

from .conftest import BUS, CAM, build


@pytest.fixture
def api(tmp_path):
    root = tmp_path / "recordings"
    rec = build(root)
    svc = Service(root=root)
    httpd = make_server(svc, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", rec.name, svc
    httpd.shutdown()
    httpd.server_close()


def call(base, method, path, body=None, headers=None):
    req = urllib.request.Request(base + path, method=method, headers={"Content-Type": "application/json", **(headers or {})},
                                 data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            data = r.read()
            return r.status, dict(r.headers), (json.loads(data) if r.headers["Content-Type"] == "application/json" else data)
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), json.loads(e.read())


def test_health_and_live_without_capture(api):
    base, _, _ = api
    code, _, h = call(base, "GET", "/health")
    assert code == 200 and h["station_reachable"] is False and h["recording"] is None
    assert call(base, "GET", "/live")[0] == 503
    assert call(base, "GET", "/live/frame?cam=28ff0ca8")[0] == 404


def test_recording_lifecycle(api):
    base, existing, svc = api
    code, _, r = call(base, "POST", "/recordings", {"name": "calib run", "note": "n"})
    assert code == 201 and r["id"].endswith("_calib-run")
    rid = r["id"]
    assert call(base, "POST", "/recordings", {"name": "x"})[0] == 409
    code, _, m = call(base, "POST", f"/recordings/{rid}/markers", {"label": "calibration start", "data": {"j": 1}})
    assert code == 201 and m["event"]["type"] == "marker" and m["event"]["t"] >= 0
    assert call(base, "POST", "/recordings/active/markers", {"label": "second"})[0] == 201
    assert call(base, "POST", f"/recordings/{rid}/markers", {})[0] == 400
    assert call(base, "POST", f"/recordings/{existing}/markers", {"label": "x"})[0] == 409
    code, _, lst = call(base, "GET", "/recordings")
    assert lst["active"] == rid and {m["id"] for m in lst["recordings"]} == {rid, existing}
    assert call(base, "GET", "/health")[2]["recording"]["id"] == rid
    code, _, s = call(base, "POST", f"/recordings/{rid}/stop")
    assert code == 200 and s["session"]["stopped"] and s["session"]["counts"]["markers"] == 2
    assert call(base, "POST", f"/recordings/{rid}/stop")[0] == 404
    code, _, d = call(base, "GET", f"/recordings/{rid}")
    assert [e["label"] for e in d["markers"]] == ["calibration start", "second"]


def test_read_endpoints(api):
    base, rid, _ = api
    code, _, d = call(base, "GET", f"/recordings/{rid}")
    assert code == 200 and d["session"]["id"] == rid and d["summary"]["state_rows"] == 100
    assert d["summary"]["cameras"][CAM[:8]]["fps"] == pytest.approx(10, rel=0.01)
    _, _, st = call(base, "GET", f"/recordings/{rid}/state?from=0.5&to=1.0&every=0.1")
    assert st["count"] == 6 and "regs" not in st["rows"][0]["buses"][BUS]["j1"]
    _, _, st = call(base, "GET", f"/recordings/{rid}/state?to=0.02&regs=1")
    assert "regs" in st["rows"][0]["buses"][BUS]["j1"]
    _, _, at = call(base, "GET", f"/recordings/{rid}/at?t=0.715")
    assert at["state"]["t"] == pytest.approx(0.72) and at["frames"][CAM[:8]]["n"] == 7
    code, hdr, jpg = call(base, "GET", f"/recordings/{rid}/frame?cam=wrist&t=0.71")
    assert code == 200 and hdr["Content-Type"] == "image/jpeg" and jpg.startswith(b"\xff\xd8") and hdr["X-Frame-N"] == "7"
    code, hdr, jpg = call(base, "GET", f"/recordings/{rid}/frames/{CAM[:8]}/3.jpg")
    assert code == 200 and b"-3" in jpg and "immutable" in hdr["Cache-Control"]
    _, _, ix = call(base, "GET", f"/recordings/{rid}/index")
    assert len(ix["cameras"][CAM[:8]]["t"]) == 20
    _, _, ev = call(base, "GET", f"/recordings/{rid}/events")
    assert ev["events"][0]["type"] == "start"
    code, hdr, html = call(base, "GET", f"/recordings/{rid}/viewer")
    assert code == 200 and hdr["Content-Type"].startswith("text/html") and b"<canvas" in html
    assert call(base, "GET", "/recordings/nope/viewer")[0] == 404
    assert call(base, "GET", f"/recordings/{rid}/at?t=abc")[0] == 400
    assert call(base, "GET", f"/recordings/{rid}/frame?cam=zzz&t=1")[0] == 404


def test_cors(api):
    base, _, _ = api
    _, hdr, _ = call(base, "GET", "/health", headers={"Origin": "http://localhost:5173"})
    assert hdr.get("Access-Control-Allow-Origin") == "http://localhost:5173"
    _, hdr, _ = call(base, "GET", "/health", headers={"Origin": "https://evil.example"})
    assert "Access-Control-Allow-Origin" not in hdr
    req = urllib.request.Request(base + "/recordings", method="OPTIONS", headers={"Origin": "http://127.0.0.1:3000"})
    with urllib.request.urlopen(req, timeout=5) as r:
        assert r.status == 204 and "POST" in r.headers["Access-Control-Allow-Methods"]
