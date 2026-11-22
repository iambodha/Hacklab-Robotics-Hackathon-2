"""Console: scripted advisor, overlay projection, HTTP API, CLI, page and launcher sanity (fake backend only)."""
import re
import socket
import subprocess
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from executor import __main__ as cli
from executor.console.demo_scene import DemoScene
from executor.console.scripted import ScriptedAdvisor, target_colour

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def scripted(ex, task="Put the blue object in the bin"):
    adv = ScriptedAdvisor("", ex.events, {}, {}, executor=ex)
    adv.task = task
    return adv


def kinds(ex, kind):
    return [e for e in ex.events.since(0, 5000) if e["kind"] == kind]


def test_target_colour_from_prompt():
    assert target_colour("Put the blue object in the bin") == "blue"
    assert target_colour("move the RED cube to the box") == "red"
    assert target_colour("tidy up") == "blue"


def test_scripted_puts_blue_object_in_bin(make_ex):
    ex, fake = make_ex(object_norm=0.3)
    ex.locator = DemoScene()
    why = scripted(ex).run()
    assert why.startswith("DONE"), why
    results = [(e["skill"], e["status"]) for e in kinds(ex, "result")]
    assert results == [("pick", "SUCCEEDED"), ("place", "SUCCEEDED"), ("park", "SUCCEEDED")]
    outcomes = {e["metric"]: e["success"] for e in kinds(ex, "outcome")}
    assert outcomes.get("held") is True and outcomes.get("release_error_mm") is True
    calls = [e["tool"] for e in kinds(ex, "advisor_tool_call")]
    assert calls[:2] == ["observe", "submit"] and "evaluate" in calls
    place = next(e["proposal"] for e in kinds(ex, "proposal") if e["proposal"]["skill"] == "place")
    assert place["params"]["xyz_world"][:2] == [0.06, -0.24] and place["params"]["xyz_world"][2] >= 0.15


def test_scripted_blocks_without_a_target_and_never_proposes(make_ex):
    ex, fake = make_ex()
    why = scripted(ex).run()
    assert why.startswith("BLOCKED") and "blue" in why
    assert not kinds(ex, "proposal") and not fake.motion_writes()


def test_scripted_retries_a_failed_pick_then_gives_up(make_ex):
    ex, fake = make_ex()                          # no object_norm: the jaws close on nothing
    ex.locator = DemoScene()
    why = scripted(ex).run()
    assert why == "BLOCKED: pick failed 3 times"
    assert [e["status"] for e in kinds(ex, "result")] == ["BLOCKED"] * 3


def test_overlay_projects_grid_estimates_and_last_path(make_ex):
    from executor.console.overlay import build
    ex, _ = make_ex(object_norm=0.3)
    ex.locator = DemoScene()
    obs = ex.observe()
    console = SimpleNamespace(ex=ex, estimates=lambda: obs.objects)
    o = build(console)
    if o["camera"] is None:
        pytest.skip("no overview camera model (calibration/scene_calib.json)")
    assert o["grid"] and any(p for g in o["grid"] for p in g["pts"])
    target = next(x for x in o["objects"] if x["kind"] == "target")
    assert target["uv"] is not None and 0 <= target["uv"][0] <= 1 and 0 <= target["uv"][1] <= 1
    assert next(x for x in o["objects"] if x["kind"] == "bin")["box"]
    assert o["path"] is None
    scripted(ex).run()
    o = build(console)
    assert o["path"]["state"] == "last" and o["path"]["skill"] == "park" and o["path"]["segments"]


def test_sim_frames_are_rendered_on_one_thread():
    from concurrent.futures import ThreadPoolExecutor
    from executor.backends.sim import SimBackend
    seen = []
    b = SimBackend.__new__(SimBackend)          # no twin: only the render routing is under test
    b.render_pool = ThreadPoolExecutor(max_workers=1)
    b._render = lambda: seen.append(threading.current_thread().name) or {}
    ts = [threading.Thread(target=b.frames) for _ in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(seen) == 6 and len(set(seen)) == 1


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def console_url(kin):
    from executor.console.server import Console, make_handler
    port = _free_port()
    c = Console("fake", port, local_rec_port=_free_port())
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(c))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        body = r.read()
    return body if r.headers.get_content_type() == "image/jpeg" else body.decode()


def test_page_and_health(console_url):
    import json
    page = _get(console_url + "/")
    for text in ("Robot logic", "Live view", "Put the blue object in the bin", "3D arm"):
        assert text in page
    h = json.loads(_get(console_url + "/api/health"))
    assert {"station", "recorder", "cameras", "arm", "backend", "advisor", "setup_id", "mirror"} <= set(h)
    assert h["backend"]["name"] == "fake" and h["arm"]["calibrated"]
    assert h["advisor"]["available"]["scripted"]
    _get(console_url + "/api/cam/overview.jpg")
    time.sleep(0.3)
    _get(console_url + "/api/cam/overview.jpg")
    assert json.loads(_get(console_url + "/api/health"))["cameras"]["overview"]["fps"]


def test_cli_against_the_console(console_url, monkeypatch, capsys):
    monkeypatch.setattr(cli, "URL", console_url)
    cli.main(["status"])
    out = capsys.readouterr().out
    assert "backend      fake" in out and "setup" in out
    cli.main(["observe"])
    out = capsys.readouterr().out
    assert "blue object" in out and "bin" in out and "world frame" in out
    cli.main(["skills"])
    assert "pick (v" in capsys.readouterr().out
    cli.main(["cancel"])
    assert "nothing is moving" in capsys.readouterr().out
    # The Gemini refusal path needs "no key": hide any real key (env or the hackathon secrets.env).
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("MODELS_SECRETS_FILE", "/nonexistent/secrets.env")
    with pytest.raises(SystemExit) as e:
        cli.main(["task", "x", "--advisor", "gemini", "--no-follow"])
    assert "GEMINI_API_KEY" in str(e.value) or "console refused" in str(e.value)


def test_cli_reports_a_missing_console(monkeypatch):
    monkeypatch.setattr(cli, "URL", f"http://127.0.0.1:{_free_port()}")
    with pytest.raises(SystemExit) as e:
        cli.main(["status"])
    assert "no executor console" in str(e.value)


def test_event_lines_cover_the_logic():
    ev = lambda **d: {"t": time.time(), **d}  # noqa: E731
    assert "REJECTED: expired" in cli.event_line(ev(kind="validation", ok=False, reasons=["expired"], checks={}))
    assert "outcome" in cli.event_line(ev(kind="outcome", metric="held", value=1, unit="bool", success=True))
    assert cli.event_line(ev(kind="phase", name="execute", status="running")) is None
    assert cli.event_line(ev(kind="observation")) is None


def test_page_svg_has_no_unclosed_elements():
    """An unquoted attribute value right before '/>' (fill=none/>) swallows the '/' and leaves the SVG element
    open, hiding everything drawn after it."""
    html = (HERE.parent / "index.html").read_text()
    assert not re.findall(r"=[\w#.-]+/>", html)


def test_stack_sh_parses_and_documents_itself():
    sh = REPO / "scripts" / "stack.sh"
    assert subprocess.run(["bash", "-n", str(sh)]).returncode == 0
    out = subprocess.run(["bash", str(sh), "--help"], capture_output=True, text=True).stdout
    assert "up" in out and "--station" in out and "--real-motion" in out


def test_3d_arm_inputs(console_url):
    """The 3D view's data: joint angles for the URDF (measured and commanded), and asset routes that stay in bounds."""
    import json
    import urllib.error
    import urllib.request
    live = json.loads(_get(console_url + "/api/live"))
    assert len(live["q"]) == len(live["q_goal"]) == len(live["joints"]) == 8
    assert all(abs(a - b) < 0.2 for a, b in zip(live["q"], live["q_goal"]))      # idle: commanded pose = measured pose
    assert "ColladaLoader" in _get(console_url + "/vendor/ColladaLoader.js")     # stub: served without a download
    for bad in ("/vendor/nope.js", "/robot/..%2F..%2FREADME.md", "/robot/assets/missing.stl"):
        try:
            urllib.request.urlopen(console_url + bad, timeout=5)
            raise AssertionError(f"{bad} was served")
        except urllib.error.HTTPError as e:
            assert e.code == 404


def test_voice_token_and_sim_only_start(console_url, monkeypatch):
    """Voice: no key = unavailable; the token request carries the whole setup and never the key in the reply;
    a spoken start is refused on the real backend."""
    import io
    import json
    import types
    import pytest
    from executor.console import voice
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("MODELS_SECRETS_FILE", "/nonexistent/secrets.env")
    assert voice.describe()["available"] is False
    with pytest.raises(voice.VoiceError) as e:
        voice.token()
    assert e.value.code == 503
    assert json.loads(_get(console_url + "/api/voice"))["start_backends"] == ["fake", "sim"]

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    seen = {}

    class Reply(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout=0):
        seen["body"], seen["key"], seen["url"] = json.loads(req.data), req.get_header("X-goog-api-key"), req.full_url
        return Reply(json.dumps({"name": "auth_tokens/abc"}).encode())

    out = voice.token(opener)
    assert out["token"] == "auth_tokens/abc" and "test-key-not-real" not in json.dumps(out) and "key=" not in seen["url"]
    assert seen["key"] == "test-key-not-real" and seen["body"]["uses"] == 1
    setup = seen["body"]["bidiGenerateContentSetup"]
    assert {d["name"] for d in setup["tools"][0]["functionDeclarations"]} == {"set_task", "start_task", "stop", "get_status"}

    events, started = [], []
    fake = types.SimpleNamespace(ex=types.SimpleNamespace(backend=types.SimpleNamespace(name="real")),
                                 events=types.SimpleNamespace(add=lambda kind, **kw: events.append((kind, kw))),
                                 start_task=lambda *a: started.append(a) or {"advisor": a[1], "backend": a[3]})
    with pytest.raises(voice.VoiceError) as e:
        voice.start_by_voice(fake, {"prompt": "Put the blue object in the bin"})
    assert e.value.code == 403 and not started and events
    fake.ex.backend.name = "sim"
    assert voice.start_by_voice(fake, {"prompt": "Park", "advisor": "claude"}) == {"advisor": "claude", "backend": "sim"}

