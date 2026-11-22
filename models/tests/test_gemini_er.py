import json

import pytest

from models import _secrets
from models.gemini_er import (FakeTransport, GeminiER, GeminiError, GenAITransport, Image, ParseError, cost_usd,
                              image_size, parse_json)

from .conftest import FAKE_KEY, tiny_jpeg


def er_with(replies, log_dir):
    return GeminiER(transport=FakeTransport(replies), log_dir=log_dir)


def test_image_size_jpeg_and_png(jpeg):
    assert image_size(jpeg) == (299, 224)
    png = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + (640).to_bytes(4, "big") + (480).to_bytes(4, "big")
    assert image_size(png) == (640, 480)


def test_parse_json_variants():
    assert parse_json('[{"point": [1, 2]}]') == [{"point": [1, 2]}]
    assert parse_json('Sure:\n```json\n{"done": true}\n```') == {"done": True}
    assert parse_json('Here you go: [{"a": 1}] hope it helps') == [{"a": 1}]
    with pytest.raises(ParseError):
        parse_json("no json here")


def test_points_are_y_first_and_scaled_to_pixels(jpeg, log_dir):
    er = er_with(['[{"point": [500, 250], "label": "cup"}, {"point": [0, 1000], "label": "corner"}, {"label": "junk"}]'],
                 log_dir)
    pts = er.point(Image(jpeg, role="overview"), ["cup"])
    assert [p.label for p in pts] == ["cup", "corner"]
    assert (pts[0].u, pts[0].v) == (pytest.approx(74.75), pytest.approx(112.0))   # x=250/1000*299, y=500/1000*224
    assert (pts[1].u, pts[1].v) == (299.0, 0.0) and pts[0].camera == "overview"


def test_boxes_accept_both_documented_formats(jpeg, log_dir):
    er = er_with(['[{"box_2d": [100, 200, 300, 400], "label": "a"}, {"y": 300, "x": 400, "y2": 100, "x2": 200, "label": "b"}]'],
                 log_dir)
    a, b = er.boxes(Image(jpeg, role="overview"))
    assert (a.u0, a.v0, a.u1, a.v1) == pytest.approx((59.8, 22.4, 119.6, 67.2))
    assert (b.u0, b.v0, b.u1, b.v1) == pytest.approx((59.8, 22.4, 119.6, 67.2))       # swapped corners sorted
    assert a.area_px == pytest.approx(59.8 * 44.8)


def test_judgement_parsing(jpeg, log_dir):
    er = er_with(['{"done": false, "confidence": 0.8, "failure": "stain left on the right", "evidence": "brown patch",'
                  ' "progress": "60-80", "camera": "overview", "residual": [{"box_2d": [400, 500, 600, 700], "label": "stain"}]}',
                  '{"done": null, "confidence": 0.2, "evidence": "arm covers the spot"}'], log_dir)
    imgs = [Image(jpeg, role="overview"), Image(tiny_jpeg(640, 480), role="wrist")]
    j = er.judge(imgs, "wipe the stain")
    assert j.done is False and j.failure.startswith("stain") and j.progress == "60-80"
    assert len(j.residual) == 1 and j.residual[0].camera == "overview" and j.residual[0].u0 == pytest.approx(149.5)
    j2 = er.judge(imgs, "wipe the stain")
    assert j2.done is None and j2.residual == []


def test_step_vocabulary_enforced(jpeg, log_dir):
    imgs = [Image(jpeg, role="overview"), Image(jpeg, role="wrist")]
    er = er_with(['{"action": "pick", "args": {"object": "purple wrapper", "point": [500, 500], "camera": "overview"},'
                  ' "rationale": "it is free", "confidence": 0.9}',
                  '{"action": "wipe_region", "args": {"box_2d": [400, 400, 600, 600], "camera": "overview"}}',
                  '{"action": "fly_away", "args": {}}',
                  '{"action": "pick", "args": {"object": "x"}}',
                  '{"action": "pick", "args": {"object": "x", "point": [1, 2], "camera": "side"}}'], log_dir)
    s = er.propose_step(imgs, "tidy up")
    assert s.action == "pick" and s.args["u"] == pytest.approx(149.5) and s.args["v"] == pytest.approx(112.0)
    w = er.propose_step(imgs, "wipe")
    assert w.args["box"].u0 == pytest.approx(119.6)
    for _ in range(3):
        with pytest.raises(ParseError):
            er.propose_step(imgs, "x")


def test_every_call_is_logged_with_usage_and_cost(jpeg, log_dir):
    er = er_with(['[{"point": [1, 1], "label": "a"}]', "not json"], log_dir)
    er.point(Image(jpeg, role="overview", ref="recording:abc@1.5"))
    with pytest.raises(ParseError):
        er.point(Image(jpeg, role="overview"))
    rows = [json.loads(l) for f in log_dir.glob("*.jsonl") for l in f.read_text().splitlines()]
    assert len(rows) == 2
    r = rows[0]
    assert r["kind"] == "point" and r["model"] == "gemini-robotics-er-2-preview"
    assert r["images"][0]["ref"] == "recording:abc@1.5" and r["images"][0]["w"] == 299 and "sha1" in r["images"][0]
    assert "Point to" in r["prompt"] and r["parsed"] == [{"point": [1, 1], "label": "a"}]
    assert r["usage"]["prompt_tokens"] == 1300 and r["cost_usd"] == pytest.approx(cost_usd(r["model"], r["usage"]))
    assert r["latency_s"] >= 0 and rows[1]["error"].startswith("parse")
    assert er.last.error.startswith("parse")


def test_cost():
    assert cost_usd("gemini-robotics-er-2-preview", {"prompt_tokens": 1_000_000, "output_tokens": 0}) == 1.0
    assert cost_usd("gemini-robotics-er-2-preview", {"prompt_tokens": 0, "output_tokens": 100_000,
                                                      "thinking_tokens": 100_000}) == 1.0
    assert cost_usd("gemini-robotics-er-2-preview", {}) is None


def test_key_never_reaches_logs_or_errors(jpeg, log_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    assert _secrets.gemini_key() == FAKE_KEY

    def leaky(model, parts, config):            # a transport echoing the key in the answer and a URL
        return f'[{{"point": [1, 1], "label": "{FAKE_KEY} https://x/?key={FAKE_KEY}"}}]'
    er = GeminiER(transport=FakeTransport(leaky), log_dir=log_dir)
    er.point(Image(jpeg, role="overview", ref=f"/tmp/{FAKE_KEY}.jpg"), [f"thing {FAKE_KEY}"])

    def boom(model, parts, config):
        raise RuntimeError(f"401 for url https://generativelanguage.googleapis.com/v1beta?key={FAKE_KEY}")
    er2 = GeminiER(transport=FakeTransport(boom), log_dir=log_dir)
    with pytest.raises(GeminiError) as ei:
        er2.point(Image(jpeg, role="overview"))
    assert FAKE_KEY not in str(ei.value)
    text = "".join(f.read_text() for f in log_dir.glob("*.jsonl"))
    assert text and FAKE_KEY not in text and "AIza" not in text and "[REDACTED]" in text


def test_key_loaded_from_secrets_file(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    f = tmp_path / "secrets.env"
    f.write_text("# comment\nOTHER=1\nexport GEMINI_API_KEY='abc123secretvalue'\n")
    monkeypatch.setenv("MODELS_SECRETS_FILE", str(f))
    assert _secrets.gemini_key() == "abc123secretvalue"
    assert "abc123secretvalue" not in _secrets.redact("error abc123secretvalue here")
    f.write_text("GEMINI_API_KEY=\n")
    assert _secrets.gemini_key() is None


def test_genai_transport_without_key_fails_cleanly(monkeypatch, tmp_path, jpeg, log_dir):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("MODELS_SECRETS_FILE", str(tmp_path / "none.env"))
    pytest.importorskip("google.genai")
    er = GeminiER(transport=GenAITransport(), log_dir=log_dir)
    with pytest.raises(GeminiError, match="no GEMINI_API_KEY"):
        er.point(Image(jpeg, role="overview"))


def test_genai_client_uses_v1beta(monkeypatch):
    """The preview ER models are only served on v1beta; SDK 3.3.0 defaults to v1 (404 NOT_FOUND)."""
    genai = pytest.importorskip("google.genai")
    seen = {}

    class Client:
        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr(genai, "Client", Client)
    GenAITransport(api_key=FAKE_KEY)._get()
    assert seen["http_options"]["api_version"] == "v1beta"


def test_retry_delay_parsed_from_429():
    from models.gemini_er import retry_delay_s
    msg = ("ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'message': 'limit: 5 ... Please retry in 21.1160414s.', "
           "'details': [{'retryDelay': '21s'}]}}")
    assert retry_delay_s(msg) == pytest.approx(22.116, abs=1e-3)
    assert retry_delay_s("ClientError: 404 NOT_FOUND") is None
    day = ("ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'message': 'limit: 20, description: Generate content "
           "requests per day limit exceeded on Free Tier.\\nPlease retry in 7h26m56.08s.'}}")
    assert retry_delay_s(day) is None
