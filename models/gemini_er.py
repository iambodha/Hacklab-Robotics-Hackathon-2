"""Thin typed client for Gemini Robotics ER (official `google-genai` SDK) on our 299x224 camera frames.

    er = GeminiER()                                   # key from $GEMINI_API_KEY or secrets.env
    pts = er.point(Image.from_path("overview.jpg"), ["purple wrapper"])     # -> [Point(u, v px)]
    boxes = er.boxes(img, ["stain"])                  # -> [Box(u0, v0, u1, v1 px)]
    j = er.judge([ov, wr], "wipe the stain off the table")               # -> Judgement(done, failure, residual)
    step = er.propose_step([ov, wr], goal, history)   # -> Step(action in VOCABULARY, args)

Coordinates: the model answers in its documented convention ([y, x] normalised 0-1000, y first); this
module converts to image pixels (u right, v down) of the image it was asked about. Pixels -> world is
the advisor's job (perception.locate). Every call appends one JSON line to models/logs/gemini-<date>.jsonl:
prompt, image refs (role, ref, size, sha1; no bytes), response text, parsed result, latency, token usage,
cost. The API key never reaches the log: every record goes through _secrets.redact_obj.

Run with: uv run --no-project --with google-genai==3.3.0 python -m models.gemini_er --help
Tests use a fake transport, so they need no SDK and no key.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import struct
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

from ._secrets import gemini_key, redact, redact_obj

DEFAULT_MODEL = os.environ.get("GEMINI_ER_MODEL", "gemini-robotics-er-2-preview")
STREAMING_MODEL = "gemini-robotics-er-2-streaming-preview"
# google-genai 3.3.0 defaults the Gemini API to v1, where the preview ER models answer 404 NOT_FOUND.
API_VERSION = os.environ.get("GEMINI_API_VERSION", "v1beta")
LOG_DIR = Path(os.environ.get("MODELS_LOG_DIR", Path(__file__).resolve().parent / "logs"))
# USD per 1M tokens (input, output incl. thinking), paid tier, valid through 2026-12-31 (doubles 2027-01-01).
PRICES = {DEFAULT_MODEL: (1.00, 5.00), "gemini-robotics-er-2-preview": (1.00, 5.00), STREAMING_MODEL: (1.00, 5.00)}

# The only actions the next-step proposer may return. Args are in the image's normalised 0-1000 frame
# ([y, x], y first, like every other ER output) and converted to pixels on parsing.
VOCABULARY = {
    "inspect": "look again (optionally from a named camera) before acting; args: {\"camera\": \"overview\"|\"wrist\"}",
    "retry_grasp": "the last grasp missed; grasp again at the object; args: {\"object\": str, \"point\": [y, x], \"camera\": str}",
    "wipe_region": "wipe a rectangular table region; args: {\"box_2d\": [ymin, xmin, ymax, xmax], \"camera\": str}",
    "pick": "pick an object; args: {\"object\": str, \"point\": [y, x], \"camera\": str}",
    "place": "put the held object down; args: {\"target\": str, \"point\": [y, x], \"camera\": str}",
    "stop": "the goal is reached, or it is unsafe / impossible to continue; args: {\"reason\": str}",
}


class GeminiError(RuntimeError):
    """API / transport failure. The message is already redacted."""


class ParseError(ValueError):
    """The model answered, but not in the requested shape."""


# ---- images -----------------------------------------------------------------------------------------------
def image_size(data: bytes) -> tuple[int, int]:
    """(width, height) of a JPEG or PNG without decoding it."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", data[16:24])
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seg = struct.unpack(">H", data[i + 2:i + 4])[0]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h, w = struct.unpack(">HH", data[i + 5:i + 9])
            return w, h
        i += 2 + seg
    raise ValueError("not a JPEG/PNG with a readable size")


@dataclass
class Image:
    data: bytes
    role: str = ""                 # "overview" | "wrist" | free text
    ref: str = ""                  # where it came from: path, "recording:<id>@<t>", frame URL (logged, not the bytes)
    w: int = 0
    h: int = 0
    mime: str = "image/jpeg"

    def __post_init__(self):
        if not (self.w and self.h):
            self.w, self.h = image_size(self.data)
        if self.data[:4] == b"\x89PNG":
            self.mime = "image/png"

    @classmethod
    def from_path(cls, path, role: str = "") -> "Image":
        p = Path(path)
        return cls(p.read_bytes(), role=role or p.stem, ref=str(p))

    def log_ref(self) -> dict:
        return {"role": self.role, "ref": self.ref, "w": self.w, "h": self.h, "bytes": len(self.data),
                "sha1": hashlib.sha1(self.data).hexdigest()[:16]}


# ---- results ------------------------------------------------------------------------------------------------
@dataclass
class Point:
    label: str
    u: float                       # pixels, right
    v: float                       # pixels, down
    camera: str = ""
    norm_yx: tuple[float, float] = (0.0, 0.0)


@dataclass
class Box:
    label: str
    u0: float
    v0: float
    u1: float
    v1: float
    camera: str = ""

    @property
    def centre(self) -> tuple[float, float]:
        return (self.u0 + self.u1) / 2, (self.v0 + self.v1) / 2

    @property
    def area_px(self) -> float:
        return max(0.0, self.u1 - self.u0) * max(0.0, self.v1 - self.v0)

    def corners(self) -> list[tuple[float, float]]:
        return [(self.u0, self.v0), (self.u1, self.v0), (self.u1, self.v1), (self.u0, self.v1)]


@dataclass
class Judgement:
    done: bool | None              # None = the model says it cannot tell
    confidence: float | None
    failure: str | None            # what went wrong, if not done
    evidence: str
    residual: list[Box] = field(default_factory=list)   # e.g. remaining stain, in pixels of `camera`
    progress: str | None = None    # "0-20" ... "80-100"


@dataclass
class Step:
    action: str                    # a key of VOCABULARY
    args: dict                     # pixel-converted: point -> {"u","v"}, box_2d -> Box
    rationale: str
    confidence: float | None = None


@dataclass
class CallInfo:
    call_id: str
    kind: str
    model: str
    latency_s: float
    prompt_tokens: int | None
    output_tokens: int | None
    thinking_tokens: int | None
    cost_usd: float | None
    error: str | None = None


# ---- transport ---------------------------------------------------------------------------------------------------
@dataclass
class RawReply:
    text: str
    usage: dict                    # prompt_tokens, output_tokens, thinking_tokens, total_tokens


class Transport(Protocol):
    def __call__(self, model: str, parts: list, config: dict) -> RawReply: ...


class GenAITransport:
    """google-genai `models.generate_content`. Imported lazily so tests and offline tools need no SDK."""

    def __init__(self, api_key: str | None = None, wait_429_s: float = 0.0):
        """wait_429_s > 0: on 429 RESOURCE_EXHAUSTED, sleep the server's retryDelay and retry, until
        that much total waiting is used up. 0 (default) fails at once, which is what a live advisor wants."""
        self._key = api_key
        self._client = None
        self.wait_429_s = wait_429_s

    def _get(self):
        if self._client is None:
            key = self._key or gemini_key()
            if not key:
                raise GeminiError("no GEMINI_API_KEY (env or secrets.env); use a fake transport or recorded replies")
            from google import genai
            self._client = genai.Client(api_key=key, http_options={"api_version": API_VERSION})
        return self._client

    def __call__(self, model: str, parts: list, config: dict) -> RawReply:
        from google.genai import types
        contents = [types.Part.from_bytes(data=p.data, mime_type=p.mime) if isinstance(p, Image) else p for p in parts]
        cfg = {"thinking_config": types.ThinkingConfig(thinking_level=config.get("thinking_level", "low"))}
        if config.get("json", True):
            cfg["response_mime_type"] = "application/json"
        if config.get("temperature") is not None:
            cfg["temperature"] = config["temperature"]
        if config.get("media_resolution"):
            cfg["media_resolution"] = config["media_resolution"]
        budget = self.wait_429_s
        while True:
            try:
                r = self._get().models.generate_content(model=model, contents=contents,
                                                        config=types.GenerateContentConfig(**cfg))
                break
            except GeminiError:
                raise
            except Exception as e:  # the SDK raises many types; keep the message, never the key
                msg = f"{type(e).__name__}: {e}"
                delay = retry_delay_s(msg)
                if delay is None or delay > budget:
                    raise GeminiError(redact(msg, (self._key or "",))) from None
                budget -= delay
                time.sleep(delay)
        u = r.usage_metadata
        return RawReply(text=r.text or "", usage={
            "prompt_tokens": getattr(u, "prompt_token_count", None), "output_tokens": getattr(u, "candidates_token_count", None),
            "thinking_tokens": getattr(u, "thoughts_token_count", None), "total_tokens": getattr(u, "total_token_count", None),
            "rate_limit_wait_s": round(self.wait_429_s - budget, 2)})


_RETRY = re.compile(r"retry in ([\d.]+)s|'retryDelay': '(\d+)s'")


def retry_delay_s(message: str) -> float | None:
    """Seconds to wait before retrying a 429 RESOURCE_EXHAUSTED error, or None if it is not one or it is the
    free tier's daily cap (20 requests/day/model; waiting means hours)."""
    if "429" not in message or "RESOURCE_EXHAUSTED" not in message or "per day" in message.lower():
        return None
    m = _RETRY.search(message)
    return (float(m.group(1) or m.group(2)) if m else 30.0) + 1.0


class FakeTransport:
    """Canned replies for tests / offline runs: a list (consumed in order), or fn(model, parts, config) -> text."""

    def __init__(self, replies: list[str] | Callable[[str, list, dict], str], usage: dict | None = None):
        self.replies, self.calls = replies, []
        self.usage = usage or {"prompt_tokens": 1300, "output_tokens": 80, "thinking_tokens": 0, "total_tokens": 1380}

    def __call__(self, model: str, parts: list, config: dict) -> RawReply:
        self.calls.append({"model": model, "parts": parts, "config": config})
        text = self.replies(model, parts, config) if callable(self.replies) else self.replies.pop(0)
        return RawReply(text=text, usage=dict(self.usage))


# ---- parsing ---------------------------------------------------------------------------------------------------
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json(text: str):
    """JSON from a model reply: plain, inside ``` fences, or the first {...} / [...] in the text."""
    t = (text or "").strip()
    m = _FENCE.search(t)
    if m:
        t = m.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (t.find("["), t.find("{")) if i >= 0]
    for i in sorted(starts):
        try:
            return json.JSONDecoder().raw_decode(t[i:])[0]
        except json.JSONDecodeError:
            continue
    raise ParseError(f"no JSON in reply: {t[:120]!r}")


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def norm_to_px(y: float, x: float, w: int, h: int) -> tuple[float, float]:
    """[y, x] in 0-1000 -> (u, v) pixels."""
    return _clamp(float(x), 0, 1000) / 1000 * w, _clamp(float(y), 0, 1000) / 1000 * h


def _items(obj) -> list:
    if isinstance(obj, dict):
        for k in ("points", "objects", "boxes", "items", "detections", "residual"):
            if isinstance(obj.get(k), list):
                return obj[k]
        return [obj]
    return obj if isinstance(obj, list) else []


def parse_points(obj, w: int, h: int, camera: str = "") -> list[Point]:
    out = []
    for it in _items(obj):
        p = it.get("point") if isinstance(it, dict) else None
        if not (isinstance(p, (list, tuple)) and len(p) == 2):
            continue
        u, v = norm_to_px(p[0], p[1], w, h)
        out.append(Point(str(it.get("label", "")), round(u, 2), round(v, 2), camera, (float(p[0]), float(p[1]))))
    return out


def _box_yxyx(it: dict):
    if isinstance(it.get("box_2d"), (list, tuple)) and len(it["box_2d"]) == 4:
        return it["box_2d"]
    if all(k in it for k in ("y", "x", "y2", "x2")):
        return [it["y"], it["x"], it["y2"], it["x2"]]
    if isinstance(it.get("box"), (list, tuple)) and len(it["box"]) == 4:
        return it["box"]
    return None


def parse_boxes(obj, w: int, h: int, camera: str = "") -> list[Box]:
    out = []
    for it in _items(obj):
        b = _box_yxyx(it) if isinstance(it, dict) else None
        if b is None:
            continue
        u0, v0 = norm_to_px(b[0], b[1], w, h)
        u1, v1 = norm_to_px(b[2], b[3], w, h)
        u0, u1 = sorted((u0, u1))
        v0, v1 = sorted((v0, v1))
        out.append(Box(str(it.get("label", "")), round(u0, 2), round(v0, 2), round(u1, 2), round(v1, 2), camera))
    return out


def _tri(x) -> bool | None:
    if isinstance(x, bool) or x is None:
        return x
    s = str(x).strip().lower()
    return True if s in ("true", "yes", "done", "success", "1") else False if s in ("false", "no", "0", "failed", "failure") else None


def parse_judgement(obj, images: list[Image]) -> Judgement:
    if not isinstance(obj, dict):
        raise ParseError("judgement is not a JSON object")
    cam = str(obj.get("camera") or (images[0].role if images else ""))
    img = next((i for i in images if i.role == cam), images[0] if images else None)
    residual = parse_boxes(obj.get("residual") or [], img.w, img.h, img.role) if img else []
    conf = obj.get("confidence")
    return Judgement(done=_tri(obj.get("done")), confidence=float(conf) if isinstance(conf, (int, float)) else None,
                     failure=obj.get("failure") or None, evidence=str(obj.get("evidence", "")), residual=residual,
                     progress=obj.get("progress"))


def parse_step(obj, images: list[Image], vocabulary=VOCABULARY) -> Step:
    if not isinstance(obj, dict) or "action" not in obj:
        raise ParseError("step has no 'action'")
    action = str(obj["action"]).strip()
    if action not in vocabulary:
        raise ParseError(f"action {action!r} is not in the vocabulary {sorted(vocabulary)}")
    args = dict(obj.get("args") or {})
    cam = str(args.get("camera") or (images[0].role if images else ""))
    img = next((i for i in images if i.role == cam), None)
    if action in ("pick", "place", "retry_grasp", "wipe_region") and img is None:
        raise ParseError(f"{action}: camera {cam!r} is not one of the images {[i.role for i in images]}")
    if action in ("pick", "place", "retry_grasp"):
        p = args.get("point")
        if not (isinstance(p, (list, tuple)) and len(p) == 2):
            raise ParseError(f"{action} needs args.point [y, x]")
        u, v = norm_to_px(p[0], p[1], img.w, img.h)
        args.update(camera=img.role, u=round(u, 2), v=round(v, 2))
        if action in ("pick", "retry_grasp") and not args.get("object"):
            raise ParseError(f"{action} needs args.object")
        if action == "place" and not args.get("target"):
            raise ParseError("place needs args.target")
    if action == "wipe_region":
        boxes = parse_boxes([args], img.w, img.h, img.role)
        if not boxes or boxes[0].area_px <= 0:
            raise ParseError("wipe_region needs a non-empty args.box_2d [ymin, xmin, ymax, xmax]")
        args.update(camera=img.role, box=boxes[0])
    conf = obj.get("confidence")
    return Step(action, args, str(obj.get("rationale", "")), float(conf) if isinstance(conf, (int, float)) else None)


# ---- prompts ------------------------------------------------------------------------------------------------------
COORDS = "Coordinates are [y, x] (y first) normalised to 0-1000 over the image."


def prompt_point(targets: list[str] | None, max_items: int) -> str:
    what = ("the following: " + "; ".join(targets)) if targets else f"no more than {max_items} items"
    return (f"Point to {what}. If something asked for is not visible, leave it out (never guess). "
            f"Answer JSON: [{{\"point\": [y, x], \"label\": <name>}}, ...]. {COORDS}")


def prompt_boxes(targets: list[str] | None, max_items: int) -> str:
    what = ("each of: " + "; ".join(targets)) if targets else f"up to {max_items} objects"
    return (f"Detect {what} and return tight 2D bounding boxes. Leave out anything not visible. "
            f"Answer JSON: [{{\"box_2d\": [ymin, xmin, ymax, xmax], \"label\": <name>}}, ...]. {COORDS}")


def prompt_judge(task: str, criteria: str | None, roles: list[str]) -> str:
    return (f"You check a robot's work from its camera images ({', '.join(roles)}). Task: {task}.\n"
            + (f"Success means: {criteria}.\n" if criteria else "")
            + "Judge only from what is visible. If the evidence is occluded, blurred or missing, set done to null.\n"
            "Answer JSON: {\"done\": true|false|null, \"confidence\": 0..1, \"failure\": <what is wrong, or null>, "
            "\"evidence\": <one sentence on what you see>, \"progress\": \"0-20\"|\"20-40\"|\"40-60\"|\"60-80\"|\"80-100\", "
            "\"camera\": <image the residual boxes refer to>, "
            "\"residual\": [{\"box_2d\": [ymin, xmin, ymax, xmax], \"label\": <what remains>}]}. "
            f"residual lists what still has to be done (e.g. remaining stain), empty if done. {COORDS}")


def prompt_step(goal: str, history: list[str], roles: list[str], vocabulary=VOCABULARY) -> str:
    acts = "\n".join(f"- {k}: {v}" for k, v in vocabulary.items())
    hist = "\n".join(f"- {h}" for h in history[-8:]) or "- (nothing yet)"
    return (f"You advise a robot arm (single arm, parallel gripper, table top). Images: {', '.join(roles)}.\n"
            f"Goal: {goal}\nWhat happened so far:\n{hist}\n"
            f"Choose exactly ONE next action from this list (nothing else is allowed):\n{acts}\n"
            "A local executor checks reachability and safety and runs the skill; you only choose. "
            "If a person's hands are in the workspace, choose stop.\n"
            "Answer JSON: {\"action\": <name>, \"args\": {...}, \"rationale\": <one sentence>, \"confidence\": 0..1}. "
            f"{COORDS}")


# ---- client ------------------------------------------------------------------------------------------------------
class GeminiER:
    def __init__(self, model: str = DEFAULT_MODEL, transport: Transport | None = None, log_dir: Path | None = LOG_DIR,
                 thinking_level: str = "low", source_tag: str = ""):
        self.model, self.thinking_level, self.source_tag = model, thinking_level, source_tag
        self.transport = transport or GenAITransport()
        self.log_dir = Path(log_dir) if log_dir else None
        self.last: CallInfo | None = None

    # -- public calls --
    def point(self, image: Image, targets: list[str] | None = None, max_items: int = 10, **cfg) -> list[Point]:
        obj = self._call("point", [image, prompt_point(targets, max_items)], cfg)
        return parse_points(obj, image.w, image.h, image.role)

    def boxes(self, image: Image, targets: list[str] | None = None, max_items: int = 10, **cfg) -> list[Box]:
        obj = self._call("boxes", [image, prompt_boxes(targets, max_items)], cfg)
        return parse_boxes(obj, image.w, image.h, image.role)

    def judge(self, images: list[Image], task: str, criteria: str | None = None, **cfg) -> Judgement:
        roles = [i.role for i in images]
        obj = self._call("judge", _labelled(images) + [prompt_judge(task, criteria, roles)], cfg)
        return parse_judgement(obj, images)

    def propose_step(self, images: list[Image], goal: str, history: list[str] | None = None,
                     vocabulary=VOCABULARY, **cfg) -> Step:
        roles = [i.role for i in images]
        obj = self._call("step", _labelled(images) + [prompt_step(goal, history or [], roles, vocabulary)], cfg)
        return parse_step(obj, images, vocabulary)

    def ask_json(self, kind: str, parts: list, **cfg):
        """Any other prompt that answers JSON (used by the auto-labeller). Logged like the rest."""
        return self._call(kind, parts, cfg)

    # -- core --
    def _call(self, kind: str, parts: list, cfg: dict):
        config = {"thinking_level": cfg.pop("thinking_level", self.thinking_level), "json": True, **cfg}
        call_id = "gem-" + uuid.uuid4().hex[:10]
        t0 = time.monotonic()
        text, usage, err, parsed = "", {}, None, None
        try:
            reply = self.transport(self.model, parts, config)
            text, usage = reply.text, reply.usage
            parsed = parse_json(text)
            return parsed
        except ParseError as e:
            err = f"parse: {e}"
            raise
        except GeminiError as e:
            err = str(e)
            raise
        except Exception as e:
            err = redact(f"{type(e).__name__}: {e}")
            raise GeminiError(err) from None
        finally:
            lat = time.monotonic() - t0 - float(usage.get("rate_limit_wait_s") or 0)   # model time, not back-off
            self.last = CallInfo(call_id, kind, self.model, round(lat, 3), usage.get("prompt_tokens"),
                                 usage.get("output_tokens"), usage.get("thinking_tokens"), cost_usd(self.model, usage), err)
            self._log(call_id, kind, parts, config, text, parsed, lat, usage, err)

    def _log(self, call_id, kind, parts, config, text, parsed, lat, usage, err):
        if not self.log_dir:
            return
        rec = {"call_id": call_id, "t_wall": round(time.time(), 3), "kind": kind, "model": self.model,
               "source_tag": self.source_tag, "config": config,
               "prompt": "\n".join(p for p in parts if isinstance(p, str)),
               "images": [p.log_ref() for p in parts if isinstance(p, Image)],
               "response": text, "parsed": parsed, "latency_s": round(lat, 3), "usage": usage,
               "cost_usd": cost_usd(self.model, usage), "error": err}
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_dir / f"gemini-{datetime.now():%Y%m%d}.jsonl", "a") as f:
            f.write(json.dumps(redact_obj(_jsonable(rec))) + "\n")


def _labelled(images: list[Image]) -> list:
    out = []
    for i, im in enumerate(images, 1):
        out += [f"Image {i}: {im.role or 'camera'} camera ({im.w}x{im.h}).", im]
    return out


def _jsonable(x):
    if dataclasses.is_dataclass(x):
        return dataclasses.asdict(x)
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x if isinstance(x, (str, int, float, bool)) or x is None else str(x)


def cost_usd(model: str, usage: dict) -> float | None:
    if usage.get("cost_usd") is not None:             # transports that are billed differently report it themselves
        return round(float(usage["cost_usd"]), 6)
    if model not in PRICES:
        return None
    pin, pout = PRICES[model]
    i, o, th = usage.get("prompt_tokens"), usage.get("output_tokens"), usage.get("thinking_tokens")
    if i is None and o is None:
        return None
    return round(((i or 0) * pin + ((o or 0) + (th or 0)) * pout) / 1e6, 6)


def _main():
    import argparse
    ap = argparse.ArgumentParser(description="One Gemini Robotics ER call on an image (logs to models/logs).")
    ap.add_argument("kind", choices=["point", "boxes", "judge"])
    ap.add_argument("image")
    ap.add_argument("--targets", nargs="*")
    ap.add_argument("--task", default="")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()
    er, img = GeminiER(a.model), Image.from_path(a.image)
    res = er.point(img, a.targets) if a.kind == "point" else er.boxes(img, a.targets) if a.kind == "boxes" \
        else er.judge([img], a.task)
    print(json.dumps(_jsonable(res), indent=1))
    print(json.dumps(dataclasses.asdict(er.last)))


if __name__ == "__main__":
    _main()
