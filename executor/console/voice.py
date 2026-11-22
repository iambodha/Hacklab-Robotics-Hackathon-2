"""Voice for the console, on the Gemini Live API.

The browser talks to Google directly over a WebSocket (microphone audio up, speech down). This module is the
server side of that: it keeps the API key here and hands the page a single-use, short-lived token that carries the
whole session setup (model, instructions, the four tools, transcription), so the page cannot change them. The page runs the tools by
calling the console API; a task started by voice goes through `start_by_voice`, which refuses on the real arm.

Key: $GEMINI_API_KEY or $GOOGLE_API_KEY in the console's environment (models/_secrets.py). Model:
CONSOLE_VOICE_MODEL (default gemini-3.8-live). Voice: CONSOLE_VOICE_NAME.
"""
from __future__ import annotations

import datetime
import json
import os
import urllib.error
import urllib.request

API = "https://generativelanguage.googleapis.com/v1beta"
WS_URL = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContentConstrained"
MODEL = os.environ.get("CONSOLE_VOICE_MODEL", "gemini-3.8-live")
VOICE = os.environ.get("CONSOLE_VOICE_NAME", "")
VOICE_BACKENDS = ("fake", "sim")          # a spoken command may start a task only on these

INSTRUCTIONS = """You are the voice of a robot arm's operator console. A person at the table talks to you; you help
them state a task for the arm and you tell them, briefly, what the arm is doing.

You are only the voice. You do not plan or execute anything: once a task is started, a separate agent (Claude)
proposes skills (pick, place, park, move), the executor validates each one, checks it in a simulated twin, runs it
and an independent check reports the outcome. A task is a short instruction such as "Put the blue object in the bin". Backends: "fake" and "sim" are simulations; "real" is the physical arm.

Tools:
- set_task(prompt): write the task into the console's prompt box. Rephrase what the person said into one clear
  imperative sentence; keep their objects and places exactly.
- start_task(): start the task that is in the prompt box. This only works on the simulated backends. On the real
  arm it is refused: tell the person to press Start on the console themselves.
- stop(): stop the task and cancel motion. Call it at once when the person says stop, halt, cancel or anything
  that sounds like alarm. Never ask for confirmation before stopping.
- get_status(): the current backend, task state, recent results and the objects the cameras see.

Rules:
- When the person gives a task, call set_task, then say the task back in a few words. Start it only when they
  clearly ask you to run it (in the same sentence or after).
- If the request is unclear or names something get_status does not show, ask one short question instead of guessing.
- Lines that begin with [console] are events from the console, not speech from the person. Turn each batch into
  one short spoken sentence. Skip routine detail; always mention rejections, faults, failures and the final outcome.
- Report only what the tools and events say. Never say a task succeeded unless an event or get_status says so.
- You cannot move the arm yourself and you cannot change safety settings. Say so if asked.
- Speak plainly, one or two short sentences at a time. Answer in the language the person speaks."""

TOOLS = [{"functionDeclarations": [
    {"name": "set_task", "description": "Write the task into the console's prompt box (does not start it).",
     "parameters": {"type": "OBJECT", "properties": {"prompt": {"type": "STRING", "description": "One clear imperative sentence."}},
                    "required": ["prompt"]}},
    {"name": "start_task", "description": "Start the task in the prompt box. Simulated backends only; refused on the real arm."},
    {"name": "stop", "description": "Stop the running task and cancel motion now. The arm holds its position."},
    {"name": "get_status", "description": "Backend, task state, recent skill results and the objects perception sees."},
]}]


class VoiceError(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


def _key() -> str | None:
    from models._secrets import gemini_key
    return gemini_key()


def describe() -> dict:
    return {"available": _key() is not None, "model": MODEL, "start_backends": list(VOICE_BACKENDS),
            "why": None if _key() else "Needs GEMINI_API_KEY or GOOGLE_API_KEY in the console environment"}


def setup() -> dict:
    """The session setup, locked into the token."""
    gen: dict = {"responseModalities": ["AUDIO"]}
    if VOICE:
        gen["speechConfig"] = {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": VOICE}}}
    return {"setup": {"model": f"models/{MODEL}", "generationConfig": gen,
                      "systemInstruction": {"parts": [{"text": INSTRUCTIONS}]}, "tools": TOOLS,
                      "inputAudioTranscription": {}, "outputAudioTranscription": {}}}


def token(opener=urllib.request.urlopen) -> dict:
    """A single-use token for one Live session on MODEL: 1 minute to connect, 30 minutes to talk."""
    key = _key()
    if not key:
        raise VoiceError(503, describe()["why"])
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    iso = lambda d: d.strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    body = {"uses": 1, "expireTime": iso(now + datetime.timedelta(minutes=30)),
            "newSessionExpireTime": iso(now + datetime.timedelta(minutes=1)),
            "bidiGenerateContentSetup": setup()["setup"]}     # no field mask: the session uses exactly this setup
    req = urllib.request.Request(f"{API}/auth_tokens", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    try:
        with opener(req, timeout=10) as r:
            name = json.loads(r.read())["name"]
    except urllib.error.HTTPError as e:
        from models._secrets import redact
        detail = redact(e.read().decode(errors="replace"))[:300]
        raise VoiceError(502, f"Google refused the voice token ({e.code}): {detail}") from None
    except (OSError, KeyError, ValueError) as e:
        raise VoiceError(502, f"could not get a voice token: {type(e).__name__}") from None
    # the page still has to open with a setup message; Google ignores its contents in favour of the token's
    return {"token": name, "url": WS_URL, "setup": {"setup": {"model": f"models/{MODEL}"}}, "model": MODEL}


def start_by_voice(console, body: dict) -> dict:
    """Start a task from a spoken command: simulated backends only. The real arm needs the Start button."""
    name = console.ex.backend.name
    if name not in VOICE_BACKENDS:
        console.events.add("voice", role="console", text=f"refused a voice start on the {name} backend")
        raise VoiceError(403, f"voice can start tasks only on {' or '.join(VOICE_BACKENDS)}; the backend is {name}: "
                              "press Start on the console")
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise VoiceError(400, "there is no task in the prompt box")
    return console.start_task(prompt, body.get("advisor", "scripted"), body.get("model"), name)
