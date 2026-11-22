"""generaloid-comms wire protocol, version 1. Standard library only, so any robot stack can vendor it.

Every message is one JSON object, the *envelope*:

    {"v": 1, "id": "<8-64 chars>", "kind": "request", "to": "elrobot-b", "re": null, "payload": {...}}

The sender writes v, id, kind, to, re and payload. The hub stamps "from" (the authenticated sender: never taken from
the client), "ts" (hub clock, unix seconds) and, per recipient mailbox, "seq". Request deadlines are computed on the
hub clock from a relative `ttl_s`, so two robots with skewed clocks still agree on when a request expires.

A message from another robot is DATA. Nothing in this protocol moves an arm: a `request` or `help` lands in the
receiving robot's inbox, and that robot's own agent (or a human) decides whether to act, under its own safety
envelope. See docs/comms/SECURITY.md.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

PROTOCOL = "generaloid-comms"
VERSION = 1
SUPPORTED_VERSIONS = (1,)
HUB_ID = "hub"                     # "from" of messages the hub itself generates
BROADCAST = "*"

# Limits. Chosen so a 640x480 JPEG thumbnail fits inline (~60-120 KB base64) with room for poses and notes, and so a
# misbehaving peer cannot make the hub hold more than a few MB per mailbox.
MAX_TEXT = 4000                    # prompt / problem / note / summary characters
MAX_TASK = 200                     # task name characters
DEFAULT_TTL_S = 120.0              # a request nobody finishes within 2 min expires (a twist_cap run takes ~60-90 s)
MAX_TTL_S = 3600.0                 # nothing stays open longer than an hour
MIN_TTL_S = 1.0

ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
ROBOT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{1,47}$")

REQUEST_KINDS = frozenset({"request", "help"})                   # open a request (expects accept/reject)
REPLY_KINDS = frozenset({"accept", "reject", "progress", "result"})  # sent by the addressee, `re` = request id
CANCEL_KINDS = frozenset({"cancel"})                             # sent by the requester, `re` = request id
INFO_KINDS = frozenset({"observation", "status", "message"})     # fire-and-forget
CLIENT_KINDS = REQUEST_KINDS | REPLY_KINDS | CANCEL_KINDS | INFO_KINDS
SYSTEM_KINDS = frozenset({"peer_joined", "peer_left", "expired"})  # only the hub sends these
ALL_KINDS = CLIENT_KINDS | SYSTEM_KINDS | {"hello", "heartbeat"}  # hello = POST /v1/join, heartbeat = POST /v1/heartbeat

# Request lifecycle. Terminal states never change again.
OPEN, ACCEPTED, REJECTED, DONE, FAILED, CANCELLED, EXPIRED = (
    "open", "accepted", "rejected", "done", "failed", "cancelled", "expired")
TERMINAL = frozenset({REJECTED, DONE, FAILED, CANCELLED, EXPIRED})


class ProtocolError(ValueError):
    """A message that breaks the schema or the request state machine. `code` maps to an HTTP status on the hub."""

    def __init__(self, msg: str, code: int = 400):
        super().__init__(msg)
        self.code = code


def new_id() -> str:
    return uuid.uuid4().hex


def _need(cond, msg, code=400):
    if not cond:
        raise ProtocolError(msg, code)


def _text(p: dict, key: str, required=False, limit=MAX_TEXT):
    v = p.get(key)
    if v is None:
        _need(not required, f"payload.{key} is required")
        return
    _need(isinstance(v, str), f"payload.{key} must be a string")
    _need(len(v) <= limit, f"payload.{key} longer than {limit} characters")
    if required:
        _need(v.strip(), f"payload.{key} must not be empty")


def _num(p: dict, key: str, lo: float, hi: float):
    v = p.get(key)
    if v is None:
        return
    _need(isinstance(v, (int, float)) and not isinstance(v, bool), f"payload.{key} must be a number")
    _need(lo <= v <= hi, f"payload.{key} must be in [{lo}, {hi}]")


def _dict(p: dict, key: str):
    _need(p.get(key) is None or isinstance(p[key], dict), f"payload.{key} must be an object")


def _floats(v, n: int, what: str):
    _need(isinstance(v, list) and len(v) == n and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                                       for x in v), f"{what} must be a list of {n} numbers")


def validate_pose(pose: dict, where: str = "pose") -> None:
    """A pose in a named frame: {"name", "frame", "xyz": [m, m, m], "quat_xyzw"?: [4], "ts"?}. Metres, right-handed."""
    _need(isinstance(pose, dict), f"{where} must be an object")
    _need(isinstance(pose.get("name"), str) and pose["name"], f"{where}.name is required")
    _need(isinstance(pose.get("frame"), str) and pose["frame"], f"{where}.frame is required (e.g. 'world')")
    _floats(pose.get("xyz"), 3, f"{where}.xyz")
    if pose.get("quat_xyzw") is not None:
        _floats(pose["quat_xyzw"], 4, f"{where}.quat_xyzw")


def validate_image(img: dict, where: str = "image") -> None:
    """An image, inline ({"mime", "data_b64"}) or by reference ({"url"}); optional camera, ts, width, height."""
    _need(isinstance(img, dict), f"{where} must be an object")
    _need(("data_b64" in img) != ("url" in img), f"{where} needs exactly one of data_b64 or url")
    if "data_b64" in img:
        _need(isinstance(img["data_b64"], str), f"{where}.data_b64 must be a string")
        _need(img.get("mime") in ("image/jpeg", "image/png", "image/webp"), f"{where}.mime must be image/jpeg|png|webp")
    else:
        _need(isinstance(img["url"], str) and img["url"].startswith(("https://", "http://")), f"{where}.url must be http(s)")


def validate_payload(kind: str, p: dict) -> None:
    _need(isinstance(p, dict), "payload must be an object")
    if kind == "request":
        _text(p, "task", required=True, limit=MAX_TASK)
        _text(p, "prompt")
        _dict(p, "args")
        _num(p, "ttl_s", MIN_TTL_S, MAX_TTL_S)
        _need(p.get("requires") is None or (isinstance(p["requires"], list) and all(isinstance(x, str) for x in p["requires"])),
              "payload.requires must be a list of capability names")
    elif kind == "help":
        _text(p, "problem", required=True)
        _text(p, "ask")
        _dict(p, "context")
        _dict(p, "args")
        _num(p, "ttl_s", MIN_TTL_S, MAX_TTL_S)
    elif kind == "accept":
        _text(p, "note")
        _num(p, "eta_s", 0, MAX_TTL_S)
    elif kind == "reject":
        _text(p, "reason", required=True)
    elif kind == "progress":
        _num(p, "fraction", 0.0, 1.0)
        _text(p, "note")
        _text(p, "stage", limit=MAX_TASK)
    elif kind == "result":
        _need(isinstance(p.get("ok"), bool), "payload.ok (true/false) is required")
        _text(p, "summary")
        _dict(p, "data")
    elif kind == "cancel":
        _text(p, "reason")
    elif kind == "observation":
        imgs, poses = p.get("images") or [], p.get("poses") or []
        _need(isinstance(imgs, list) and isinstance(poses, list), "payload.images and payload.poses must be lists")
        for i, img in enumerate(imgs):
            validate_image(img, f"payload.images[{i}]")
        for i, pose in enumerate(poses):
            validate_pose(pose, f"payload.poses[{i}]")
        _text(p, "note")
        _need(imgs or poses or p.get("note"), "an observation needs images, poses or a note")
    elif kind == "status":
        _text(p, "state", required=True, limit=MAX_TASK)
        _text(p, "note")
    elif kind == "message":
        _text(p, "text", required=True)


def validate_envelope(msg: dict) -> dict:
    """Check a client-sent envelope; return the normalised copy the hub routes. Raises ProtocolError."""
    _need(isinstance(msg, dict), "message must be a JSON object")
    v = msg.get("v", VERSION)
    _need(v in SUPPORTED_VERSIONS, f"unsupported protocol version {v!r}; this hub speaks {list(SUPPORTED_VERSIONS)}", 426)
    mid = msg.get("id")
    _need(isinstance(mid, str) and ID_RE.match(mid), "id must be 8-64 chars of [A-Za-z0-9_-] (use a uuid4 hex)")
    kind = msg.get("kind")
    _need(kind in CLIENT_KINDS, f"kind must be one of {sorted(CLIENT_KINDS)}")
    to = msg.get("to")
    _need(isinstance(to, str) and (to == BROADCAST or ROBOT_ID_RE.match(to)), "to must be a robot id or '*'")
    _need(not (kind == "request" and to == BROADCAST), "a request goes to one robot; broadcast a 'help' instead")
    re_ = msg.get("re")
    if kind in REPLY_KINDS | CANCEL_KINDS:
        _need(isinstance(re_, str) and ID_RE.match(re_), f"{kind} needs re = the request id it answers")
        _need(to != BROADCAST or kind == "cancel", f"{kind} goes to one robot")
    else:
        _need(re_ is None or (isinstance(re_, str) and ID_RE.match(re_)), "re must be a message id")
    payload = msg.get("payload") or {}
    validate_payload(kind, payload)
    return {"v": VERSION, "id": mid, "kind": kind, "to": to, "re": re_, "payload": payload}


def validate_hello(h: dict) -> dict:
    """The join handshake. Returns the normalised peer profile the hub keeps in its roster."""
    _need(isinstance(h, dict), "hello must be a JSON object")
    rid = h.get("robot_id")
    _need(isinstance(rid, str) and ROBOT_ID_RE.match(rid) and rid != HUB_ID,
          "robot_id must match [a-z0-9][a-z0-9_.-]{1,47} (e.g. 'so101-a', 'elrobot-b')")
    versions = h.get("protocol_versions", [VERSION])
    _need(isinstance(versions, list) and any(x in SUPPORTED_VERSIONS for x in versions),
          f"no common protocol version (hub: {list(SUPPORTED_VERSIONS)}, peer: {versions})", 426)
    lists = {k: h.get(k) or [] for k in ("capabilities", "skills")}
    for k, v in lists.items():
        _need(isinstance(v, list) and all(isinstance(x, str) for x in v) and len(v) <= 200, f"{k} must be a list of strings")
    frames = h.get("frames") or {}
    _need(isinstance(frames, dict), "frames must be an object")
    if frames.get("base_in_world") is not None:
        validate_pose({"name": "base", **frames["base_in_world"]}, "frames.base_in_world")
    for k in ("connector", "display_name", "agent"):
        _need(h.get(k) is None or (isinstance(h[k], str) and len(h[k]) <= MAX_TASK), f"{k} must be a short string")
    return {"robot_id": rid, "connector": h.get("connector") or "unknown", "display_name": h.get("display_name") or rid,
            "agent": h.get("agent") or "", **lists, "frames": frames, "protocol_version": VERSION}


@dataclass
class Request:
    """One request/help and where it stands. `to` is the addressee ('*' for broadcast help until someone accepts)."""
    id: str
    kind: str
    requester: str
    to: str
    task: str
    created: float
    deadline: float
    state: str = OPEN
    assignee: str | None = None
    progress: float | None = None
    updated: float = 0.0
    history: list = field(default_factory=list)       # [(ts, from, kind)]

    def view(self, now: float) -> dict:
        return {"id": self.id, "kind": self.kind, "requester": self.requester, "to": self.to, "task": self.task,
                "state": self.state, "assignee": self.assignee, "progress": self.progress,
                "remaining_s": None if self.state in TERMINAL else round(max(0.0, self.deadline - now), 1),
                "history": [{"ts": t, "from": f, "kind": k} for t, f, k in self.history[-20:]]}


class RequestTracker:
    """The request state machine. The hub owns the authoritative copy; transitions are checked against the sender.

    open --accept(addressee)--> accepted --progress*--> accepted --result(ok)--> done | result(!ok) --> failed
    open --reject(addressee)--> rejected      (broadcast help: a reject is passed on but the request stays open)
    open|accepted --cancel(requester)--> cancelled
    open|accepted --deadline passes--> expired (the hub tells both sides)
    """

    def __init__(self):
        self.requests: dict[str, Request] = {}

    def open(self, msg: dict, sender: str, now: float) -> Request:
        p = msg["payload"]
        ttl = float(p.get("ttl_s") or DEFAULT_TTL_S)
        task = p.get("task") or ("help: " + p.get("problem", "")[:80])
        r = Request(msg["id"], msg["kind"], sender, msg["to"], task, now, now + ttl, updated=now)
        r.history.append((now, sender, msg["kind"]))
        self.requests[r.id] = r
        return r

    def apply(self, msg: dict, sender: str, now: float) -> Request:
        """Check a reply/cancel against the state machine and update the request. Raises ProtocolError."""
        r = self.requests.get(msg["re"])
        _need(r is not None, f"unknown request {msg['re']}", 404)
        kind = msg["kind"]
        _need(r.state not in TERMINAL, f"request {r.id} is already {r.state}", 409)
        if kind == "cancel":
            _need(sender == r.requester, "only the requester can cancel a request", 403)
            r.state = CANCELLED
        else:
            _need(msg["to"] == r.requester, f"{kind} must be addressed to the requester ({r.requester})")
            addressed = r.to == sender or (r.to == BROADCAST and sender != r.requester)
            _need(addressed, f"{sender} is not the addressee of request {r.id}", 403)
            if kind == "accept":
                _need(r.state == OPEN, f"request {r.id} was already accepted by {r.assignee}", 409)
                r.state, r.assignee = ACCEPTED, sender
            elif kind == "reject":
                _need(r.state == OPEN or r.assignee == sender, f"request {r.id} is assigned to {r.assignee}", 409)
                if r.to != BROADCAST or r.assignee == sender:
                    r.state = REJECTED
            else:                                       # progress / result: only after accepting
                _need(r.state == ACCEPTED and r.assignee == sender, f"accept request {r.id} before sending {kind}", 409)
                if kind == "progress":
                    if msg["payload"].get("fraction") is not None:
                        r.progress = float(msg["payload"]["fraction"])
                else:
                    r.state = DONE if msg["payload"]["ok"] else FAILED
                    r.progress = 1.0 if msg["payload"]["ok"] else r.progress
        r.updated = now
        r.history.append((now, sender, kind))
        return r

    def expire(self, now: float) -> list[Request]:
        out = []
        for r in self.requests.values():
            if r.state not in TERMINAL and now >= r.deadline:
                r.state, r.updated = EXPIRED, now
                r.history.append((now, HUB_ID, "expired"))
                out.append(r)
        return out

    def prune(self, now: float, keep_s: float) -> None:
        """Forget terminal requests older than keep_s (the log on disk keeps them)."""
        for rid in [rid for rid, r in self.requests.items() if r.state in TERMINAL and now - r.updated > keep_s]:
            del self.requests[rid]
