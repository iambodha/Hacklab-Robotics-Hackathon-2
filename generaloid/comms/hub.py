"""The comms hub: one robot hosts it, the others join over HTTP(S). Roster, routing, request tracking, a log.

Built on starlette + uvicorn, which the `mcp` package already depends on: no new dependencies.

    POST /v1/join       Bearer <join token | per-peer key>   {"hello": {...}, "resume": "<old session>"?}
    POST /v1/send       Bearer <session>                      envelope (see protocol.py)
    GET  /v1/inbox      Bearer <session>  ?since=<seq>&wait=<s>&limit=<n>   long-poll; `since` also acks
    GET  /v1/stream     Bearer <session>  ?since=<seq>        the same mailbox as server-sent events
    POST /v1/heartbeat  Bearer <session>                      {"status"?: {...}}
    GET  /v1/peers      Bearer <session>
    GET  /v1/requests   Bearer <session>                      requests this robot sent, received or holds
    GET  /v1/log        Bearer <session>  ?limit=<n>          this robot's slice of the hub log
    POST /v1/leave      Bearer <session>
    GET  /health        (no auth; no robot names)

Security (docs/comms/SECURITY.md): nothing is accepted without a token; the shared join token (or a per-peer key)
only buys a session key; every other call needs that session key; the hub stamps `from`, so a peer cannot speak for
another; per-peer token buckets and per-address auth-failure lockout; bodies capped. The hub binds 127.0.0.1 by
default: to reach it from another machine, put cloudflared in front (docs/comms/CLOUDFLARE.md).

The hub routes messages. It has no access to any arm, and nothing it receives can move one.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import secrets
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request as HTTPRequest
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from . import protocol as P


@dataclass
class HubConfig:
    """Tunables. Every number says where it came from."""
    # Long-poll cap. Cloudflare closes a proxied request after 100 s without a response byte (its documented proxy
    # read timeout); 25 s keeps far below that and below the ~30-60 s idle cut of many home routers and corporate
    # proxies.
    max_wait_s: float = 25.0
    # A peer that has not called anything for 45 s (two missed 20 s long-polls + slack) shows as offline.
    peer_stale_s: float = 45.0
    # ...and is dropped from the roster after 5 min: long enough to ride out a Wi-Fi hiccup or a cloudflared restart
    # and resume with its queued messages, short enough to free the robot id after a crash.
    peer_drop_s: float = 300.0
    sweep_s: float = 0.5                   # deadline/staleness check period: request timeouts fire within 0.5 s
    keep_terminal_s: float = 600.0         # finished requests stay queryable for 10 min (the log keeps them forever)
    mailbox_max: int = 1000                # undelivered messages per robot; beyond that the oldest are dropped
    seen_ids_max: int = 20000              # idempotency window: (sender, id) pairs remembered
    max_body_bytes: int = 512 * 1024       # one request body; fits a few inline JPEG thumbnails
    # Sends: 10 msg/s sustained, burst 30. An agent sends a handful per task; progress every ~1 s is plenty.
    send_rate: float = 10.0
    send_burst: float = 30.0
    # All authenticated calls (polls, peers, heartbeats): 20/s, burst 60.
    call_rate: float = 20.0
    call_burst: float = 60.0
    # 5 wrong tokens from one address within 60 s lock that address out for 60 s (a 192-bit token is not
    # guessable anyway; this stops log spam and slow brute force through a public tunnel URL).
    auth_fail_max: int = 5
    auth_fail_window_s: float = 60.0
    # Behind cloudflared every request comes from 127.0.0.1; trust CF-Connecting-IP only when the hub is known to
    # sit behind Cloudflare (otherwise a client could spoof the header to dodge the lockout).
    trust_proxy: bool = False


class TokenBucket:
    def __init__(self, rate: float, burst: float, now: float):
        self.rate, self.burst, self.tokens, self.t = rate, burst, burst, now

    def take(self, now: float) -> bool:
        self.tokens = min(self.burst, self.tokens + (now - self.t) * self.rate)
        self.t = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


@dataclass
class Peer:
    profile: dict
    session_hash: str
    joined: float
    last_seen: float
    send_bucket: TokenBucket
    call_bucket: TokenBucket
    mailbox: deque = field(default_factory=deque)
    next_seq: int = 1
    dropped: int = 0                        # messages lost to mailbox overflow
    polling: int = 0                        # open long-polls / streams: a peer waiting on the hub is alive
    status: dict = field(default_factory=dict)
    wake: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def robot_id(self) -> str:
        return self.profile["robot_id"]


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _strip_images(payload: dict) -> dict:
    """Log copy of a payload: inline image data replaced by its size, so the log stays small."""
    if not isinstance(payload.get("images"), list):
        return payload
    imgs = [{**{k: v for k, v in i.items() if k != "data_b64"}, "data_b64_len": len(i["data_b64"])}
            if isinstance(i, dict) and "data_b64" in i else i for i in payload["images"]]
    return {**payload, "images": imgs}


class HubError(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


class Hub:
    """All state lives on the hub's event loop; handlers never block, so no locks are needed."""

    def __init__(self, token: str | None = None, peer_keys: dict[str, str] | None = None,
                 log_path: str | Path | None = None, config: HubConfig | None = None, clock=time.time):
        self.peer_keys = dict(peer_keys or {})
        self.token = token or (None if self.peer_keys else secrets.token_urlsafe(24))
        self.config = config or HubConfig()
        self.clock = clock
        self.peers: dict[str, Peer] = {}
        self.sessions: dict[str, str] = {}              # sha256(session key) -> robot id
        self.tracker = P.RequestTracker()
        self.seen: OrderedDict = OrderedDict()           # (sender, id) -> ack
        self.auth_fails: dict[str, deque] = {}
        self.log_path = Path(log_path) if log_path else None
        self.log: deque = deque(maxlen=5000)
        self.started = self.clock()

    # ---------- log ----------
    def _log(self, event: str, **kw):
        rec = {"ts": round(self.clock(), 3), "event": event, **kw}
        self.log.append(rec)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")

    # ---------- auth ----------
    def _client_addr(self, req: HTTPRequest) -> str:
        if self.config.trust_proxy and req.headers.get("cf-connecting-ip"):
            return req.headers["cf-connecting-ip"]
        return req.client.host if req.client else "?"

    def _check_lockout(self, addr: str, now: float):
        fails = self.auth_fails.get(addr)
        if fails:
            while fails and now - fails[0] > self.config.auth_fail_window_s:
                fails.popleft()
            if len(fails) >= self.config.auth_fail_max:
                raise HubError(429, "too many failed authentications from this address; wait a minute")

    def _auth_failed(self, addr: str, now: float, why: str):
        self.auth_fails.setdefault(addr, deque()).append(now)
        self._log("auth_fail", addr=addr, why=why)
        raise HubError(401, why)

    @staticmethod
    def _bearer(req: HTTPRequest) -> str:
        auth = req.headers.get("authorization", "")
        return auth[7:].strip() if auth[:7].lower() == "bearer " else ""

    def _session(self, req: HTTPRequest, now: float) -> Peer:
        """A valid session key always gets through, even while its address is locked out: behind cloudflared every
        peer shares 127.0.0.1, and a stranger spamming bad tokens must not cut off robots already in session."""
        key = self._bearer(req)
        rid = self.sessions.get(_hash(key)) if key else None
        if rid is None or rid not in self.peers:
            addr = self._client_addr(req)
            self._check_lockout(addr, now)
            self._auth_failed(addr, now, "unknown or expired session; join again")
        peer = self.peers[rid]
        if not peer.call_bucket.take(now):
            raise HubError(429, "rate limit: too many calls")
        peer.last_seen = now
        return peer

    # ---------- delivery ----------
    def _deliver(self, peer: Peer, msg: dict):
        m = {**msg, "seq": peer.next_seq}
        peer.next_seq += 1
        peer.mailbox.append(m)
        while len(peer.mailbox) > self.config.mailbox_max:
            peer.mailbox.popleft()
            peer.dropped += 1
        peer.wake.set()
        peer.wake = asyncio.Event()

    def _system(self, to: list[Peer], kind: str, payload: dict, re: str | None = None):
        msg = {"v": P.VERSION, "id": P.new_id(), "kind": kind, "from": P.HUB_ID, "to": "*", "re": re,
               "ts": round(self.clock(), 3), "payload": payload}
        for p in to:
            self._deliver(p, msg)
        self._log("message", **{k: msg[k] for k in ("id", "kind", "from", "to", "re")}, payload=payload)

    def _alive(self, p: Peer, now: float, limit: float) -> bool:
        return p.polling > 0 or now - p.last_seen < limit

    def roster(self, now: float) -> list[dict]:
        return [{**p.profile, "online": self._alive(p, now, self.config.peer_stale_s),
                 "last_seen_s": round(now - p.last_seen, 1), "status": p.status} for p in self.peers.values()]

    # ---------- operations (called by the HTTP handlers; also usable directly in tests) ----------
    def join(self, req_key: str, body: dict, addr: str) -> dict:
        now = self.clock()
        self._check_lockout(addr, now)
        try:
            profile = P.validate_hello(body.get("hello") or {})
        except P.ProtocolError as e:
            raise HubError(e.code, str(e))
        rid = profile["robot_id"]
        if self.peer_keys:
            expected = self.peer_keys.get(rid)
            if expected is None or not hmac.compare_digest(req_key.encode(), expected.encode()):
                self._auth_failed(addr, now, f"no valid key for robot {rid}")
        elif not (req_key and hmac.compare_digest(req_key.encode(), self.token.encode())):
            self._auth_failed(addr, now, "wrong or missing join token")
        old = self.peers.get(rid)
        resumed = False
        if old is not None:
            resume = body.get("resume") or ""
            if resume and hmac.compare_digest(_hash(resume), old.session_hash):
                resumed = True
            elif self._alive(old, now, self.config.peer_stale_s):
                raise HubError(409, f"robot id {rid} is already connected; pick another id or wait "
                                    f"{self.config.peer_stale_s:.0f} s for it to go stale")
            self.sessions.pop(old.session_hash, None)
        session = secrets.token_urlsafe(32)
        if old is not None:                          # same robot back: keep its mailbox and sequence numbers
            old.profile, old.session_hash, old.last_seen = profile, _hash(session), now
            peer = old
        else:
            c = self.config
            peer = Peer(profile, _hash(session), now, now, TokenBucket(c.send_rate, c.send_burst, now),
                        TokenBucket(c.call_rate, c.call_burst, now))
            self.peers[rid] = peer
        self.sessions[peer.session_hash] = rid
        self._log("join", robot_id=rid, addr=addr, resumed=resumed, rejoined=old is not None,
                  connector=profile["connector"])
        others = [p for p in self.peers.values() if p is not peer]
        self._system(others, "peer_joined", {"peer": profile, "resumed": resumed})
        return {"ok": True, "session": session, "robot_id": rid, "resumed": resumed, "rejoined": old is not None,
                "cursor": peer.next_seq - 1 - len(peer.mailbox),
                "hub": {"protocol": P.PROTOCOL, "version": P.VERSION, "max_wait_s": self.config.max_wait_s,
                        "max_body_bytes": self.config.max_body_bytes, "peer_stale_s": self.config.peer_stale_s},
                "peers": self.roster(now)}

    def send(self, peer: Peer, body: dict) -> dict:
        now = self.clock()
        try:
            env = P.validate_envelope(body)
        except P.ProtocolError as e:
            raise HubError(e.code, str(e))
        key = (peer.robot_id, env["id"])
        if key in self.seen:                         # idempotent retry: same answer, no second delivery
            return {**self.seen[key], "duplicate": True}
        if not peer.send_bucket.take(now):
            raise HubError(429, f"rate limit: more than {self.config.send_rate:g} messages/s")
        to = env["to"]
        if to != P.BROADCAST and to not in self.peers:
            raise HubError(404, f"no robot {to!r} in this session (see /v1/peers)")
        if to == peer.robot_id:
            raise HubError(400, "a robot cannot message itself")
        kind, request = env["kind"], None
        try:
            if kind in P.REQUEST_KINDS:
                if env["id"] in self.tracker.requests:
                    raise HubError(409, f"request id {env['id']} already used")
                request = self.tracker.open(env, peer.robot_id, now)
            elif kind in P.REPLY_KINDS | P.CANCEL_KINDS:
                request = self.tracker.apply(env, peer.robot_id, now)
                if kind == "cancel":                 # a cancel goes wherever the request went
                    to = request.assignee or request.to
        except P.ProtocolError as e:
            raise HubError(e.code, str(e))
        msg = {**env, "to": to, "from": peer.robot_id, "ts": round(now, 3)}
        targets = [p for p in self.peers.values() if p is not peer] if to == P.BROADCAST else [self.peers[to]]
        for t in targets:
            self._deliver(t, msg)
        self._log("message", **{k: msg[k] for k in ("id", "kind", "from", "to", "re")},
                  payload=_strip_images(env["payload"]), delivered_to=[t.robot_id for t in targets])
        ack = {"ok": True, "id": env["id"], "ts": msg["ts"], "delivered_to": [t.robot_id for t in targets],
               "request": request.view(now) if request else None}
        self.seen[key] = ack
        while len(self.seen) > self.config.seen_ids_max:
            self.seen.popitem(last=False)
        return ack

    def inbox(self, peer: Peer, since: int, limit: int) -> dict:
        while peer.mailbox and peer.mailbox[0]["seq"] <= since:       # `since` acknowledges everything up to it
            peer.mailbox.popleft()
        msgs = [m for m in peer.mailbox if m["seq"] > since][:limit]
        cursor = msgs[-1]["seq"] if msgs else max(since, peer.next_seq - 1 - len(peer.mailbox))
        return {"messages": msgs, "cursor": cursor, "dropped": peer.dropped}

    def leave(self, peer: Peer, reason: str = "left"):
        self.peers.pop(peer.robot_id, None)
        self.sessions.pop(peer.session_hash, None)
        self._log("leave", robot_id=peer.robot_id, reason=reason)
        self._system(list(self.peers.values()), "peer_left", {"robot_id": peer.robot_id, "reason": reason})
        peer.wake.set()

    def sweep(self):
        now = self.clock()
        for r in self.tracker.expire(now):
            to = [self.peers[x] for x in {r.requester, r.assignee or r.to} if x in self.peers]
            if r.to == P.BROADCAST and not r.assignee:
                to = list(self.peers.values())
            self._system(to, "expired", {"request": r.view(now), "reason": "deadline passed"}, re=r.id)
        for p in [p for p in self.peers.values() if not self._alive(p, now, self.config.peer_drop_s)]:
            self.leave(p, reason="timeout")
        self.tracker.prune(now, self.config.keep_terminal_s)

    def requests_for(self, rid: str) -> list[dict]:
        now = self.clock()
        return [r.view(now) for r in self.tracker.requests.values()
                if rid in (r.requester, r.to, r.assignee) or (r.to == P.BROADCAST and not r.assignee)]

    def log_for(self, rid: str, limit: int) -> list[dict]:
        mine = [e for e in self.log if e.get("robot_id") == rid or e.get("from") == rid
                or e.get("to") in (rid, P.BROADCAST) or rid in e.get("delivered_to", ())]
        return mine[-limit:]


def create_app(hub: Hub) -> Starlette:
    cfg = hub.config

    async def body_json(req: HTTPRequest) -> dict:
        if int(req.headers.get("content-length") or 0) > cfg.max_body_bytes:
            raise HubError(413, f"body larger than {cfg.max_body_bytes} bytes")
        raw = await req.body()
        if len(raw) > cfg.max_body_bytes:
            raise HubError(413, f"body larger than {cfg.max_body_bytes} bytes")
        try:
            data = json.loads(raw or b"{}")
        except ValueError:
            raise HubError(400, "body is not JSON")
        if not isinstance(data, dict):
            raise HubError(400, "body must be a JSON object")
        return data

    def handler(fn):
        async def wrapped(req: HTTPRequest):
            try:
                return await fn(req)
            except HubError as e:
                return JSONResponse({"ok": False, "error": str(e)}, status_code=e.code)
        return wrapped

    @handler
    async def health(req):
        return JSONResponse({"ok": True, "service": P.PROTOCOL, "version": P.VERSION,
                             "peers_online": sum(1 for p in hub.roster(hub.clock()) if p["online"])})

    @handler
    async def join(req):
        hub._check_lockout(hub._client_addr(req), hub.clock())
        body = await body_json(req)
        return JSONResponse(hub.join(hub._bearer(req), body, hub._client_addr(req)))

    @handler
    async def send(req):
        peer = hub._session(req, hub.clock())
        return JSONResponse(hub.send(peer, await body_json(req)))

    def _int(req, name, default, lo, hi):
        try:
            return max(lo, min(hi, int(float(req.query_params.get(name, default)))))
        except ValueError:
            raise HubError(400, f"{name} must be a number")

    @handler
    async def inbox(req):
        peer = hub._session(req, hub.clock())
        since = _int(req, "since", 0, 0, 2**62)
        limit = _int(req, "limit", 100, 1, 1000)
        try:
            wait = max(0.0, min(cfg.max_wait_s, float(req.query_params.get("wait", 0))))
        except ValueError:
            raise HubError(400, "wait must be a number")
        deadline = hub.clock() + wait
        peer.polling += 1
        try:
            while True:
                ev = peer.wake
                out = hub.inbox(peer, since, limit)
                remaining = deadline - hub.clock()
                if out["messages"] or remaining <= 0 or hub.peers.get(peer.robot_id) is not peer:
                    return JSONResponse(out)
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(ev.wait(), remaining)
        finally:
            peer.polling -= 1
            peer.last_seen = hub.clock()

    @handler
    async def stream(req):
        peer = hub._session(req, hub.clock())
        since = _int(req, "since", req.headers.get("last-event-id", 0), 0, 2**62)

        async def events():
            cursor = since
            peer.polling += 1
            try:
                async for chunk in _events(cursor):
                    yield chunk
            finally:
                peer.polling -= 1
                peer.last_seen = hub.clock()

        async def _events(cursor):
            while hub.peers.get(peer.robot_id) is peer:
                ev = peer.wake
                out = hub.inbox(peer, cursor, 100)
                for m in out["messages"]:
                    yield f"id: {m['seq']}\nevent: {m['kind']}\ndata: {json.dumps(m)}\n\n"
                cursor = out["cursor"]
                peer.last_seen = hub.clock()
                if not out["messages"]:
                    try:
                        await asyncio.wait_for(ev.wait(), cfg.max_wait_s * 0.6)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @handler
    async def heartbeat(req):
        peer = hub._session(req, hub.clock())
        body = await body_json(req)
        if isinstance(body.get("status"), dict):
            peer.status = body["status"]
        return JSONResponse({"ok": True, "peers_online": sum(1 for p in hub.roster(hub.clock()) if p["online"])})

    @handler
    async def peers(req):
        peer = hub._session(req, hub.clock())
        return JSONResponse({"you": peer.robot_id, "peers": hub.roster(hub.clock())})

    @handler
    async def requests(req):
        peer = hub._session(req, hub.clock())
        return JSONResponse({"requests": hub.requests_for(peer.robot_id)})

    @handler
    async def log(req):
        peer = hub._session(req, hub.clock())
        return JSONResponse({"log": hub.log_for(peer.robot_id, _int(req, "limit", 200, 1, 5000))})

    @handler
    async def leave(req):
        hub.leave(hub._session(req, hub.clock()))
        return JSONResponse({"ok": True})

    @contextlib.asynccontextmanager
    async def lifespan(app):
        async def sweeper():
            while True:
                await asyncio.sleep(cfg.sweep_s)
                hub.sweep()
        task = asyncio.create_task(sweeper())
        hub._log("hub_start", auth="peer_keys" if hub.peer_keys else "token")
        try:
            yield
        finally:
            task.cancel()

    return Starlette(routes=[
        Route("/health", health, methods=["GET"]),
        Route("/v1/join", join, methods=["POST"]),
        Route("/v1/send", send, methods=["POST"]),
        Route("/v1/inbox", inbox, methods=["GET"]),
        Route("/v1/stream", stream, methods=["GET"]),
        Route("/v1/heartbeat", heartbeat, methods=["POST"]),
        Route("/v1/peers", peers, methods=["GET"]),
        Route("/v1/requests", requests, methods=["GET"]),
        Route("/v1/log", log, methods=["GET"]),
        Route("/v1/leave", leave, methods=["POST"]),
    ], lifespan=lifespan)


class HubServer:
    """Run a hub on its own thread and event loop (so it can live inside an MCP server process)."""

    def __init__(self, hub: Hub, host: str = "127.0.0.1", port: int = 0):
        import uvicorn
        self.hub = hub
        self.server = uvicorn.Server(uvicorn.Config(create_app(hub), host=host, port=port, log_level="warning",
                                                    lifespan="on", access_log=False))
        self.host = host
        self.thread = threading.Thread(target=self.server.run, name="comms-hub", daemon=True)

    def start(self, timeout: float = 10.0) -> "HubServer":
        self.thread.start()
        t0 = time.time()
        while not self.server.started:
            if not self.thread.is_alive() or time.time() - t0 > timeout:
                raise RuntimeError("comms hub failed to start (port in use?)")
            time.sleep(0.02)
        return self

    @property
    def port(self) -> int:
        return self.server.servers[0].sockets[0].getsockname()[1]

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "") else self.host
        return f"http://{host}:{self.port}"

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=10)
