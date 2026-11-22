"""A comms client: joins a hub, long-polls its mailbox on a background thread, sends with retries.

Thread-based on purpose: it runs the same inside an MCP server's event loop, a plain script or a test, without
touching the host's loop. Uses httpx (already a dependency of `mcp`).

Delivery is at-least-once with de-duplication: the hub numbers each robot's messages (`seq`), the client acks by
polling with `since=<last seq>`, and every message id is remembered so a redelivery after a reconnect is dropped.
Sends are idempotent: a retry reuses the message id and the hub answers with the original ack.

Reconnect: on a network error the poller backs off (0.5 s doubling to 10 s) and retries; if the hub answers 401
(session unknown: the hub restarted, or dropped us after `peer_drop_s`), the client joins again with its old session
as `resume`, which keeps its mailbox if the hub still has it.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict, deque

import httpx

from . import protocol as P

POLL_WAIT_S = 20.0          # long-poll length; the hub caps it at HubConfig.max_wait_s (25 s)
BACKOFF_MIN_S = 0.5
BACKOFF_MAX_S = 10.0         # a robot back online is noticed within 10 s
SEND_RETRIES = 3             # network-level retries of one send (same id, so never delivered twice)
LOCAL_KEEP = 2000            # messages kept locally for comms_inbox
SEEN_KEEP = 10000

logging.getLogger("httpx").setLevel(logging.WARNING)   # one INFO line per long-poll is noise on an MCP server's stderr


class CommsError(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(f"{code}: {msg}" if code else msg)
        self.code = code


class CommsClient:
    def __init__(self, url: str, token: str, hello: dict, *, poll_wait: float = POLL_WAIT_S,
                 transport: httpx.BaseTransport | None = None, timeout: float = 10.0):
        self.url = url.rstrip("/")
        self.token = token
        self.hello = {"protocol_versions": list(P.SUPPORTED_VERSIONS), **hello}
        self.robot_id = hello["robot_id"]
        self.poll_wait = poll_wait
        self.http = httpx.Client(transport=transport, timeout=timeout, follow_redirects=False,
                                 headers={"User-Agent": f"{P.PROTOCOL}/{P.VERSION}"})
        self.session: str | None = None
        self.generation = 0  # discard a poll response from a session replaced by a concurrent send/rejoin
        self.cursor = 0                       # last hub seq received (acked on the next poll)
        self.n = 0                            # local message counter, what comms_inbox(since) refers to
        self.messages: deque = deque(maxlen=LOCAL_KEEP)
        self.seen: OrderedDict = OrderedDict()
        self.incoming: dict[str, dict] = {}   # request/help messages addressed to us, by id
        self.closed: set[str] = set()         # incoming requests we finished/rejected, or that were cancelled/expired
        self.peers_cache: list[dict] = []
        self.connected = False
        self.last_error = ""
        self.reconnects = 0
        self.cond = threading.Condition()
        self.stop_ev = threading.Event()
        self.join_lock = threading.Lock()
        self.thread: threading.Thread | None = None

    # ---------- HTTP ----------
    def _call(self, method: str, path: str, *, auth: str | None = None, json=None, params=None, timeout=None):
        headers = {"Authorization": f"Bearer {auth if auth is not None else self.session or ''}"}
        r = self.http.request(method, self.url + path, json=json, params=params, headers=headers,
                              timeout=timeout or self.http.timeout)
        try:
            data = r.json()
        except ValueError:
            data = {"error": r.text[:200]}
        if r.status_code >= 400:
            raise CommsError(r.status_code, data.get("error", r.reason_phrase))
        return data

    # ---------- session ----------
    def join(self, start_polling: bool = True) -> dict:
        body = {"hello": self.hello}
        if self.session:
            body["resume"] = self.session
        data = self._call("POST", "/v1/join", auth=self.token, json=body)
        with self.cond:
            self.session = data["session"]
            self.generation += 1
            if not data["resumed"]:
                self.cursor = data["cursor"]
                self.incoming.clear()
                self.closed.clear()
        self.peers_cache = data["peers"]
        self.connected, self.last_error = True, ""
        if start_polling and (self.thread is None or not self.thread.is_alive()):
            self.stop_ev.clear()
            self.thread = threading.Thread(target=self._poll_loop, name=f"comms-{self.robot_id}", daemon=True)
            self.thread.start()
        return data

    def _rejoin(self, failed_session):
        with self.join_lock:
            if self.stop_ev.is_set() or self.session != failed_session:
                return
            self.reconnects += 1
            self.join(start_polling=False)

    def leave(self):
        self.stop_ev.set()
        if self.session:
            try:
                self._call("POST", "/v1/leave", timeout=5)
            except (CommsError, httpx.HTTPError):
                pass
        self.session, self.connected = None, False
        if self.thread:
            self.thread.join(timeout=self.poll_wait + 5)
        self.http.close()

    # ---------- receive ----------
    def _poll_loop(self):
        backoff = BACKOFF_MIN_S
        while not self.stop_ev.is_set():
            with self.cond:
                session, generation, cursor = self.session, self.generation, self.cursor
            try:
                data = self._call("GET", "/v1/inbox", auth=session, params={"since": cursor, "wait": self.poll_wait},
                                  timeout=self.poll_wait + 15)
                with self.cond:
                    if generation != self.generation:
                        continue
                    self._ingest(data["messages"])
                    self.cursor = max(self.cursor, data["cursor"])
                self.connected, backoff = True, BACKOFF_MIN_S
            except CommsError as e:
                if self.stop_ev.is_set():
                    return
                self.connected, self.last_error = False, str(e)
                if e.code == 401:
                    try:
                        self._rejoin(session)
                        continue
                    except (CommsError, httpx.HTTPError) as e2:
                        self.last_error = f"rejoin failed: {e2}"
                self.stop_ev.wait(backoff)
                backoff = min(BACKOFF_MAX_S, backoff * 2)
            except httpx.HTTPError as e:
                if self.stop_ev.is_set():
                    return
                self.connected, self.last_error = False, f"{type(e).__name__}: {e}"
                self.stop_ev.wait(backoff)
                backoff = min(BACKOFF_MAX_S, backoff * 2)

    def _ingest(self, msgs: list[dict]):
        with self.cond:
            for m in msgs:
                key = (m.get("from"), m["id"])
                if key in self.seen:
                    continue
                self.seen[key] = True
                while len(self.seen) > SEEN_KEEP:
                    self.seen.popitem(last=False)
                self.n += 1
                local = {**m, "n": self.n}
                self.messages.append(local)
                if m["kind"] in P.REQUEST_KINDS:
                    self.incoming[m["id"]] = local
                elif m["kind"] in ("cancel", "expired", "request_claimed") and m.get("re"):
                    self.closed.add(m["re"])
                elif m["kind"] == "peer_joined":
                    rid = m["payload"]["peer"]["robot_id"]
                    self.peers_cache = [p for p in self.peers_cache if p["robot_id"] != rid] + \
                        [{**m["payload"]["peer"], "online": True}]
                elif m["kind"] == "peer_left":
                    self.peers_cache = [p for p in self.peers_cache if p["robot_id"] != m["payload"]["robot_id"]]
            # Keep request bookkeeping bounded alongside the replay cache. The hub remains authoritative.
            while len(self.incoming) > LOCAL_KEEP:
                self.incoming.pop(next(iter(self.incoming)))
            self.closed.intersection_update(self.incoming)
            self.cond.notify_all()

    def inbox(self, since: int = 0, wait: float = 0.0, kinds: set | None = None) -> list[dict]:
        """Messages with local number > since (optionally only some kinds), waiting up to `wait` s for one."""
        deadline = time.time() + wait
        with self.cond:
            while True:
                out = [m for m in self.messages if m["n"] > since and (not kinds or m["kind"] in kinds)]
                remaining = deadline - time.time()
                if out or remaining <= 0:
                    return out
                self.cond.wait(remaining)

    def wait_for(self, pred, timeout: float = 10.0, since: int = 0) -> dict:
        """Block until a message matching pred arrives (scripted agents and tests). Raises TimeoutError."""
        deadline = time.time() + timeout
        with self.cond:
            while True:
                for m in self.messages:
                    if m["n"] > since and pred(m):
                        return m
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError(f"{self.robot_id}: no matching message within {timeout} s")
                self.cond.wait(remaining)

    # ---------- send ----------
    def send(self, to: str, kind: str, payload: dict | None = None, re: str | None = None,
             msg_id: str | None = None) -> dict:
        env = {"v": P.VERSION, "id": msg_id or P.new_id(), "kind": kind, "to": to, "re": re, "payload": payload or {}}
        P.validate_envelope(env)                           # fail fast, locally, with the same rules as the hub
        last: Exception | None = None
        for attempt in range(SEND_RETRIES + 1):
            session = self.session
            try:
                return self._call("POST", "/v1/send", auth=session, json=env)
            except CommsError as e:
                if e.code != 401 or attempt == SEND_RETRIES:
                    raise
                last = e
                self._rejoin(session)
            except httpx.HTTPError as e:                   # the hub may or may not have it: retry with the same id
                last = e
                time.sleep(min(BACKOFF_MAX_S, BACKOFF_MIN_S * 2 ** attempt))
        raise CommsError(0, f"send failed after {SEND_RETRIES + 1} attempts: {last}")

    def request(self, to: str, task: str, prompt: str = "", args: dict | None = None, ttl_s: float | None = None,
                **extra) -> dict:
        payload = {"task": task, "prompt": prompt, "args": args or {}, **extra}
        if ttl_s is not None:
            payload["ttl_s"] = ttl_s
        return self.send(to, "request", payload)

    def help(self, to: str, problem: str, ask: str = "", context: dict | None = None, ttl_s: float | None = None,
             **extra) -> dict:
        payload = {"problem": problem, "ask": ask, "context": context or {}, **extra}
        if ttl_s is not None:
            payload["ttl_s"] = ttl_s
        return self.send(to, "help", payload)

    def reply(self, request_id: str, kind: str, payload: dict | None = None) -> dict:
        """accept / reject / progress / result for a request someone sent us."""
        if kind not in P.REPLY_KINDS:
            raise CommsError(0, f"reply kind must be one of {sorted(P.REPLY_KINDS)}")
        req = self.incoming.get(request_id)
        if req is None:
            raise CommsError(0, f"no incoming request {request_id} (see comms_inbox)")
        ack = self.send(req["from"], kind, payload or {}, re=request_id)
        if kind in ("reject", "result"):
            self.closed.add(request_id)
        return ack

    def waiting_for_me(self) -> list[dict]:
        """Incoming requests still waiting for this robot's decision or result."""
        with self.cond:
            return [m for rid, m in self.incoming.items() if rid not in self.closed]

    def cancel(self, request_id: str, reason: str = "") -> dict:
        return self.send(P.BROADCAST, "cancel", {"reason": reason}, re=request_id)

    # ---------- queries ----------
    def peers(self) -> list[dict]:
        self.peers_cache = self._call("GET", "/v1/peers")["peers"]
        return self.peers_cache

    def requests(self) -> list[dict]:
        return self._call("GET", "/v1/requests")["requests"]

    def log(self, limit: int = 200) -> list[dict]:
        return self._call("GET", "/v1/log", params={"limit": limit})["log"]

    def heartbeat(self, status: dict | None = None) -> dict:
        return self._call("POST", "/v1/heartbeat", json={"status": status} if status else {})

    def state(self) -> dict:
        return {"robot_id": self.robot_id, "hub": self.url, "connected": self.connected, "cursor": self.cursor,
                "local_messages": self.n, "reconnects": self.reconnects, "last_error": self.last_error}
