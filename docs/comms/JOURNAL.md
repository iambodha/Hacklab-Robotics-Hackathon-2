# r5-comms journal

Times are local (22 Nov 2026). Newest at the bottom.

**08:14** Read the brief and `prompts/r5-common.md`. Worktree `ws/r5-comms` on `r5-comms` at `e3cdeaa`. Neither this
worktree nor `ws/r5-generaloid` has a `generaloid/` package yet, so the hook signature is mine to define:
`generaloid.comms.register(server, connector)`, as the r5-generaloid brief already names it. Created an empty
`generaloid/__init__.py` (the brief allows that when it does not exist).

**08:16** Read the starting points: `origin/MCP:robot_mcp/http_server.py` (FastMCP over streamable HTTP behind a
bearer-token ASGI wrapper, `hmac.compare_digest`, token file with mode 600), `origin/kerandia:scripts/
station_https_relay.py` (Station's UI uses `ws://` and breaks behind an https tunnel; the relay rewrites one string)
and `scripts/lan_forward.py` (raw TCP forwarder with a loud "no auth" warning). Lessons kept: bearer tokens compared
in constant time, a health endpoint without auth, bind loopback, and plain `ws://` is a trap behind https tunnels.

Baseline test run started in the background (`baseline.txt` under `DATA_ROOT`).

**08:20** Decisions on transport:
- **HTTP + long-polling as the primary channel, SSE as an extra**, not WebSocket. Reasons: starlette, uvicorn, httpx
  and sse-starlette all ship with `mcp` (checked: mcp 1.30, starlette 1.10, uvicorn 0.54, httpx 0.28), so there is
  no new dependency; long-polls pass through Cloudflare quick tunnels, corporate proxies and anything that speaks
  HTTP/1.1, and do not have the `ws://`-on-https trap the kerandia relay had to patch; a robot exchanges a handful of
  messages a minute, so the ~1 RTT extra latency does not matter. Long-poll cap 25 s (Cloudflare's proxy read
  timeout is 100 s).
- **The hub is authoritative for request state.** It stamps `from` (a peer cannot speak for another), computes
  deadlines from a relative `ttl_s` on its own clock (robots' clocks may be skewed), enforces the request state
  machine (only the addressee accepts/progresses/finishes, only the requester cancels) and sends `expired` to both
  sides when a deadline passes.
- **Delivery: per-robot mailbox with sequence numbers**, acked by `since=<seq>` on the next poll; at-least-once with
  client-side de-duplication by message id; sends idempotent by `(sender, id)`.
- **Client is thread-based** (httpx.Client + a poller thread) so it runs inside the MCP server's event loop, a script
  or a test without touching the host loop.
- **Core safety principle:** comms only reads the connector's identity (name, skills, capabilities, frames). No code
  path in `generaloid/comms` calls anything that moves an arm. Incoming requests are surfaced through `comms_inbox`,
  labelled as untrusted peer content, and the local agent decides with its own tools.

**08:45** Wrote `protocol.py` (stdlib only), `hub.py`, `client.py`, `tools.py`, `__main__.py`, `twin_server.py`.
Smoke run on localhost: A sends help, B accepts, progress 0.5, result ok; A sees accept/progress/result; hub state
`done`. Committed `a1aa0ba`.
