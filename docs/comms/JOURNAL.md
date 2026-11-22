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

**2026-11-22 08:42 CET — Codex takeover.** Devin stopped at WIP `993614f` after quota exhaustion. Read the common
brief, full branch diff/stat, existing journal, and tail of `logs/r5-comms.log`; working tree initially clean.
No commits will be made, per this run's explicit instruction. Read the three upstream transport examples again.
The sibling Generaloid branch is at `16d6651`: `core/hooks.py` calls `register(server, connector)` after mounting
connector tools, matching this module. The original empty `generaloid/__init__.py` remains empty.
Started the prescribed full baseline and all comms tests with uv and Station disabled; outputs on external disk:
`takeover-baseline.txt`, `takeover-comms-before.txt`. No live services or tunnels started.
Critical review in progress: malformed JSON shapes can currently raise internal exceptions, deadline enforcement
relies only on the sweeper, cancellation can address a departed peer, and the HTTP body limit buffers before
checking streamed bodies. Documentation beyond the journal/evidence is missing and will be completed.

**08:47 CET — Review results and repairs.** Original comms suite: **53 passed, 1 skipped, 42.06 s**; the skipped
test needs the sibling Generaloid core. Full baseline: **278 passed, 6 failed, 1 skipped, 3 errors, 99.40 s**.
Failures include MuJoCo CGL `invalid CoreGraphics connection` in executor/learning rendering; the known table-hit
preflight assertion also fails. These are outside owned files. No sandbox permission escalation attempted.
Fixed malformed kind/payload/frame shapes, strict finite JSON numbers and base64 validation, incremental request
body limits including chunked transfer, exact deadline enforcement at send time, cancellation after a recipient
leaves, and conflicting reuse of an idempotency id (409). Added broadcast `request_claimed` notification so other
candidates stop showing accepted help as waiting. Cancellation of broadcast help now reaches all candidates.
Bounded client request bookkeeping and guarded concurrent send/poller reconnects against stale session responses.
Added targeted regression tests and started the full comms suite again.

**08:49 CET — Integration and docs.** Repaired suite: **66 passed, 1 skipped, 30.13 s**. Created a temporary combined
package by copying sibling `generaloid/` plus this branch's comms, with other repo entries symlinked read-only by
convention; no sibling worktree edited. Combined SO-101/ElRobot Generaloid process test: **1 passed, 6.69 s**.
Added README, full protocol/lifecycle/sequence references, intended-hardware walkthrough, security limitations and
Cloudflare commands. Checked Cloudflare's official Quick Tunnel docs: SSE is unsupported there, so the built-in
long-poll client is essential. The optional SSE endpoint is localhost-tested only. Updated demo result wording to
avoid implying the gripper test physically held a bottle. Fixed token-file creation permissions, thumbnail budget
and IPv6 URL formatting. Added deterministic reconnect-race and CLI credential tests. Final comms and regression
runs started. An attempted process inventory with `ps` was denied by the sandbox; session polling and output files
provide progress instead. No escalation or live-service inspection is needed.

**08:52 CET — Regression comparison.** Final full common suite matches baseline exactly: **278 passed, 6 failed,
1 skipped, 3 errors, 150.86 s**; the same six failure names and three error names, no new failures. Comms run before
the last boundary-case additions: **70 passed, 1 skipped, 43.93 s**. Added overflow-safe pose/reach validation,
robust audit stripping for extra image metadata, a bounded demo result wait and a real standalone hub/client
subprocess check. The final combined-package suite is running against all final executable sources, including
both distinct Generaloid twins. It runs with `nice -n 10`. Final captures will replace Devin's WIP captures so
that result wording and frame labels agree with what was actually demonstrated.

**08:52 CET — Sandbox note.** `nice -n 10` reported `setpriority: Operation not permitted`; its child test command
still runs at the inherited priority. Workaround: keep the remaining verification/demo jobs sequential, after the
full baseline/regression processes have completed. `ps` is also denied; tracked command sessions and log files
are sufficient. Neither restriction requires elevated access or contact with a live service.

**08:53 CET — Complete.** Final combined suite: **74 passed in 46.92 s**, no skips/failures. This includes all
boundary-case additions, deterministic stale-poll/concurrent-rejoin regressions, standalone hub/client processes,
two legacy twin MCPs and the separate SO-101/ElRobot Generaloid connectors. Local-only run before the last three
regressions: 70 passed, 1 expected core-absent skip. Final combined verification validates those three as well.
Executed the final demos sequentially with Station disabled: `demo --twin` succeeded in **20.83 s**;
`demo --generaloid` succeeded in **6.99 s** in the temporary combined package. Copied both JSONL/timeline pairs to
`docs/comms/evidence/`, replacing WIP captures. Added evidence provenance and measured results. All jobs exited.
`git diff --check` passes, local Markdown links resolve, and all changed/untracked source files are inside the
owned `generaloid/comms/` and `docs/comms/` paths. No real arm, Station client or public tunnel was used.

Integrator handoff:
- No new commits or git mutations; all takeover changes remain in the working tree. Last commit stays `993614f`.
- Files changed: comms protocol, hub, client, tools, CLI, demo/twin labels, integration-test documentation; new CLI
  and client-race tests, additional hub/protocol tests; five documentation guides, journal and five evidence files.
- The empty `generaloid/__init__.py` is inherited from Devin's initial package creation; preserve the sibling
  initializer if it later differs. `register(server, connector)` matches sibling core at `16d6651` (rechecked).
- Optional core improvement outside owned scope: retain the returned CommsTools state and call `leave()` during
  mount shutdown. Current core ignores the return; explicit `comms_leave` and process exit still work and are tested.
- Full common regression names/counts match baseline. Graphics-dependent failures remain environmental; no attempt
  was made to change the platform renderer or unrelated preflight expectations.
- Remaining limits are intentional and documented: no hardware grasp/cooperation proof, no internet tunnel test,
  transient hub state (audit log is not recovery), bounded delivery/idempotency, no distributed collision planner
  or frame calibration, and no global resource quotas against authorized malicious peers. No blocking question.
