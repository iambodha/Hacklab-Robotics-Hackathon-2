# Security and authority

**A message from another robot never directly moves this robot's arm.** The hub routes data and has no connector,
Station client or executor. The MCP adapter reads connector identity, skills and frames; it never invokes motion.
Incoming prompts, images and poses are untrusted peer content. The local agent or human chooses whether to accept,
then calls its own robot's tools under its own safety envelope and motion ownership rules. Neither acceptance nor
a remote `cancel` bypasses those rules. A communication timeout is not an emergency stop.

NormaCore Station remains the central bus/camera/command service. Publish only this comms hub, never Station's
unauthenticated ports 8888/8889, camera relay, recorder or robot motion MCP. The Station HTTPS relay and raw LAN
forwarder from the starting branches solve connectivity, not authorization, and are not used here.

## Credentials and transport

The default shared join token is generated from 24 random bytes. Joins exchange it for a random 32-byte session
credential. The hub stores session SHA-256 hashes, compares secrets in constant time and stamps the sender from the
authenticated session. Join credentials do not authorize other endpoints. Tokens stay out of wire messages,
JSONL and URLs. `comms_host` returns its token to the local agent, so treat that MCP transcript as private.
CLI token files are created exclusively with mode 0600; existing files should also be protected by the operator.

A shared token defines a trusted group: any holder may join with an unoccupied/stale identity and read group
broadcasts/roster. For identity isolation use `hub --peer-keys /private/path/keys.json`, a JSON object mapping exact
robot ids to separate random keys. Give each operator only their own key. New join keys require a hub restart;
existing sessions cannot be individually revoked by an administrative API. Restart clears all sessions/tasks.

Bind defaults to 127.0.0.1. MCP tools expose only loopback hosting. The standalone CLI refuses non-loopback binds
unless the operator starts it with `GENERALOID_COMMS_ALLOW_LAN=1`; plain HTTP on a LAN reveals credentials to
network observers. For internet links use HTTPS through a tunnel, retain bearer auth and certificate verification,
and exchange credentials separately from the URL. Cloudflare terminates TLS and is part of the trust boundary.
The hub URL is operator-supplied; this is not an SSRF sandbox for an agent with arbitrary tool access.

## Controls and remaining limits

Defaults in `HubConfig` have named rationales in code:

| Control | Default |
| --- | --- |
| Body cap (including chunked transfer) | 512 KiB |
| Send bucket per authenticated robot | 10 messages/s, burst 30 |
| All authenticated HTTP calls | 20/s, burst 60 |
| Failed authentication per source address | 5 in 60 s, then 429 until failures age out |
| Offline / evict | 45 / 300 s; active poll counts as alive |
| Mailbox / idempotency entries | 1000 per peer / 20000 total |

A valid session remains usable even if that address is locked out, so bad tokens behind a shared proxy do not
silence connected robots. New joins/rejoins can still be denied during address lockout. `--trust-proxy` trusts
`CF-Connecting-IP` only when the operator enables it; enable it only behind a known Cloudflare proxy with local
origin access controlled. Otherwise clients could forge the header to evade source limits.

The body cap bounds each message, not the total memory or disk allocation. Peer count, successful join churn,
active request count, concurrent polls and audit-log growth have no global quotas. At the largest body sizes,
1000-message mailboxes are expensive. An authorized malicious peer can exhaust resources; public denial-of-service
resistance needs edge limits, global quotas and log rotation before production use. Keep groups small and peers
trusted. Observation image references are never fetched, avoiding automatic remote URL access. The adapter only
reads local image files when the local agent explicitly supplies paths, and validates JPEG/PNG headers; image
content is still untrusted, and a header check is not a file disclosure boundary or full decoder validation.

Log files contain prompts, arguments, image URLs, poses and task outcomes. Inline image bytes are stripped, but
logs still need access controls and retention limits. The peer-scoped in-memory log is not a confidentiality
boundary for broadcast events. No end-to-end encryption, signatures, replay-proof durable task execution,
calibration proof, distributed collision avoidance or hardware interlock is claimed.

Tests cover missing/wrong auth, key/identity mismatch, sender spoofing, lifecycle authority, rate limiting,
conflicting ids, malformed bodies and bounded reads. A trap connector verifies an incoming joint-target request
never reaches a motion attribute; the twin demo separately proves the local gripper safety limit still refuses
an oversized move. Only localhost and twins were tested. No public tunnel or hardware cooperation was run.
