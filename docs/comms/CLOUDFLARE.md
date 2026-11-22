# Put the comms hub behind Cloudflare

These are the intended remote-deployment steps. **Only localhost/twins were tested; no tunnel was started.**
Keep NormaCore Station and robot motion tools local. The only port to publish here is the authenticated comms hub
(default 8930). Never substitute Station's ports or the unauthenticated Station relay/forwarder.

## Host machine (ElRobot's operator)

1. From the repo root, start a standalone hub. This command is safe for twin-only development:

   ```sh
   export STATION_HOST=station-disabled-in-tests.invalid
   umask 077
   uv run --no-project --with 'mcp>=1.12,<2' --with httpx python -m generaloid.comms hub \
     --bind 127.0.0.1 --port 8930 --token-file /tmp/comms-token --log /tmp/comms-hub.jsonl
   ```

   Alternatively, ask the ElRobot agent's existing MCP to call `comms_host(port=8930)`. It returns the token and
   automatically joins the host robot. In that case do not start a second standalone hub on the same port. A
   standalone hub has no robot identity: both robot MCPs must join it using `comms_join`.

2. In another terminal check `curl http://127.0.0.1:8930/health`. Install `cloudflared` using
   [Cloudflare's downloads](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)
   if necessary, then start the tunnel **when the operator intends to expose the hub**:

   ```sh
   cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8930
   ```

   Copy the printed `https://<random>.trycloudflare.com` URL. Do not publish the token. Send URL and token to the
   other operator over a private channel; a URL alone does not authorize joining. `/health` deliberately needs
   no auth and reveals only service/version/online count.

## Joining machine (SO-101's operator)

3. Run the local robot MCP in **sim mode for this walkthrough**. After both branches are integrated:

   ```sh
   export STATION_HOST=station-disabled-in-tests.invalid
   export GENERALOID_ROBOT_ID=so101-b
   uv run --no-project --with 'mcp>=1.12,<2' --with httpx --with numpy --with scipy \
     --with pillow --with mujoco python -m generaloid serve --connector so101 --mode sim
   ```

   Connect your local MCP agent over stdio and call:

   ```text
   comms_join(url="https://<random>.trycloudflare.com", token="<privately received token>", robot_id="so101-b")
   comms_peers()
   comms_send(peer="elrobot-a", kind="message", payload={"text":"ready to coordinate"})
   ```

   With a standalone hub, the ElRobot agent must also call `comms_join` using its distinct `elrobot-a` id.
   For a non-robot connectivity check, place the received token in a private 0600 file without adding it to shell
   history, then run:

   ```sh
   uv run --no-project --with 'mcp>=1.12,<2' --with httpx python -m generaloid.comms client \
     --url https://<random>.trycloudflare.com --robot-id remote-observer \
     --token-file /private/path/comms-token peers
   ```

4. Use the [scenario](SCENARIO.md). Receiving a request is a local decision point; a tunnel grants no motor access.
   At the end call `comms_leave()` on each robot, stop `cloudflared` with Ctrl-C, then stop the standalone hub if
   used. Protect/delete the temporary token file and retain/rotate the sensitive log according to operator policy.

## Transport and security details

Built-in clients use ordinary HTTP long-polling (20 s waits; hub maximum 25 s). Optional `/v1/stream` SSE works on
localhost, but **Quick Tunnels do not support SSE**. Quick Tunnels are temporary development services, with a
200 in-flight request cap, no uptime guarantee and a new hostname on restart. Browser email gates cannot be used
by these non-interactive robot clients. These constraints were checked against
[Cloudflare's Quick Tunnel documentation](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/)
on 22 Nov 2026. For a stable endpoint use an operator-managed named tunnel; any extra Access policy must explicitly
support machine clients (the built-in client currently implements only the hub bearer authentication).

HTTPS protects the public hop; the local origin remains HTTP on loopback. Never disable TLS verification.
`--trust-proxy` is optional for a standalone hub: it attributes failed auth to Cloudflare's connecting-IP header
instead of every visitor sharing 127.0.0.1. Leave it off unless origin access is restricted to the trusted proxy.
Valid sessions bypass source lockout; a stranger can otherwise temporarily prevent new joins from that address.
Per-peer keys provide better robot identity isolation than a shared group token; see [SECURITY](SECURITY.md).

Troubleshooting: 401 means wrong join credential or expired session (clients automatically rejoin); 409 commonly
means a duplicate robot id; 429 means rate/auth lockout or tunnel concurrency limits. A stopped hub loses all task
state even when its log survives. A restarted Quick Tunnel also changes URL, requiring explicit leave/join with
the new URL. Reconnection to the same live endpoint is automatic; physical task recovery is always a local decision.
