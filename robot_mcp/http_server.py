"""Serve robot_mcp over HTTP (MCP streamable-HTTP transport) with a shared bearer token.

Any MCP client can connect — Claude Code, Codex, Cursor, Gemini CLI, the OpenAI Agents SDK,
or a plain Python `mcp` client — from this machine or, with --lan, from other laptops.

  python -m robot_mcp --http          # this machine only: http://127.0.0.1:8765/mcp
  python -m robot_mcp --http --lan    # all interfaces: http://<laptop-ip>:8765/mcp

Security model (hackathon grade):
  * Every request except GET /health must carry  Authorization: Bearer <token>.
  * The token comes from ROBOT_MCP_TOKEN, else robot_mcp/data/http_token.local.json
    (generated on first run, gitignored, reused across restarts so client configs keep working).
  * Plain HTTP: anyone sniffing the same Wi-Fi can read the token. Don't use beyond a trusted LAN.
  * Station itself (:8888/:8889) stays on loopback; only this server's safe tool surface is exposed.
  * One motion source per arm is still enforced by the robot's motion lock, across all clients.
"""
from __future__ import annotations

import hmac
import json
import os
import secrets
import socket

from . import config

TOKEN_FILE = config.DATA_DIR / "http_token.local.json"
DEFAULT_PORT = int(os.environ.get("ROBOT_MCP_PORT", "8765"))


def get_token() -> str:
    env = os.environ.get("ROBOT_MCP_TOKEN", "").strip()
    if env:
        return env
    if TOKEN_FILE.exists():
        return json.loads(TOKEN_FILE.read_text())["token"]
    tok = secrets.token_urlsafe(24)
    TOKEN_FILE.write_text(json.dumps({"token": tok}))
    TOKEN_FILE.chmod(0o600)
    return tok


class BearerAuth:
    """ASGI wrapper: 401 unless the Authorization header carries the token.
    Lifespan and non-HTTP scopes pass straight through (the MCP session manager needs them)."""

    def __init__(self, app, token: str):
        self.app = app
        self.expected = f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            if scope.get("path") == "/health" and scope.get("method") == "GET":
                return await self._send(send, 200, {"ok": True, "service": "robot-mcp"})
            auth = dict(scope.get("headers") or []).get(b"authorization", b"")
            if not hmac.compare_digest(auth, self.expected):
                return await self._send(send, 401, {"error": "missing or wrong bearer token"})
        await self.app(scope, receive, send)

    @staticmethod
    async def _send(send, status: int, body: dict):
        data = json.dumps(body).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(data)).encode())]})
        await send({"type": "http.response.body", "body": data})


def lan_ip() -> str:
    """This machine's LAN address (a UDP connect sends no packets)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def build_app(mcp, open_hosts: bool):
    """The authenticated ASGI app. open_hosts disables the SDK's localhost-only Host check,
    which would otherwise reject clients that address this laptop by its LAN IP."""
    if open_hosts:
        from mcp.server.transport_security import TransportSecuritySettings
        mcp.settings.transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    return BearerAuth(mcp.streamable_http_app(), get_token())


def serve(mcp, lan: bool = False, port: int = DEFAULT_PORT, host: str | None = None) -> None:
    import uvicorn
    host = host or ("0.0.0.0" if lan else "127.0.0.1")
    app = build_app(mcp, open_hosts=lan or host not in ("127.0.0.1", "localhost"))
    where = lan_ip() if lan else "127.0.0.1"
    url = f"http://{where}:{port}{mcp.settings.streamable_http_path}"
    tok = get_token()
    print(f"""
robot-mcp over HTTP  ->  {url}   ({'LAN: reachable from other machines' if lan else 'this machine only'})
token: {tok}
check reachability:  curl http://{where}:{port}/health

Claude Code:  claude mcp add --transport http robot {url} --header "Authorization: Bearer {tok}"
Any other MCP client: streamable-HTTP URL {url} with header  Authorization: Bearer <token>
(see robot_mcp/README.md for Codex / Cursor / Gemini CLI / Python examples)
""", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning")
