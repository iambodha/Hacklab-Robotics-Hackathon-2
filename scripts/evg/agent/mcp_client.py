"""Minimal MCP (streamable HTTP, JSON-RPC) client and the arm helpers built on it. Stdlib only.

Config: scripts/evg/robot.json (url, calibrated ranges; no secrets). The bearer token comes ONLY from the environment (EVG_MCP_TOKEN).
"""
import json, os, re, urllib.error, urllib.request
from pathlib import Path

CFG_PATH = Path(__file__).resolve().parents[1] / "robot.json"
CAMERAS = {"5cd55831": "over", "e091eeda": "grip"}      # camera id prefix -> our name
ROW = re.compile(r"\s*j(\d)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+\[\s*(\d+),\s*(\d+)\]\s+(\w)\s+(\d+)\s+(0x[0-9a-fA-F]+|\d+)\s+(-?\d+)\s+(\d+)")


class McpError(Exception):
    pass


def load_cfg():
    try:
        return json.loads(CFG_PATH.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def backend():
    return os.environ.get("EVG_BACKEND") or load_cfg().get("backend") or "station"


class Mcp:
    def __init__(self, url, token, timeout=30):
        if not token:
            raise McpError("EVG_MCP_TOKEN is not set in the environment")
        self.url, self.token, self.timeout, self.sid, self._id = url, token, timeout, None, 0
        self._rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "evg", "version": "0.1"}})
        self._rpc("notifications/initialized", notify=True)

    def _rpc(self, method, params=None, notify=False, timeout=None):
        self._id += 1
        body = {"jsonrpc": "2.0", "method": method, **({"params": params} if params is not None else {}), **({} if notify else {"id": self._id})}
        h = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.sid: h["Mcp-Session-Id"] = self.sid
        try:
            with urllib.request.urlopen(urllib.request.Request(self.url, json.dumps(body).encode(), h), timeout=timeout or self.timeout) as r:
                self.sid = r.headers.get("Mcp-Session-Id") or self.sid
                raw, ct = r.read().decode(), r.headers.get("content-type", "")
        except urllib.error.HTTPError as e:
            raise McpError(f"HTTP {e.code} from arm server ({'bad token?' if e.code in (401, 403) else e.reason})")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise McpError(f"arm server unreachable: {e}")
        if notify or not raw.strip():
            return None
        msg = None
        if "event-stream" in ct:
            for line in raw.splitlines():
                if line.startswith("data:"): msg = json.loads(line[5:]); break
        else:
            msg = json.loads(raw)
        if not msg: raise McpError("empty reply from arm server")
        if "error" in msg: raise McpError(f"{method}: {msg['error'].get('message')}")
        return msg["result"]

    def call(self, name, args=None, timeout=None):
        """-> (text, images) ; images = [(label_text, jpeg_bytes)]. Raises McpError if the tool reports isError."""
        import base64
        res = self._rpc("tools/call", {"name": name, "arguments": args or {}}, timeout=timeout)
        texts, images, label = [], [], ""
        for c in res.get("content", []):
            if c["type"] == "text": texts.append(c["text"]); label = c["text"]
            elif c["type"] == "image": images.append((label, base64.b64decode(c["data"])))
        text = "\n".join(texts)
        if res.get("isError"): raise McpError(text or f"{name} failed")
        return text, images


def connect():
    cfg = load_cfg().get("mcp") or {}
    if not cfg.get("url"): raise McpError(f"no MCP url in {CFG_PATH}")
    return Mcp(cfg["url"], os.environ.get(cfg.get("token_env", "EVG_MCP_TOKEN")))


def health(timeout=3):
    """True/False, no token needed."""
    url = (load_cfg().get("mcp") or {}).get("url")
    if not url: return False
    try:
        with urllib.request.urlopen(url.rsplit("/mcp", 1)[0] + "/health", timeout=timeout) as r:
            return json.loads(r.read()).get("ok") is True
    except Exception:
        return False


def parse_table(text):
    """joint table text -> {j: dict(pos, goal, cur(load), mn, mx, torque, status, in_range)}"""
    st = {}
    for line in text.splitlines():
        m = ROW.match(line)
        if m:
            st[int(m[1])] = dict(pos=int(m[3]), goal=int(m[4]), mn=int(m[5]), mx=int(m[6]), in_range=m[7] == "y", torque=int(m[8]),
                                 status=m[9], cur=int(m[10]), temp=int(m[11]), frozen=True)
    return st


def hold_arm():
    """Torque on = goal := present on every motor (the server does this first, so nothing jumps). Never disables torque."""
    return connect().call("set_torque", {"enabled": True})[0]
