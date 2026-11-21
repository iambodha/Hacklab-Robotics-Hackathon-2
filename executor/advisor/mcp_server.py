"""MCP server (stdio) for advisors: the executor's API and nothing else. It forwards to the console's HTTP API
(EXECUTOR_API, default http://127.0.0.1:8910), so the advisor process never touches Station or the arm.

    uv run --no-project --with 'mcp>=1.12,<2' python executor/advisor/mcp_server.py
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from mcp.server.fastmcp import FastMCP, Image

API = os.environ.get("EXECUTOR_API", "http://127.0.0.1:8910")
server = FastMCP("executor")


def _call(method: str, path: str, body: dict | None = None, timeout: float = 60.0):
    req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", "X-Advisor": "claude"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
            return json.loads(data) if r.headers.get("Content-Type", "").startswith("application/json") else data
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}"}


def _obs_text(o: dict) -> str:
    j = o.get("joints", {})
    frames = {r: {"seq": f["seq"], "age_s": f["age_s"], "stale": f["stale"]} for r, f in o.get("frames", {}).items()}
    objs = [{k: v for k, v in x.items() if k in ("name", "xyz_world", "size", "yaw", "uncertainty_mm", "visible", "source")}
            for x in o.get("objects", [])]
    return json.dumps({"obs_id": o["obs_id"], "setup_id": o["setup_id"], "freshness_s": o["freshness_s"], "frames": frames,
                       "tcp_world": j.get("tcp_world"), "gripper_open": j.get("gripper_open"),
                       "joints_norm": dict(zip(j.get("names", []), j.get("norm", []))),
                       "load": dict(zip(j.get("names", []), j.get("load", []))), "objects": objs, "notes": o.get("notes")}, indent=1)


@server.tool(annotations={"readOnlyHint": True})
def observe() -> str:
    """A new observation WITHOUT images: obs_id, setup_id, freshness, camera frame sequence numbers (stale = unknown),
    joints, TCP position (world frame, metres), gripper opening, object estimates."""
    return _obs_text(_call("POST", "/api/observe", {}))


@server.tool(annotations={"readOnlyHint": True})
def look() -> list:
    """A new observation WITH the camera images it is based on (wrist camera first, then overview). Use its obs_id
    and setup_id in submit() right away."""
    o = _call("POST", "/api/observe", {})
    out: list = [_obs_text(o)]
    for role in ("wrist", "overview"):
        f = o.get("frames", {}).get(role)
        if not f:
            out.append(f"{role} camera: not available")
            continue
        data = _call("GET", f["path_or_recording_ref"])
        if isinstance(data, bytes):
            out += [f"{role} camera, frame seq {f['seq']}" + (" (STALE: content unknown)" if f["stale"] else ""),
                    Image(data=data, format="jpeg")]
    return out


@server.tool()
def submit(skill: str, params: dict, obs_id: str, setup_id: str, rationale: str, ttl_s: float = 15.0,
           preconditions: list[str] | None = None) -> str:
    """Propose ONE skill. The executor validates it (freshness of obs_id, expiry after ttl_s, setup_id, limits,
    reachability of the whole path) and runs it, or rejects it with reasons and numbers. Returns execution_id."""
    r = _call("POST", "/api/submit", {"skill": skill, "params": params, "based_on_obs": obs_id, "setup_id": setup_id,
                                      "rationale": rationale, "ttl_s": ttl_s, "preconditions": preconditions or [],
                                      "source": "claude"})
    return json.dumps(r, indent=1)


@server.tool(annotations={"readOnlyHint": True})
def status(execution_id: str, wait_s: float = 0.0) -> str:
    """Execution result: status, termination, measured end state, residuals, phases. wait_s > 0 waits (at most
    that long) until it is final."""
    return json.dumps(_call("GET", f"/api/status/{execution_id}?wait={min(max(wait_s, 0), 60)}", timeout=75), indent=1)


@server.tool()
def cancel(execution_id: str) -> str:
    """Stop the execution now: no further commands, the arm holds its position."""
    return json.dumps(_call("POST", f"/api/cancel/{execution_id}", {}))


@server.tool(annotations={"readOnlyHint": True})
def list_skills() -> str:
    """Skills, their versions, parameters (units, world frame), freshness limits and the cameras they need."""
    return json.dumps(_call("GET", "/api/skills"), indent=1)


@server.tool(annotations={"readOnlyHint": True})
def scene() -> str:
    """World frame, workspace limits, calibration source, robot profile, tool attached, setup_id."""
    return json.dumps(_call("GET", "/api/scene"), indent=1)


if __name__ == "__main__":
    server.run()
