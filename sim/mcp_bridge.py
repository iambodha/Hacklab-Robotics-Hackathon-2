"""HTTP/CLI bridge for running persistent MCP tool calls against so101-sim or so101-real.

Usage:
  # Start persistent MCP server over Streamable HTTP (keeps MuJoCo physics / Station connection warm):
  uv run --no-project --with 'mcp>=1.12,<2' --with pillow --with numpy --with mujoco \\
      python3 sim/mcp_bridge.py serve --target sim --port 8771

  # Call any MCP tool from CLI / VLM policy loop (saves returned images to --img-dir):
  uv run --no-project --with 'mcp>=1.12,<2' \\
      python3 sim/mcp_bridge.py call --port 8771 look --img-dir /tmp/policy_views
  uv run --no-project --with 'mcp>=1.12,<2' \\
      python3 sim/mcp_bridge.py call --port 8771 pixel_to_table '{"camera":"wrist","u":410,"v":280,"height":0.01}'
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def run_serve(target: str, host: str, port: int):
    os.environ.setdefault("ROBOT_MODEL", "so101")
    if target == "sim":
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("ROBOT_SIM_TRUTH_LOG", str(Path.home() / ".cache/so101-sim/truth.jsonl"))
        Path(os.environ["ROBOT_SIM_TRUTH_LOG"]).parent.mkdir(parents=True, exist_ok=True)
        import so101_sim_mcp as mod
    elif target == "real":
        os.environ.setdefault("MUJOCO_GL", "egl")
        import so101_real_mcp as mod
    else:
        raise SystemExit(f"Unknown target {target!r}; expected 'sim' or 'real'")

    srv = mod.server
    srv.settings.host = host
    srv.settings.port = port
    try:
        srv.run(transport="streamable-http")
    finally:
        if target == "sim":
            mod._srv.arm.running = False
            if mod._srv.recorder:
                mod._srv.recorder.close()
                import make_replay
                make_replay.build(mod._srv.recorder.dir)


async def run_call(url: str, tool: str | None, raw_args: str, img_dir: Path | None):
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            if not tool or tool == "--list":
                tools = await session.list_tools()
                for t in tools.tools:
                    print(f"{t.name}: {(t.description or '').splitlines()[0]}")
                return
            args = json.loads(raw_args) if raw_args else {}
            res = await session.call_tool(tool, args)
            if res.isError:
                print("MCP TOOL ERROR:", file=sys.stderr)
            idx = 0
            for item in res.content:
                kind = getattr(item, "type", "")
                if kind == "text":
                    print(item.text)
                elif kind == "image":
                    idx += 1
                    raw = base64.b64decode(item.data)
                    if img_dir is not None:
                        img_dir.mkdir(parents=True, exist_ok=True)
                        out_p = img_dir / f"{tool}_{idx}.jpg"
                        out_p.write_bytes(raw)
                        print(f"[saved image -> {out_p} ({len(raw)} bytes)]")
                    else:
                        print(f"[image {idx}: {len(raw)} bytes]")
            if res.isError:
                raise SystemExit(1)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("serve")
    ps.add_argument("--target", choices=("sim", "real"), default="sim")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8771)

    pc = sub.add_parser("call")
    pc.add_argument("--port", type=int, default=8771)
    pc.add_argument("--url", default="")
    pc.add_argument("--img-dir", type=Path, default=Path("/tmp/so101_mcp_views"))
    pc.add_argument("tool", nargs="?", default="--list")
    pc.add_argument("args", nargs="?", default="{}")

    args = p.parse_args()
    if args.cmd == "serve":
        run_serve(args.target, args.host, args.port)
    else:
        url = args.url or f"http://127.0.0.1:{args.port}/mcp"
        asyncio.run(run_call(url, args.tool, args.args, args.img_dir))


if __name__ == "__main__":
    main()
