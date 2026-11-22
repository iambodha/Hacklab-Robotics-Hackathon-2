"""Scripted MCP client checks against a Generaloid server in the TWIN (never real mode).

    python -m generaloid.checks tools  --connector elrobot           # list tools over MCP stdio
    python -m generaloid.checks pick   --connector elrobot|so101     # pick the cube, lift it, judge by sim truth
    python -m generaloid.checks twist  --connector elrobot           # bottle twin: episode + twist_cap + check_cap

Each check starts `python -m generaloid serve --connector <c> --mode sim` as a subprocess and talks to it as any MCP
client (Claude, Gemini, Devin) would: tool calls over stdio. The client only uses what an agent sees; the verdict
comes from the twin's ground-truth log (ROBOT_SIM_TRUTH_LOG), which the agent never sees. Prints one JSON line.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = Path(__file__).resolve().parents[1]
GRIP_STEP = 0.3            # max gripper change per call on the ElRobot twin (its gripper obeys ROBOT_MAX_DELTA 0.35)
LIFTED_Z = 0.05            # m: the cube's centre this high = picked up (it starts at 0.01)
HOVER_Z, GRASP_Z, LIFT_Z = 0.08, 0.014, 0.12   # m: TCP heights for the scripted pick (2 cm cube on the table)
# Firm grip as learned in learning/sim_loop.py; 10 strokes as the real demo takes needed for a fully closed cap
# (logs/learning/bottle/SUMMARY.md: 10-11 strokes). sim_loop's 6 x 90 deg opens it only after two earlier attempts.
TWIST_PARAMS = {"grip_torque_limit": 0.45, "strokes": 10, "twist_deg_per_stroke": 90}


def _env(tmp: Path, extra: dict | None = None) -> dict:
    return {**os.environ, "STATION_HOST": "station-disabled-in-tests.invalid",
            "ROBOT_CAM_URL": "http://camera-relay-disabled-in-tests.invalid:8890", "EXECUTOR_REAL_MOTION": "0",
            "ROBOT_SIM_RECORD": "0", "BOTTLE_GEMINI_JUDGE": "0", "MUJOCO_GL": os.environ.get("MUJOCO_GL", "cgl"),
            "ROBOT_SIM_TRUTH_LOG": str(tmp / "truth.jsonl"), "SKILLCFG_DIR": str(tmp / "skillcfg"),
            "BOTTLE_SIM_LEARN_DIR": str(tmp / "learn"), "PYTHONPATH": str(REPO), **(extra or {})}


def _text(res) -> str:
    return "\n".join(c.text for c in res.content if getattr(c, "type", "") == "text")


def _truth(tmp: Path) -> dict:
    lines = (tmp / "truth.jsonl").read_text().splitlines() if (tmp / "truth.jsonl").exists() else []
    return json.loads(lines[-1]) if lines else {}


class Client:
    def __init__(self, session: ClientSession):
        self.s, self.log = session, []

    async def call(self, tool: str, **args) -> str:
        res = await self.s.call_tool(tool, args)
        text = _text(res)
        self.log.append({"tool": tool, "args": args, "error": bool(res.isError), "head": text.splitlines()[:1]})
        if res.isError:
            raise RuntimeError(f"{tool}{args}: {text}")
        return text

    async def joint_norm(self, joint: str) -> float:
        m = re.search(rf"^\s*{re.escape(joint)}\s+([0-9.]+)", await self.call("joint_state"), re.M)
        return float(m.group(1))

    async def gripper_to(self, joint: str, target: float) -> str:
        """Gripper in steps the server accepts (the ElRobot twin limits a gripper call to ROBOT_MAX_DELTA)."""
        out = ""
        for _ in range(6):
            now = await self.joint_norm(joint)
            if abs(now - target) < 0.02:
                break
            out = await self.call("gripper", open_fraction=round(now + max(-GRIP_STEP, min(GRIP_STEP, target - now)), 3))
        return out


async def _session(connector: str, tmp: Path, layout: str | None, fn):
    headless = os.environ.get("GENERALOID_CHECK_HEADLESS") == "1"
    launcher = "generaloid.tests.run_headless_server" if headless else "generaloid"
    args = ["-m", launcher, "serve", "--connector", connector, "--mode", "sim"] + (["--layout", layout] if layout else [])
    params = StdioServerParameters(command=sys.executable, args=args, cwd=str(REPO), env=_env(tmp))
    with open(tmp / "server.stderr", "w") as err:
        async with stdio_client(params, errlog=err) as (r, w), ClientSession(r, w) as s:
            await s.initialize()
            return await fn(Client(s))


async def check_tools(connector: str, tmp: Path, layout=None) -> dict:
    async def run(c: Client):
        tools = (await c.s.list_tools()).tools
        return {"tools": sorted(t.name for t in tools),
                "specs": {t.name: {"description": t.description, "inputSchema": t.inputSchema} for t in tools}}
    return await _session(connector, tmp, layout, run)


async def check_pick(connector: str, tmp: Path, layout=None) -> dict:
    from generaloid import connectors
    conn = connectors.get(connector)
    lay = json.loads(Path(layout or conn.twin.layout).read_text())
    cube = next(o for o in lay["objects"] if o["name"].endswith("cube"))
    x, y = cube["pos"][:2]
    jaws = conn.gripper.joint

    async def run(c: Client):
        await c.call("look")
        await c.gripper_to(jaws, 0.6)
        await c.call("move_tcp", x=x, y=y, z=HOVER_Z, top_down=True, yaw_deg=float(cube.get("yaw", 0)))
        await c.call("move_tcp", x=x, y=y, z=max(GRASP_Z, conn.safety.min_tcp_z), top_down=True,
                     yaw_deg=float(cube.get("yaw", 0)))
        before = _truth(tmp)["objects"][cube["name"]]["xyz"]
        closed = await c.gripper_to(jaws, 0.0)
        lifted = await c.call("move_tcp", x=x, y=y, z=LIFT_Z, top_down=True, yaw_deg=float(cube.get("yaw", 0)))
        await asyncio.sleep(1.0)
        tcp = await c.call("tcp_pose")
        after = _truth(tmp)["objects"][cube["name"]]["xyz"]
        return {"cube_before": before, "cube_after": after, "picked": after[2] > LIFTED_Z,
                "gripper": closed.splitlines()[0], "lift": lifted.splitlines()[0], "tcp": tcp.splitlines()[0]}
    return await _session(connector, tmp, layout, run)


async def check_twist(connector: str, tmp: Path, layout=None) -> dict:
    sys.path.insert(0, str(REPO))
    from learning import bottle_twin
    lay_path = bottle_twin.make_layout(Path(layout) if layout else tmp / "bottle_twin.json")
    b = json.loads(lay_path.read_text())["screw_cap"]
    x, y, h = b["pos"][0], b["pos"][1], 2 * b["half_height"]
    top = b["pos"][2] + h

    async def run(c: Client):
        hist = await c.call("episode_history", last_n=5)
        begin = await c.call("episode_begin", params=TWIST_PARAMS, hypothesis="generaloid check: firm grip, 10 x 90 deg")
        twist = await c.call("twist_cap", x=x, y=y, cap_top_z=top, cap_height=h)
        check = await c.call("check_cap")
        cap = _truth(tmp).get("screw_cap", {})
        end = await c.call("episode_end", outcome="opened" if cap.get("free") else "loosened",
                           lesson="scripted generaloid check", next_change={})
        return {"history": hist.splitlines()[0], "begin": begin.splitlines()[0], "twist": twist.splitlines()[0],
                "check_cap": check.splitlines()[0], "end": end.splitlines()[0], "cap": cap,
                "opened": bool(cap.get("free")), "turned_deg": cap.get("turned_deg")}
    return await _session(connector, tmp, str(lay_path), run)


CHECKS = {"tools": check_tools, "pick": check_pick, "twist": check_twist}


def run(check: str, connector: str, layout: str | None = None, workdir: Path | None = None) -> dict:
    tmp = Path(workdir or tempfile.mkdtemp(prefix=f"generaloid_{check}_"))
    tmp.mkdir(parents=True, exist_ok=True)
    out = asyncio.run(CHECKS[check](connector, tmp, layout))
    return {"check": check, "connector": connector, "workdir": str(tmp),
            "rendering": "diagnostic placeholder" if os.environ.get("GENERALOID_CHECK_HEADLESS") == "1" else "camera", **out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=sorted(CHECKS))
    ap.add_argument("--connector", required=True)
    ap.add_argument("--layout")
    ap.add_argument("--workdir", type=Path)
    a = ap.parse_args(argv)
    if a.check == "twist" and a.connector != "elrobot":
        ap.error("twist check requires the elrobot bottle twin")
    result = run(a.check, a.connector, a.layout, a.workdir)
    print(json.dumps(result))
    return 0 if result.get("picked", result.get("opened", True)) else 1


if __name__ == "__main__":
    sys.exit(main())
