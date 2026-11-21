"""Claude Code as a background advisor: `claude -p` headless, with ONE MCP server (executor/advisor/mcp_server.py)
that exposes only the executor's API (observe, look, submit, status, cancel, list_skills, scene). No shell, no
files, no other MCP servers, an empty working directory. The stream-json output is parsed into the console's
timeline (assistant messages, tool calls, tool results) and logged to executor/logs/advisor_claude_<time>.jsonl."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from .base import Advisor

HERE = Path(__file__).resolve().parent
SERVER_NAME = "executor"


def mcp_config(api_url: str) -> dict:
    """Run the MCP server with uv (it needs the `mcp` package), else with this interpreter."""
    server = str(HERE / "mcp_server.py")
    if os.environ.get("EXECUTOR_MCP_PYTHON"):
        cmd, args = os.environ["EXECUTOR_MCP_PYTHON"], [server]
    else:
        cmd, args = "uv", ["run", "--no-project", "-q", "--with", "mcp>=1.12,<2", "python", server]
    return {"mcpServers": {SERVER_NAME: {"command": cmd, "args": args, "env": {"EXECUTOR_API": api_url}}}}


class ClaudeAdvisor(Advisor):
    name = "claude"

    def __init__(self, *a, model: str = "sonnet", max_turns: int = 60, effort: str | None = None, **kw):
        super().__init__(*a, **kw)
        self.model, self.max_turns, self.effort = model, max_turns, effort
        self.proc: subprocess.Popen | None = None

    def command(self, cfg_path: str) -> list[str]:
        cmd = ["claude", "-p", self.task, "--model", self.model, "--max-turns", str(self.max_turns),
               "--output-format", "stream-json", "--verbose", "--tools", "", "--strict-mcp-config",
               "--mcp-config", cfg_path, "--allowedTools", f"mcp__{SERVER_NAME}", "--append-system-prompt", self.system]
        if self.effort:
            cmd += ["--effort", self.effort]
        return cmd

    def run(self):
        work = tempfile.mkdtemp(prefix="executor-advisor-")
        cfg = Path(work) / "mcp.json"
        cfg.write_text(json.dumps(mcp_config(self.api)))
        cmd = self.command(str(cfg))
        self.log("command", argv=[c if len(c) < 200 else c[:200] + "..." for c in cmd])
        self.proc = subprocess.Popen(cmd, cwd=work, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, bufsize=1, start_new_session=True)
        for line in self.proc.stdout:
            if self._stop.is_set():
                break
            self.handle(line)
        if self._stop.is_set():
            self.kill()
            return "stopped by the operator"
        self.proc.wait()
        err = self.proc.stderr.read()[-800:] if self.proc.stderr else ""
        if self.proc.returncode:
            self.log("error", text=f"claude exited {self.proc.returncode}: {err}")
            return f"claude exited with {self.proc.returncode}"
        return None

    def handle(self, line: str) -> None:
        try:
            d = json.loads(line)
        except ValueError:
            return
        t = d.get("type")
        if t == "assistant":
            for b in d.get("message", {}).get("content", []):
                if b.get("type") == "text" and b.get("text", "").strip():
                    self.log("message", text=b["text"].strip())
                elif b.get("type") == "tool_use":
                    self.log("tool_call", tool=b.get("name", "").removeprefix(f"mcp__{SERVER_NAME}__"), input=b.get("input"),
                             id=b.get("id"))
        elif t == "user":
            for b in d.get("message", {}).get("content", []) if isinstance(d.get("message", {}).get("content"), list) else []:
                if b.get("type") == "tool_result":
                    c = b.get("content")
                    text = " ".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text") if isinstance(c, list) else str(c)
                    n_img = sum(1 for x in c if isinstance(x, dict) and x.get("type") == "image") if isinstance(c, list) else 0
                    self.log("tool_result", id=b.get("tool_use_id"), text=text[:1500], images=n_img, error=bool(b.get("is_error")))
        elif t == "result":
            self.log("result", text=d.get("result", ""), turns=d.get("num_turns"), seconds=round(d.get("duration_ms", 0) / 1000),
                     cost_usd=d.get("total_cost_usd"))
        elif t == "system" and d.get("subtype") == "init":
            self.log("init", model=d.get("model"), tools=d.get("tools"), mcp_servers=d.get("mcp_servers"))

    def kill(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

    def stop(self) -> None:
        super().stop()
        self.kill()


if __name__ == "__main__":     # print the exact command line for a manual run
    print(" ".join(ClaudeAdvisor("http://127.0.0.1:8910", None, {}, {}).command("mcp.json")), file=sys.stderr)
