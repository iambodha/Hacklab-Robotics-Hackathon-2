"""Claude as a drop-in transport for GeminiER's prompts, through the Claude Code CLI (`claude -p`).

Used by the benchmark so both models get the same prompts and the same parsers:

    er = GeminiER(model="claude-sonnet-5-5", transport=ClaudeCLITransport())

Images are passed as file paths that Claude reads with its Read tool (the only tool allowed). Cost and
token counts come from the CLI's JSON output (`total_cost_usd`, `usage`).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from .gemini_er import GeminiError, Image, RawReply

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")


class ClaudeCLITransport:
    def __init__(self, timeout_s: float = 180.0, tmp_dir: Path | None = None):
        self.timeout_s = timeout_s
        self.tmp = Path(tmp_dir or tempfile.gettempdir()) / "models-claude-frames"

    def _path(self, im: Image) -> str:
        if im.ref and Path(im.ref).is_file():
            return str(Path(im.ref).resolve())
        self.tmp.mkdir(parents=True, exist_ok=True)
        p = self.tmp / (hashlib.sha1(im.data).hexdigest()[:16] + (".png" if im.mime == "image/png" else ".jpg"))
        if not p.exists():
            p.write_bytes(im.data)
        return str(p)

    def __call__(self, model: str, parts: list, config: dict) -> RawReply:
        lines = []
        for p in parts:
            if isinstance(p, Image):
                lines.append(f"[Read the image file {self._path(p)} ({p.w}x{p.h} px); it is the image described just above.]")
            else:
                lines.append(p)
        lines.append("Reply with the JSON only, no prose, no code fences.")
        cmd = [CLAUDE_BIN, "-p", "\n".join(lines), "--output-format", "json", "--allowedTools", "Read"]
        if model and model != "claude":
            cmd += ["--model", model]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_s)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise GeminiError(f"claude CLI: {type(e).__name__}: {e}") from None
        if r.returncode != 0:
            raise GeminiError(f"claude CLI exit {r.returncode}: {(r.stderr or r.stdout)[-300:]}")
        try:
            out = json.loads(r.stdout)
        except json.JSONDecodeError:
            raise GeminiError(f"claude CLI: non-JSON output {r.stdout[:200]!r}") from None
        u = out.get("usage") or {}
        prompt = sum(int(u.get(k) or 0) for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        return RawReply(text=str(out.get("result", "")), usage={
            "prompt_tokens": prompt, "output_tokens": u.get("output_tokens"), "thinking_tokens": None,
            "total_tokens": prompt + int(u.get("output_tokens") or 0), "cost_usd": out.get("total_cost_usd"),
            "num_turns": out.get("num_turns")})
