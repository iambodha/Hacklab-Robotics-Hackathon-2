"""Isolated git worktrees for the improvement agent (candidate) and for the incumbent (baseline arm)."""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


def git(*args, cwd, check=True) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=check).stdout.strip()


@dataclass
class Sandbox:
    repo: Path
    path: Path
    base: str

    @classmethod
    def create(cls, repo: str | Path, path: str | Path, base: str = "HEAD") -> "Sandbox":
        repo, path = Path(repo).resolve(), Path(path).resolve()
        sha = git("rev-parse", base, cwd=repo)
        path.parent.mkdir(parents=True, exist_ok=True)
        git("worktree", "add", "--detach", str(path), sha, cwd=repo)
        return cls(repo, path, sha)

    def diff(self) -> str:
        git("add", "-A", "--intent-to-add", ".", cwd=self.path, check=False)     # untracked files show in the diff
        return subprocess.run(["git", "diff", self.base], cwd=self.path, capture_output=True, text=True).stdout

    def run(self, args: list[str], timeout: float = 600, env: dict | None = None) -> dict:
        """Run a command inside the sandbox with only the sandbox on the Python path."""
        e = {**os.environ, "PYTHONPATH": str(self.path), "SKILLS_LOOP_INNER": "1", "PYTHONDONTWRITEBYTECODE": "1",
             **(env or {})}
        try:
            r = subprocess.run(args, cwd=self.path, capture_output=True, text=True, timeout=timeout, env=e)
            return {"cmd": args, "returncode": r.returncode, "stdout": r.stdout[-6000:], "stderr": r.stderr[-4000:]}
        except subprocess.TimeoutExpired as ex:
            return {"cmd": args, "returncode": None, "stdout": (ex.stdout or "")[-6000:] if isinstance(ex.stdout, str) else "",
                    "stderr": f"timeout after {timeout}s"}

    def python(self, *args, **kw) -> dict:
        return self.run([sys.executable, *args], **kw)

    def remove(self):
        git("worktree", "remove", "--force", str(self.path), cwd=self.repo, check=False)
        git("worktree", "prune", cwd=self.repo, check=False)
