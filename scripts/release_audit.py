"""Read-only release audit; never print candidate credential values."""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 5_000_000
SIGNATURES = {
    "google": rb"AIza[0-9A-Za-z_-]{25,}",
    "openai": rb"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}",
    "github": rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})",
    "aws_access_id": rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "slack": rb"\bxox[baprs]-[A-Za-z0-9-]{20,}",
    "private_key": rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
}


def git(*args: str, input: bytes | None = None) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT, input=input)


def known_fixture(value: bytes, location: str) -> bool:
    # Exact mock string only; never broadly exempt a directory or key prefix.
    fixture = ROOT / "models/tests/conftest.py"
    match = re.search(rb'FAKE_KEY = "([^"]+)"', fixture.read_bytes())
    return bool(match and b"fake" in match[1].lower()
                and value == match[1] and "models/tests/conftest.py" in location)


def main() -> int:
    files = sorted(set(x.decode() for x in git("ls-files", "-co", "--exclude-standard", "-z").split(b"\0") if x))
    report: dict = {"shipping_files": len(files), "limit_bytes": LIMIT,
                    "large_shipping": [], "credential_candidates": [], "known_fake_fixture": [],
                    "credential_filenames": [], "keyword_lines": 0, "absolute_path_files": [],
                    "large_history": [], "history_credential_candidates": [], "history_fake_fixture": []}
    for name in files:
        path = ROOT / name
        if not path.is_file():
            continue
        if path.stat().st_size > LIMIT:
            report["large_shipping"].append({"path": name, "bytes": path.stat().st_size})
        data = path.read_bytes()
        for kind, pattern in SIGNATURES.items():
            for match in re.finditer(pattern, data):
                item = {"path": name, "line": data[:match.start()].count(b"\n") + 1, "kind": kind}
                report["known_fake_fixture" if known_fixture(match[0], name) else "credential_candidates"].append(item)
        if re.search(r"(^|/)(?:\.env(?:\..*)?|secrets?\.env|.*\.(?:pem|key))$", name):
            report["credential_filenames"].append(name)
        if b"\0" not in data:
            report["keyword_lines"] += len(re.findall(rb"(?im)^.*(?:AIza|sk-|secret|token|\.env).*$", data))
            if b"/Users/bodha/" in data or b"/Volumes/Dell HDD/" in data:
                report["absolute_path_files"].append(name)
    # Inspect every reachable ref, a superset of commits destined for main.
    history = git("log", "--all", "-p", "--format=commit %H")
    for kind, pattern in SIGNATURES.items():
        for match in re.finditer(pattern, history):
            before = history[:match.start()]
            commit = before[before.rfind(b"commit "):].splitlines()[0].decode(errors="replace")
            marker = max(before.rfind(b"+++ "), before.rfind(b"--- "))
            location = before[marker:].splitlines()[0].decode(errors="replace")
            item = {"commit": commit, "file": location, "kind": kind}
            report["history_fake_fixture" if known_fixture(match[0], location) else "history_credential_candidates"].append(item)
    objects = git("rev-list", "--objects", "--all").decode().splitlines()
    sizes = git("cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)",
                input=("\n".join(line.split()[0] for line in objects) + "\n").encode()).decode().splitlines()
    for obj, size in zip(objects, sizes):
        parts = size.split()
        if len(parts) == 3 and parts[1] == "blob" and int(parts[2]) > LIMIT:
            report["large_history"].append({"object": obj, "bytes": int(parts[2])})
    print(json.dumps(report, indent=2))
    return int(any(report[k] for k in ("large_shipping", "credential_candidates", "credential_filenames",
                                      "large_history", "history_credential_candidates")))


if __name__ == "__main__":
    raise SystemExit(main())
