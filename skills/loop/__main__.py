"""python -m skills.loop synthetic [--agent heuristic|claude|devin] [--keep]
   python -m skills.loop run BUNDLE [--agent ...]
   python -m skills.loop approve PROMOTION_ID --by NAME
   python -m skills.loop status"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..memory import Memory
from . import runner
from .agent import make


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m skills.loop")
    ap.add_argument("--loop-dir", default=os.environ.get("SKILLS_LOOP_DIR", str(runner.REPO / "skills" / ".loop")))
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synthetic")
    s.add_argument("--agent", default="heuristic")
    s.add_argument("--keep", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("bundle")
    r.add_argument("--agent", default="heuristic")
    r.add_argument("--keep", action="store_true")
    a = sub.add_parser("approve")
    a.add_argument("promotion_id")
    a.add_argument("--by", required=True)
    sub.add_parser("status")
    args = ap.parse_args(argv)
    ld = Path(args.loop_dir)
    mem = Memory(ld / "memory")
    if args.cmd == "synthetic":
        rep = runner.synthetic(ld, make(args.agent), keep=args.keep, memory=mem)
    elif args.cmd == "run":
        rep = runner.run(args.bundle, agent=make(args.agent), loop_dir=ld, keep=args.keep, memory=mem)
    elif args.cmd == "approve":
        print(json.dumps(runner.approve(mem, args.promotion_id, args.by), indent=1))
        return
    else:
        for p in mem.promotions():
            print(p["id"], p["decision"], p["skill"], p["candidate"], p.get("approved_by") or "")
        return
    print(Path(rep["run_dir"], "CANDIDATE.md").read_text())


if __name__ == "__main__":
    main()
