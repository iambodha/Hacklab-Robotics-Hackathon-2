"""Human-side judge: read the ground-truth log written by sim_arm_mcp.py (ROBOT_SIM_TRUTH_LOG).

  python3 sim/judge.py [~/.cache/elrobot-sim/truth.jsonl] [--all]
"""
import json
import os
import sys
from pathlib import Path

args = [a for a in sys.argv[1:] if not a.startswith('--')]
path = Path(args[0] if args else os.environ.get('ROBOT_SIM_TRUTH_LOG', Path.home() / '.cache/elrobot-sim/truth.jsonl')).expanduser()
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
for row in rows if '--all' in sys.argv else rows[-1:]:
    objs = ', '.join(f"{k} {v['xyz']} in_bin={v['in_bin']}" for k, v in row['objects'].items())
    print(f"{row['wall']} t={row['sim_time']}s {row['tool']} {json.dumps(row['args'])} | {objs}")
calls = sum(r['tool'] != 'operator_reset_scene' for r in rows)
done = all(v['in_bin'] for v in rows[-1]['objects'].values()) if rows else False
print(f'{len(rows)} logged calls ({calls} agent actions); task complete: {done}')
