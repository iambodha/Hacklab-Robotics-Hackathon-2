"""Turn System 1 decision traces into an Open-Jev training set (choice rows, split by goal).

  python system2/export_jev_data.py system2/traces/* --out data/elrobot-system1 [--all-goals]

Each decision stores the exact model input (state + candidate texts). By default only decisions
from goals that SUCCEEDED are kept, labelled with the candidate that was executed: imitation of
working behaviour (e.g. geometric runs), the first step before outcome-weighted data. Splits are
by goal (train 80 % / calibration 5 % / validation 5 % / test 10 %), so no goal leaks across
splits. Validate and train with the Open-Jev checkout (see system2/README.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s2.scorers import QUESTION  # noqa: E402


def split_of(group: str) -> str:
    h = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    return 'train' if h < 80 else 'calibration' if h < 85 else 'validation' if h < 90 else 'test'


def rows_from(trace: Path, all_goals: bool):
    path = trace / 'decisions.jsonl'
    if not path.exists():
        return
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    outcome = {r['goal']: r['status'] for r in lines if r.get('event') == 'goal_end'}
    for n, r in enumerate(lines, 1):
        mi = r.get('model_input')
        if r.get('event') != 'decision' or not mi or r['chosen'] not in mi['options']:
            continue
        if not all_goals and outcome.get(r['goal']) != 'succeeded':
            continue
        if str(r.get('result', '')).startswith(('UNREACHABLE', 'REFUSED')):
            continue
        group = f'{trace.name}/{r["goal"]}'
        ids = list(mi['options'])
        yield {'id': f'{group}/{n}', 'group_id': group, 'split': split_of(group), 'source': 'elrobot-system1',
               'state': mi['state'], 'question': QUESTION, 'kind': 'choice',
               # Same "name: description" strings the inference API builds from criteria.
               'options': [f'{k}: {mi["options"][k]}' for k in ids],
               'target': [float(k == r['chosen']) for k in ids],
               'metadata': {'scorer': r.get('scorer'), 'goal_status': outcome.get(r['goal']),
                            'provenance': {'type': 'import', 'license': 'CC0-1.0', 'split_policy': 'by goal (sha256 of trace/goal)',
                                           'input_sha256': digest, 'source_url': f'local-trace:{trace.name}',
                                           'original_id': f'{trace.name}:{n}'}}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('traces', nargs='+', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--all-goals', action='store_true', help='also keep decisions from goals that failed')
    args = p.parse_args()
    rows = [row for t in args.traces if t.is_dir() for row in rows_from(t, args.all_goals)]
    if not rows:
        sys.exit('no decisions with model inputs found (traces from before this exporter existed have none)')
    args.out.mkdir(parents=True, exist_ok=True)
    counts = Counter(r['split'] for r in rows)
    for split in counts:
        with (args.out / f'{split}.jsonl').open('w') as f:
            for r in rows:
                if r['split'] == split:
                    f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n')
    print(json.dumps({'rows': len(rows), 'splits': counts, 'goals': len({r['group_id'] for r in rows}), 'out': str(args.out)}))


if __name__ == '__main__':
    main()
