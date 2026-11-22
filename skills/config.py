"""Shared strategy reader; isolated wipe improvement sandboxes ship skills/ alone.

Normal execution uses versioned skillcfg. In an isolated package the same checked-in
baseline file is sufficient; no hardware or runtime configuration is available there.
"""
import json
from pathlib import Path


def get(key):
    try:
        from learning.skillcfg import get as active
    except ModuleNotFoundError as error:
        if error.name not in ('learning', 'learning.skillcfg'):
            raise
        return json.loads(Path(__file__).with_name('strategy_defaults.json').read_text())['entries'][key]['value']
    return active(key)
