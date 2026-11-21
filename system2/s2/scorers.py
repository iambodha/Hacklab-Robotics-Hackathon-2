"""Scorers rank System 1's candidates. All share one interface, so the runner, the guards and
the logs do not care which one decides:

  geometric  pick the feasible candidate with the lowest predicted error (the baseline)
  jev        Open-Jev ranks the feasible candidates from a structured description of the
             goal, the observations, recent outcomes and each candidate's prediction
  shadow     geometric decides; Open-Jev's choice is logged next to it for comparison
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import httpx
import numpy as np

from .candidates import Candidate
from .goals import Goal, State

QUESTION = ('Choose the next arm motion. It should make the most progress toward meeting every condition '
            'without breaking a constraint. Predictions come from the arm kinematic model; if recent actions '
            'did not have their predicted effect, prefer actions that gather evidence or recover.')


@dataclass
class Ranking:
    order: list[str]                      # candidate ids, best first (feasible only)
    scorer: str
    probabilities: dict[str, float] = field(default_factory=dict)
    latency_s: float = 0.0
    shadow: dict | None = None            # the other scorer's view (shadow mode)
    note: str = ''


def geometric(cands: list[Candidate], cost_now: float) -> Ranking:
    feas = [c for c in cands if c.feasible]
    # Lowest predicted error; ties go to the candidate listed first (bigger, more direct steps).
    order = sorted(feas, key=lambda c: (round(c.cost, 5), cands.index(c)))
    return Ranking([c.id for c in order], 'geometric')


def jev_state(goal: Goal, s: State, evals, world_summary: dict) -> dict:
    objs = {n: {k: v for k, v in e.items() if k in ('kind', 'xyz_m', 'visible', 'seen_by', 'last_seen_s_ago', 'held', 'note')}
            for n, e in world_summary['entities'].items()}
    g = world_summary['gripper']
    return {
        'task': goal.description,
        'conditions': [{'condition': c.spec, 'now': e.text, 'met': e.met} for c, e in zip(goal.conditions, evals)],
        'constraints': [c.spec for c in goal.constraints],
        'gripper': {'tcp_cm': [round(v * 100, 1) for v in g['tcp_xyz_m']], 'jaw_open': g['jaw_open'],
                    'something_between_jaws': g['something_between_jaws']},
        'objects': objs,
        'recent_actions': goal.history[-3:],
    }


class JevScorer:
    def __init__(self, url: str, timeout: float):
        self.url, self.timeout = url.rstrip('/'), timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        self.latencies: list[float] = []

    async def health(self) -> dict:
        r = await self.client.get(f'{self.url}/health', timeout=5)
        r.raise_for_status()
        return r.json()

    async def rank(self, cands: list[Candidate], cost_now: float, state: dict) -> Ranking:
        feas = [c for c in cands if c.feasible]
        criteria = {c.id: c.describe(cost_now) for c in feas}
        body = {'model': 'open-jev', 'state': state,
                'questions': {'next_motion': {'type': 'choice', 'instructions': QUESTION, 'criteria': criteria}}}
        t0 = time.monotonic()
        r = await self.client.post(f'{self.url}/v1/systemone', json=body)
        latency = time.monotonic() - t0
        if r.status_code != 200:
            raise RuntimeError(f'Open-Jev HTTP {r.status_code}: {r.text[:300]}')
        probs = r.json()['answers']['next_motion']['probabilities']
        if set(probs) != set(criteria):
            raise RuntimeError('Open-Jev answered a different candidate set')
        self.latencies.append(latency)
        order = sorted(probs, key=lambda k: -probs[k])
        return Ranking(order, 'jev', {k: round(float(v), 4) for k, v in probs.items()}, latency)

    def stats(self) -> dict:
        if not self.latencies:
            return {'calls': 0}
        a = np.array(self.latencies)
        return {'calls': len(a), 'p50_s': round(float(np.percentile(a, 50)), 3),
                'p95_s': round(float(np.percentile(a, 95)), 3)}

    async def aclose(self):
        await self.client.aclose()
