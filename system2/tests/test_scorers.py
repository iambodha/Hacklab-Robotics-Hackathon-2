import asyncio
import json

import httpx
import pytest

from conftest import make_world
from s2.candidates import generate
from s2.goals import State, parse_goal
from s2.scorers import JevScorer, geometric, jev_state


def _setup():
    w = make_world(objects={'cube': ((0.02, 0.16, 0.01), 'object', 0.02, (0.02, 0.02))})
    g = parse_goal({'description': 'above the cube', 'conditions': [{'type': 'reach', 'b': 'cube', 'offset': [0, 0, 0.06]}]}, w)
    s = State.from_world(w, 20)
    return w, g, s, generate(g, s)


def test_geometric_ranks_only_feasible_by_predicted_cost():
    w, g, s, cands = _setup()
    r = geometric(cands, g.cost(s))
    by = {c.id: c for c in cands}
    costs = [by[i].cost for i in r.order]
    assert costs == sorted(costs) and all(by[i].feasible for i in r.order)


def test_jev_request_shape_and_mapping():
    w, g, s, cands = _setup()
    seen = {}

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        seen['body'] = body
        crit = body['questions']['next_motion']['criteria']
        probs = {k: (0.9 if i == 1 else 0.1 / (len(crit) - 1)) for i, k in enumerate(crit)}
        return httpx.Response(200, json={'answers': {'next_motion': {'type': 'choice', 'probabilities': probs}}})

    jev = JevScorer('http://jev.test', 5)
    jev.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    state = jev_state(g, s, g.evaluate(s), w.summary())
    r = asyncio.run(jev.rank(cands, g.cost(s), state))
    feas = [c.id for c in cands if c.feasible]
    assert set(seen['body']['questions']['next_motion']['criteria']) == set(feas)
    assert r.order[0] == feas[1] and r.scorer == 'jev'
    assert seen['body']['state']['task'] == 'above the cube'


def test_jev_answer_for_other_candidates_is_rejected():
    w, g, s, cands = _setup()
    jev = JevScorer('http://jev.test', 5)
    jev.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(
        200, json={'answers': {'next_motion': {'probabilities': {'zzz': 1.0}}}})))
    with pytest.raises(RuntimeError, match='different candidate set'):
        asyncio.run(jev.rank(cands, g.cost(s), {}))
