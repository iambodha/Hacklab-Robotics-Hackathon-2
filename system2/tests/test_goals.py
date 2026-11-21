import numpy as np
import pytest

from conftest import make_world
from s2.candidates import generate, predict
from s2.goals import GoalError, State, parse_goal
from s2.scorers import geometric


def goal(world, conditions, constraints=()):
    return parse_goal({'description': 't', 'conditions': list(conditions), 'constraints': list(constraints)}, world)


def state(world):
    return State.from_world(world, belief_ttl=20)


def test_validation_rejects_unknown_entities_and_types(cube_world):
    with pytest.raises(GoalError, match='unknown entity'):
        goal(cube_world, [{'type': 'reach', 'b': 'sphere'}])
    with pytest.raises(GoalError, match='unknown condition type'):
        goal(cube_world, [{'type': 'pick_up', 'object': 'cube'}])
    with pytest.raises(GoalError, match='non-empty'):
        goal(cube_world, [])
    with pytest.raises(GoalError, match='outside'):
        goal(cube_world, [{'type': 'reach', 'b': 'cube', 'tol': 0.0001}])


def test_reach_met_within_tolerance(cube_world):
    g = goal(cube_world, [{'type': 'reach', 'b': 'cube', 'offset': [0, 0, 0.06], 'tol': 0.01}])
    s = state(cube_world).moved(delta=np.array([0.02, 0.16, 0.07]) - cube_world.gripper.tcp)
    assert g.evaluate(s)[0].met
    assert not g.evaluate(s.moved(delta=np.array([0.02, 0, 0])))[0].met


def test_holding_cost_falls_monotonically_along_the_approach():
    """Regression: the cost used to jump up when the cube came between the jaws."""
    path = [(0.02, 0.16, z) for z in np.linspace(0.08, 0.018, 15)]
    costs = []
    for tcp in path:
        w = make_world(tcp=tcp, objects={'cube': ((0.02, 0.16, 0.01), 'object', 0.02, (0.02, 0.02))})
        costs.append(goal(w, [{'type': 'gripper', 'state': 'holding', 'object': 'cube'}]).cost(state(w)))
    assert all(b < a + 1e-9 for a, b in zip(costs, costs[1:])), costs


def test_closing_in_position_is_chosen_and_predicted_to_grip():
    w = make_world(tcp=(0.02, 0.16, 0.02), objects={'cube': ((0.02, 0.16, 0.01), 'object', 0.02, (0.02, 0.02))})
    g = goal(w, [{'type': 'gripper', 'state': 'holding', 'object': 'cube'}])
    s = state(w)
    cands = generate(g, s)
    best = next(c for c in cands if c.id == geometric(cands, g.cost(s)).order[0])
    assert best.kind == 'jaw' and best.jaw == 0.0
    assert 'cube' in predict(g, s, best).holding


def test_does_not_close_before_reaching_the_object(cube_world):
    g = goal(cube_world, [{'type': 'gripper', 'state': 'holding', 'object': 'cube'}])
    s = state(cube_world)
    cands = generate(g, s)
    best = next(c for c in cands if c.id == geometric(cands, g.cost(s)).order[0])
    assert best.kind == 'move'


def test_approach_lines_up_above_before_descending():
    w = make_world(tcp=(0.10, 0.10, 0.10), objects={'cube': ((0.02, 0.16, 0.01), 'object', 0.02, (0.02, 0.02))})
    g = goal(w, [{'type': 'gripper', 'state': 'holding', 'object': 'cube'}])
    tgt = g.conditions[0].target(state(w))
    assert tgt[2] >= 0.06 and np.allclose(tgt[:2], [0.02, 0.16])


def test_rest_pose_offers_an_entry_move(cube_world):
    w = make_world(tcp=(0.006, 0.049, 0.492), objects={'cube': ((0.02, 0.16, 0.01), 'object', 0.02, (0.02, 0.02))})
    g = goal(w, [{'type': 'reach', 'b': 'cube', 'offset': [0, 0, 0.06]}])
    s = state(w)
    cands = generate(g, s)
    best = next(c for c in cands if c.id == geometric(cands, g.cost(s)).order[0])
    assert 'working volume' in best.label and (s.tcp + best.delta)[2] <= 0.3


def test_min_height_and_keep_holding_constraints(cube_world):
    g = goal(cube_world, [{'type': 'reach', 'b': 'cube'}], [{'type': 'min_height', 'z': 0.08}])
    s = state(cube_world)
    assert all(c.delta[2] >= -1e-9 or (s.tcp + c.delta)[2] >= 0.08 - 0.002 for c in generate(g, s) if c.feasible and c.kind == 'move')
    held = State(s.tcp, 0.03, 0.0, 0.0, frozenset({'cube'}), s.pos, s.known, s.entities)
    g2 = goal(cube_world, [{'type': 'height', 'a': 'cube', 'min': 0.2}], [{'type': 'keep_holding', 'object': 'cube'}])
    opens = [c for c in generate(g2, held) if c.kind == 'jaw' and c.jaw > 0.5]
    assert opens and not any(c.feasible for c in opens)


def test_blocked_direction_is_ruled_out(cube_world):
    g = goal(cube_world, [{'type': 'reach', 'xyz': [0.2, 0.15, 0.10]}])
    cands = generate(g, state(cube_world), [np.array([1.0, 0, 0])])
    assert all(not c.feasible for c in cands if c.kind == 'move' and c.delta[0] > 0.9 * np.linalg.norm(c.delta))


def test_unobservable_entity_is_reported_not_guessed(cube_world):
    cube_world.entities['cube'].visible = False
    cube_world.entities['cube'].last_seen = -100
    g = goal(cube_world, [{'type': 'reach', 'b': 'cube'}])
    assert g.evaluate(state(cube_world))[0].met is None


def test_containers_stay_known_when_hidden(cube_world):
    cube_world.entities['box'].visible = False
    cube_world.entities['box'].last_seen = -1000
    assert state(cube_world).known['box']
