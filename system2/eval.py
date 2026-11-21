"""Closed-loop evaluation of System 1 in the MuJoCo twin, with a scripted stand-in for System 2.

  python system2/eval.py --episodes 5 --policy geometric [--task pick_place|reach_away] [--seed 0]

Per episode: reset the scene (randomized), pick the cube and the box out of the detections
(smallest / largest overview blob), then submit the same goal sequence a planner would. The
runner, perception, guards and arm tools are exactly the ones the MCP server uses. Success is
scored ONLY from the simulator's ground-truth log (ROBOT_SIM_TRUTH_LOG), never shown to the
runner. Writes report.json next to the decision traces.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s2.arm import ArmClient  # noqa: E402
from s2.config import Config  # noqa: E402
from s2.goals import parse_goal  # noqa: E402
from s2.runner import Runner, Trace  # noqa: E402
from s2.scorers import JevScorer  # noqa: E402
from s2.world import World  # noqa: E402

PICK_PLACE = [
    ('jaws open, 6 cm above the cube, lined up with its faces',
     [{'type': 'gripper', 'state': 'open'}, {'type': 'reach', 'a': 'gripper', 'b': 'cube', 'offset': [0, 0, 0.06], 'tol': 0.008},
      {'type': 'yaw_aligned', 'a': 'gripper', 'b': 'cube', 'tol_deg': 8}], []),
    ('holding the cube', [{'type': 'gripper', 'state': 'holding', 'object': 'cube'}], []),
    ('cube lifted 8 cm', [{'type': 'height', 'a': 'cube', 'min': 0.08}], [{'type': 'keep_holding', 'object': 'cube'}]),
    ('cube carried over the box opening', [{'type': 'reach', 'a': 'cube', 'b': 'box', 'offset': [0, 0, 0.12], 'tol': 0.02, 'tol_z': 0.03}],
     [{'type': 'keep_holding', 'object': 'cube'}, {'type': 'min_height', 'z': 0.1}]),
    ('cube released inside the box', [{'type': 'within', 'a': 'cube', 'b': 'box', 'margin': 0.02},
                                      {'type': 'gripper', 'state': 'open'}], []),
]
REACH_AWAY = [
    ('gripper 5 cm above the cube', [{'type': 'reach', 'a': 'gripper', 'b': 'cube', 'offset': [0, 0, 0.05], 'tol': 0.01}], []),
    ('gripper at least 15 cm from the cube, sideways', [{'type': 'distance', 'a': 'gripper', 'b': 'cube', 'axes': 'xy', 'min': 0.15}],
     [{'type': 'min_height', 'z': 0.04}]),
]
TASKS = {'pick_place': PICK_PLACE, 'grasp': PICK_PLACE[:3], 'reach_away': REACH_AWAY}


def truth_log_path() -> Path:
    return Path(os.environ['ROBOT_SIM_TRUTH_LOG'])


def truth(path: Path) -> dict:
    lines = path.read_text().splitlines() if path.exists() else []
    return json.loads(lines[-1]) if lines else {}


async def episode(arm, runner: Runner, world: World, task, randomize: bool, log) -> dict:
    await arm.reset_scene(randomize)
    world.entities.clear()
    world.gripper.jaw_cmd = None
    runner.yaw = None
    await asyncio.sleep(0.5)
    await runner.observe()
    over = sorted((d for d in world.detections.values() if d.camera == 'overview'), key=lambda d: d.area)
    if len(over) < 2:
        return {'success': False, 'reason': f'perception found {len(over)} overview objects'}
    world.track('cube', over[0].id, 0.02, 'object')
    world.track('box', over[-1].id, 0.09, 'container')
    log(f'  tracked cube {world.entities["cube"].xyz.round(3).tolist()} box {world.entities["box"].xyz.round(3).tolist()} '
        f'{world.entities["box"].size.round(3).tolist()}')
    goals = []
    for desc, conds, cons in task:
        g = parse_goal({'description': desc, 'conditions': conds, 'constraints': cons, 'timeout_s': 120, 'max_steps': 40}, world)
        runner.submit(g)
        await runner.wait(200)
        goals.append({'goal': desc, 'status': g.status, 'reason': g.reason, 'steps': g.steps,
                      'seconds': round(time.monotonic() - g.created, 1)})
        log(f'  {g.status:9s} {g.steps:2d} steps {goals[-1]["seconds"]:5.1f}s  {desc}: {g.reason[:150]}')
        if any(c.type == 'yaw_aligned' for c in g.conditions):
            await arm.text('joint_state')  # logs a fresh ground-truth line
            gt = truth(truth_log_path()).get('objects', {}).get('black_cube', {})
            cube = world.entities['cube']
            goals[-1].update(yaw_estimate=cube.yaw, jaw_yaw=world.gripper.jaw_yaw_deg, yaw_truth=gt.get('yaw_deg'),
                             xy_estimate=cube.xyz[:2].round(4).tolist(), xy_truth=gt.get('xyz', [None, None])[:2])
            log(f'    cube yaw est {cube.yaw} jaw {world.gripper.jaw_yaw_deg} truth {gt.get("yaw_deg")}; '
                f'xy est {cube.xyz[:2].round(4).tolist()} truth {gt.get("xyz")}')
        if g.status != 'succeeded':
            break
    return {'goals': goals, 'all_goals_succeeded': all(g['status'] == 'succeeded' for g in goals) and len(goals) == len(task)}


async def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--policy', choices=('geometric', 'jev', 'shadow'), default=None)
    p.add_argument('--task', choices=sorted(TASKS), default='pick_place')
    p.add_argument('--no-randomize', action='store_true', help='first episode uses the default placement')
    args = p.parse_args()
    if args.policy:
        os.environ['S2_POLICY'] = args.policy
    truth_log = Path(os.environ.setdefault('ROBOT_SIM_TRUTH_LOG', f'/tmp/s2-eval-truth-{os.getpid()}.jsonl'))
    cfg = Config()
    cfg.validate()
    trace = Trace(cfg.trace_dir)
    jev = JevScorer(cfg.jev_url, cfg.jev_timeout) if cfg.policy != 'geometric' else None
    results = []
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    async with ArmClient(cfg.arm_command[0], cfg.arm_command[1:], cfg.arm_env(), cfg.camera_roles) as arm:
        world = World()
        runner = Runner(arm, world, cfg, jev, trace)
        for k in range(args.episodes):
            t0 = time.monotonic()
            log(f'episode {k + 1}/{args.episodes} policy={cfg.policy} task={args.task}')
            r = await episode(arm, runner, world, TASKS[args.task], not (args.no_randomize and k == 0), log)
            await arm.text('joint_state')  # flush a final ground-truth line
            gt = truth(truth_log).get('objects', {}).get('black_cube', {})
            r.update(seconds=round(time.monotonic() - t0, 1), cube_truth=gt)
            if args.task == 'pick_place':
                r['success'] = gt.get('in_bin') == 'grey_box'
            elif args.task == 'grasp':
                r['success'] = r['all_goals_succeeded'] and gt.get('xyz', [0, 0, 0])[2] > 0.06
            else:
                r['success'] = r['all_goals_succeeded']
            log(f'  => success={r["success"]} truth={gt}')
            results.append(r)
    summary = {'policy': cfg.policy, 'task': args.task, 'episodes': len(results),
               'successes': sum(r['success'] for r in results), 'runner': runner.stats,
               'jev_latency': jev.stats() if jev else None, 'results': results}
    (trace.dir / 'report.json').write_text(json.dumps(summary, indent=2))
    log(json.dumps({k: v for k, v in summary.items() if k != 'results'}, indent=1))
    log(f'report: {trace.dir / "report.json"}')
    if jev:
        await jev.aclose()


if __name__ == '__main__':
    asyncio.run(main())
