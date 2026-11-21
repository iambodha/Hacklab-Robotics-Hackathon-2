"""Harness self-test: drive a pick-and-place through the MCP tool functions.

Uses privileged IK (object/bin coordinates) ONLY to prove the simulated physics and
tool semantics allow the task; the agent never gets this information.

  python3 sim/selftest.py [--randomize] [--images DIR]
"""
import argparse
import asyncio
import json
from pathlib import Path

import numpy as np

import sim_arm_mcp as srv
from build_scene import ArmIK


def norms(q):
    return {f'j{i+1}': float(v) for i, v in enumerate(srv.arm.norm(np.asarray(q, float)))}


async def go(q, label, log, step=0.02, dt=0.12):
    """Move to joint pose q (radians) in chunks that respect ROBOT_MAX_DELTA."""
    target = norms(q)
    target.pop(srv.GRIPPER)
    for _ in range(10):
        cur = {k: v['norm'] for k, v in srv.arm.state().items()}
        if max(abs(target[k] - cur[k]) for k in target) < 0.01:
            break
        chunk = {k: cur[k] + np.clip(target[k] - cur[k], -0.3, 0.3) for k in target}
        out = await srv.move_to(chunk, step, dt)
        log.append((label, out.splitlines()[0]))
        if out.startswith('ABORTED'):
            return False
    return True


async def main():
    p = argparse.ArgumentParser()
    p.add_argument('--randomize', action='store_true')
    p.add_argument('--images', type=Path)
    args = p.parse_args()
    if args.randomize:
        await srv.operator_reset_scene(True)
    open_val = 0.55 if srv.arm.robot_model == 'so101' else 1.0
    if srv.arm.state()[srv.GRIPPER]['norm'] < open_val - 0.1:
        await srv.gripper(open_val)
    truth = srv.arm.truth()
    cube = np.array(truth['objects']['black_cube']['xyz'])
    box = srv.arm.layout['bins'][0]
    ik = ArmIK(srv.arm.model)
    log, ok = [], True
    q = srv.arm.state()
    q_now = np.array([srv.arm.rad(i, q[f'j{i+1}']['norm']) for i in range(srv.arm.model.nu)])
    plan = [('above cube', [*cube[:2], 0.09], True, 0.02, 0.12),
            ('descend', [*cube[:2], 0.018 if srv.arm.robot_model == 'so101' else 0.022], True, 0.01, 0.15),
            ('close', None, None, None, None),
            ('lift', [*cube[:2], 0.13 if srv.arm.robot_model == 'so101' else 0.10], False, 0.015, 0.15),
            ('over box', [*box['pos'], box['size'][2] + 0.06], False, 0.02, 0.12),
            ('open', None, None, None, None)]
    for label, xyz, top_down, step, dt in plan:
        if label in ('close', 'open'):
            out = await srv.gripper(0.0 if label == 'close' else open_val)
            log.append((label, out.splitlines()[0]))
            await asyncio.sleep(0.5 if label == 'close' else 1.0)
            continue
        sol = ik.solve(xyz, top_down=top_down, q0=q_now)
        if label == 'over box':  # a tilted gripper can leave the cube lying on the lower pad: prefer pointing down
            for z in (0.14, 0.13, 0.12):
                for ln in (0, 10, 20, 30):
                    down = ik.solve([*xyz[:2], z], top_down=True, q0=q_now, lean_deg=ln)
                    if down is not None:
                        sol = down; break
                if down is not None:
                    break
        if sol is None:
            log.append((label, 'IK failed')); ok = False; break
        q_now = sol
        if not await go(sol, label, log, step, dt):
            ok = False; break
        with srv.arm.lock:
            d, m = srv.arm.data, srv.arm.model
            log.append((label, {'tcp_target': np.round(xyz, 3).tolist(),
                                'tcp_actual': np.round(d.site_xpos[m.site('grasp_tcp').id], 3).tolist(),
                                'pads': [np.round(d.geom_xpos[m.geom(n).id], 3).tolist() for n in ('left_pad', 'right_pad')]}))
        if args.images:
            args.images.mkdir(parents=True, exist_ok=True)
            for cam, *_ in srv.CAMERAS:
                (args.images / f"{label.replace(' ', '_')}_{cam}.jpg").write_bytes(srv.arm.snapshot(cam))
    final = srv.arm.truth()
    success = final['objects']['black_cube']['in_bin'] == box['name']
    print(json.dumps({'start': truth, 'final': final, 'steps': log, 'success': success and ok}, indent=2))
    print(srv.arm.table())
    if srv.recorder:
        srv.recorder.close()
        import make_replay
        print('replay:', make_replay.build(srv.recorder.dir))
    srv.arm.running = False
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
