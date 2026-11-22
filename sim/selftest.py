"""Harness self-test: drive a pick-and-place through the MCP tool functions.

Uses privileged IK (object/bin coordinates) ONLY to prove the simulated physics and
tool semantics allow the task; the agent never gets this information.

  python3 sim/selftest.py [--layout venue|home] [--randomize] [--images DIR]

venue: black cube into the grey box. home (no loose objects on the real table yet): the rendered overview shows the
ArUco markers where the model puts them, the arm reaches above the two base markers through the tool functions, and
the wrist camera then sees the marker under the gripper.
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np

_p = argparse.ArgumentParser()
_p.add_argument('--layout', default=os.environ.get('SIM_LAYOUT', 'venue'))
_p.add_argument('--randomize', action='store_true')
_p.add_argument('--images', type=Path)
ARGS = _p.parse_args()
os.environ['SIM_LAYOUT'] = ARGS.layout          # sim_arm_mcp picks its layout at import
os.environ.pop('ROBOT_SIM_LAYOUT', None)
sys.argv = sys.argv[:1]

import sim_arm_mcp as srv  # noqa: E402
from build_scene import ArmIK  # noqa: E402


def norms(q):
    return {f'j{i+1}': float(v) for i, v in enumerate(srv.arm.norm(np.asarray(q, float)))}


async def go(q, label, log, step=0.02, dt=0.12):
    """Move to joint pose q (radians) in chunks that respect ROBOT_MAX_DELTA."""
    target = norms(q)
    target.pop('j8')
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


def detect_markers(rgb, dictionary):
    import cv2
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)), cv2.aruco.DetectorParameters())
    c, ids, _ = det.detectMarkers(cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY))
    return {} if ids is None else {int(i): cc.reshape(4, 2) for i, cc in zip(ids.ravel(), c)}


def model_corners(cam, mid, d=None):
    """Projected corner sites of marker mid in camera cam (pinhole from fovy, as MuJoCo renders)."""
    import build_scene
    m, d = srv.arm.model, (d if d is not None else srv.arm.data)
    W, H = srv.arm.layout['image_size']
    cid = m.camera(cam).id
    R = d.cam_xmat[cid].reshape(3, 3) @ build_scene.CV_TO_MJ
    pc = (np.array([d.site_xpos[m.site(f'aruco_{mid}_c{k}').id] for k in range(4)]) - d.cam_xpos[cid]) @ R
    f = H / 2 / np.tan(np.radians(m.cam_fovy[cid]) / 2)
    return pc[:, :2] / pc[:, 2:3] * f + [W / 2, H / 2]


async def home_check(args):
    """Markers, reach and wrist view on the home table (no objects to move there yet)."""
    mk = srv.arm.layout['markers']
    log, ok = [], True
    with srv.arm.lock:
        seen = detect_markers(srv.arm.frame('overview_cam'), mk['dictionary'])
        err = {i: float(np.linalg.norm(seen[i] - model_corners('overview_cam', i), axis=1).mean()) for i in seen}
    in_layout = {it['id'] for it in mk['items']}
    log.append(('overview markers', {'seen': sorted(seen), 'corner_err_px': {i: round(e, 2) for i, e in err.items()}}))
    ok &= len(set(seen) & in_layout) >= 3 and max(err.values()) < 1.5
    ik = ArmIK(srv.arm.model)
    q = srv.arm.state()
    q_now = np.array([srv.arm.rad(i, q[f'j{i+1}']['norm']) for i in range(8)])
    base_markers = sorted((it for it in mk['items'] if it.get('role') == 'base'),
                          key=lambda it: it['id']) or mk['items'][:2]
    import mujoco
    import real_joints
    kin = mujoco.MjData(srv.arm.model)
    W, H = srv.arm.layout['image_size']
    for it in base_markers:
        xy = np.asarray(it['T'])[:2, 3] if 'T' in it else np.asarray(it['pos'][:2])
        # a reachable TCP point on the way to the marker from which the model's wrist camera sees the whole marker
        pick = None
        for frac in (0.55, 0.45, 0.65, 0.35):
            for z in (0.14, 0.18, 0.10):
                target = [*(xy * frac), z]
                sol, lean = ik.solve_least_lean(target, q_now)
                if sol is None:
                    continue
                real_joints.set_arm_q(srv.arm.model, kin, sol)
                px = model_corners('wrist_cam', it['id'], kin)
                if np.isfinite(px).all() and (px > 12).all() and (px < [W - 12, H - 12]).all():
                    pick = (target, sol, lean); break
            if pick:
                break
        if pick is None:
            log.append((f"view marker {it['id']}", 'no reachable pose sees it')); ok = False; continue
        target, sol, lean = pick
        q_now = sol
        moved = await go(sol, f"view marker {it['id']}", log)
        await asyncio.sleep(0.8)
        with srv.arm.lock:
            tcp = srv.arm.data.site_xpos[srv.arm.model.site('grasp_tcp').id].copy()
            wrist = detect_markers(srv.arm.frame('wrist_cam'), mk['dictionary'])
            werr = float(np.linalg.norm(wrist[it['id']] - model_corners('wrist_cam', it['id']), axis=1).mean()) if it['id'] in wrist else None
        miss = float(np.linalg.norm(tcp - target))
        log.append((f"view marker {it['id']}", {'tcp_target': np.round(target, 3).tolist(), 'lean_deg': lean, 'tcp_err_m': round(miss, 4),
                                               'wrist_sees': sorted(wrist), 'wrist_corner_err_px': werr and round(werr, 2)}))
        ok &= moved and miss < 0.02 and werr is not None and werr < 2.0
        if args.images:
            args.images.mkdir(parents=True, exist_ok=True)
            for cam, *_ in srv.CAMERAS:
                (args.images / f"home_marker{it['id']}_{cam}.jpg").write_bytes(srv.arm.snapshot(cam))
    print(json.dumps({'layout': str(srv.LAYOUT), 'steps': log, 'success': bool(ok)}, indent=2))
    srv.arm.running = False
    return 0 if ok else 1


async def main():
    args = ARGS
    if 'markers' in srv.arm.layout and not srv.arm.layout['objects']:
        return await home_check(args)
    if args.randomize:
        await srv.operator_reset_scene(True)
    truth = srv.arm.truth()
    cube = np.array(truth['objects']['black_cube']['xyz'])
    box = srv.arm.layout['bins'][0]
    ik = ArmIK(srv.arm.model)
    log, ok = [], True
    q = srv.arm.state()
    q_now = np.array([srv.arm.rad(i, q[f'j{i+1}']['norm']) for i in range(8)])
    plan = [('above cube', [*cube[:2], 0.09], True, 0.02, 0.12),
            ('descend', [*cube[:2], 0.022], True, 0.01, 0.15),
            ('close', None, None, None, None),
            ('lift', [*cube[:2], 0.10], False, 0.01, 0.15),
            ('over box', [*box['pos'], box['size'][2] + 0.06], False, 0.02, 0.12),
            ('open', None, None, None, None)]
    for label, xyz, top_down, step, dt in plan:
        if label in ('close', 'open'):
            for v in ((0.7, 0.4, 0.1, 0.0) if label == 'close' else (0.3, 0.6, 0.9, 1.0)):  # gripper() refuses > MAX_DELTA
                out = await srv.gripper(v); log.append((label, out.splitlines()[0]))
            await asyncio.sleep(0.5 if label == 'close' else 1.0)
            continue
        sol = ik.solve(xyz, top_down=top_down, q0=q_now)
        if label == 'over box':  # a tilted gripper can leave the cube lying on the lower pad: prefer pointing down
            for z in (0.14, 0.13, 0.12):
                down = ik.solve([*xyz[:2], z], top_down=True, q0=q_now)
                if down is not None:
                    sol = down; break
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
