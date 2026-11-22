"""Task placements on a site's table, as tape measurements from its markers, checked with the skills' own IK.

    python sim/site_tasks.py place --site office      # -> sites/office/tasks.json (bottle, kit sheet, pick points)
    python sim/site_tasks.py pick  --site office      # scripted pick + lift of a 2 cm cube at each pick point (twin)

Everything is computed from the site layout at run time (markers, table, base frame), so a refined calibration only
needs a re-run. Positions are in the arm base frame; the arm is the same on every table, so the same base-frame point
gives the same arm pose and gripper lean as at home.

  bottle   the home demo cap position (learning/profiles/bottle_cap.json object.cap_xy_m: attempts 9-12 opened the
           cap there at a measured 32-33 deg wrist lean), checked with the twist_cap IK (hover, then grasp at the same
           lean), plus the other spots on the office table where the IK gives the same lean.
  kitting  the A4 sheet at the spec's home pose (tasks/kitting/kit_a4.json placement), checked: on the table, clear of
           every table marker, every cell and start spot reachable at grasp and hover height for the nominal and the
           extreme randomized sheet poses.
  pick     a few marker-relative points (between markers 22 and 23 toward the far markers, and farther out).
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import itertools
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'cgl' if sys.platform == 'darwin' else 'egl')
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(HERE), str(REPO)):
    if p not in sys.path:
        sys.path.append(p)

import site_reach as SR  # noqa: E402
from sites import get_site  # noqa: E402

GRASP_BELOW_TOP = 0.010      # learning/skillcfg twist.grasp_below_top_m
APPROACH_Z_OFFSET = -0.0065  # learning/profiles/bottle_cap.json params.default.approach_z_offset_mm (learned)
BOTTLE_R = 0.036             # learning/bottle_twin.REAL_BOTTLE body_r: footprint radius of the demo bottle
BOTTLE_CLEAR = 0.05          # m from the bottle wall to a marker / fixed object: room for the hand placing it
MARKER_CLEAR = 0.01          # m around a marker the sheet must leave free (the quiet zone the detector needs)
TABLE_MARGIN = 0.02          # m between the sheet / an object and the table edge
KIT_GRASP_Z = 0.014          # TCP z of a kit grasp (fingertips 10 mm lower; kit profile grasp_z_mm 8-16)
KIT_HOVER_Z = 0.08           # tasks/kitting/plan.TRAVEL_Z: travel and hover height
LEAN_TOL = 5.0               # deg: a lean within this of the home demo lean "reproduces" it
PICK_ACROSS = (0.12, 0.30)   # m forward of the 22-23 marker line: near the base, and farther out
PICK_SIDE_ALONG = 0.12       # m from marker 23 toward 22 for the side point


# --- IK (the skill's full least-lean search) ---------------------------------------------------------------------
class IK:
    def __init__(self, site: str):
        import build_scene
        import mujoco
        st = get_site(site)
        scene, _ = build_scene.build(st.layout, Path(tempfile.gettempdir()) / f'site_tasks_{site}', report=False)
        m = mujoco.MjModel.from_xml_path(str(scene))
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key('rest').id)
        self.ik = build_scene.ArmIK(m)
        self.rest = d.qpos[self.ik.qids].copy()

    def lean(self, xyz, q0=None, yaw=0.0):
        q, lean = self.ik.solve_least_lean(np.asarray(xyz, float), self.rest if q0 is None else q0, yaw)
        return q, lean

    def keep_lean(self, xyz, lean, q0, yaw=0.0):
        """A straight-down move keeping the orientation (twist_cap's guarded descent): the same lean at xyz."""
        xyz = np.asarray(xyz, float)
        return self.ik.solve_rot(xyz, self.ik.target_rot(xyz, lean, yaw), q0, 0)


# --- polygons ----------------------------------------------------------------------------------------------------
def overlaps(a: np.ndarray, b: np.ndarray) -> bool:
    """Separating-axis test for two convex polygons (k, 2)."""
    for poly in (a, b):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-e[1], e[0]])
            pa, pb = a @ n, b @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def grow(square: np.ndarray, by: float) -> np.ndarray:
    c = square.mean(0)
    d = square - c
    return c + d * (1 + by / (np.linalg.norm(d, axis=1, keepdims=True) / math.sqrt(2)))


# --- bottle ------------------------------------------------------------------------------------------------------
def bottle(lay: dict, ik: IK, rows: list[dict] | None) -> dict:
    from learning import skillcfg as SC
    os.environ.setdefault('SKILL_PROFILES', 'bottle_cap')
    obj = next(p for p in SC.profiles() if p.get('profile') == 'bottle_cap')['object']
    xy = [float(v) for v in obj['cap_xy_m']]
    top = obj['cap_top_z_m']
    hover_z, grasp_z = top + SR.HOVER_ABOVE_CAP, top - GRASP_BELOW_TOP + APPROACH_Z_OFFSET
    q_h, lean_h = ik.lean([*xy, hover_z])
    q_g = ik.keep_lean([*xy, grasp_z], lean_h, q_h) if q_h is not None else None
    mk = SR.markers(lay)
    half = lay['markers']['size_m'] / 2
    clear = {str(i): round(float(np.linalg.norm(np.asarray(xy) - m['xy'])) - BOTTLE_R - half, 3) for i, m in mk.items()}
    fixed = {f['name']: round(float(np.hypot(*(np.asarray(xy) - f['pos'][:2]))) - BOTTLE_R - f['size'][0], 3)
             for f in lay.get('fixed', [])}
    same = []
    for p in rows or []:
        if p['lean'] is not None and abs(abs(p['lean']) - abs(lean_h or 0)) <= LEAN_TOL and SR.on_table(lay, p['xy'], BOTTLE_R + TABLE_MARGIN) \
                and all(np.linalg.norm(np.asarray(p['xy']) - m['xy']) > BOTTLE_R + half + BOTTLE_CLEAR for m in mk.values()):
            same.append({'xy_m': p['xy'], 'r_m': p['r'], 'az_deg': p['az'], 'map_lean_deg': p['lean']})
    radii = [s['r_m'] for s in same]
    return {'cap_xy_m': xy, 'cap_top_z_m': top, 'source': 'learning/profiles/bottle_cap.json object.cap_xy_m (home demo, attempts 9-12)',
            'ik': {'hover_z_m': round(hover_z, 4), 'hover_lean_deg': lean_h, 'grasp_z_m': round(grasp_z, 4),
                   'grasp_reachable_at_hover_lean': q_g is not None,
                   'note': 'IK lean steps are the skill config ik.leans (10/5 deg); the real arm measured 32-33 deg here '
                           '(SUMMARY.md attempts 9-10: 33.0 / 32.1 deg)'},
            'on_table': SR.on_table(lay, xy, BOTTLE_R + TABLE_MARGIN),
            'clearance_to_marker_edge_m': clear, 'clearance_to_fixed_m': fixed,
            'tape': {'from_22_toward_23': SR.to_marker(lay, xy, 22, 23, (20, 21)),
                     'from_23_toward_22': SR.to_marker(lay, xy, 23, 22, (20, 21))},
            'same_lean_ring': {'map_points': len(same), 'r_m': [min(radii), max(radii)] if radii else None,
                               'note': f'map points (sim/site_reach.py bottle_hover) within {LEAN_TOL} deg of the hover lean, '
                                       'on the table and clear of markers'}}


# --- kitting -----------------------------------------------------------------------------------------------------
def kit(lay: dict, ik: IK, site: str) -> dict:
    from tasks.kitting.spec import Pose2D, Spec
    spec = Spec.load()
    nom = spec.site_sheet_pose(site)
    r = spec.raw['placement']['randomize']
    # the nominal pose, then the four corners of the randomization box, each at a yaw extreme
    poses = [nom] + [Pose2D(nom.x + dx, nom.y + dy, nom.yaw_deg + dw) for (dx, dy), dw in
                     zip(itertools.product(r['x_m'], r['y_m']), [r['yaw_deg'][1], r['yaw_deg'][0], r['yaw_deg'][0], r['yaw_deg'][1]])]
    W, H = (v / 2000 for v in spec.paper_mm)
    page = np.array([[-W, -H], [W, -H], [W, H], [-W, H]])
    mk = SR.markers(lay)
    checks, seeds = [], {}
    for k, pose in enumerate(poses):
        corners = pose.apply(page)
        hit = [i for i, m in mk.items() if overlaps(corners, grow(m['corners'], MARKER_CLEAR))]
        pts = {f'cell {c}': spec.cell_pose(pose, c) for c in spec.cells()} | {f'spot {s}': spec.spot_pose(pose, s) for s in spec.spots()}
        leans = {}
        for name, p in pts.items():
            if k == 0:      # the skill's full search; the randomized poses continue from these solutions (cheaper)
                seeds[name], leans[name] = ik.lean([p.x, p.y, KIT_GRASP_Z], yaw=p.yaw_deg)
                _, leans[name + ' hover'] = ik.lean([p.x, p.y, KIT_HOVER_Z], yaw=p.yaw_deg)
            else:
                _, leans[name] = SR.continuation_lean(ik.ik, [p.x, p.y, KIT_GRASP_Z],
                                                      [s for s in (seeds[name], ik.rest) if s is not None])
        bad = [n for n, v in leans.items() if v is None]
        worst = max([abs(v) for v in leans.values() if v is not None] + [0])
        checks.append({'sheet_pose': pose.as_list(), 'on_table': all(SR.on_table(lay, c, TABLE_MARGIN) for c in corners),
                       'covers_markers': hit, 'unreachable': bad, 'max_lean_deg': worst,
                       **({'leans_deg': leans} if k == 0 else {})})
    corners = nom.apply(page)
    return {'sheet_xy_m': [nom.x, nom.y], 'sheet_yaw_deg': nom.yaw_deg,
            'source': 'tasks/kitting/kit_a4.json placement (home pose, arm base frame)',
            'sheet_corners_m': np.round(corners, 4).tolist(),
            'cells_m': {c: [round(spec.cell_pose(nom, c).x, 4), round(spec.cell_pose(nom, c).y, 4)] for c in spec.cells()},
            'spots_m': {s: [round(spec.spot_pose(nom, s).x, 4), round(spec.spot_pose(nom, s).y, 4)] for s in spec.spots()},
            'ok': all(c['on_table'] and not c['covers_markers'] and not c['unreachable'] for c in checks),
            'checks': checks,
            'tape': {'sheet_centre': SR.to_marker(lay, [nom.x, nom.y], 22, 23, (20, 21)),
                     'near_left_corner': SR.to_marker(lay, corners[0], 22, 23, (20, 21)),
                     'near_right_corner': SR.to_marker(lay, corners[1], 22, 23, (20, 21)),
                     'edges': 'long (297 mm) edges parallel to the arm x axis, the printed arrow / top edge away from the arm'}}


# --- pick points -------------------------------------------------------------------------------------------------
def pick_points(lay: dict) -> list[dict]:
    mk = SR.markers(lay)
    mid = (mk[22]['xy'] + mk[23]['xy']) / 2
    _, u, v = SR.marker_frame(lay, 22, 23, (20, 21))
    pts = [(f'mid22-23 +{a * 100:.0f}cm', mid + a * v, f'from the midpoint of markers 22 and 23, {a * 100:.0f} cm toward 20/21')
           for a in PICK_ACROSS]
    side = SR.from_marker(lay, 23, 22, PICK_SIDE_ALONG, PICK_ACROSS[0], (20, 21))
    pts.append(('side 23', side, f'{PICK_SIDE_ALONG * 100:.0f} cm from marker 23 toward 22, then {PICK_ACROSS[0] * 100:.0f} cm toward 20/21'))
    return [{'name': n, 'xy_m': np.round(p, 4).tolist(), 'how': how, 'tape': SR.to_marker(lay, p, 22, 23, (20, 21))}
            for n, p, how in pts]


def place(site: str) -> dict:
    st = get_site(site)
    lay = json.loads(st.layout.read_text())
    reach = st.directory / 'reach.json'
    rows = json.loads(reach.read_text())['samples']['bottle_hover'] if reach.is_file() else None
    ik = IK(site)
    old = json.loads((st.directory / 'tasks.json').read_text()) if (st.directory / 'tasks.json').is_file() else {}
    pick = pick_points(lay)
    for p in pick:
        _, p['grasp_lean_deg'] = ik.lean([*p['xy_m'], 0.014])
        p['sim'] = next((o.get('sim') for o in old.get('pick', {}).get('points', []) if o['name'] == p['name'] and o['xy_m'] == p['xy_m']), None)
    out = {'_about': f'Task placements on the {site} table (sim/site_tasks.py place; base frame = arm base, metres). '
                     'tape: along / across from a marker centre, measured on the table; dist_cm: straight distance '
                     'to each marker centre (two of them fix the point).',
           'site': site, 'layout': str(st.layout.relative_to(REPO)), 'generated': time.strftime('%Y-%m-%d %H:%M'),
           'bottle': {**old.get('bottle', {}), **bottle(lay, ik, rows)},
           'kitting': {**old.get('kitting', {}), **kit(lay, ik, site)},
           'pick': {'object': '2 cm cube (generaloid.checks pick: hover 0.08, grasp TCP 0.014, lift 0.12; picked = cube centre > 0.05)',
                    'points': pick}}
    return out


# --- scripted pick in the twin -----------------------------------------------------------------------------------
CUBE = {'mass': 0.006, 'yaw': 0,
        'parts': [{'type': 'box', 'size': [0.01, 0.01, 0.01], 'rgba': [0.06, 0.06, 0.07, 1]}]}


def pick(site: str, names: list[str] | None, work: Path) -> list[dict]:
    from generaloid import checks
    st = get_site(site)
    lay = json.loads(st.layout.read_text())
    tasks = json.loads((st.directory / 'tasks.json').read_text())
    res = []
    for p in tasks['pick']['points']:
        if names and p['name'] not in names:
            continue
        L = copy.deepcopy(lay)
        L['objects'] = [{'name': 'pick_cube', **copy.deepcopy(CUBE), 'pos': [*p['xy_m'], 0.01]}]
        L['_notes'] = [f"GENERATED by sim/site_tasks.py pick: {site} layout + a 2 cm cube at {p['name']}"]
        d = work / p['name'].replace(' ', '_').replace('+', 'p')
        d.mkdir(parents=True, exist_ok=True)
        path = d / f'pick_{site}.json'
        path.write_text(json.dumps(L, indent=1))
        t0 = time.time()
        try:
            r = asyncio.run(checks.check_pick('elrobot', d, str(path)))
            r = {'picked': bool(r['picked']), 'cube_before': r['cube_before'], 'cube_after': r['cube_after'],
                 'lift': r['lift'], 'tcp': r['tcp']}
        except Exception as e:  # a failed tool call is a failed pick, reported
            r = {'picked': False, 'error': str(e)[-400:]}
        r['s'] = round(time.time() - t0, 1)
        print(json.dumps({p['name']: r}), flush=True)
        p['sim'] = {**r, 'date': time.strftime('%Y-%m-%d %H:%M')}
        res.append({'name': p['name'], **r})
    (st.directory / 'tasks.json').write_text(json.dumps(tasks, indent=1) + '\n')
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['place', 'pick'])
    ap.add_argument('--site', default=None)
    ap.add_argument('--only', nargs='*', help='pick: point names')
    ap.add_argument('--work', type=Path, default=Path(tempfile.gettempdir()) / 'site_tasks')
    a = ap.parse_args()
    st = get_site(a.site)
    if a.cmd == 'place':
        out = place(st.name)
        (st.directory / 'tasks.json').write_text(json.dumps(out, indent=1) + '\n')
        print(json.dumps({k: out[k] for k in ('bottle', 'pick')}, indent=1)[:6000])
        print(json.dumps({k: v for k, v in out['kitting'].items() if k != 'checks'}, indent=1)[:3000])
        print('kit checks:', [(c['sheet_pose'], c['on_table'], c['covers_markers'], c['unreachable'], c['max_lean_deg']) for c in out['kitting']['checks']])
    else:
        print(json.dumps(pick(st.name, a.only, a.work), indent=1))


if __name__ == '__main__':
    main()
