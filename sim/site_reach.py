"""Where the arm can grasp on a site's table, and how far the gripper leans there: the least-lean jaws-down IK the
skills use (build_scene.ArmIK.solve_least_lean = sim_arm_mcp.move_tcp / twist_cap, learning/skillcfg.lean_search with
the active ik.leans / ik.restarts / ik.search) evaluated over the table of the site layout.

    python sim/site_reach.py --site office                 # -> sites/office/reach.json + sites/office/reach_map.png
    python sim/site_reach.py --site home --out docs/sites/home_reach
    python sim/site_reach.py --site office --plot-only     # redraw from reach.json (+ the site's tasks.json overlays)

Geometry (table rectangle, markers, base frame) is read from the site layout at run time. Points are sampled on rays
from the arm base (polar: RADII x azimuth step), marching outwards with the previous solution as the IK seed
(continuation_lean: the skill's lean order and solver without its random restarts, ~50x cheaper; it can only
over-estimate the lean). The forward ray is also run with the full skill search (forward_ray_check in the JSON), and
task points are checked with the full search by sim/site_tasks.py. A ray stops at the table edge or after its first
unreachable point beyond a reachable one (the outer reach boundary). The IK checks joint limits only, not collisions: a
grasp at the table with a steep lean
would put the gripper body into the table, so the map marks "steep" separately and the physics trials
(sim/site_pick.py, tasks/kitting/sim_env.py, learning/validate_bottle_twin.py) are the evidence that a point works.

Heights (TCP z, metres above the table top; base frame):
  table_grasp   0.014   a small object on the table: generaloid.checks GRASP_Z (2 cm cube; fingertips 10 mm below TCP)
  bottle_grasp  0.1885  the twist_cap grasp on the demo cap: cap top 0.205 (learning/profiles/bottle_cap.json)
                        - twist.grasp_below_top_m 0.010 + approach_z_offset_mm -6.5. The guarded descent from the hover
                        re-solves the least-lean IK at every step (sim_arm_mcp.descend), so the cap is grasped at this
                        height's lean (the hover, 5 cm higher, leans more: sim/site_tasks.py reports both)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'cgl' if sys.platform == 'darwin' else 'egl')
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(HERE), str(REPO)):
    if p not in sys.path:
        sys.path.append(p)

from sites import get_site  # noqa: E402

CAP_TOP_Z = 0.205            # learning/profiles/bottle_cap.json object.cap_top_z_m (the demo bottle)
HOVER_ABOVE_CAP = 0.05       # learning/skillcfg twist.hovers_m[0]: the first (highest) hover twist_cap tries
GRASP_BELOW_TOP = 0.010      # learning/skillcfg twist.grasp_below_top_m
APPROACH_Z_OFFSET = -0.0065  # learning/profiles/bottle_cap.json params.default.approach_z_offset_mm (learned)
HEIGHTS = {'table_grasp': 0.014, 'bottle_grasp': round(CAP_TOP_Z - GRASP_BELOW_TOP + APPROACH_Z_OFFSET, 4)}
RADII = np.round(np.arange(0.06, 0.601, 0.02), 3)   # m from the base axis; 0.6 m is beyond the arm's ~0.45 m reach
AZ_STEP_DEG = 10.0           # ray spacing; at 0.4 m a 10 deg step is 7 cm apart (interpolated in the map)
MAX_MISSES = 3               # consecutive unreachable points before a ray that never reached anything stops
YAW_DEG = 0.0                # jaw closing axis yaw of the samples (j7 rolls the jaws: the lean barely depends on it)
# Lean classes (deg from vertical). 'steep' > 45: the gripper body comes close to the table at grasp height.
CLASSES = [('vertical', 10), ('lean_30', 30), ('lean_45', 45), ('steep', 90)]
CLASS_RGBA = {'vertical': '#1a9850', 'lean_30': '#91cf60', 'lean_45': '#fee08b', 'steep': '#fc8d59',
              'unreachable': '#d9d9d9'}
MAP_STEP = 0.01              # m pixel of the rasterised map


# --- geometry from the layout ------------------------------------------------------------------------------------
def table_corners(lay: dict) -> np.ndarray:
    """(4, 2) table-top corners in the base frame (layout table: centre, full size, yaw deg)."""
    t = lay['table']
    w, h = t['size'][0] / 2, t['size'][1] / 2
    a = math.radians(t.get('yaw', 0.0))
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) @ R.T + t['center']


def on_table(lay: dict, xy, margin: float = 0.0) -> bool:
    t = lay['table']
    a = math.radians(t.get('yaw', 0.0))
    d = np.asarray(xy, float) - t['center']
    u = d @ [math.cos(a), math.sin(a)], d @ [-math.sin(a), math.cos(a)]
    return abs(u[0]) <= t['size'][0] / 2 - margin and abs(u[1]) <= t['size'][1] / 2 - margin


def markers(lay: dict) -> dict[int, dict]:
    """id -> {'xy', 'corners' (4, 2), 'R' (2x2 marker x/y axes)} of the table markers (layout markers.items T)."""
    h = lay['markers']['size_m'] / 2
    out = {}
    for m in lay['markers']['items']:
        T = np.asarray(m['T'], float)
        R, c = T[:2, :2], T[:2, 3]
        out[int(m['id'])] = {'xy': c, 'R': R, 'corners': np.array([[-h, h], [h, h], [h, -h], [-h, -h]]) @ R.T + c}
    return out


def marker_frame(lay: dict, origin: int, toward: int, away_from: tuple[int, ...] = ()) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(origin xy, unit 'along' origin->toward, unit 'across' perpendicular to it, pointing to the side of the
    away_from markers' centroid, else to the arm-forward side). How a person measures on the table with a tape."""
    mk = markers(lay)
    o = mk[origin]['xy']
    u = mk[toward]['xy'] - o
    u = u / np.linalg.norm(u)
    v = np.array([-u[1], u[0]])
    ref = np.mean([mk[i]['xy'] for i in away_from], axis=0) - o if away_from else np.array([0.0, 1.0])
    return o, u, (v if v @ ref >= 0 else -v)


def to_marker(lay: dict, xy, origin: int, toward: int, away_from: tuple[int, ...] = ()) -> dict:
    """A base-frame point as tape measurements from marker centres: 'along' origin->toward and 'across' it."""
    o, u, v = marker_frame(lay, origin, toward, away_from)
    d = np.asarray(xy, float) - o
    mk = markers(lay)
    return {'origin_marker': origin, 'toward_marker': toward, 'along_cm': round(float(d @ u) * 100, 1),
            'across_cm': round(float(d @ v) * 100, 1), 'across_toward': list(away_from) or 'arm forward',
            'dist_cm': {str(i): round(float(np.linalg.norm(np.asarray(xy) - m['xy'])) * 100, 1) for i, m in mk.items()}}


def from_marker(lay: dict, origin: int, toward: int, along_m: float, across_m: float,
                away_from: tuple[int, ...] = ()) -> np.ndarray:
    o, u, v = marker_frame(lay, origin, toward, away_from)
    return o + along_m * u + across_m * v


def lean_class(lean) -> str:
    if lean is None:
        return 'unreachable'
    return next(name for name, hi in CLASSES if abs(lean) <= hi)


# --- the IK sweep ------------------------------------------------------------------------------------------------
_IK = None


def _ik(scene: str):
    global _IK
    if _IK is None:
        import mujoco
        import build_scene
        m = mujoco.MjModel.from_xml_path(scene)
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key('rest').id)
        ik = build_scene.ArmIK(m)
        _IK = (ik, d.qpos[ik.qids].copy())
    return _IK


def continuation_lean(ik, xyz, seeds) -> tuple:
    """(q, lean): the skill's lean search (same solver and ik.leans, least lean first) without random restarts, from
    each seed in turn. ~50x cheaper than solve_least_lean's 12 restarts per smaller lean; it can only report a lean
    larger than the skill would find (or miss a pose it would find), never a smaller one."""
    from learning import skillcfg as SC
    xyz = np.asarray(xyz, float)
    for q0 in seeds:
        sol, lean = SC.lean_search(lambda ln, rs: ik.solve_rot(xyz, ik.target_rot(xyz, ln, YAW_DEG), q0, rs), restarts=0)
        if sol is not None:
            return sol, lean
    return None, None


def _ray(args) -> list[dict]:
    scene, lay, z, az, exact = args
    ik, rest = _ik(scene)
    a = math.radians(az)
    q, out, misses, reached = rest.copy(), [], 0, False
    for r in RADII:
        xy = [float(r * math.sin(a)), float(r * math.cos(a))]
        if not on_table(lay, xy):
            break
        if exact:
            sol, lean = ik.solve_least_lean(np.array([*xy, z]), q, YAW_DEG)
        else:
            sol, lean = continuation_lean(ik, [*xy, z], [q, rest] if reached else [rest])
        out.append({'r': float(r), 'az': az, 'xy': [round(v, 4) for v in xy], 'lean': lean})
        if sol is None:
            misses += 1
            if reached or misses >= MAX_MISSES:
                break
            continue
        q, misses, reached = sol, 0, True
    return out


def sweep(site: str, workers: int, az_step: float = AZ_STEP_DEG, heights: dict = HEIGHTS, log=print) -> dict:
    import build_scene
    from learning import skillcfg as SC
    st = get_site(site)
    lay = json.loads(st.layout.read_text())
    scene, _ = build_scene.build(st.layout, Path('/tmp') / f'site_reach_{site}', report=False)
    jobs = [(str(scene), lay, z, float(az), False) for z in heights.values()
            for az in np.arange(-180.0, 180.0, az_step)]
    # the forward ray once more with the skill's full search: how far the continuation map is from it
    checks = [(str(scene), lay, z, 0.0, True) for z in heights.values()]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        rays = list(ex.map(_ray, checks + jobs))
    exact, rays = rays[:len(checks)], rays[len(checks):]
    samples = {name: [p for (s, _, z, _, _), ray in zip(jobs, rays) if z == hz for p in ray] for name, hz in heights.items()}
    log(f'{sum(len(v) for v in samples.values())} map points in {time.time() - t0:.0f} s')
    compare = {}
    for name, ray in zip(heights, exact):
        cont = {p['r']: p['lean'] for p in samples[name] if abs(p['az']) < 1e-6}
        compare[name] = [{'r': p['r'], 'skill_search': p['lean'], 'map': cont.get(p['r'])} for p in ray]
    vals = SC.active()['values']
    return {'site': site, 'layout': str(st.layout.relative_to(REPO)), 'frame': 'arm base (sim frame): +y forward, +x right',
            'ik': {'solver': 'sim/build_scene.ArmIK (solve_rot + skillcfg.lean_search)', 'skillcfg_target': SC.target(),
                   'leans': vals['ik.leans'], 'restarts_skill': vals['ik.restarts'], 'search': vals['ik.search'],
                   'map_seeding': 'continuation along each ray, then the rest pose; no random restarts '
                                  '(sim/site_reach.continuation_lean); forward_ray_check = the full skill search',
                   'yaw_deg': YAW_DEG},
            'heights_m': dict(heights), 'az_step_deg': az_step, 'radii_m': RADII.tolist(), 'classes_deg': dict(CLASSES),
            'forward_ray_check': compare, 'samples': samples, 'minutes': round((time.time() - t0) / 60, 1)}


# --- summary and map ---------------------------------------------------------------------------------------------
def raster(lay: dict, pts: list[dict], az_step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lean on a MAP_STEP grid over the table (NaN = unreachable, inf = off the table): the polar sample with the
    nearest azimuth and radius (within half a step), else unreachable."""
    c = table_corners(lay)
    xs = np.arange(c[:, 0].min(), c[:, 0].max() + MAP_STEP, MAP_STEP)
    ys = np.arange(c[:, 1].min(), c[:, 1].max() + MAP_STEP, MAP_STEP)
    by = {(round(p['az'], 3), round(p['r'], 3)): p['lean'] for p in pts}
    dr = float(RADII[1] - RADII[0])
    grid = np.full((len(ys), len(xs)), np.inf)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            if not on_table(lay, [x, y]):
                continue
            r, az = math.hypot(x, y), math.degrees(math.atan2(x, y))
            azn = round(((round((az + 180) / az_step) * az_step) % 360) - 180, 3)
            rn = round(float(RADII[np.argmin(np.abs(RADII - r))]), 3)
            lean = by.get((azn, rn)) if abs(rn - r) <= dr / 2 + 1e-9 or r < RADII[0] else None
            grid[i, j] = np.nan if lean is None else abs(lean)
    return xs, ys, grid


def summarize(res: dict, lay: dict) -> dict:
    out = {}
    for name, pts in res['samples'].items():
        _, _, g = raster(lay, pts, res['az_step_deg'])
        on = np.isfinite(g) | np.isnan(g)
        reach = np.isfinite(g)
        frac = {cls: round(float(((g <= hi) & (g > lo) if lo >= 0 else (g <= hi)).sum() / on.sum()), 3)
                for (cls, hi), lo in zip(CLASSES, [-1] + [h for _, h in CLASSES[:-1]])}
        frac['unreachable'] = round(float((on & ~reach).sum() / on.sum()), 3)
        ok = [p for p in pts if p['lean'] is not None]
        maxr = {cls: max([p['r'] for p in ok if abs(p['lean']) <= hi] or [0.0]) for cls, hi in CLASSES}
        fwd = [p for p in ok if abs(p['az']) < 1e-6]
        out[name] = {'z_tcp_m': res['heights_m'][name], 'table_fraction': frac,
                     'max_radius_m_by_lean': maxr,
                     'forward_ray_lean_deg': {f"{p['r']:.2f}": p['lean'] for p in fwd},
                     'azimuth_range_deg': [min((p['az'] for p in ok), default=None), max((p['az'] for p in ok), default=None)]}
    return out


def plot(res: dict, lay: dict, out_png: Path, tasks: dict | None = None) -> Path:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch, Polygon
    names = list(res['samples'])
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 5.4), layout='constrained')
    cmap = ListedColormap([CLASS_RGBA[c] for c, _ in CLASSES])
    norm = BoundaryNorm([0] + [h for _, h in CLASSES], cmap.N)
    corners = table_corners(lay)
    mk = markers(lay)
    for ax, name in zip(np.atleast_1d(axes), names):
        xs, ys, g = raster(lay, res['samples'][name], res['az_step_deg'])
        ext = [xs[0] - MAP_STEP / 2, xs[-1] + MAP_STEP / 2, ys[0] - MAP_STEP / 2, ys[-1] + MAP_STEP / 2]
        ax.imshow(np.where(np.isnan(g), 1, np.nan), extent=ext, origin='lower', cmap=ListedColormap([CLASS_RGBA['unreachable']]),
                  vmin=0, vmax=2, interpolation='nearest')
        ax.imshow(np.where(np.isfinite(g), g, np.nan), extent=ext, origin='lower', cmap=cmap, norm=norm, interpolation='nearest')
        ax.add_patch(Polygon(corners, closed=True, fill=False, ec='k', lw=1.5))
        for i, m in mk.items():
            ax.add_patch(Polygon(m['corners'], closed=True, fc='k', ec='k', alpha=0.85))
            ax.text(*m['xy'], str(i), color='w', ha='center', va='center', fontsize=9, weight='bold')
        ax.plot(0, 0, 'k^', ms=9)
        ax.text(0.012, -0.03, 'arm base', fontsize=8)
        ax.annotate('', xy=(0, 0.09), xytext=(0, 0.0), arrowprops=dict(arrowstyle='->', lw=1))
        kit = (tasks or {}).get('kitting')
        if kit and name == 'table_grasp':
            ax.add_patch(Polygon(np.asarray(kit['sheet_corners_m']), closed=True, fill=False, ec='#2166ac', lw=1.6, ls='--'))
            for cell, xy in kit.get('cells_m', {}).items():
                ax.plot(*xy, 's', mfc='none', mec='#2166ac', ms=5)
            for xy in kit.get('spots_m', {}).values():
                ax.plot(*xy, 'o', mfc='none', mec='#2166ac', ms=4)
            ax.text(kit['sheet_xy_m'][0], kit['sheet_xy_m'][1] + 0.115, 'A4 kit sheet', color='#2166ac', ha='center', fontsize=8)
        for p in (tasks or {}).get('pick', {}).get('points', []) if name == 'table_grasp' else []:
            ax.plot(*p['xy_m'], 'P', color='#762a83', ms=8)
            ax.text(p['xy_m'][0] + 0.015, p['xy_m'][1] - 0.01, p['name'], color='#762a83', fontsize=7)
        bot = (tasks or {}).get('bottle')
        if bot and name == 'bottle_grasp':
            ax.add_patch(plt.Circle(bot['cap_xy_m'], 0.036, fill=False, ec='#2166ac', lw=1.6))
            ax.text(bot['cap_xy_m'][0] + 0.04, bot['cap_xy_m'][1], 'bottle', color='#2166ac', fontsize=8, va='center')
        z = res['heights_m'][name]
        ax.set_title(f"{res['site']}: {name.replace('_', ' ')} (TCP z {z:.3f} m)\nleast-lean jaws-down IK, lean from vertical",
                     fontsize=9)
        ax.set_xlabel('x (m, arm right)')
        ax.set_ylabel('y (m, arm forward)')
        ax.set_aspect('equal')
        ax.set_xlim(corners[:, 0].min() - 0.03, corners[:, 0].max() + 0.03)
        ax.set_ylim(corners[:, 1].min() - 0.03, corners[:, 1].max() + 0.03)
        ax.grid(alpha=0.25, lw=0.5)
    labels = {'vertical': '<= 10 deg', 'lean_30': '10-30 deg', 'lean_45': '30-45 deg', 'steep': '> 45 deg (body near table)',
              'unreachable': 'unreachable'}
    fig.legend(handles=[Patch(color=CLASS_RGBA[k], label=v) for k, v in labels.items()], loc='lower center', ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, -0.06))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=80, bbox_inches='tight')
    plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--site', default=None, help='default: SITE, home if unset')
    ap.add_argument('--out', type=Path, default=None, help='folder for reach.json + reach_map.png (default sites/<site>)')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--az-step', type=float, default=AZ_STEP_DEG)
    ap.add_argument('--plot-only', action='store_true')
    ap.add_argument('--heights', nargs='*', choices=list(HEIGHTS), help='recompute only these maps, merged into reach.json')
    a = ap.parse_args()
    st = get_site(a.site)
    out = a.out or st.directory
    lay = json.loads(st.layout.read_text())
    path = out / 'reach.json'
    if a.plot_only:
        res = json.loads(path.read_text())
    else:
        res = sweep(st.name, a.workers, a.az_step, {k: HEIGHTS[k] for k in (a.heights or HEIGHTS)})
        if a.heights and path.is_file():
            old = json.loads(path.read_text())
            for key in ('samples', 'heights_m', 'forward_ray_check'):
                res[key] = {**{k: v for k, v in old[key].items() if k in HEIGHTS}, **res[key]}
            res['samples'] = {k: res['samples'][k] for k in HEIGHTS if k in res['samples']}
        res['summary'] = summarize(res, lay)
        out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(res, indent=1) + '\n')
    tasks_path = st.directory / 'tasks.json'
    tasks = json.loads(tasks_path.read_text()) if tasks_path.is_file() and out == st.directory else None
    png = plot(res, lay, out / 'reach_map.png', tasks)
    print(json.dumps({'json': str(path), 'png': str(png), 'png_kb': round(png.stat().st_size / 1024), **res['summary']}, indent=1))


if __name__ == '__main__':
    main()
