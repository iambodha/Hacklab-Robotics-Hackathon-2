"""Stir skill (IK-planned): a spoon stands in a cup; grasp its handle, straighten it, stir circles inside the cup.

Uses only what the real arm has: the arm's kinematic model, the joint readings and the jaw stall feedback. The
caller gives the cup centre (e.g. pixel_to_table on the middle of the opening) and the side the handle leans to.

Geometry: the handle rests on the rim on the side it leans to and the bowl lies on the cup bottom against the
opposite wall, so the handle line follows from the cup centre, the cup size and the spoon length (to ~1 cm).
The arm can only hold a tool tilted with its top towards the arm base around the cup, so that is the default.

  1. grasp      - jaws aligned WITH the handle (approach axis along it, closing across its thin side), come in
                  along the handle from above, close; if the jaws close on nothing, retry with small variations.
  2. straighten - keep the bowl where it is and swing the handle towards vertical as far as the arm allows.
  3. stir       - steer the bowl in horizontal circles around the cup axis (the spoon is rigid in the jaws, so the
                  gripper pose follows from the tool vector), `turns` times; then open and retreat.

The wrist camera was tried for finding the handle and dropped: from every pose the arm can reach above the cup,
the jaws hide the part of the cup the handle sticks out of.
"""
from __future__ import annotations

import asyncio
import time

import numpy as np

UP = np.array([0.0, 0.0, 1.0])
# Code baseline of the stir motion (the r4 values; learning/stir/practice.py learns better ones in the twin and the
# profile learning/profiles/stir_cup.json carries them, read through the skill config keys sim.stir.*)
STIR_HEIGHT = 0.04      # bowl centre height while stirring: 2 cm above the r4 mug's bottom, under its drink
STEP_DT = 0.05          # s per streamed joint target on the circle (the follow() default)
RING_POINTS = 16        # IK points per stirring circle
WALL_MARGIN = 0.006     # the handle stays this far inside the rim; caps the stirring radius
MIN_RADIUS = 0.004      # never stir a smaller circle than this, whatever the cap says
LEAN_STEPS_DEG = (0, 4, 8, 12)   # extra lean tried when the circle does not fit (IK)


def _rot(s, j_hint):
    """Gripper rotation whose approach axis (+y) is s and whose jaw closing axis (+x) is j_hint made perpendicular
    to s (so the jaw axis stays continuous while the tool tilts)."""
    j = np.asarray(j_hint, float) - (np.asarray(j_hint, float) @ s) * s
    j /= np.linalg.norm(j)
    return np.column_stack([j, s, np.cross(j, s)])


def _width_axis(s):
    """Across the handle's WIDTH: perpendicular to the handle, in its (vertical) lean plane. The simulated pads
    cannot close below ~7 mm, so the jaws must close on the 12 mm width, not the 4 mm thickness."""
    side = np.cross(s, UP)
    if np.linalg.norm(side) < 1e-6:
        return np.array([1.0, 0.0, 0.0])
    w = np.cross(side, s)
    return w / np.linalg.norm(w)


def _lean_dir(theta, az):
    """Unit vector pointing DOWN along a spoon leaning theta from vertical, its top towards azimuth az."""
    return np.array([-np.sin(theta) * np.cos(az), -np.sin(theta) * np.sin(az), -np.cos(theta)])


def _first_line(text):
    return text.splitlines()[0] if isinstance(text, str) and text else str(text)


def _components(mask):
    """Connected components (8-neighbour) of a boolean image as arrays of (row, col)."""
    from collections import deque
    seen, out = np.zeros_like(mask), []
    h, w = mask.shape
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        q, pts = deque([(y0, x0)]), []
        seen[y0, x0] = True
        while q:
            y, x = q.popleft()
            pts.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
        out.append(np.array(pts))
    return out


def handle_line(cup, rim, inner_r, spoon_len, az, bowl_shift=0.0, bowl_z=0.02):
    """(handle end, bowl centre) of a spoon leaning on the rim at azimuth az, bowl against the opposite wall."""
    e = np.array([np.cos(az), np.sin(az), 0.0])
    bowl = np.array([cup[0], cup[1], bowl_z]) - (inner_r * 0.4 + bowl_shift) * e
    rim_pt = np.array([cup[0], cup[1], rim]) + inner_r * e
    d = (rim_pt - bowl) / np.linalg.norm(rim_pt - bowl)
    return bowl + spoon_len * d, bowl


class StirSkill:
    def __init__(self, arm, move_tcp, gripper):
        self.arm, self.move_tcp, self.gripper = arm, move_tcp, gripper

    # --- mug centre from the fixed overview camera -------------------------------------------
    def refine_cup(self, guess, rim, outer_r, max_shift=0.10):
        """Refine a rough cup centre with the overview image: the mug is the strongly red blob nearest the guess.
        Its lowest pixel is the front of its base (z = 0), its highest the back of its rim (z = rim); each ray, moved
        one outer radius further from / closer to the camera, gives the centre. Returns (xy, note)."""
        arm = self.arm
        cid = arm.model.camera('overview_cam').id
        with arm.lock:
            pos, mat = arm.data.cam_xpos[cid].copy(), arm.data.cam_xmat[cid].reshape(3, 3).copy()
        img = arm.frame('overview_cam').astype(np.float32)
        h, w = img.shape[:2]
        import build_scene   # the pinhole MuJoCo renders with: a calibrated camera's principal point is off centre
        K = build_scene.camera_K(arm.model, cid, w, h)   # (office overview: 4 px right, 6 px down = ~4 cm on the table)
        r_, g_, b_ = img[..., 0], img[..., 1], img[..., 2]
        red = (r_ > 90) & (r_ > 1.6 * g_) & (r_ > 1.6 * b_)
        comps = _components(red)
        if not comps:
            return guess, 'no red mug in the overview image; using the given centre'

        def ray(u, v):
            # u, v in pixel-edge coordinates (pixel k spans k..k+1); K is in pixel-centre coordinates
            d = mat @ np.array([(u - 0.5 - K[0, 2]) / K[0, 0], -(v - 0.5 - K[1, 2]) / K[1, 1], -1.0])
            return d / np.linalg.norm(d)

        def hit(u, v, z):
            d = ray(u, v)
            return pos + ((z - pos[2]) / d[2]) * d

        best = None
        for c in comps:
            if len(c) < 15:
                continue
            vmax, vmin = c[:, 0].max(), c[:, 0].min()
            front = hit(c[c[:, 0] >= vmax - 1, 1].mean() + 0.5, vmax + 1.0, 0.0)
            back = hit(c[c[:, 0] <= vmin + 1, 1].mean() + 0.5, vmin, rim)
            away = lambda p: (p[:2] - pos[:2]) / np.linalg.norm(p[:2] - pos[:2])
            centre = 0.5 * ((front[:2] + outer_r * away(front)) + (back[:2] - outer_r * away(back)))
            dist = float(np.linalg.norm(centre - guess))
            if best is None or dist < best[0]:
                best = (dist, centre, len(c))
        if best is None or best[0] > max_shift:
            return guess, 'red blob too far from the given centre; using the given centre'
        return best[1], f'mug centre refined from the overview image ({best[2]} red px): moved {best[0] * 100:.1f} cm'

    # --- motion helpers -----------------------------------------------------------------
    async def follow(self, qs, dt=0.05, tol=0.25):
        """Stream arm joint targets (rad, 7 arm joints; the jaws keep their own command). Aborts on tracking error."""
        arm = self.arm
        for k, q in enumerate(qs):
            with arm.lock:
                arm.data.ctrl[:7] = q[:7]
            await asyncio.sleep(dt)
            err = float(np.abs(arm.measured_q()[:7] - q[:7]).max())
            if err > tol or arm.status.any():
                with arm.lock:
                    arm.data.ctrl[:7] = arm.data.qpos[arm.qids][:7]
                return f'aborted at point {k}/{len(qs)}: tracking error {err:.2f} rad'
        await asyncio.sleep(0.3)
        return None

    @staticmethod
    def interp(q0, q1, max_step=0.015):
        n = max(1, int(np.ceil(np.abs(q1[:7] - q0[:7]).max() / max_step)))
        return [q0 + (q1 - q0) * (k / n) for k in range(1, n + 1)]

    def _solve(self, xyz, R, q_seed):
        """Pose IK from good seeds first (current joints, a position-only solution), random restarts last."""
        ik = self.arm.ik
        seeds = [q_seed]
        q_pos = ik.solve(np.asarray(xyz), top_down=False, q0=q_seed, attempts=3)
        if q_pos is not None:
            seeds.append(q_pos)
        for q0 in seeds:
            q = ik.solve_rot(xyz, R, q0, 0)
            if q is not None:
                return q
        return ik.solve_rot(xyz, R, seeds[-1], 3)

    def _plan_ring(self, cx, cy, z, r, L, s, Rk, n=RING_POINTS):
        q_prev, lead, ring = self.arm.measured_q(), [], []
        for k in range(n + 1):
            a = 2 * np.pi * k / n
            tip = np.array([cx + r * np.cos(a), cy + r * np.sin(a), z])
            q = self.arm.ik.solve_rot(tip - L * s, Rk, q_prev, 0 if k else 2)
            if q is None or (k and np.abs(q[:7] - q_prev[:7]).max() > 0.3):
                return None
            (lead if k == 0 else ring).extend(self.interp(q_prev, q, 0.01))
            q_prev = q
        return lead, ring

    async def _open(self):
        while self.arm.norm(self.arm.measured_q())[7] < 0.97:
            await self.gripper(open_fraction=float(min(1.0, self.arm.norm(self.arm.measured_q())[7] + 0.34)))

    def _plan_grasp(self, top, bowl, grasp_from_top, q_now):
        s0 = (bowl - top) / np.linalg.norm(bowl - top)
        for g_off in (grasp_from_top, grasp_from_top - 0.008, grasp_from_top + 0.006):   # lower: jaws hit the cup
            g = top + g_off * s0
            for sign in (1, -1):
                R = _rot(s0, sign * _width_axis(s0))
                q_pre = self._solve(g - 0.05 * s0, R, q_now)
                q_g = self.arm.ik.solve_rot(g, R, q_pre, 2) if q_pre is not None else None
                if q_g is not None:
                    return g, s0, q_pre, q_g, g_off
        return None

    # --- the skill ----------------------------------------------------------------------
    async def run(self, cup_x, cup_y, handle_azimuth_deg=None, rim_height=0.09, inner_radius=0.03,
                  spoon_length=0.145, turns=3, radius=0.009, grasp_from_top=0.035, stir_height=STIR_HEIGHT,
                  step_dt=STEP_DT, ring_points=RING_POINTS, wall_margin=WALL_MARGIN, lean_extra_deg=0.0,
                  on_phase=None):
        """stir_height: bowl centre height above the table while stirring (m); step_dt: seconds per streamed joint
        target on the stirring circle (speed); ring_points: IK points per circle; wall_margin: the handle stays this
        far inside the rim (it caps the radius); lean_extra_deg: lean added to the straightened spoon before stirring.
        on_phase(phase, status, info): called at the end of every phase (locate, grasp, straighten, stir, release)."""
        arm, ik, log, t0 = self.arm, self.arm.ik, [], time.monotonic()

        def phase(name, status='ok', **info):
            if on_phase is not None:
                on_phase(name, status, {'t_s': round(time.monotonic() - t0, 2), **info})
        cup, note = self.refine_cup(np.array([cup_x, cup_y], float), rim_height, inner_radius + 0.004)
        cup_x, cup_y = float(cup[0]), float(cup[1])
        log.append(f'{note} -> cup centre [{cup_x:.3f}, {cup_y:.3f}]')
        phase('locate', cup_xy=[round(cup_x, 4), round(cup_y, 4)], note=note)
        az0 = (np.radians(handle_azimuth_deg) if handle_azimuth_deg is not None
               else float(np.arctan2(-cup[1], -cup[0])))           # default: handle towards the arm base
        await self._open()
        # 1. grasp, retrying with small variations when the jaws close on nothing
        holding, tries = False, [(0.0, 0.0), (0.0, 0.01), (0.0, -0.01), (12.0, 0.0), (-12.0, 0.0)]
        for k, (daz, shift) in enumerate(tries):
            az = az0 + np.radians(daz)
            top, bowl = handle_line(cup, rim_height, inner_radius, spoon_length, az, shift)
            plan = await asyncio.to_thread(self._plan_grasp, top, bowl, grasp_from_top, arm.measured_q())
            if plan is None:
                log.append(f'try {k + 1}: no arm pose reaches along the handle (azimuth {np.degrees(az):.0f} deg)')
                continue
            g, s0, q_pre, q_g, g_off = plan
            res = await self.follow(self.interp(arm.measured_q(), q_pre))
            if not res:
                res = await self.follow(self.interp(q_pre, q_g, 0.006), dt=0.06)
            if res:
                log.append(f'try {k + 1}: approach stopped ({res})')
                await self.follow(self.interp(arm.measured_q(), q_pre))
                continue
            for v in (0.7, 0.4, 0.1, 0.0):
                await self.gripper(open_fraction=v)
            await asyncio.sleep(0.5)
            st = arm.state()['j8']
            holding = st['norm'] > 0.012 or st['load'] > 150   # stalled short of closed, or squeezing hard
            log.append(f'try {k + 1}: handle end estimated at {np.round(top, 3).tolist()}, grasped {g_off * 100:.1f} cm '
                       f'below it: j8 {st["norm"]:.3f}, load {st["load"]} -> {"HOLDING" if holding else "empty"}')
            if holding:
                break
            await self._open()
            await self.follow(self.interp(arm.measured_q(), q_pre))
        if not holding:
            phase('grasp', 'fail', tries=k + 1)
            return 'STIR stopped: could not get hold of the spoon handle.', log
        phase('grasp', tries=k + 1, grasp_below_top_m=round(float(g_off), 4), j8=st['norm'], load=st['load'])
        log.append(f'[t={time.monotonic() - t0:.0f}s]')
        # the spoon is rigid in the jaws from now on: steer its bowl
        # the spoon is held along the approach axis: steer it from where it actually is in the jaws
        tcp, Rg = ik.fk(arm.measured_q())
        L = float(np.linalg.norm(bowl - g))
        j_axis, s_now = Rg[:, 0], Rg[:, 1]
        theta0 = float(np.arccos(np.clip(-s_now[2], -1, 1)))
        az = float(np.arctan2(-s_now[1], -s_now[0])) if theta0 > np.radians(3) else az
        tip_now = tcp + L * s_now
        # stir 2 cm above the bottom: the spoon is held by its edges and hinges about the jaw axis, so any drag on
        # the bottom or the wall pulls the bowl off its circle
        tip0 = np.array([cup_x, cup_y, float(stir_height)])
        # 2. straighten: bowl to the cup centre while the handle swings towards vertical
        path, q_prev, theta_ok = [], arm.measured_q(), None
        lift = await asyncio.to_thread(ik.solve_rot, tcp - 0.015 * s_now, Rg, q_prev, 0)   # bowl off the bottom first
        if lift is not None:
            path.extend(self.interp(q_prev, lift, 0.008))
            q_prev, tip_now = lift, tip_now - 0.015 * s_now
        steps = np.radians(np.arange(np.degrees(theta0), -0.1, -3.0))
        for i, th in enumerate(steps):
            f = (i + 1) / len(steps)
            tip = tip_now + min(1.0, 2 * f) * (tip0 - tip_now)
            s = _lean_dir(th, az)
            q = await asyncio.to_thread(ik.solve_rot, tip - L * s, _rot(s, j_axis), q_prev, 0)
            if q is None or np.abs(q[:7] - q_prev[:7]).max() > 0.25:
                if theta_ok is None:
                    continue
                break
            path.extend(self.interp(q_prev, q, 0.01))
            q_prev, theta_ok = q, th
        if theta_ok is None:
            phase('straighten', 'fail', reason='IK')
            return 'STIR stopped: cannot move the held spoon at all (IK).', log
        res = await self.follow(path)
        if res:
            phase('straighten', 'abort', reason=res)
            return f'STIR stopped while straightening the spoon: {res}', log
        phase('straighten', lean_deg=round(float(np.degrees(theta_ok)), 1), from_deg=round(float(np.degrees(theta0)), 1))
        # 3. stir circles with the bowl, the spoon translating rigidly; lean more if the circle does not fit
        ring_plan = None
        for extra in LEAN_STEPS_DEG:
            th = theta_ok + np.radians(lean_extra_deg + extra)
            offset_at_rim = (rim_height - tip0[2]) * np.tan(th)
            r = float(min(radius, max(MIN_RADIUS, inner_radius - wall_margin - offset_at_rim)))
            s = _lean_dir(th, az)
            Rk = _rot(s, j_axis)
            ring_plan = await asyncio.to_thread(self._plan_ring, cup_x, cup_y, tip0[2], r, L, s, Rk, int(ring_points))
            if ring_plan is not None:
                break
        if ring_plan is None:
            phase('stir', 'fail', reason='no circle fits (IK)')
            return 'STIR stopped: no stirring circle fits inside the cup at this position (IK).', log
        lead, ring = ring_plan
        log.append(f'straightened from {np.degrees(theta0):.0f} to {np.degrees(th):.0f} deg from vertical; '
                   f'stir radius {r * 100:.1f} cm')
        res = await self.follow(lead)
        if not res:
            phase('lead_in', lean_deg=round(float(np.degrees(th)), 1))
            res = await self.follow(ring * int(turns), dt=float(step_dt))
        log.append(f'stirred {int(turns)} turns' if not res else f'stir interrupted: {res}')
        phase('stir', 'abort' if res else 'ok', radius_m=round(r, 4), radius_asked_m=round(float(radius), 4),
              lean_deg=round(float(np.degrees(th)), 1), turns=int(turns), points_per_turn=len(ring),
              reason=res or '')
        # release and retreat along the handle, then up
        await self._open()
        q_back = await asyncio.to_thread(ik.solve_rot, tip0 - (L + 0.06) * s, Rk, arm.measured_q(), 2)
        if q_back is not None:
            await self.follow(self.interp(arm.measured_q(), q_back))
        out = await self.move_tcp(x=float(cup_x), y=float(cup_y) - 0.06, z=0.22, top_down=True)
        log.append(f'released and retreated: {_first_line(out)} [t={time.monotonic() - t0:.0f}s]')
        phase('release')
        return ('DONE stirring' if not res else 'STIR partly done'), log
