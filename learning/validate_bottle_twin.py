"""Headless physical twin acceptance probe; never imports the real-arm server.

Run through uv with the simulation dependencies from prompts/r5-common.md.
Returns a nonzero exit code when opening/extraction acceptance is not met.
Use --small for the historical short-bottle regression fixture; the default
is the real measured geometry. --preposition isolates grasp from park travel.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--small', action='store_true')
    ap.add_argument('--preposition', action='store_true')
    a = ap.parse_args()
    os.environ['STATION_HOST'] = 'station-disabled-in-tests.invalid'
    os.environ['SKILL_PROFILES'] = 'bottle_cap'
    root = Path(tempfile.mkdtemp(prefix='bottle-validation-'))
    os.environ.update(SKILLCFG_DIR=str(root/'cfg'), ROBOT_SIM_RECORD='0', BOTTLE_RECORD='0', BOTTLE_GEMINI_JUDGE='0')
    from . import bottle_twin, episodes as E, skillcfg as SC
    obj = SC.profiles()[0]['object']
    layout = (bottle_twin.make_layout(root/'layout.json', {'cap_h': obj['cap_height_m']},
              {'turns_to_free': obj['thread_free_deg']/360}) if a.small else bottle_twin.learned_layout(root/'layout.json'))
    os.environ['SIM_LAYOUT'] = os.environ['ROBOT_SIM_LAYOUT'] = str(layout)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'sim'))
    import sim_arm_mcp as S
    from .twist import Twister

    async def run():
        import numpy as np
        import mujoco
        p = E.defaults()
        b = S.arm.layout['screw_cap']
        x,y = b['pos'][:2]
        top = b['pos'][2] + 2*b['half_height']
        if a.preposition:
            # Offline fixture placement, not an actuator command or relaxed safety threshold.
            q, _ = S.arm.ik.solve_least_lean(np.array([x,y,top+min(SC.get('twist.hovers_m'))]), S.arm.measured_q(), p['yaw_start_deg'])
            if q is None: raise RuntimeError('fixture hover unreachable')
            with S.arm.lock:
                S.arm.data.qpos[S.arm.qids] = q
                S.arm.data.ctrl[:] = q
                S.arm.data.qvel[:] = 0
                mujoco.mj_forward(S.arm.model,S.arm.data)
        prior, calls = [], []
        # At most one call per stroke, even under a very small configured budget.
        for _ in range(p['strokes']):
            tw = Twister(S.RIG,p,lambda *args: None,prior_strokes=prior)
            result = await tw.twist_cap(x,y,top,2*b['half_height'])
            prior += tw.strokes
            calls.append(result)
            if result.get('stopped_by') != 'time budget' or not tw.strokes: break
        lift = await Twister(S.RIG,p,lambda *args:None).extract_cap()
        truth = S.RIG.truth()
        ok = bool(truth['free'] and lift['cap_free'] and lift['cap_lift_mm'] >= SC.get('extract.lift_m')*1000*SC.get('lift.free_rise_fraction'))
        print(json.dumps({'ok':ok,'geometry':'short fixture' if a.small else 'learned real bottle',
                          'prepositioned':a.preposition,'profile':SC.profile_label(),'calls':calls,'extract':lift,'truth':truth},indent=1))
        return ok
    try:
        return 0 if asyncio.run(run()) else 1
    finally:
        S.arm.running = False


if __name__ == '__main__':
    raise SystemExit(main())
