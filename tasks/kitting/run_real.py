"""Offline review or operator-enabled executor run of the exported kit profile.

No Station imports or connections in --image mode. --serve creates the existing
executor console; each motion is a validated proposal through NormaCore Station.
"""
from __future__ import annotations
import argparse
import io
import json
import threading
import time
from pathlib import Path
import numpy as np
from PIL import Image
from . import profile
from .locate import camera_model, locate_sheet
from .plan import order, plan_pick_place
from .spec import Spec


def commands(spec, pose, params, kin, mode='dry', only=None):
    """One shared plan, converted from table/base to executor world coordinates."""
    out = []
    def move(xy, z, yaw):
        return {'xyz_world': kin.to_world([*xy,z]).tolist(), 'yaw_deg': yaw + kin.yaw_offset}
    for t in order(spec.targets(pose)):
        if only and t['name'] != only:
            continue
        p = plan_pick_place(t, params[t['class']], spec.class_dims(t['class'])['round'])
        def add(phase, skill, args):
            out.append(dict(object=p.name, phase=phase, skill=skill, params=args))
        def descend(phase, xy, z, yaw):
            w = move(xy,z,yaw)
            add(phase,'descend_guarded',dict(xy_world=w['xyz_world'][:2],z_world=w['xyz_world'][2],yaw_deg=w['yaw_deg']))
        add('open','gripper',{'open_fraction':1.0})
        add('approach','move_tcp',move(p.pick_xy,p.hover_z,p.yaw_pick))
        # Dry mode remains 20 mm above the learned grasp and never closes.
        descend('grasp_height',p.pick_xy,p.grasp_z + (.020 if mode=='dry' else 0),p.yaw_pick)
        if mode!='dry':
            add('close','gripper',{'open_fraction':0.,'force':p.grip_force})
        add('lift','move_tcp',move(p.pick_xy,max(p.travel_z,p.hover_z),p.yaw_pick))
        xy = p.pick_xy if mode=='pick' else p.place_xy
        z = p.grasp_z + .001 if mode=='pick' else p.place_z
        add('carry','move_tcp',move(xy,max(p.travel_z,z+.03),p.yaw_place))
        descend('place_height',xy,z + (.020 if mode=='dry' else 0),p.yaw_place)
        add('release','gripper',{'open_fraction':1.0 if mode=='dry' else p.release_open})
        add('retreat','move_tcp',move(xy,max(p.travel_z,z+.03),p.yaw_place))
    return out


def execute(ex, cmds):
    """Stop on every rejected/partial/fault result; do not auto-retry a physical grasp."""
    from executor.contracts import ActionProposal, Status, to_dict
    records=[]
    for c in cmds:
        obs=ex.observe()
        p=ActionProposal(ActionProposal.new_id(),'script:kitting',obs.obs_id,obs.setup_id,c['skill'],'',c['params'],
                         ['sheet fixed since location','operator verified printed start spots'],ex.now()+30,
                         f"kitting {c['object']} {c['phase']}")
        eid=ex.submit(p)
        r=ex.wait(eid,180)
        if r is None or not r.status.final:
            ex.cancel(eid)
            raise RuntimeError(f"{c['phase']}: timed out; cancelled")
        records.append({'command':c,'result':to_dict(r)})
        if r.status != Status.SUCCEEDED:
            raise RuntimeError(f"{c['object']} {c['phase']}: {r.status.value}: {r.termination}; holding")
        if c['phase']=='close' and not r.achieved.get('holding'):
            raise RuntimeError(f"{c['object']}: empty grasp; holding")
    return records


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile',type=Path,default=profile.PROFILE)
    ap.add_argument('--image',type=Path,help='offline overview image: write commands, never move')
    ap.add_argument('--twin',action='store_true',help='offline image has twin pinhole calibration')
    ap.add_argument('--serve',action='store_true',help='real executor console; operator must enable motion')
    ap.add_argument('--mode',choices=['dry','pick','full'],default='dry')
    ap.add_argument('--only',help='one object name, e.g. cap_1')
    ap.add_argument('--port',type=int,default=8930)
    ap.add_argument('--out',type=Path,default=Path('/tmp/kitting_commands.json'))
    a=ap.parse_args()
    if bool(a.image)==bool(a.serve):
        ap.error('choose exactly one of --image or --serve')
    S=Spec.load(); params=profile.load(a.profile)
    if a.only and a.only not in [o['name'] for o in S.objects()]:
        ap.error('unknown --only object')
    from sites import get_site
    layout=json.loads(get_site().layout.read_text())
    from executor.kinematics import Kinematics
    kin=Kinematics()
    cam=camera_model(layout,twin=a.twin)
    def locate(img):
        if img.shape[:2] != (cam.H,cam.W):
            raise RuntimeError(f'image dimensions must be {cam.W}x{cam.H}; use matching calibration')
        r=locate_sheet(img,S,cam)
        if not r['ok'] or len(r.get('markers',[]))<2:
            raise RuntimeError(f'two sheet markers required for handoff: {r}')
        return r
    if a.image:
        r=locate(np.asarray(Image.open(a.image).convert('RGB')))
        result={'sheet_pose':r['pose'].as_list(),'profile':str(a.profile),'mode':a.mode,
                'commands':commands(S,r['pose'],params,kin,a.mode,a.only)}
        a.out.write_text(json.dumps(result,indent=2)+'\n')
        print(a.out)
        return
    # Deliberately only imported after explicit --serve. Never invoked during training/tests.
    from http.server import ThreadingHTTPServer
    from executor.console.server import Console, make_handler
    from tasks.blue_to_bin.skills import install
    console=Console('real',a.port,a.port+1)
    install(console.ex)
    def worker():
        try:
            deadline=time.monotonic()+600
            while not console.ex.backend.can_move:
                if time.monotonic()>deadline:
                    raise RuntimeError('operator never enabled arm')
                time.sleep(.5)
            frame=console.ex.backend.frames()['overview']
            r=locate(np.asarray(Image.open(io.BytesIO(frame.jpeg)).convert('RGB')))
            cmds=commands(S,r['pose'],params,console.ex.kin,a.mode,a.only)
            a.out.write_text(json.dumps({'sheet_pose':r['pose'].as_list(),'commands':cmds},indent=2)+'\n')
            records=execute(console.ex,cmds)
            a.out.with_suffix('.results.json').write_text(json.dumps(records,indent=2,default=str))
            print('Motion sequence finished; human placement inspection still required.',flush=True)
        except Exception as e:
            console.ex.cancel()
            print(f'Kitting stopped: {e}',flush=True)
    threading.Thread(target=worker,daemon=True).start()
    print(f'Operator console: http://127.0.0.1:{a.port}; enable only after reviewing REAL_RUN.md',flush=True)
    srv=ThreadingHTTPServer(('127.0.0.1',a.port),make_handler(console))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        console.ex.cancel()
    finally:
        srv.server_close()

if __name__=='__main__':
    main()
