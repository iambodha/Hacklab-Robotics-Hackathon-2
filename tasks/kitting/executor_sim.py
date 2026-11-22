"""Headless MuJoCo kitting backend for the existing executor.

No Station client and no global real-time twin module. Physics advances during
backend.sleep; actuator calibration and frames come from this task's KitSim.
This is a handoff check of executor primitives, separate from fast training.
"""
import io
import threading
import math
from PIL import Image
import numpy as np
from executor.backends.base import Backend, FrameSample, MotorRead
from .sim_env import KitSim, JAW_TORQUE


class KittingBackend(Backend):
    name='sim'
    def __init__(self,sim):
        self.sim=sim
        self.lock=threading.RLock()
        self.seq=0
    def sleep(self,s):
        with self.lock:
            self.sim.step_s(max(.002,s))
    def read_state(self):
        with self.lock:
            s=self.sim; q=s.q(); out={}
            for i in range(8):
                ticks=lambda v: int(round(2048+(-v if i==7 else v)*4096/(2*math.pi)))
                lo,hi=sorted([ticks(s.lo[i]),ticks(s.hi[i])])
                norm=(q[i]-s.lo[i])/(s.hi[i]-s.lo[i]) if i<7 else 1-q[i]/s.hi[i]
                raw=ticks(q[i])
                out[i+1]=dict(raw=raw,goal=ticks(s.d.ctrl[i]),lo=lo,hi=hi,norm=float(norm),
                              in_range=lo-20<=raw<=hi+20,torque=1,status=0,temp=31,volt=12.4,
                              load=int(min(1000,s.loads()[i])))
            return MotorRead(self.now(),out)
    def write_goals(self,goals):
        with self.lock:
            for mid,n in goals.items():
                i=mid-1; s=self.sim; n=np.clip(n,0,1)
                s.d.ctrl[i]=s.lo[i]+n*(s.hi[i]-s.lo[i]) if i<7 else s.hi[i]*(1-n)
    def set_torque_limit(self,mid,fraction):
        if mid==8:
            with self.lock:
                self.sim.m.actuator_forcerange[7]=[-JAW_TORQUE*fraction,JAW_TORQUE*fraction]
    def frames(self):
        with self.lock:
            out={}; self.seq+=1
            for role,cam in [('overview','overview_cam'),('wrist','wrist_cam')]:
                buf=io.BytesIO()
                Image.fromarray(self.sim.render(cam)).save(buf,format='JPEG',quality=90)
                out[role]=FrameSample(role,f'kitting/{cam}',self.seq,self.now(),buf.getvalue())
            return out
    def truth(self):
        with self.lock:
            return {'objects':{name:{'xyz':self.sim.obj_pose(name)['xyz'].tolist()} for name in self.sim.obj}}


def run(sim,params,only='box_1',mode='full',root=None):
    """Run one shared-plan object through Executor, recording every proposal/result."""
    from pathlib import Path
    from executor.core import Executor
    from executor.kinematics import Kinematics
    from executor.recording import RecorderLink
    from tasks.blue_to_bin.skills import install
    from .run_real import commands,execute
    from .locate import locate_sheet
    from .spec import Pose2D
    root=Path(root or '/tmp/kitting-executor')
    # Camera/table in the twin are already in base frame; kinematics handles world conversion.
    kin=Kinematics()
    backend=KittingBackend(sim)
    ex=Executor(backend,kin,recorder=RecorderLink('local',root=root/'recordings'),
                locator=lambda *a,**k: [],table_memory_path=root/'table.json',twin=False)
    install(ex)
    loc=locate_sheet(sim.render(),sim.spec,sim.cam)
    if not loc['ok']:
        raise RuntimeError(f'sheet not located: {loc}')
    try:
        return execute(ex,commands(sim.spec,loc['pose'],params,kin,mode,only))
    finally:
        ex.cancel()
