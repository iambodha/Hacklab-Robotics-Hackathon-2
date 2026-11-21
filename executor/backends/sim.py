"""The MuJoCo twin (sim/sim_arm_mcp.SimArm, real-time physics on its own thread) behind the backend interface.

Joint conventions are the sim's own, which match the real arm: norm in the calibrated range, j8 1 = open.
Camera frames are rendered on demand; seq counts renders, t_capture is the render time. No camera stamps exist
in the twin, so a frozen sim camera cannot happen (the fake backend covers that case).
"""
from __future__ import annotations

import io
import os
import sys
import threading
from pathlib import Path

from .base import Backend, FrameSample, MotorRead

SIM = Path(__file__).resolve().parents[2] / "sim"
CAMS = {"wrist": "wrist_cam", "overview": "overview_cam"}


class SimBackend(Backend):
    name = "sim"

    def __init__(self, layout: str | None = None):
        os.environ.setdefault("ROBOT_SIM_RECORD", "0")
        os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
        if layout:
            os.environ["ROBOT_SIM_LAYOUT"] = str(layout)
        import recorder  # noqa: F401  (pin the recorder package before sim/ (with its own recorder.py) joins sys.path)
        if str(SIM) not in sys.path:
            sys.path.append(str(SIM))
        import sim_arm_mcp   # builds the scene if needed and starts the physics thread (module-level SimArm)
        self.mod, self.arm = sim_arm_mcp, sim_arm_mcp.arm
        self.render_lock = threading.Lock()
        self.seq = {r: 0 for r in CAMS}
        self.layout_name = Path(os.environ.get("ROBOT_SIM_LAYOUT", "layout.json")).stem

    def read_state(self) -> MotorRead:
        rows = self.arm.state()
        t = self.now()
        out = {}
        for i, name in enumerate(self.mod.NAMES):
            r = rows[name]
            out[i + 1] = {k: r[k] for k in ("raw", "goal", "lo", "hi", "norm", "in_range", "torque", "status", "temp", "load")}
            out[i + 1]["volt"] = 12.4
        return MotorRead(t, out)

    def frames(self) -> dict[str, FrameSample]:
        from PIL import Image
        out = {}
        for role, cam in CAMS.items():
            with self.render_lock:
                t = self.now()
                joints = [float(v) for v in self.arm.norm(self.arm.measured_q())]
                img = self.arm.frame(cam)
                self.seq[role] += 1
                seq = self.seq[role]
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=80)
            out[role] = FrameSample(role, f"sim/{cam}", seq, t, buf.getvalue(), [round(j, 4) for j in joints])
        return out

    def write_goals(self, goals: dict[int, float]) -> None:
        with self.arm.lock:
            for mid, n in goals.items():
                i = mid - 1
                self.arm.data.ctrl[i] = self.arm.rad(i, float(min(1.0, max(0.0, n))))

    def camera_info(self, kin) -> dict:
        """The twin's overview camera as s2.geometry.Camera info (OpenCV axes) in the WORLD frame."""
        import numpy as np
        m, d = self.arm.model, self.arm.data
        cid = m.camera(CAMS["overview"]).id
        with self.arm.lock:
            pos, mat = d.cam_xpos[cid].copy(), d.cam_xmat[cid].reshape(3, 3).copy()
        w, h = self.arm.layout["image_size"]
        f = (h / 2) / np.tan(np.radians(m.cam_fovy[cid]) / 2)
        R = kin.R @ mat @ np.diag([1.0, -1.0, -1.0])
        return {"overview": {"width": w, "height": h, "fx": f, "fy": f, "cx": w / 2, "cy": h / 2, "R_base_cam": R.tolist(),
                             "t_base_cam": kin.to_world(pos).tolist(), "source": "twin camera"}}

    def truth(self) -> dict:
        """Simulator ground truth: for the independent evaluator only, never shown to an advisor."""
        return self.arm.truth()

    def describe(self) -> dict:
        return {**super().describe(), "layout": self.layout_name}
