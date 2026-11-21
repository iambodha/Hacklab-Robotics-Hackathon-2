import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s2.arm import Frame, Joint  # noqa: E402
from s2.geometry import Camera  # noqa: E402
from s2.world import Entity, World  # noqa: E402

# The twin's overview camera: at (0, 0.6, 0.4) looking at (0, 0.1, 0), 299x224, 62 deg vertical FOV.
OVERVIEW = {'width': 299, 'height': 224, 'fx': 186.3993, 'fy': 186.3993, 'cx': 149.5, 'cy': 112.0,
            'R_base_cam': [[-1, 0, 0], [0, 0.624695, -0.780869], [0, -0.780869, -0.624695]], 't_base_cam': [0, 0.6, 0.4]}


@pytest.fixture
def overview():
    return Camera.from_info(OVERVIEW)


def make_world(tcp=(0.0, 0.15, 0.10), jaw=1.0, jaw_cmd=None, objects=None, load=0):
    w = World()
    joints = {f'j{i}': Joint(0.5, 2048, 2048, True, 0, 0) for i in range(1, 9)}
    joints['j8'] = Joint(jaw, 2048, 2048, True, 0, load)
    cam = Camera.from_info(OVERVIEW)
    frame = Frame(0.0, {}, {}, joints, np.array(tcp, float), 0.0, 0.0, {'overview': cam, 'wrist': cam})
    w.gripper.jaw_cmd = jaw_cmd
    w.update(frame)
    for name, (xyz, kind, h, size) in (objects or {}).items():
        w.entities[name] = Entity(name, kind, h, np.array(xyz, float), size=np.array(size, float),
                                  last_seen=0.0, visible=True)
    return w


@pytest.fixture
def cube_world():
    return make_world(objects={'cube': ((0.02, 0.16, 0.01), 'object', 0.02, (0.02, 0.02)),
                               'box': ((0.24, 0.06, 0.045), 'container', 0.09, (0.18, 0.13))})
