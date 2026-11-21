import asyncio
import io
import json

import numpy as np
from PIL import Image

from conftest import OVERVIEW
from s2.arm import ArmClient
from test_perception_world import SIM_TABLE

TCP = ('tcp xyz = [0.000, 0.190, 0.240] m; jaws point 3 deg from straight down; '
       'jaw closing axis yaw = 0 deg (2 deg from level).')


def _jpeg(shade):
    buf = io.BytesIO()
    Image.fromarray(np.full((224, 299, 3), shade, np.uint8)).save(buf, format='JPEG')
    return buf.getvalue()


def _client(look_items):
    c = ArmClient('true', [], None, {'df52ffda': 'wrist', 'c98bf1cf': 'overview'})
    answers = {'look': look_items, 'tcp_pose': [TCP], 'camera_info': [json.dumps({'wrist': OVERVIEW})]}

    async def raw(tool, **args):
        return answers[tool]
    c.raw = raw
    return c


def test_real_arm_look_with_a_camera_not_live():
    """elrobot-real labels: 'wrist camera <hash8> fresh, age=..s'; a dead camera has a label and no image."""
    items = ['wrist camera df52ffda fresh, age=0.12s, captured 12:00:00', _jpeg(200),
             'overview camera c98bf1cf: not live on the relay, not shown', SIM_TABLE]
    f = asyncio.run(_client(items).frame())
    assert set(f.images) == {'wrist'} and abs(f.image_age['wrist'] - 0.12) < 1e-9
    assert f.jaw_level_deg == 2 and 'overview' in f.notes[0] and len(f.joints) == 8


def test_sim_look_labels_still_parse():
    c = _client([])
    c.camera_roles = {'322ab551': 'wrist', '8c697f6e': 'overview'}
    items = ['camera 322ab5514b8e14014c3743c1abfd4a07 (sim-wrist-uvc) fresh, age=0.03s', _jpeg(10),
             'camera 8c697f6e0c79a3d2fab2a4ea977ee738 (sim-overview) fresh, age=0.03s', _jpeg(240), SIM_TABLE]
    async def raw(tool, **args):
        return {'look': items, 'tcp_pose': [TCP], 'camera_info': [json.dumps({'wrist': OVERVIEW})]}[tool]
    c.raw = raw
    f = asyncio.run(c.frame())
    assert set(f.images) == {'wrist', 'overview'} and f.images['overview'].mean() > 200
