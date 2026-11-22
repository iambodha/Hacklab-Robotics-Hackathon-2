import json
import os
import xml.etree.ElementTree as ET

from generaloid.connectors.so101 import LAYOUT
from generaloid.connectors.so101.scene import build


def test_switching_to_an_older_layout_rebuilds_scene(tmp_path):
    data = json.loads(LAYOUT.read_text())
    output = tmp_path / "scene"
    build(output, LAYOUT)
    cube = next(o for o in data["objects"] if o["name"] == "black_cube")
    cube["pos"][0] = 0.08
    alternate = tmp_path / "alternate.json"
    alternate.write_text(json.dumps(data))
    os.utime(alternate, (1, 1))
    path = build(output, alternate)
    body = ET.parse(path).find('.//body[@name="black_cube"]')
    assert float(body.get("pos").split()[0]) == 0.08
