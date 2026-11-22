"""SO-101 twin scene: norma-core's own SO-101 follower URDF (the model the Station viewer renders) as a MuJoCo arm with
six position servos, on the same table/bins/objects/cameras code as the ElRobot twin (sim/build_scene.py).

Ported from origin/simulation_s101 (sim/build_scene.py, `robot_model == 'so101'` branches, e8789ec), without editing
sim/build_scene.py: the arm and gripper builders are passed into build_scene.build_xml for one call.
Values below come from that branch, where they were fitted against the live SO-101 table:
  base pose      joint-1 axis on the table origin, +y = straight ahead (URDF base turned 90 deg about z)
  grasp_tcp      between the fixed jaw (-0.008 x) and the open moving jaw, 9.3 cm down the gripper link; the site's
                 +y axis (approach) = gripper -z, +x = jaw closing axis (the TCP convention of every Generaloid arm)
  jaw pads       boxes at the fingertips do the gripping (the jaw meshes stay visible but do not collide)
  wrist camera   wrist_camera.json (SO-101 top wrist mount + Innomaker U20CAM, 398x224, fovy 52)
"""
from __future__ import annotations

import contextlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from generaloid.core.paths import ensure_sim_on_path
from generaloid.core.station import norma_core

HERE = Path(__file__).resolve().parent
URDF_REL = "software/station/clients/station-viewer/public/devices/so101/so101_robot_follower.urdf"
BASE_POS = (-0.168068, 0.124202, -0.0300817)   # puts the joint-1 axis at the table origin (origin/simulation_s101)
BASE_YAW_DEG = 90.0                            # URDF base frame -> twin frame (+y ahead)
SERVO_TORQUE_NM = 2.94                         # STS3215 stall torque at 12 V, as for the ElRobot servos
GRIPPER_TORQUE_NM = 0.5                        # jaw servo limit in the twin: stalls on an object (origin/simulation_s101)
SERVO_KP, SERVO_KV = "30", "2"                 # position servo gains, as the ElRobot twin
URDF_RED = ("0.7 0.7 0.7 1", "1 0 0 1")        # URDF placeholder colours, replaced by the layout's arm_rgba


def find_urdf() -> Path:
    nc = norma_core()
    if nc is None or not (nc / URDF_REL).exists():
        raise FileNotFoundError(f"{Path(URDF_REL).name} not found: set NORMA_CORE_REPO to a full norma-core clone")
    return nc / URDF_REL


def base_arm(out_dir: Path):
    """URDF -> MJCF with six position servos; robot self-collision off, contact with the scene on."""
    import build_scene as B
    urdf_path = find_urdf()
    urdf = ET.parse(urdf_path)
    for mesh in urdf.findall('.//mesh'):
        mesh.set('filename', str((urdf_path.parent / mesh.get('filename')).resolve()))
    ext = ET.SubElement(urdf.getroot(), 'mujoco')
    ET.SubElement(ext, 'compiler', discardvisual='false', fusestatic='false', balanceinertia='true',
                  boundmass='0.001', boundinertia='0.0000001')
    converted = out_dir / 'so101_import.urdf'
    urdf.write(converted, encoding='unicode')
    base_xml = out_dir / 'so101_base.xml'
    mujoco.mj_saveLastXML(str(base_xml), mujoco.MjModel.from_xml_path(str(converted)))
    root = ET.parse(base_xml).getroot()
    option = root.find('option')
    if option is None:
        option = ET.SubElement(root, 'option')
    option.attrib.update(timestep='0.002', integrator='implicitfast', cone='elliptic', impratio='10',
                         iterations='80', noslip_iterations='5')
    ET.SubElement(root.find('asset'), 'texture', type='skybox', builtin='gradient', rgb1='0.86 0.86 0.84',
                  rgb2='0.55 0.55 0.55', width='256', height='256')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='960', offheight='720')
    ET.SubElement(visual, 'headlight', ambient='0.3 0.3 0.3', diffuse='0.3 0.3 0.3')
    for geom in root.find('worldbody').iter('geom'):
        geom.attrib.update(contype='1', conaffinity='2', friction='1 0.01 0.001')
        if geom.get('rgba') in URDF_RED:
            geom.set('rgba', '0.7 0.7 0.7 1')                  # build_xml recolours this one to arm_rgba
    base = root.find('.//body[@name="base"]')
    base.set('pos', B.fmt(BASE_POS))
    base.set('quat', B.fmt(B.yaw_quat(BASE_YAW_DEG)))
    actuators = ET.SubElement(root, 'actuator')
    for i in range(1, 7):
        joint = root.find(f'.//joint[@name="{i}"]')
        joint.attrib.update(damping='0.3', armature='0.01')
        limit = GRIPPER_TORQUE_NM if i == 6 else SERVO_TORQUE_NM
        ET.SubElement(actuators, 'position', name=f'motor_{i}', joint=joint.get('name'), kp=SERVO_KP, kv=SERVO_KV,
                      ctrllimited='true', ctrlrange=joint.get('range'), forcelimited='true', forcerange=f'-{limit} {limit}')
    return root


def add_gripper_parts(root, layout):
    import build_scene as B
    gripper = root.find('.//body[@name="gripper"]')
    ET.SubElement(gripper, 'site', name='grasp_tcp', pos='0.014 0 -0.093', xyaxes='1 0 0 0 0 -1',
                  size='.004', rgba='0 1 0 1', group='3')
    for geom in gripper.findall('geom'):
        if geom.get('mesh') == 'wrist_roll_follower_so101_v1':
            geom.attrib.update(B.NO_CONTACT)
    pad = dict(rgba='.15 .15 .16 1', friction='1.4 .02 .002', condim='4', solref='.004 1', solimp='.95 .99 .001',
               mass='.005', **B.PAD_CONTACT)
    ET.SubElement(gripper, 'geom', name='left_pad', type='box', pos='-0.012 0 -0.090', size='.004 .008 .014', **pad)
    jaw = root.find('.//body[@name="moving_jaw_so101_v1"]')
    for geom in jaw.findall('geom'):
        geom.attrib.update(B.NO_CONTACT)
    ET.SubElement(jaw, 'geom', name='right_pad', type='box', pos='-0.0083 -0.0666 0.0188', size='.004 .014 .008', **pad)
    cam = layout['wrist_camera']
    T = np.array(json.loads((HERE / cam['pose_file']).read_text())['T_link_camera'])
    # OpenCV (x right, y down, z forward) -> MuJoCo camera (x right, y up, looks along -z).
    ET.SubElement(gripper, 'camera', name='wrist_cam', pos=B.fmt(T[:3, 3]), xyaxes=B.fmt(np.r_[T[:3, 0], -T[:3, 1]]),
                  fovy=str(cam['fovy']))


@contextlib.contextmanager
def _arm_builders(B):
    saved = B.base_arm, B.add_gripper_parts
    B.base_arm, B.add_gripper_parts = base_arm, add_gripper_parts
    try:
        yield
    finally:
        B.base_arm, B.add_gripper_parts = saved


def build(out_dir: Path, layout_path: Path) -> Path:
    """Write <out_dir>/workspace.xml, reusing it only for the same unchanged sources."""
    ensure_sim_on_path()
    import build_scene as B
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = out_dir / 'workspace.xml'
    sources = [Path(layout_path), Path(__file__), HERE / 'wrist_camera.json', Path(B.__file__), find_urdf()]
    signature = {str(p.resolve()): [p.stat().st_mtime_ns, p.stat().st_size] for p in sources}
    manifest = out_dir / 'sources.json'
    if scene.exists() and manifest.exists() and json.loads(manifest.read_text()) == signature:
        return scene
    layout = json.loads(Path(layout_path).read_text())
    with _arm_builders(B):
        root = B.build_xml(layout, out_dir)
    B.add_rest_keyframe(root, layout)
    ET.indent(root)
    scene.write_text(ET.tostring(root, encoding='unicode'))
    manifest.write_text(json.dumps(signature, sort_keys=True))
    return scene
