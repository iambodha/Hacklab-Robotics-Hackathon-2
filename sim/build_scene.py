"""Build the MuJoCo twin of the hackathon table: ElRobot follower, table, bins, objects, cameras.

  python3 sim/build_scene.py [--layout sim/layout.json | home] [--out sim/generated] [--photo overview.jpg]
  SIM_LAYOUT=home python3 sim/build_scene.py        # the home table (sim/layouts/home.json)

1. Converts the official URDF (NORMA_CORE_REPO/hardware/elrobot/simulation) to MJCF with
   eight position servos (URDF limits, ST3215 torque limit 2.94 N m, no gravity compensation).
2. Adds the table, bins and loose objects from layout.json, box contact pads on both jaws
   (CAD jaw meshes are visual only), the wrist camera at the CAD pose from
   camera_pose/wrist_camera.json, and the overview/top cameras.
3. Checks that objects settle, checks reach (top-down IK for objects, any-orientation IK
   for bin drop points) and renders previews. Everything is written to --out.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault('MUJOCO_GL', 'cgl' if sys.platform == 'darwin' else 'osmesa')
import mujoco
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from sites import get_site
OBJECT_CONTACT = {'contype': '2', 'conaffinity': '3'}
STATIC_CONTACT = {'contype': '2', 'conaffinity': '1'}
PAD_CONTACT = {'contype': '4', 'conaffinity': '2'}
NO_CONTACT = {'contype': '0', 'conaffinity': '0'}
GRIPPER_TORQUE = 0.4  # N m on rev_motor_08 (0 rad = jaws open)
CV_TO_MJ = np.diag([1.0, -1.0, -1.0])  # OpenCV camera axes (x right, y down, z forward) <-> MuJoCo (y up, looks along -z)


def layout_path(choice=None):
    """Explicit path/name, then SIM_LAYOUT / ROBOT_SIM_LAYOUT, then the selected site's layout."""
    choice = choice or os.environ.get('SIM_LAYOUT') or os.environ.get('ROBOT_SIM_LAYOUT')
    if choice is None:
        return get_site().layout
    choice = str(choice)
    if choice == Path(choice).name and (REPO / 'sites' / choice / 'site.json').is_file():
        return get_site(choice).layout
    if choice in ('venue', 'layout'):
        return HERE / 'layout.json'
    p = Path(choice)
    if p.exists():
        return p.resolve()
    named = HERE / 'layouts' / (choice if choice.endswith('.json') else choice + '.json')
    if named.exists():
        return named
    raise FileNotFoundError(f'layout {choice!r} not found (tried {p} and {named})')


def generated_dir(layout_file):
    """sim/generated for the venue layout, sim/generated/<stem> for any other."""
    layout_file = Path(layout_file)
    if layout_file.parent.parent == REPO / 'sites':
        return HERE / 'generated' / ('site_' + layout_file.parent.name)
    return HERE / 'generated' if layout_file.name == 'layout.json' else HERE / 'generated' / layout_file.stem


def find_urdf():
    candidates = [os.environ.get('NORMA_CORE_REPO', ''), REPO / 'norma-core', REPO.parent / 'norma-core',
                  REPO.parent.parent / 'repo/norma-core', '/mnt/d/Robotics-Hackathon/repos/norma-core']
    for base in candidates:
        path = Path(base) / 'hardware/elrobot/simulation/elrobot_follower.urdf'
        if base and path.exists():
            return path
    raise FileNotFoundError('elrobot_follower.urdf not found; set NORMA_CORE_REPO to a full norma-core clone')


def fmt(values):
    return ' '.join(f'{float(v):.6g}' for v in values)


def yaw_quat(deg):
    a = math.radians(deg) / 2
    return [math.cos(a), 0, 0, math.sin(a)]


def camera_axes(pos, lookat):
    z = np.subtract(pos, lookat); z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z); x /= np.linalg.norm(x)
    return np.r_[x, np.cross(z, x)]


# MuJoCo only uses focal/principal when a sensorsize is given; only their ratios matter, so 1 mm "pixels".
SENSOR_PITCH_M = 1e-3


def camera_lens(cam, image_size):
    """MJCF lens attributes of a layout camera: its calibrated pinhole when it has one (K_normalized: OpenCV,
    pixel-centre coordinates, normalised by the image width/height), else fovy. fovy alone puts the principal point
    at the image centre (the overview's is 4 px right of and 6 px below it) with fx = fy. MuJoCo scales the
    intrinsics to any render size and derives model.cam_fovy from fy; camera_K() returns the matrix it renders with.
    Lens distortion is not modelled (compare against undistorted real frames)."""
    k = cam.get('K_normalized')
    if not k or not image_size:
        return {'fovy': str(cam['fovy'])}
    W, H = image_size
    fx, fy, cx, cy = k['fx'] * W, k['fy'] * H, k['cx'] * W, k['cy'] * H
    # MuJoCo's principal offset is measured from the image centre ((W-1)/2 in pixel-centre coordinates) and is
    # positive towards image-left/up (checked by rendering a point with MuJoCo 4.0.1: within 0.1 px at 1x and 2x).
    return {'resolution': fmt([W, H]), 'sensorsize': fmt([W * SENSOR_PITCH_M, H * SENSOR_PITCH_M]),
            'focalpixel': fmt([fx, fy]), 'principalpixel': fmt([(W - 1) / 2 - cx, (H - 1) / 2 - cy])}


def camera_K(model, cam_id, W, H):
    """3x3 OpenCV pinhole (pixel-centre coordinates) that MuJoCo renders camera cam_id with at W x H."""
    sensor = model.cam_sensorsize[cam_id]
    if sensor[0] > 0 and sensor[1] > 0:
        fx, fy, px, py = model.cam_intrinsic[cam_id]
        sx, sy = W / sensor[0], H / sensor[1]
        return np.array([[fx * sx, 0, (W - 1) / 2 - px * sx], [0, fy * sy, (H - 1) / 2 - py * sy], [0, 0, 1.0]])
    f = H / 2 / math.tan(math.radians(model.cam_fovy[cam_id]) / 2)
    return np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1.0]])


def base_arm(out_dir):
    """URDF -> MJCF with position servos; robot self-collision off, contact with the scene on."""
    urdf_path = find_urdf()
    urdf = ET.parse(urdf_path)
    for mesh in urdf.findall('.//mesh'):
        mesh.set('filename', str((urdf_path.parent / mesh.get('filename')).resolve()))
    ext = ET.SubElement(urdf.getroot(), 'mujoco')
    ET.SubElement(ext, 'compiler', discardvisual='false', fusestatic='false', balanceinertia='true',
                  boundmass='0.001', boundinertia='0.0000001')
    converted = out_dir / 'elrobot_import.urdf'
    urdf.write(converted, encoding='unicode')
    mujoco.mj_saveLastXML(str(out_dir / 'elrobot_base.xml'), mujoco.MjModel.from_xml_path(str(converted)))
    root = ET.parse(out_dir / 'elrobot_base.xml').getroot()
    option = root.find('option')
    if option is None: option = ET.SubElement(root, 'option')
    option.attrib.update(timestep='0.002', integrator='implicitfast', cone='elliptic', impratio='10',
                         iterations='80', noslip_iterations='5')
    # Light grey "room" instead of a black void, so up-facing camera views look like the lab.
    ET.SubElement(root.find('asset'), 'texture', type='skybox', builtin='gradient', rgb1='0.86 0.86 0.84',
                  rgb2='0.55 0.55 0.55', width='256', height='256')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='960', offheight='720')
    ET.SubElement(visual, 'headlight', ambient='0.3 0.3 0.3', diffuse='0.3 0.3 0.3')
    for geom in root.find('worldbody').iter('geom'):
        geom.attrib.update(contype='1', conaffinity='2', friction='1 0.01 0.001')
    equality = root.find('equality')
    if equality is None: equality = ET.SubElement(root, 'equality')
    for name, multiplier in [('rev_motor_08_1', -0.0115), ('rev_motor_08_2', 0.0115)]:
        for old in list(equality):
            if old.get('joint1') == name: equality.remove(old)
        ET.SubElement(equality, 'joint', joint1=name, joint2='rev_motor_08', polycoef=f'0 {multiplier} 0 0 0', solref='0.004 1')
    actuators = ET.SubElement(root, 'actuator')
    for i in range(1, 9):
        joint = root.find(f'.//joint[@name="rev_motor_{i:02d}"]')
        joint.attrib.update(damping='0.3', armature='0.01')
        # Jaw servo: 0.4 N m through the 11.5 mm/rad linkage = ~35 N squeeze, so it stalls on an object.
        limit = GRIPPER_TORQUE if i == 8 else 2.94
        ET.SubElement(actuators, 'position', name=f'motor_{i}', joint=joint.get('name'), kp='30', kv='2',
                      ctrllimited='true', ctrlrange=joint.get('range'), forcelimited='true', forcerange=f'-{limit} {limit}')
    return root


def add_gripper_parts(root, layout):
    gripper = root.find('.//body[@name="Gripper_Base_v1_1"]')
    ET.SubElement(gripper, 'site', name='grasp_tcp', pos='0 .09 0', size='.004', rgba='0 1 0 1', group='3')
    # Jaw meshes stay visible but do not collide; box pads at the fingertips do the gripping.
    for name, pos, desired, label in [('Gripper_Jaw_02_v1_1', [.311734, -.199648, .032849], [-.035, .09, 0], 'left_pad'),
                                      ('Gripper_Jaw_01_v1_1', [.308573, -.243367, -.049391], [.035, .09, 0], 'right_pad')]:
        body = root.find(f'.//body[@name="{name}"]')
        for geom in body.findall('geom'): geom.attrib.update(NO_CONTACT)
        ET.SubElement(body, 'geom', name=label, type='box', pos=fmt(np.subtract(desired, pos)), size='.006 .016 .018',
                      rgba='.15 .15 .16 1', friction='1.2 .02 .002', condim='4', solref='.004 1', solimp='.95 .99 .001',
                      mass='.005', **PAD_CONTACT)
    cam = layout['wrist_camera']
    # Inline T_link_cam (e.g. synced from scene_calib.json) or the CAD pose file; link = Gripper_Base_v1_1.
    pose_file = cam.get('pose_file')
    # Frozen legacy layouts use this token for the active site's mount; custom poses stay explicit.
    pose = get_site().wrist_mount if pose_file == 'camera_pose/wrist_camera.json' else REPO / (pose_file or get_site().wrist_mount)
    T = np.array(cam['T_link_cam'] if 'T_link_cam' in cam else json.loads(pose.read_text())['T_link_camera'])
    if cam.get('pitch_offset_deg'):  # mount tilted vs the CAD: rotate about the camera's own x axis
        a = math.radians(cam['pitch_offset_deg'])
        T = T.copy(); T[:3, :3] = T[:3, :3] @ np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    # OpenCV (x right, y down, z forward) -> MuJoCo camera (x right, y up, looks along -z).
    ET.SubElement(gripper, 'camera', name='wrist_cam', pos=fmt(T[:3, 3]), xyaxes=fmt(np.r_[T[:3, 0], -T[:3, 1]]),
                  **camera_lens(cam, layout.get('image_size')))


def object_quat(spec):
    """Start orientation: 'quat' [w, x, y, z], or 'euler' [roll, pitch, yaw] in degrees (xyz), else 'yaw'."""
    if 'quat' in spec:
        return spec['quat']
    if 'euler' not in spec:
        return yaw_quat(spec.get('yaw', 0))
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.radians(spec['euler']), 'xyz')
    return q


def shape_parts(spec):
    """(parts, sites) of an object: explicit 'parts', or generated from 'shape' ('cup' or 'spoon').
    cup:   open cylinder standing on its bottom; the body origin is its centre (inner radius, height, wall).
    spoon: flat handle along the body z axis with an ellipsoid bowl at its lower end; site <name>_tip = bowl centre."""
    shape = spec.get('shape')
    if shape is None:
        return spec['parts'], {}
    rgba = spec.get('rgba', [0.8, 0.8, 0.8, 1])
    if shape == 'cup':
        r, h, t, n = spec['radius'], spec['height'], spec.get('wall', 0.004), spec.get('segments', 20)
        parts = [{'type': 'cylinder', 'size': [r + t, t / 2], 'pos': [0, 0, -h / 2 + t / 2], 'rgba': rgba}]
        half_w = math.pi * (r + t / 2) / n * 1.08
        for k in range(n):
            a = 2 * math.pi * k / n
            parts.append({'type': 'box', 'size': [t / 2, half_w, h / 2], 'rgba': rgba, 'euler': [0, 0, math.degrees(a)],
                          'pos': [(r + t / 2) * math.cos(a), (r + t / 2) * math.sin(a), 0]})
        if spec.get('liquid_rgba'):  # visual-only liquid surface
            parts.append({'type': 'cylinder', 'size': [r - 0.001, 0.0005], 'rgba': spec['liquid_rgba'], 'visual_only': True,
                          'pos': [0, 0, -h / 2 + spec.get('liquid_level', 0.6) * h]})
        return parts, {f"{spec['name']}_rim": [0, 0, h / 2]}
    if shape == 'spoon':
        L, w, th = spec['handle_length'], spec.get('handle_width', 0.01), spec.get('handle_thickness', 0.004)
        bw, bd, bl = spec.get('bowl', [0.022, 0.008, 0.034])           # full width, depth, length
        parts = [{'type': 'box', 'size': [w / 2, th / 2, L / 2], 'pos': [0, 0, 0], 'rgba': rgba},
                 {'type': 'ellipsoid', 'size': [bw / 2, bd / 2, bl / 2], 'pos': [0, 0, -L / 2 - bl / 2 + 0.004], 'rgba': rgba}]
        return parts, {f"{spec['name']}_tip": [0, 0, -L / 2 - bl / 2 + 0.004], f"{spec['name']}_top": [0, 0, L / 2]}
    raise ValueError(f"unknown shape {shape!r}")


def add_part(body, part, name, mass=None):
    attrs = {'name': name, 'type': part['type'], 'size': fmt(part['size']), 'rgba': fmt(part['rgba']),
             'pos': fmt(part.get('pos', [0, 0, 0])), 'euler': fmt(np.radians(part.get('euler', [0, 0, 0])))}
    if part.get('visual_only'):
        attrs.update(NO_CONTACT, group='2', mass='0')
    else:
        attrs.update(OBJECT_CONTACT, friction='1.2 0.02 0.002', condim='4')
        if mass is not None: attrs['mass'] = f'{mass:.6g}'
    ET.SubElement(body, 'geom', attrs)


def add_bin(world, spec):
    w, d, h = spec['size']; t = spec['wall']
    body = ET.SubElement(world, 'body', name=spec['name'], pos=fmt([*spec['pos'], 0]), quat=fmt(yaw_quat(spec['yaw'])))
    walls = {'bottom': ([0, 0, t/2], [w/2, d/2, t/2]),
             'wall_back': ([0, -(d-t)/2, h/2], [w/2, t/2, h/2]),
             'wall_front': ([0, (d-t)/2, h/2], [w/2, t/2, h/2]),
             'wall_left': ([-(w-t)/2, 0, h/2], [t/2, d/2 - t, h/2]),
             'wall_right': ([(w-t)/2, 0, h/2], [t/2, d/2 - t, h/2])}
    for label, (pos, size) in walls.items():
        ET.SubElement(body, 'geom', name=f"{spec['name']}_{label}", type='box', pos=fmt(pos), size=fmt(size),
                      rgba=fmt(spec['rgba']), friction='0.8 0.01 0.001', **STATIC_CONTACT)
    ET.SubElement(body, 'geom', name=f"{spec['name']}_inside", type='box', pos=fmt([0, 0, t + 0.0005]),
                  size=fmt([w/2 - t, d/2 - t, 0.0005]), rgba=fmt(spec['inner_rgba']), group='2', **NO_CONTACT)
    # Flap hinged on a wall's top edge: 0 deg = closed over the opening, 90 = upright, 180 = folded outward.
    sides = {'back': (0, d/2, w), 'front': (180, d/2, w), 'left': (-90, w/2, d), 'right': (90, w/2, d)}
    for i, flap in enumerate(spec.get('flaps', [])):
        yaw, offset, width = sides[flap['side']]
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        hinge = ET.SubElement(body, 'body', name=f"{spec['name']}_flap{i}_{flap['side']}",
                              pos=fmt([s*offset, -c*offset, h]), quat=fmt(yaw_quat(yaw)))
        a = math.radians(flap['angle']); half = flap['length'] / 2
        ET.SubElement(hinge, 'geom', name=f"{spec['name']}_flap{i}", type='box',
                      pos=fmt([0, half*math.cos(a), half*math.sin(a)]), euler=fmt([a, 0, 0]),
                      size=fmt([width/2, half, t/2]), rgba=fmt(spec['rgba']), friction='0.8 0.01 0.001', **STATIC_CONTACT)


def build_xml(layout, out_dir):
    root = base_arm(out_dir)
    world = root.find('worldbody')
    for geom in world.iter('geom'):
        if geom.get('rgba') == '0.7 0.7 0.7 1': geom.set('rgba', fmt(layout['arm_rgba']))
    add_gripper_parts(root, layout)

    table = layout['table']; tw, td = table['size']; th = table['thickness']
    # 'center' [x, y] + 'yaw' (deg, table axes vs the arm frame), or the venue form center_x + back_edge_y.
    cx, cy = table['center'] if 'center' in table else (table.get('center_x', 0.0), table['back_edge_y'] + td/2)
    tbody = ET.SubElement(world, 'body', name='table', pos=fmt([cx, cy, 0]), quat=fmt(yaw_quat(table.get('yaw', 0))))
    ET.SubElement(tbody, 'geom', name='table_top', type='box', pos=fmt([0, 0, -th/2]), size=fmt([tw/2, td/2, th/2]),
                  rgba=fmt(table['rgba']), friction='0.8 0.01 0.001', **STATIC_CONTACT)
    if table.get('texture'):
        # Real table top (orthomosaic from the phone video, scripts/realsim/table_texture.py), stretched once over
        # the top face: image right = +x, image up = +y.
        asset = root.find('asset')
        ET.SubElement(asset, 'texture', name='table_tex', type='2d', file=str((HERE / table['texture']).resolve()))
        ET.SubElement(asset, 'material', name='table_mat', texture='table_tex', texuniform='false', texrepeat='1 1',
                      specular=str(table.get('specular', 0.15)), shininess='0.3', reflectance='0')
        ET.SubElement(tbody, 'geom', name='table_skin', type='plane', size=fmt([tw/2, td/2, 0.05]), pos='0 0 0.0002',
                      material='table_mat', group='0', **NO_CONTACT)
    for i, (sx, sy) in enumerate([(-1, -1), (1, -1), (-1, 1), (1, 1)]):
        ET.SubElement(tbody, 'geom', name=f'table_leg{i}', type='box', size=fmt([0.02, 0.02, (table['height']-th)/2]),
                      pos=fmt([sx*(tw/2-0.05), sy*(td/2-0.05), -th - (table['height']-th)/2]),
                      rgba='0.75 0.75 0.75 1', **STATIC_CONTACT)
    ET.SubElement(world, 'geom', name='floor', type='plane', size='3 3 0.1', pos=fmt([cx, cy, -table['height']]),
                  rgba=fmt(table.get('floor_rgba', [0.45, 0.45, 0.47, 1])), **STATIC_CONTACT)
    lighting = layout.get('lighting', {})
    if 'headlight' in lighting:
        root.find('visual/headlight').attrib.update({k: fmt(v) for k, v in lighting['headlight'].items()})
    if 'skybox' in lighting:
        sky = root.find('asset/texture[@type="skybox"]')
        sky.attrib.update(rgb1=fmt(lighting['skybox'][0]), rgb2=fmt(lighting['skybox'][1]))
    lights = lighting.get('lights', [
        {'name': 'ceiling_a', 'directional': True, 'pos': [0.3, cy, 2], 'dir': [-0.15, -0.1, -1],
         'diffuse': [0.4, 0.4, 0.4], 'specular': [0.1, 0.1, 0.1]},
        {'name': 'ceiling_b', 'directional': True, 'pos': [-0.3, cy, 2], 'dir': [0.2, 0.3, -1],
         'diffuse': [0.2, 0.2, 0.2], 'specular': [0, 0, 0], 'castshadow': False}])
    for spec in lights:
        attrs = {k: (str(v).lower() if isinstance(v, bool) else fmt(v)) for k, v in spec.items() if k != 'name'}
        ET.SubElement(world, 'light', name=spec['name'], **attrs)

    for spec in layout['bins']: add_bin(world, spec)
    for spec in layout['fixed']:
        ET.SubElement(world, 'geom', name=spec['name'], type=spec['type'], pos=fmt(spec['pos']), size=fmt(spec['size']),
                      quat=fmt(yaw_quat(spec['yaw'])), rgba=fmt(spec['rgba']), **STATIC_CONTACT)
    for spec in layout['objects']:
        body = ET.SubElement(world, 'body', name=spec['name'], pos=fmt(spec['pos']), quat=fmt(object_quat(spec)))
        ET.SubElement(body, 'freejoint', name=f"{spec['name']}_free")
        parts, sites = shape_parts(spec)
        physical = [p for p in parts if not p.get('visual_only')]
        for i, part in enumerate(parts):
            add_part(body, part, f"{spec['name']}_g{i}", None if part.get('visual_only') else spec['mass']/len(physical))
        for site_name, pos in sites.items():
            ET.SubElement(body, 'site', name=site_name, pos=fmt(pos), size='.003', rgba='1 0 1 1', group='3')
    if layout.get('markers'):
        add_markers(root, layout['markers'], out_dir)
    for cam in layout['cameras']:
        if 'T_cv' in cam:  # full pose, OpenCV axes, in the sim frame (from a marker calibration)
            T = np.asarray(cam['T_cv'], float)
            pos, axes = T[:3, 3], np.r_[T[:3, 0], -T[:3, 1]]
        else:
            pos, axes = cam['pos'], camera_axes(cam['pos'], cam['lookat'])
        ET.SubElement(world, 'camera', name=cam['name'], pos=fmt(pos), xyaxes=fmt(axes),
                      **camera_lens(cam, layout.get('image_size')))
    return root


def marker_pose(item):
    """4x4 T_sim_marker of a marker item: 'T' (4x4), or 'pos' [x, y(, z)] + 'yaw' (deg). Marker axes as in OpenCV
    ArUco: x = image right, y = image up, z out of the paper."""
    if 'T' in item:
        return np.asarray(item['T'], float)
    a = math.radians(item.get('yaw', 0))
    T = np.eye(4)
    T[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    T[:3, 3] = [*item['pos'][:2], item['pos'][2] if len(item['pos']) > 2 else 0.0]
    return T


def marker_texture(dictionary, marker_id, out_dir, px=240, levels=(0.0, 1.0)):
    """PNG of an ArUco marker (black square incl. its 1-cell border = the nominal size), generated with OpenCV;
    levels = (ink, paper) grey levels in 0..1 (printed markers are not pure black and white under room light)."""
    import cv2
    ink, paper = (int(round(255 * v)) for v in levels)
    path = Path(out_dir) / 'markers' / f'{dictionary}_{marker_id}_{ink}_{paper}.png'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
        img = cv2.aruco.generateImageMarker(d, int(marker_id), px).astype(np.float32) / 255
        cv2.imwrite(str(path), cv2.cvtColor((ink + img * (paper - ink)).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    return path


def add_markers(root, spec, out_dir):
    """ArUco markers as visual-only textured planes lying on the table, with the white paper margin around them."""
    asset, world = root.find('asset'), root.find('worldbody')
    s, margin = spec['size_m'], spec.get('paper_margin_m', 0.012)
    levels = spec.get('levels', [0.0, 1.0])
    for item in spec['items']:
        T, mid = marker_pose(item), item['id']
        tex = marker_texture(spec['dictionary'], mid, out_dir, levels=levels)
        ET.SubElement(asset, 'texture', name=f'aruco_tex_{mid}', type='2d', file=str(tex.resolve()))
        ET.SubElement(asset, 'material', name=f'aruco_mat_{mid}', texture=f'aruco_tex_{mid}', texuniform='false',
                      texrepeat='1 1', specular='0.1', shininess='0.2', reflectance='0')
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, T[:3, :3].flatten())
        body = ET.SubElement(world, 'body', name=f'aruco_{mid}', pos=fmt(T[:3, 3]), quat=fmt(quat))
        ET.SubElement(body, 'geom', name=f'aruco_{mid}_paper', type='box', size=fmt([s/2 + margin, s/2 + margin, 0.0002]),
                      pos='0 0 0.0003', rgba=fmt(spec.get('paper_rgba', [levels[1]] * 3 + [1])), group='0', **NO_CONTACT)
        ET.SubElement(body, 'geom', name=f'aruco_{mid}', type='plane', size=fmt([s/2, s/2, 0.01]), pos='0 0 0.0006',
                      material=f'aruco_mat_{mid}', group='0', **NO_CONTACT)
        for k, (cx_, cy_) in enumerate([(-1, 1), (1, 1), (1, -1), (-1, -1)]):  # OpenCV corner order: TL, TR, BR, BL
            ET.SubElement(body, 'site', name=f'aruco_{mid}_c{k}', pos=fmt([cx_ * s/2, cy_ * s/2, 0.0006]),
                          size='.002', rgba='1 0 0 1', group='4')


def add_reach_rings(root, radii):
    world = root.find('worldbody')
    for name, radius, rgba in radii:
        n = 72; seg = 2*math.pi*radius/n
        for i in range(n):
            a = 2*math.pi*i/n
            ET.SubElement(world, 'geom', name=f'{name}_{i}', type='box', size=fmt([seg/2, 0.002, 0.0002]),
                          pos=fmt([radius*math.cos(a), radius*math.sin(a), 0.0003]),
                          quat=fmt(yaw_quat(math.degrees(a) + 90)), rgba=fmt(rgba), group='2', **NO_CONTACT)


def add_rest_keyframe(root, layout):
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    qpos = model.qpos0.copy()
    qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]] = layout['rest_pose']
    keyframe = root.find('keyframe')
    if keyframe is None: keyframe = ET.SubElement(root, 'keyframe')
    ET.SubElement(keyframe, 'key', name='rest', qpos=fmt(qpos), ctrl=fmt(layout['rest_pose']))


class ArmIK:
    """Damped least-squares IK on the grasp_tcp site (7 arm joints, gripper untouched)."""

    def __init__(self, model):
        self.model, self.data = model, mujoco.MjData(model)
        joints = model.actuator_trnid[:, 0]
        self.qids, self.dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints][:7]
        self.site = model.site('grasp_tcp').id
        self.low, self.high = model.actuator_ctrlrange.T
        self.down = np.array([[1., 0, 0], [0, 0, 1], [0, -1, 0]])  # tcp +y (approach) pointing at the table

    def fk(self, q):
        """Measured joints (rad) -> (tcp xyz, tcp rotation matrix) in the base frame."""
        self.data.qpos[self.qids] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy()

    def solve(self, xyz, top_down=True, q0=None, attempts=16, seed=3, yaw_deg=0.0):
        rng = np.random.default_rng(seed)
        c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        down = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ self.down  # jaw closing axis rotated about vertical
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for attempt in range(attempts):
            q = (np.zeros(self.model.nu) if q0 is None else q0.copy()) if attempt == 0 else rng.uniform(self.low, self.high)
            for _ in range(300):
                self.data.qpos[self.qids] = q
                mujoco.mj_kinematics(self.model, self.data); mujoco.mj_comPos(self.model, self.data)
                err = np.asarray(xyz) - self.data.site_xpos[self.site]
                mujoco.mj_jacSite(self.model, self.data, jp, jr, self.site)
                if top_down:
                    quat = np.zeros(4)
                    mujoco.mju_mat2Quat(quat, (down @ self.data.site_xmat[self.site].reshape(3, 3).T).reshape(-1))
                    if quat[0] < 0: quat *= -1
                    length = np.linalg.norm(quat[1:])
                    err_r = quat[1:] * (2*np.arctan2(length, quat[0])/max(length, 1e-9))
                    if np.linalg.norm(err) < 0.002 and np.linalg.norm(err_r) < 0.02: return q
                    jac, e = np.vstack([jp[:, self.dofs], .1*jr[:, self.dofs]]), np.r_[err, .1*err_r]
                else:
                    if np.linalg.norm(err) < 0.003: return q
                    jac, e = jp[:, self.dofs], err
                dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4*np.eye(len(e)), e)
                q[:7] = np.clip(q[:7] + np.clip(dq, -.08, .08), self.low[:7] + .001, self.high[:7] - .001)
        return None

    # Least-lean IK, the same algorithm as sim/real_arm_mcp.py (_target_rot/_solve_rot/_solve_tcp) so the twin
    # behaves like the hardware: solve() falls back to ignoring orientation, which can leave the gripper steeply
    # tilted; here the lean away from (or toward) the base is stepped up gradually. The lean list and the search order
    # are the shared skill config (learning/skillcfg.py: ik.leans, ik.search, ik.restarts), read at call time.
    @staticmethod
    def target_rot(xyz, lean_deg, yaw_deg):
        """TCP rotation whose approach axis (+y) points down, leaned lean_deg away from the base (negative: toward it)."""
        r = np.asarray(xyz[:2], float)
        r = r / np.linalg.norm(r) if np.linalg.norm(r) > 1e-6 else np.array([0.0, 1.0])
        t, yw = np.radians(lean_deg), np.radians(yaw_deg)
        a = np.array([np.sin(t) * r[0], np.sin(t) * r[1], -np.cos(t)])
        j = np.array([np.cos(yw), np.sin(yw), 0.0])
        j = j - (j @ a) * a
        if np.linalg.norm(j) < 1e-6:
            j = np.array([-r[1], r[0], 0.0])
        j /= np.linalg.norm(j)
        return np.column_stack([j, a, np.cross(j, a)])

    def solve_rot(self, xyz, Rt, q0, restarts=0, seed=3):
        """Damped least squares to position + rotation Rt. First from q0; with restarts, the solution nearest q0."""
        m, d = self.model, self.data
        rng = np.random.default_rng(seed)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        best = None
        for attempt in range(1 + restarts):
            q = np.array(q0, float)
            if attempt:
                q[:7] = rng.uniform(self.low[:7], self.high[:7])
            for _ in range(300):
                d.qpos[self.qids] = q
                mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
                err = np.asarray(xyz) - d.site_xpos[self.site]
                mujoco.mj_jacSite(m, d, jp, jr, self.site)
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, (Rt @ d.site_xmat[self.site].reshape(3, 3).T).reshape(-1))
                if quat[0] < 0: quat *= -1
                ln = np.linalg.norm(quat[1:])
                err_r = quat[1:] * (2 * np.arctan2(ln, quat[0]) / max(ln, 1e-9))
                if np.linalg.norm(err) < 0.002 and np.linalg.norm(err_r) < 0.02:
                    if best is None or np.abs(q[:7] - q0[:7]).max() < np.abs(best[:7] - q0[:7]).max():
                        best = q.copy()
                    break
                jac, e = np.vstack([jp[:, self.dofs], .1 * jr[:, self.dofs]]), np.r_[err, .1 * err_r]
                dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), e)
                q[:7] = np.clip(q[:7] + np.clip(dq, -.08, .08), self.low[:7] + .001, self.high[:7] - .001)
            if attempt == 0 and best is not None:
                return best
        return best

    def solve_least_lean(self, xyz, q0, yaw_deg=0.0, max_jump=0.6):
        """(q, lean_deg) of the least-leaning reachable pose by skillcfg.lean_search; (None, None) if none. A solution
        from the current joints counts as near only if no joint moves more than max_jump rad, so a short TCP move
        never swings the arm through space onto another branch."""
        q0 = np.asarray(q0, float)
        return _skillcfg().lean_search(lambda lean, rs: self.solve_rot(xyz, self.target_rot(xyz, lean, yaw_deg), q0, rs),
                                       near=lambda q: np.abs(q[:7] - q0[:7]).max() <= max_jump)


def _skillcfg():
    try:
        from learning import skillcfg
    except ImportError:
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from learning import skillcfg
    return skillcfg


def sample_reach(model, samples, top_down_deg, seed=0):
    data = mujoco.MjData(model)
    qids = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    low, high = model.actuator_ctrlrange.T
    site = model.site('grasp_tcp').id
    rng = np.random.default_rng(seed)
    points, down = np.zeros((samples, 3)), np.zeros(samples, bool)
    cos_tol = math.cos(math.radians(top_down_deg))
    for i in range(samples):
        data.qpos[qids] = rng.uniform(low, high)
        mujoco.mj_kinematics(model, data)
        points[i] = data.site_xpos[site]
        down[i] = -data.site_xmat[site][7] > cos_tol
    keep = points[:, 2] > 0.005
    return points[keep], down[keep]


def ring_radius(points, mask, z_low, z_high):
    sel = mask & (points[:, 2] > z_low) & (points[:, 2] < z_high)
    return float(np.percentile(np.hypot(points[sel, 0], points[sel, 1]), 99.5)) if sel.any() else 0.0


def reach_report(model, layout):
    ik, clearance = ArmIK(model), layout['reach']['bin_drop_clearance']

    def check(xyz, top_down):
        return {'target_xyz': [round(float(v), 4) for v in xyz], 'distance_from_base_m': round(float(np.hypot(*xyz[:2])), 4),
                'reachable': ik.solve(xyz, top_down=top_down) is not None}

    return {'objects_grasp_point (top-down)': {s['name']: check([*s['pos'][:2], max(s['pos'][2], 0.024)], True)
                                               for s in layout['objects']},
            'bins_drop_point (any orientation)': {s['name']: check([*s['pos'], s['size'][2] + clearance], False)
                                                  for s in layout['bins']}}


def settle(model, seconds=3.0):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('rest').id)
    mujoco.mj_forward(model, data)
    names = [model.body(b).name for b in range(model.nbody) if model.body_jntnum[b] and
             model.jnt_type[model.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE]
    start = {n: data.body(n).xpos.copy() for n in names}
    while data.time < seconds:
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all(), 'simulation diverged'
    return data, {n: round(float(np.linalg.norm(data.body(n).xpos - start[n])), 4) for n in names}


def render(model, data, out_dir, layout, photo=None):
    w, h = layout['image_size']
    paths = {}
    with mujoco.Renderer(model, 480, 640) as big, mujoco.Renderer(model, h, w) as small:
        for cam in [c['name'] for c in layout['cameras']] + ['wrist_cam']:
            renderer = small if cam == 'wrist_cam' else big
            renderer.update_scene(data, camera=cam)
            paths[cam] = out_dir / f'{cam}.png'
            Image.fromarray(renderer.render()).save(paths[cam])
    if photo:
        a = Image.open(photo).convert('RGB').resize((640, 480)); b = Image.open(paths['overview_cam'])
        both = Image.new('RGB', (1280, 480)); both.paste(a, (0, 0)); both.paste(b, (640, 0))
        both.save(out_dir / 'photo_vs_sim.png')
    return {k: str(v) for k, v in paths.items()}


def build(layout_path=HERE / 'layout.json', out_dir=HERE / 'generated', photo=None, report=True):
    layout = json.loads(Path(layout_path).read_text())
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    root = build_xml(layout, out_dir)
    rings = {}
    if layout['reach'].get('show_rings') or report:
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        points, down = sample_reach(model, layout['reach']['samples'] if report else 20000, layout['reach']['top_down_tolerance_deg'])
        rings = {'max_reach_table_height_m': ring_radius(points, np.ones(len(points), bool), 0.005, 0.05),
                 'max_reach_top_down_grasp_m': ring_radius(points, down, 0.015, 0.05)}
    if layout['reach'].get('show_rings'):
        add_reach_rings(root, [('reach_ring_any', rings['max_reach_table_height_m'], [0.95, 0.75, 0.1, 0.8]),
                               ('reach_ring_topdown', rings['max_reach_top_down_grasp_m'], [0.1, 0.75, 0.3, 0.8])])
    add_rest_keyframe(root, layout)
    ET.indent(root)
    scene = out_dir / 'workspace.xml'
    scene.write_text(ET.tostring(root, encoding='unicode'))
    if not report:
        return scene, None
    model = mujoco.MjModel.from_xml_path(str(scene))
    data, moved = settle(model)
    result = {'scene': str(scene), 'layout': str(layout_path), 'bodies': model.nbody, 'geoms': model.ngeom,
              'joint_limits_rad': {f'j{i+1}': model.actuator_ctrlrange[i].round(4).tolist() for i in range(model.nu)},
              'reach_rings': {k: round(v, 4) for k, v in rings.items()},
              'settle_displacement_m': moved, **reach_report(model, layout),
              'previews': render(model, data, out_dir, layout, photo)}
    (out_dir / 'report.json').write_text(json.dumps(result, indent=2))
    return scene, result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--layout', help='layout file or name (venue | home); default: env SIM_LAYOUT / ROBOT_SIM_LAYOUT, else SITE')
    p.add_argument('--out', type=Path, help='output folder (default sim/generated[/<layout name>])')
    p.add_argument('--photo', type=Path, help='real overview photo to put next to the overview_cam render')
    args = p.parse_args()
    path = layout_path(args.layout)
    print(json.dumps(build(path, args.out or generated_dir(path), args.photo)[1], indent=2))


if __name__ == '__main__':
    main()
