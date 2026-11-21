"""Build the MuJoCo twin of the hackathon table: ElRobot follower, table, bins, objects, cameras.

  python3 sim/build_scene.py [--layout sim/layout.json] [--out sim/generated] [--photo overview.jpg]

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
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault('MUJOCO_GL', 'osmesa')
import mujoco
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OBJECT_CONTACT = {'contype': '2', 'conaffinity': '3'}
STATIC_CONTACT = {'contype': '2', 'conaffinity': '1'}
PAD_CONTACT = {'contype': '4', 'conaffinity': '2'}
NO_CONTACT = {'contype': '0', 'conaffinity': '0'}
GRIPPER_TORQUE = 0.4  # N m on rev_motor_08 (0 rad = jaws open)


def find_urdf():
    candidates = [os.environ.get('NORMA_CORE_REPO', ''), REPO / 'norma-core', REPO.parent / 'norma-core',
                  '/mnt/d/Robotics-Hackathon/repos/norma-core']
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
    T = np.array(json.loads((REPO / cam['pose_file']).read_text())['T_link_camera'])
    # OpenCV (x right, y down, z forward) -> MuJoCo camera (x right, y up, looks along -z).
    ET.SubElement(gripper, 'camera', name='wrist_cam', pos=fmt(T[:3, 3]), xyaxes=fmt(np.r_[T[:3, 0], -T[:3, 1]]),
                  fovy=str(cam['fovy']))


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
    cy = table['back_edge_y'] + td/2
    tbody = ET.SubElement(world, 'body', name='table', pos=fmt([0, cy, 0]))
    ET.SubElement(tbody, 'geom', name='table_top', type='box', pos=fmt([0, 0, -th/2]), size=fmt([tw/2, td/2, th/2]),
                  rgba=fmt(table['rgba']), friction='0.8 0.01 0.001', **STATIC_CONTACT)
    for i, (sx, sy) in enumerate([(-1, -1), (1, -1), (-1, 1), (1, 1)]):
        ET.SubElement(tbody, 'geom', name=f'table_leg{i}', type='box', size=fmt([0.02, 0.02, (table['height']-th)/2]),
                      pos=fmt([sx*(tw/2-0.05), sy*(td/2-0.05), -th - (table['height']-th)/2]),
                      rgba='0.75 0.75 0.75 1', **STATIC_CONTACT)
    ET.SubElement(world, 'geom', name='floor', type='plane', size='3 3 0.1', pos=fmt([0, cy, -table['height']]),
                  rgba='0.45 0.45 0.47 1', **STATIC_CONTACT)
    ET.SubElement(world, 'light', name='ceiling_a', directional='true', pos=fmt([0.3, cy, 2]), dir='-0.15 -0.1 -1',
                  diffuse='0.4 0.4 0.4', specular='0.1 0.1 0.1')
    ET.SubElement(world, 'light', name='ceiling_b', directional='true', pos=fmt([-0.3, cy, 2]), dir='0.2 0.3 -1',
                  diffuse='0.2 0.2 0.2', specular='0 0 0', castshadow='false')

    for spec in layout['bins']: add_bin(world, spec)
    for spec in layout['fixed']:
        ET.SubElement(world, 'geom', name=spec['name'], type=spec['type'], pos=fmt(spec['pos']), size=fmt(spec['size']),
                      quat=fmt(yaw_quat(spec['yaw'])), rgba=fmt(spec['rgba']), **STATIC_CONTACT)
    for spec in layout['objects']:
        body = ET.SubElement(world, 'body', name=spec['name'], pos=fmt(spec['pos']), quat=fmt(yaw_quat(spec['yaw'])))
        ET.SubElement(body, 'freejoint', name=f"{spec['name']}_free")
        physical = [p for p in spec['parts'] if not p.get('visual_only')]
        for i, part in enumerate(spec['parts']):
            add_part(body, part, f"{spec['name']}_g{i}", None if part.get('visual_only') else spec['mass']/len(physical))
    for cam in layout['cameras']:
        ET.SubElement(world, 'camera', name=cam['name'], pos=fmt(cam['pos']),
                      xyaxes=fmt(camera_axes(cam['pos'], cam['lookat'])), fovy=str(cam['fovy']))
    return root


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

    def solve(self, xyz, top_down=True, q0=None, attempts=16, seed=3):
        rng = np.random.default_rng(seed)
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
                    mujoco.mju_mat2Quat(quat, (self.down @ self.data.site_xmat[self.site].reshape(3, 3).T).reshape(-1))
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
    p.add_argument('--layout', type=Path, default=HERE / 'layout.json')
    p.add_argument('--out', type=Path, default=HERE / 'generated')
    p.add_argument('--photo', type=Path, help='real overview photo to put next to the overview_cam render')
    args = p.parse_args()
    print(json.dumps(build(args.layout, args.out, args.photo)[1], indent=2))


if __name__ == '__main__':
    main()
