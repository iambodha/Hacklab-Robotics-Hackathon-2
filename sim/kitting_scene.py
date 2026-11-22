"""The kitting twin: a calibrated site table (sites.get_site().layout: SITE, default home = sim/layouts/home.json) +
the printed kit sheet as a textured plane + the kit objects (screws, caps, boxes) on their start spots, from
tasks/kitting/kit_a4.json. The sheet lies at the site's placement (spec.site_sheet_pose: sites/<site>/tasks.json
"kitting", else the spec's nominal home pose).

    python3 sim/kitting_scene.py [--spec tasks/kitting/kit_a4.json] [--base SITE] [--out DIR]
        -> sim/layouts/kit_<base>.json (the layout, nominal poses) + <out>/workspace.xml + previews

Built with sim/build_scene.build_xml (same arm, servos, cameras, markers, table texture and lights as every twin) and
three kitting-only additions done here on the XML, so build_scene.py stays unchanged:
  1. the sheet: a visual-only plane with the sheet texture (tasks/kitting/sheet.twin_texture: the same drawing as the
     print), in a static body 'kit_sheet' that randomization moves (model.body_pos/quat);
  2. the objects as explicit geom parts (screw = two shank halves + head, so ~1/3 of the mass is in the head);
  3. the fingertip pads moved to where the real ones are:
     - PAD_FIX_M: both 3.5 mm further in, so the jaws close to a ~1 mm gap like the real ones. The stock twin closes to
       an 8 mm gap and cannot hold a 4 mm screw; this is the change tasks/blue_to_bin/sim_trials.py proposes
       (apply_pad_fix).
     - PAD_TIP_SHIFT_M: both 6 mm back along the approach axis, so the pad tips end 10 mm below the TCP, as measured on
       the real arm (profile touch.nominal_table_tcp_z = 0.010: the TCP height when the fingertips touch the table, 21
       Nov). The stock pads (32 mm long, centred on the TCP) reach 16 mm below it, and a grasp height learned in such a
       twin would put the real fingertips 6 mm higher than intended.
     Applied to this scene only.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'cgl' if sys.platform == 'darwin' else 'egl')
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(HERE), str(REPO)):
    if p not in sys.path:
        sys.path.append(p)
import build_scene  # noqa: E402

from tasks.kitting.sheet import twin_texture  # noqa: E402
from sites import get_site  # noqa: E402
from tasks.kitting.spec import Pose2D, Spec  # noqa: E402

PAD_FIX_M = 0.0035          # tasks/blue_to_bin/sim_trials.PAD_FIX_M: real jaw gap ~49 mm open .. ~1 mm closed
PAD_TIP_SHIFT_M = 0.006     # stock pad tip 16 mm below the TCP -> 10 mm (profile touch.nominal_table_tcp_z)
SHEET_Z = 0.0004            # paper plane above the table top (table skin at 0.2 mm, table markers at 0.6 mm)
TEXTURE_PPM = 4.0           # sheet texture pixels per mm: 0.25 mm pixels, finer than the overview's ~3 mm/px
KIT_LAYOUT = HERE / 'layouts' / 'kit_home.json'
SCENE_ROOT = Path(os.environ.get('DATA_ROOT') or Path(__file__).resolve().parents[1] / 'data') / 'r5-kitsim'


def default_base() -> str:
    """The selected site (SITE, default home); SIM_LAYOUT is not consulted: kitting needs a calibrated site table."""
    return get_site().name


def scene_dir(base: str) -> Path:
    """Generated kitting scene per base layout; home keeps the historical r5-kitsim/scene."""
    return SCENE_ROOT / ('scene' if base == 'home' else f'scene_{base}')


def object_parts(spec: Spec, cls: str) -> tuple[list[dict], float]:
    """(build_scene 'parts' with half-sizes in metres, body z of the resting pose). Body origin: the screw's shank
    axis at its geometric centre (head towards -x); the cap's / box's centre. Long axis = body x."""
    c = spec.classes()[cls]
    rgba = c['rgba']
    if c['shape'] == 'screw':
        r, L, R, hh = c['shank_diameter'] / 2000, c['shank_length'] / 1000, c['head_diameter'] / 2000, c['head_height'] / 1000
        x0 = -(L + hh) / 2                                  # head outer face
        parts = [{'type': 'cylinder', 'size': [R, hh / 2], 'pos': [x0 + hh / 2, 0, 0], 'euler': [0, 90, 0], 'rgba': rgba},
                 {'type': 'cylinder', 'size': [r, L / 4], 'pos': [x0 + hh + L / 4, 0, 0], 'euler': [0, 90, 0], 'rgba': rgba},
                 {'type': 'cylinder', 'size': [r, L / 4], 'pos': [x0 + hh + 3 * L / 4, 0, 0], 'euler': [0, 90, 0], 'rgba': rgba}]
        return parts, R + 0.0005
    if c['shape'] == 'cylinder':
        return [{'type': 'cylinder', 'size': [c['diameter'] / 2000, c['height'] / 2000], 'rgba': rgba}], c['height'] / 2000 + 0.0005
    if c['shape'] == 'box':
        sx, sy, sz = (v / 2000 for v in c['size'])
        return [{'type': 'box', 'size': [sx, sy, sz], 'rgba': rgba}], sz + 0.0005
    raise ValueError(c['shape'])


def make_layout(spec: Spec, base: str | None = None, sheet_pose: Pose2D | None = None) -> dict:
    """The twin layout: the base table layout (a site name, default the selected site) with its loose objects
    replaced by the kit objects on their spots."""
    base = base or default_base()
    lay = copy.deepcopy(json.loads(build_scene.layout_path(base).read_text()))
    sheet_pose = sheet_pose or spec.site_sheet_pose(base)
    objs = []
    for t in spec.targets(sheet_pose):
        parts, z = object_parts(spec, t['class'])
        objs.append({'name': t['name'], 'class': t['class'], 'parts': parts, 'pos': [t['start'].x, t['start'].y, z],
                     'yaw': t['start'].yaw_deg, 'mass': spec.class_dims(t['class'])['mass'], 'src': f'kit spec {spec.name}'})
    lay['objects'] = objs
    lay['bins'] = []
    lay['_notes'] = [f"GENERATED by sim/kitting_scene.py from {base} + {spec.path.name}: do not edit, re-run it."] + \
        lay.get('_notes', [])
    lay['kitting'] = {'spec': str(spec.path.relative_to(REPO)) if spec.path.is_relative_to(REPO) else str(spec.path),
                      'sheet_pose': sheet_pose.as_list(), 'base_layout': base, 'pad_fix_m': PAD_FIX_M,
                      'pad_tip_shift_m': PAD_TIP_SHIFT_M}
    return lay


def add_sheet(root: ET.Element, spec: Spec, pose: Pose2D, out_dir: Path, levels) -> None:
    tex = twin_texture(spec, out_dir / f'{spec.name}_texture.png', TEXTURE_PPM, levels)
    asset, world = root.find('asset'), root.find('worldbody')
    ET.SubElement(asset, 'texture', name='kit_sheet_tex', type='2d', file=str(tex.resolve()))
    ET.SubElement(asset, 'material', name='kit_sheet_mat', texture='kit_sheet_tex', texuniform='false', texrepeat='1 1',
                  specular='0.1', shininess='0.2', reflectance='0')
    W, H = (v / 2000 for v in spec.paper_mm)
    body = ET.SubElement(world, 'body', name='kit_sheet', pos=build_scene.fmt([pose.x, pose.y, 0]),
                         quat=build_scene.fmt(build_scene.yaw_quat(pose.yaw_deg)))
    ET.SubElement(body, 'geom', name='kit_sheet', type='plane', size=build_scene.fmt([W, H, 0.01]), pos=f'0 0 {SHEET_Z}',
                  material='kit_sheet_mat', group='0', **build_scene.NO_CONTACT)


def pad_fix(root: ET.Element, inward_m: float = PAD_FIX_M, tip_shift_m: float = PAD_TIP_SHIFT_M) -> None:
    """Pads inward_m towards the jaw centre (closing axis = jaw x) and tip_shift_m back along the approach (jaw +y
    points from the gripper base towards the fingertips, build_scene.add_gripper_parts)."""
    for name, sign in (('left_pad', 1.0), ('right_pad', -1.0)):
        g = root.find(f'.//geom[@name="{name}"]')
        p = [float(v) for v in g.get('pos').split()]
        p[0] += sign * inward_m
        p[1] -= tip_shift_m
        g.set('pos', build_scene.fmt(p))


def build(layout: dict, out_dir: Path, spec: Spec | None = None) -> Path:
    """workspace.xml of a kitting layout (from make_layout) into out_dir."""
    spec = spec or Spec.load(REPO / layout['kitting']['spec'])
    out_dir.mkdir(parents=True, exist_ok=True)
    root = build_scene.build_xml(layout, out_dir)
    levels = layout.get('markers', {}).get('levels', [0.0, 1.0])
    add_sheet(root, spec, Pose2D(*layout['kitting']['sheet_pose']), out_dir, levels)
    pad_fix(root, layout['kitting'].get('pad_fix_m', PAD_FIX_M), layout['kitting'].get('pad_tip_shift_m', PAD_TIP_SHIFT_M))
    build_scene.add_rest_keyframe(root, layout)
    ET.indent(root)
    scene = out_dir / 'workspace.xml'
    scene.write_text(ET.tostring(root, encoding='unicode'))
    (out_dir / 'layout.json').write_text(json.dumps(layout, indent=1))
    return scene


def ensure_scene(spec: Spec | None = None, out_dir: Path | None = None, base: str | None = None) -> tuple[Path, dict]:
    """(workspace.xml, layout) of the nominal kitting twin, rebuilt when the spec, this file, build_scene, the base
    layout or the generated layout (e.g. a new site sheet placement) changed."""
    spec = spec or Spec.load()
    base = base or default_base()
    out_dir = Path(out_dir or scene_dir(base))
    scene = out_dir / 'workspace.xml'
    srcs = [spec.path, Path(__file__), HERE / 'build_scene.py', REPO / 'tasks' / 'kitting' / 'sheet.py',
            build_scene.layout_path(base)]
    lay = make_layout(spec, base)
    built = out_dir / 'layout.json'
    if (not scene.exists() or scene.stat().st_mtime < max(p.stat().st_mtime for p in srcs)
            or not built.exists() or json.loads(built.read_text()) != json.loads(json.dumps(lay))):
        build(lay, out_dir, spec)
    return scene, lay


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--spec', default=None)
    ap.add_argument('--base', default=None, help='site (default: SITE, home if unset)')
    ap.add_argument('--out', type=Path, default=None, help='default: DATA_ROOT/r5-kitsim/scene[_<base>]')
    ap.add_argument('--layout-out', type=Path, default=None, help='default: sim/layouts/kit_<base>.json')
    a = ap.parse_args()
    spec = Spec.load(a.spec)
    a.base = a.base or default_base()
    a.out = a.out or scene_dir(a.base)
    a.layout_out = a.layout_out or KIT_LAYOUT.with_name(f'kit_{a.base}.json')
    lay = make_layout(spec, a.base)
    a.layout_out.write_text(json.dumps(lay, indent=1) + '\n')
    scene = build(lay, a.out, spec)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data, moved = build_scene.settle(model, 2.0)
    paths = build_scene.render(model, data, a.out, lay)
    print(json.dumps({'layout': str(a.layout_out), 'scene': str(scene), 'settle_displacement_m': moved, 'previews': paths},
                     indent=1))


if __name__ == '__main__':
    main()
