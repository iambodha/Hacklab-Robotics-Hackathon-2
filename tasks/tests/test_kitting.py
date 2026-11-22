"""Kitting geometry, calibrated image localisation, physics and profile handoff."""
import json
from pathlib import Path
import numpy as np
import pytest
from tasks.kitting.spec import Spec, Pose2D


def test_sheet_geometry_and_object_assignment():
    from sim.kitting_scene import make_layout, object_parts
    s=Spec.load()
    assert s.paper_mm == (297.,210.)
    assert not set(s.markers()) & {20,21,22,23}
    pose=Pose2D(.012,.237,17)
    for cid in s.cells():
        corners=s.cell_corners_mm(cid)
        assert np.allclose(s.to_sheet(pose,s.to_table(pose,corners)),corners)
    layout=make_layout(s,sheet_pose=pose)
    for obj,target in zip(layout['objects'],s.targets(pose)):
        assert np.allclose(obj['pos'][:2],[target['start'].x,target['start'].y])
        assert obj['mass'] == s.class_dims(target['class'])['mass']
        parts,_=object_parts(s,target['class'])
        assert parts == obj['parts']
    assert not s.problems()


@pytest.mark.parametrize('site',['home','office'])
def test_kitting_layout_follows_the_site(monkeypatch,site):
    from sim.kitting_scene import make_layout, scene_dir
    from sites import get_site
    from tasks.kitting.spec import table_marker_ids
    monkeypatch.setenv('SITE',site)
    s=Spec.load()
    lay=make_layout(s)
    base=json.loads(get_site(site).layout.read_text())
    assert lay['kitting']['base_layout']==site and lay['markers']==base['markers'] and lay['table']==base['table']
    pose=s.site_sheet_pose(site)
    assert lay['kitting']['sheet_pose']==pose.as_list()
    assert np.allclose(lay['objects'][0]['pos'][:2],[s.targets(pose)[0]['start'].x,s.targets(pose)[0]['start'].y])
    assert {m['id'] for m in base['markers']['items']} <= set(table_marker_ids())
    assert scene_dir(site).name==('scene' if site=='home' else f'scene_{site}')


def test_home_kitting_layout_unchanged(monkeypatch):
    from sim.kitting_scene import make_layout
    monkeypatch.delenv('SITE',raising=False)
    s=Spec.load()
    assert s.site_sheet_pose('home')==s.nominal_sheet_pose()
    lay=make_layout(s)
    assert lay['kitting']['base_layout']=='home' and lay['kitting']['sheet_pose']==s.nominal_sheet_pose().as_list()
    assert lay['markers']==json.loads((Path(__file__).resolve().parents[2]/'sim/layouts/home.json').read_text())['markers']


def test_print_scale_and_marker_decoding(tmp_path):
    from tasks.kitting.sheet import write_print
    from tasks.kitting.locate import detect_sheet_markers
    from PIL import Image
    r=write_print(Spec.load(),tmp_path)
    assert np.allclose(r['page_mm'],[297,210],atol=.04)
    assert set(detect_sheet_markers(np.asarray(Image.open(r['png']).convert('RGB').resize((1000,707))),Spec.load()))=={10,11,12,13}


@pytest.fixture
def sim(monkeypatch):
    monkeypatch.setenv('KIT_RENDERER','cpu')
    from tasks.kitting.sim_env import KitSim
    sim=KitSim()
    yield sim
    sim.close()


@pytest.mark.parametrize('seed',[0,1,2])
def test_localise_rendered_twin_cells(sim,seed):
    from tasks.kitting.sim_env import DR
    from tasks.kitting.locate import locate_sheet
    truth=sim.reset(np.random.default_rng(seed),DR(sheet=True,human=True,cam_pos_m=0,cam_rot_deg=0,base_m=0,base_yaw_deg=0))
    r=locate_sheet(sim.render(),sim.spec,sim.cam)
    assert r['ok'],r
    p=Pose2D(*truth['sheet'])
    for cid,xy in r['cells'].items():
        expected=sim.spec.cell_pose(p,cid)
        assert np.linalg.norm(np.array(xy)-[expected.x,expected.y])*1000 < 3.0


def test_short_physics_episode_and_records(sim,tmp_path,monkeypatch):
    from tasks.kitting.sim_env import DR,run_episode
    from tasks.kitting import profile as P
    from tasks.kitting.kit_episodes import KitEpisodeStore,record_episode
    monkeypatch.setenv('SKILLCFG_DIR',str(tmp_path/'cfg'))
    from learning import skillcfg as SC
    old=SC.target()
    try:
        P.use_dir(tmp_path/'cfg')
        ep=run_episode(sim,P.generic(),np.random.default_rng(123),DR.none(),only=['box_1'])
        assert not ep['aborted']
        assert len(ep['objects'])==1
        o=ep['objects'][0]
        assert o.outcome=='placed', (o.outcome,o.truth,o.phases)
        assert o.truth['error_mm']<10
        assert {'grasp','lift','release'} <= {p['phase'] for p in o.phases}
        store=KitEpisodeStore(tmp_path/'episodes')
        record_episode(store,ep,P.flat(P.generic()),'Generic box transfer smoke test')
        rows=[json.loads(x) for x in store.path.read_text().splitlines()] if hasattr(store,'path') else [json.loads(x) for x in (store.root/'episodes.jsonl').read_text().splitlines()]
        assert rows[-1]['outcome']=='complete'
        proposal=SC.propose({P.key('box','place_z_mm'):32.},'Lower release after measured bounce',['smoke test'],'test','twin')
        assert proposal['ok']
        assert SC.apply(proposal['proposal']['id'],'test')['ok']
        path=P.export(tmp_path/'learned.json')
        assert P.load(path)['box']['place_z_mm']==32
    finally:
        SC.set_target(old)


def test_abort_counts_as_failure():
    from tasks.kitting.learn import metrics
    r=metrics([{'objects':[],'expected_objects':6,'aborted':'no markers'}],['screw','cap','box'])
    assert r['objects']==6 and r['success_rate']==0 and r['aborted']==1


def test_executor_handoff_uses_profile_and_world_frame():
    from tasks.kitting import profile as P
    from tasks.kitting.run_real import commands
    from executor.kinematics import Kinematics
    s=Spec.load(); k=Kinematics(); params=P.generic()
    params['cap']['grip_force']=.42
    cmds=commands(s,s.nominal_sheet_pose(),params,k,'full','cap_1')
    close=next(c for c in cmds if c['phase']=='close')
    assert close['params']['force']==.42
    approach=next(c for c in cmds if c['phase']=='approach')
    t=next(t for t in s.targets(s.nominal_sheet_pose()) if t['name']=='cap_1')
    assert np.allclose(approach['params']['xyz_world'][:2],k.to_world([t['start'].x,t['start'].y,0])[:2])
    params['cap']['release_open']=.5
    dry=commands(s,s.nominal_sheet_pose(),params,k,'dry')
    assert not any(c['phase']=='close' for c in dry)
    assert all(c['params']['open_fraction']==1 for c in dry if c['skill']=='gripper')


def test_executor_sim_handoff(sim,tmp_path):
    from tasks.kitting.executor_sim import run
    from tasks.kitting import profile as P
    from tasks.kitting.sim_env import DR,in_cell
    sim.reset(np.random.default_rng(1),DR.none())
    records=run(sim,P.generic(),only='box_1',mode='full',root=tmp_path)
    assert records and all(r['result']['status']=='SUCCEEDED' for r in records)
    target=sim.spec.object('box_1')['cell']
    sim.step_s(.6)
    ok,err,_=in_cell(sim.spec,sim.spec.nominal_sheet_pose(),target,sim.obj_pose('box_1'),'box')
    assert ok,err


def test_saved_gl_twin_image():
    from PIL import Image
    from tasks.kitting.locate import camera_model,locate_sheet
    root=Path(__file__).resolve().parents[2]
    image=np.asarray(Image.open(root/'docs/kitting/images/devin_gl_overview.png').convert('RGB').resize((299,224)))
    layout=json.loads((root/'sim/layouts/home.json').read_text())
    s=Spec.load(); r=locate_sheet(image,s,camera_model(layout,twin=True))
    assert r['ok'],r
    nominal=s.nominal_sheet_pose()
    assert np.hypot(r['pose'].x-nominal.x,r['pose'].y-nominal.y)*1000<3


def test_bad_sheet_identifiers_rejected(tmp_path):
    s=Spec.load()
    s.raw['markers']['items'][0]['id']=20
    s.raw['cells'][1]['id']=s.raw['cells'][0]['id']
    path=tmp_path/'invalid.json'; path.write_text(json.dumps(s.raw))
    with pytest.raises(ValueError,match='duplicate cell identifier'):
        Spec.load(path)


def test_ab_cannot_pass_by_aborting(monkeypatch):
    from types import SimpleNamespace
    from tasks.kitting.learn import ab_check
    from tasks.kitting import sim_env
    obj=SimpleNamespace(outcome='missed',diagnosis=['grasp_too_high'],truth={},phases=[])
    before={'objects':[obj], 'aborted':None}
    after={'objects':[], 'aborted':'numerical failure'}
    replies=iter([before,after])
    monkeypatch.setattr(sim_env,'run_episode',lambda *a,**k: next(replies))
    sim=SimpleNamespace(spec=Spec.load())
    r=ab_check(sim,'screw',{}, {},[1],None,'grasp_too_high')
    assert r['result']=='fail' and r['after']['aborted']==1


def test_select_best_uses_evaluation_and_reverts(tmp_path,monkeypatch):
    from tasks.kitting import profile as P
    from tasks.kitting.learn import select_best
    from learning import skillcfg as SC
    monkeypatch.setenv('SKILLCFG_DIR',str(tmp_path/'cfg'))
    old=SC.target()
    try:
        P.use_dir(tmp_path/'cfg')
        generic,version=P.current()
        p=SC.propose({P.key('box','place_z_mm'):32.},'Candidate profile',['test'],'test','twin')
        assert SC.apply(p['proposal']['id'],'test')['ok']
        worse,v=P.current()
        rows=[dict(iteration=0,params=generic,version=version,success_rate=.5,mean_error_mm=4.),
              dict(iteration=1,params=worse,version=v,success_rate=.4,mean_error_mm=3.)]
        chosen,changed=select_best(rows,P,SC,log=lambda _:None)
        assert changed and chosen['selected_iteration']==0
        assert P.current()[0]==generic and P.current()[1]>v
    finally:
        SC.set_target(old)
