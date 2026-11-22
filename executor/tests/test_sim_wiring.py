"""The same wiring on the MuJoCo twin (sim backend, real-time physics): perception sees the twin's blue cap through the
twin's overview camera, pick_at -> drop_at puts it in the bin (sim ground truth), a skills/wipe stroke feels the table.

Layout: sim/layouts/home.json (cameras, markers, table) plus a blue cap and a bin, written to a temporary layout file
(r2-task owns home.json's objects). One twin per process: every test here shares it, in order."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from executor.contracts import Status
from executor.tests.conftest import proposal

ROOT = Path(__file__).resolve().parents[2]
CAP_W = (0.24, -0.06)                       # world xy (x forward, y left)
BIN_W = (0.25, 0.13)
BIN_SIZE_W = [0.10, 0.12, 0.06]             # world x, world y, rim height


@pytest.fixture(scope="module")
def sim_ex(kin, tmp_path_factory):
    pytest.importorskip("mcp")
    from executor.backends.sim import SimBackend
    from executor.core import Executor
    from executor.kinematics import load_scene_calib
    from executor.perception import PerceptionLocator
    from executor.recording import RecorderLink
    lay = json.loads((ROOT / "sim/layouts/home.json").read_text())
    to_sim = lambda xy: kin.to_base([xy[0], xy[1], 0.0])[:2].round(4).tolist()          # noqa: E731
    lay["objects"] = [{"name": "blue_cap", "pos": [*to_sim(CAP_W), 0.011], "yaw": 0, "mass": 0.008,
                       "parts": [{"type": "cylinder", "size": [0.016, 0.011], "rgba": [0.08, 0.22, 0.85, 1]}]}]
    lay["bins"] = [{"name": "bin", "pos": to_sim(BIN_W), "yaw": 0, "size": [BIN_SIZE_W[1], BIN_SIZE_W[0], BIN_SIZE_W[2]],
                    "wall": 0.004, "rgba": [0.85, 0.85, 0.82, 1], "inner_rgba": [0.8, 0.8, 0.78, 1]}]
    path = ROOT / "sim" / "generated" / "r2wire_test_home.json"            # sim/generated is gitignored
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lay))
    old = {k: os.environ.get(k) for k in ("SIM_LAYOUT", "ROBOT_SIM_LAYOUT")}
    os.environ["SIM_LAYOUT"] = os.environ["ROBOT_SIM_LAYOUT"] = str(path)
    try:
        backend = SimBackend(layout=str(path))
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"twin not available: {e}")
    if Path(backend.mod.LAYOUT).resolve() != path.resolve():
        pytest.skip(f"another twin layout is already loaded in this process ({backend.mod.LAYOUT})")
    tmp = tmp_path_factory.mktemp("simrec")
    ex = Executor(backend, kin, recorder=RecorderLink("local", root=tmp / "rec"), locator=PerceptionLocator(load_scene_calib()),
                  table_memory_path=tmp / "table.json", layout="r2wire_test_home")
    ex.twin._ready.wait(30)
    yield ex
    ex.cancel()
    ex.end_task("test end")
    for k, v in old.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def test_sim_observation_sees_the_blue_cap(sim_ex):
    obs = sim_ex.observe()
    blue = [o for o in obs.objects if "blue" in o.name]
    assert blue, (obs.objects, obs.notes)
    o = blue[0]
    truth = sim_ex.kin.to_world(sim_ex.backend.truth()["objects"]["blue_cap"]["xyz"])
    err = float(np.linalg.norm(np.array(o.xyz_world[:2]) - truth[:2])) * 1000
    assert err < 12, (o, truth, err)                     # twin camera, colour footprint: about a pixel on the table
    assert o.visible and o.uncertainty_mm and "perception.locate" in o.source


def test_sim_pick_blue_and_drop_in_bin(sim_ex):
    ex = sim_ex
    obs = ex.observe()
    r = ex.wait(ex.submit(proposal(ex, obs, "pick_at", {"object": "blue", "object_height": 0.022}, ttl=60)), 180)
    assert r.status == Status.SUCCEEDED, (r.termination, r.validation.reasons, r.phases)
    assert r.validation.checks["twin_preflight"]["result"] == "pass"
    o = ex.evaluate(r.execution_id)
    assert o.success is True, o                          # sim truth: the cap is up in the air at the TCP
    obs = ex.observe()
    bin_ = {"xyz_world": [*BIN_W, 0.0], "size": BIN_SIZE_W, "yaw_deg": 0.0}
    d = ex.wait(ex.submit(proposal(ex, obs, "drop_at", {"bin": bin_, "object": "blue"}, ttl=60)), 180)
    assert d.status == Status.SUCCEEDED, (d.termination, d.validation.reasons)
    assert d.achieved["release_over_opening"] is True
    o = ex.evaluate(d.execution_id)
    assert o.success is True and "in_bin=bin" in " ".join(o.evidence), o


def test_sim_wipe_stroke_feels_the_table(sim_ex):
    from skills.wipe import WipeSkill
    ex = sim_ex
    sk = WipeSkill()
    v = sk.seed_version(w_hat=0.03)
    plan = sk.plan(v, {"region": (0.18, -0.20, 0.30, -0.17), "budget": 1})
    obs = ex.observe()
    p = sk.execute(plan, based_on_obs=obs.obs_id, setup_id=obs.setup_id, now=ex.now(), version=v.version, ttl_s=60)[0]
    r = ex.wait(ex.submit(p), 240)
    assert r.validation.ok, r.validation.reasons
    assert [ph["name"] for ph in r.phases] == ["plan", "validate", "execute", "verify", "recover"]
    assert r.achieved["contact_z"][0] is not None and abs(r.achieved["contact_z"][0] - ex.kin.P.touch.nominal_table_tcp_z) < 0.015
    # The twin's servos report far higher loads in contact than the real arm (first touch ~960+917 vs ~250), so the
    # hardware-validated contact limits (load_abort 550, hard_load_sum 520) may stop the stroke: then it must say so
    # (PARTIAL with the load) and the recovery must lift straight up. Either outcome is truthful; never a silent success.
    if r.status != Status.SUCCEEDED:
        assert r.status == Status.PARTIAL and "load" in r.termination, r.termination
        rec = r.phases[-1]
        assert rec["status"] == "done" and "straight up" in rec["note"], rec
        assert ex.kin.fk(ex.motion.q_now())[0][2] > 0.06
    assert r.achieved["strokes_world"], r.achieved
    o = ex.evaluate(r.execution_id)                      # no stain in the twin: the evaluator says unknown, not success
    assert o.metric == "contamination_reduction" and o.success is None
