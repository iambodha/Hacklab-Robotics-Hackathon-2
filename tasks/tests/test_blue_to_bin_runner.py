"""The scripted planner end to end on the executor's fake backend (virtual clock) with a stub world as the locator:
every motion must be an accepted executor proposal, a missed grasp must be retried, attempts are capped."""
import numpy as np
import pytest

pytest.importorskip("mujoco")

from executor.backends import FakeBackend, park_norm  # noqa: E402
from executor.contracts import ObjectEstimate  # noqa: E402
from executor.core import Executor  # noqa: E402
from executor.kinematics import Kinematics  # noqa: E402
from executor.perception import Locator  # noqa: E402
from executor.recording import RecorderLink  # noqa: E402
from tasks.blue_to_bin.plan import Runner  # noqa: E402
from tasks.blue_to_bin.skills import install  # noqa: E402
from tasks.blue_to_bin.task import Spec, plan_grasp  # noqa: E402


class StubWorld(Locator):
    """The lid rides with the jaws once they close at the pinch point (unless a miss is scripted); when they open it
    drops onto the tub floor if it is over the tub, else onto the table. `hidden` = steps when the camera sees nothing."""
    name = "stub world"

    def __init__(self, spec, start_xy, misses=0):
        self.spec, self.xyz = spec, np.r_[start_xy, 0.0]      # lid base
        self.misses, self.held, self.offset = misses, False, None

    def locate(self, frames, joints_norm, kin, camera_info):
        q = kin.q_from_norm(joints_norm)
        tcp = kin.fk(q)[0]
        closed = joints_norm[kin.grip_id] < kin.P.gripper.open_enough
        g = plan_grasp(self.spec, [*self.xyz[:2], 0.0])
        if not self.held and closed and np.hypot(*(tcp[:2] - g.tcp_xy)) < 0.012 and tcp[2] < g.z_grasp + 0.008:
            if self.misses:
                self.misses -= 1
                self.xyz[:2] += [0.0, 0.01]                    # the pad nudged it: replan from the new spot
            else:
                self.held, self.offset = True, self.xyz - np.r_[tcp[:2], tcp[2] - g.z_grasp]
        if self.held and not closed:
            self.held = False
            inside = np.hypot(*(self.xyz[:2] - self.spec.bin_xy)) < self.spec.bin_rim_diameter / 2 - 0.01
            self.xyz[2] = 0.002 if inside else 0.0
        if self.held:
            self.xyz = np.r_[tcp[:2], tcp[2] - g.z_grasp] + self.offset
        return [ObjectEstimate("blue_cap", [*self.xyz[:2], self.xyz[2] + self.spec.cap_height / 2], uncertainty_mm=9.0,
                               source="stub", visible=True)]


@pytest.fixture(scope="module")
def kin():
    return Kinematics()


def make(kin, tmp_path, world):
    fake = FakeBackend(start_norm=dict(park_norm(kin)), kin=kin, table_z=0.0)
    ex = Executor(fake, kin, recorder=RecorderLink("local", root=tmp_path / "rec"), locator=world,
                  table_memory_path=tmp_path / "table.json")
    install(ex)
    return ex, fake


def test_full_task_every_motion_is_a_proposal(kin, tmp_path):
    spec = Spec.load()
    world = StubWorld(spec, [0.26, -0.13])
    ex, fake = make(kin, tmp_path, world)
    rep = Runner(ex, spec).run()
    assert rep["success"], rep["why"]
    assert rep["attempts"] == 1
    names = [s["step"] for s in rep["steps"] if s["skill"]]
    order = ["park", "jaw check: close empty", "jaw check: open", "approach above", "descend (guarded)", "close (verify hold)",
             "test lift", "lift", "carry above bin", "move above bin", "lower into bin (guarded)", "release",
             "retreat (straight up)", "retreat (above park)", "park"]
    assert names == order
    assert all(s["status"] in ("SUCCEEDED",) for s in rep["steps"] if s["skill"])
    accepted = {r.execution_id for r in ex.results.values()}
    assert fake.motion_writes() and all(tag in accepted for _, _, _, tag in fake.motion_writes())   # nothing outside the executor
    assert all(ex.proposals[e].preconditions and ex.proposals[e].source == "script:blue_to_bin" for e in accepted)
    assert np.hypot(*(world.xyz[:2] - spec.bin_xy)) < spec.bin_rim_diameter / 2


def test_missed_grasp_is_retried_and_attempts_are_capped(kin, tmp_path):
    spec = Spec.load()
    ex, _ = make(kin, tmp_path, StubWorld(spec, [0.26, -0.13], misses=1))
    rep = Runner(ex, spec).run()
    assert rep["success"] and rep["attempts"] == 2
    assert any(s["step"].startswith("recover: open (missed)") for s in rep["steps"])

    ex2, _ = make(kin, tmp_path / "b", StubWorld(spec, [0.26, -0.13], misses=5))
    rep2 = Runner(ex2, spec).run()
    assert not rep2["success"] and rep2["attempts"] == spec.max_attempts


def test_dry_run_never_closes_and_pick_run_puts_it_back(kin, tmp_path):
    spec = Spec.load()
    world = StubWorld(spec, [0.26, -0.13])
    ex, _ = make(kin, tmp_path, world)
    rep = Runner(ex, spec, mode="dry", look_s=0.1).run()
    assert rep["stage_ok"], rep["why"]
    closes = [s for s in rep["steps"] if s["skill"] == "gripper" and s["params"].get("open_fraction") == 0.0]
    assert [s["step"] for s in closes] == ["jaw check: close empty"]          # only in the air, at park
    assert not world.held and np.allclose(world.xyz[:2], [0.26, -0.13])
    assert any(s["step"].startswith("dry: lower into bin") and s["status"] == "SUCCEEDED" for s in rep["steps"])

    rep = Runner(ex, spec, mode="pick").run()
    assert rep["stage_ok"] and not world.held, rep["why"]
    assert np.hypot(*(world.xyz[:2] - [0.26, -0.13])) < 0.01 and world.xyz[2] == 0.0


def test_too_wide_for_an_across_grasp_moves_nothing(kin, tmp_path):
    spec = Spec.load(grasp="across")
    ex, fake = make(kin, tmp_path, StubWorld(spec, [0.26, -0.13]))
    rep = Runner(ex, spec).run()
    assert not rep["success"] and "no across grasp" in rep["why"]
    assert not [s for s in rep["steps"] if s["skill"] in ("move_tcp", "descend_guarded")]
