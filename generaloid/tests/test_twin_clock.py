import threading
import types

from generaloid.twin import generic


def test_servo_protection_accumulates_short_physics_batches(monkeypatch):
    twin = generic.GenericTwin.__new__(generic.GenericTwin)
    twin.model = types.SimpleNamespace(opt=types.SimpleNamespace(timestep=0.002))
    twin.data = types.SimpleNamespace(time=0.0)
    twin.lock = threading.RLock()
    twin.running = True
    calls = []
    twin._servo_protection = calls.append
    clock = iter([0.0, 0.003])
    monkeypatch.setattr(generic.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(generic.time, "sleep", lambda _: None)

    def step(model, data):
        data.time += model.opt.timestep
        twin.running = False

    monkeypatch.setattr(generic.mujoco, "mj_step", step)
    twin._physics()
    assert calls == [0.004]
