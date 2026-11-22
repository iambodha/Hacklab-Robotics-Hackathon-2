"""perception.latest: one Station subscription per (client, queue), however often frames are grabbed."""
import asyncio
import types

import pytest

from perception import station_io as sio
from perception.latest import Latest


class FakeClient:
    """station_py-like client: follow() starts a task that streams entries into target.put forever."""

    def __init__(self):
        self.follows: list[str] = []
        self.tasks = []

    def follow(self, queue, target):
        self.follows.append(queue)
        errors = asyncio.Queue(maxsize=1)

        async def stream():
            k = 0
            while True:
                k += 1
                await target.put(types.SimpleNamespace(Data=k.to_bytes(4, "little")))
                await asyncio.sleep(0)

        self.tasks.append(asyncio.create_task(stream()))
        return errors


def _fake_usbvideo():
    """Envelope readers that treat the entry bytes themselves as one JPEG frame."""
    class Env:
        def __init__(self, mv):
            self.b = bytes(mv)

        def get_type(self):
            return "frames"

        def get_frames(self):
            b = self.b
            return types.SimpleNamespace(get_frames_data=lambda: [b],
                                         get_format=lambda: types.SimpleNamespace(get_kind=lambda: "jpeg"))
    usbvideo = types.SimpleNamespace(RxEnvelopeReader=Env, RxEnvelopeType=types.SimpleNamespace(ET_FRAMES="frames"))
    fr = types.SimpleNamespace(FrameFormatKind=types.SimpleNamespace(FF_JPEG="jpeg"))
    return None, usbvideo, fr


def test_500_grabs_one_subscription(monkeypatch):
    monkeypatch.setattr(sio, "_import_station", _fake_usbvideo)

    async def go():
        cli = FakeClient()
        seen = []
        for _ in range(500):
            seen += await sio.grab_jpegs(cli, "overview", 1)
        seen += await sio.grab_jpegs(cli, "overview", 5)
        seen += await sio.grab_jpegs(cli, "wrist", 3)
        for t in cli.tasks:
            t.cancel()
        return cli.follows, seen

    follows, seen = asyncio.run(go())
    assert follows == [sio.QUEUES["overview"], sio.QUEUES["wrist"]]
    ov = [int.from_bytes(b, "little") for b in seen[:505]]
    assert all(b > a for a, b in zip(ov, ov[1:]))          # every grab is a newer frame than the last


def test_latest_keeps_only_newest_and_fresh_waits():
    async def go():
        cli = FakeClient()
        sub = Latest.of(cli, "q")
        assert Latest.of(cli, "q") is sub
        first = await sub.get()
        await asyncio.sleep(0.01)                            # many entries arrive meanwhile; only the newest is kept
        newest = await sub.get()
        fresh = await sub.get(fresh=True)
        for t in cli.tasks:
            t.cancel()
        return cli.follows, first, newest, fresh, sub.seq

    follows, first, newest, fresh, seq = asyncio.run(go())
    assert follows == ["q"]
    n = lambda e: int.from_bytes(e.Data, "little")  # noqa: E731
    assert n(newest) > n(first) and n(fresh) > n(newest) and seq >= n(fresh)


def test_ended_subscription_raises_and_does_not_refollow():
    class Dead:
        follows = 0

        def follow(self, queue, target):
            Dead.follows += 1
            q = asyncio.Queue(maxsize=1)
            q.put_nowait(ConnectionError("gone"))
            return q

    async def go():
        cli = Dead()
        for _ in range(3):
            with pytest.raises(RuntimeError, match="ended"):
                await Latest.of(cli, "q").get(timeout=0.05)

    asyncio.run(go())
    assert Dead.follows == 1


def test_arm_mcp_bus_frame_uses_one_subscription(monkeypatch):
    import importlib.util
    import pathlib
    p = pathlib.Path(__file__).resolve().parent.parent / "arm_mcp" / "station.py"
    spec = importlib.util.spec_from_file_location("arm_mcp_station_under_test", p)
    stn = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(stn)
    except (SystemExit, ImportError):
        pytest.skip("norma-core not available")
    assert stn.Latest is Latest                              # shared registry with perception

    class Reader:
        def __init__(self, mv):
            pass

        def get_buses(self):
            bus = types.SimpleNamespace(get_serial_number=lambda: stn.BUS.encode())
            return [types.SimpleNamespace(get_bus=lambda: bus, get_motors=lambda: ["m"])]

    monkeypatch.setattr(stn, "st3215", types.SimpleNamespace(InferenceStateReader=Reader))

    async def go():
        cli = FakeClient()
        for _ in range(200):
            assert await stn.bus_frame(cli) == ["m"]
        for t in cli.tasks:
            t.cancel()
        return cli.follows

    assert asyncio.run(go()) == [stn.ST3215_QUEUE]


def _fake_st3215(serial: str, ids=range(1, 9)):
    """InferenceStateReader whose motors all read the entry counter as their position."""
    import struct

    class Reader:
        def __init__(self, mv):
            self.k = int.from_bytes(bytes(mv), "little")

        def get_buses(self):
            s = bytearray(0x3A)
            struct.pack_into("<H", s, 0x38, 1000 + self.k % 2000)
            motors = [types.SimpleNamespace(get_id=lambda i=i: i, get_state=lambda: bytes(s),
                                            get_range_min=lambda: 900, get_range_max=lambda: 3100) for i in ids]
            bus = types.SimpleNamespace(get_serial_number=lambda: serial.encode(), get_port_name=lambda: b"tty")
            return [types.SimpleNamespace(get_bus=lambda: bus, get_motors=lambda: motors)]
    return types.SimpleNamespace(InferenceStateReader=Reader)


def _sim_station_io():
    import importlib.util
    import pathlib
    p = pathlib.Path(__file__).resolve().parent.parent / "sim" / "station_io.py"
    spec = importlib.util.spec_from_file_location("sim_station_io_under_test", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_sim_station_io_one_subscription_per_queue(monkeypatch):
    s = _sim_station_io()
    assert s.Latest is Latest                                # shared registry with perception
    _, usbvideo, fr = _fake_usbvideo()
    monkeypatch.setattr(s, "_mods", lambda: (None, usbvideo, fr, _fake_st3215(s.BUS)))

    async def go():
        cli = FakeClient()
        jf, jf2 = s.JointFollower(cli), s.JointFollower(cli)
        ticks = [(await jf.read())[0][1] for _ in range(100)]
        assert (await jf2.read())[0][1] >= ticks[-1]          # a second reader on the same client shares the stream
        for _ in range(50):
            await s.grab_jpegs(cli, "overview")
        await s.grab_jpegs(cli, "wrist", 3)
        for t in cli.tasks:
            t.cancel()
        return cli.follows, ticks

    follows, ticks = asyncio.run(go())
    assert sorted(follows) == sorted([s.ST3215_QUEUE, s.QUEUES["overview"], s.QUEUES["wrist"]])
    assert all(b != a for a, b in zip(ticks, ticks[1:]))     # each read is a newer bus frame


def test_station_lib_reads_use_one_subscription(monkeypatch):
    from robot_mcp import station_lib as sl
    monkeypatch.setattr(sl, "_norma", lambda: (None, _fake_st3215("BUS1"), None, None, None))

    async def go():
        cli = FakeClient()
        assert [b["serial"] for b in await sl.list_buses(cli)] == ["BUS1"]
        for _ in range(100):
            joints, ranges, raws = await sl.current_st3215(cli, "BUS1")
        for t in cli.tasks:
            t.cancel()
        return cli.follows, joints, ranges

    follows, joints, ranges = asyncio.run(go())
    assert follows == [sl.ST3215_QUEUE]
    assert len(joints) == 8 and ranges[0] == (900, 3100)
