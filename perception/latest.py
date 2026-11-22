"""One long-lived Station subscription per (client, queue), read as "the newest entry".

station_py has no unfollow: every `client.follow()` streams into its target forever. Following per frame or per
request piles up subscriptions on the connection until Station stalls. `Latest.of(cli, queue)` follows a queue
once per client (never again, even after errors) and keeps only the newest entry, so memory stays flat however
rarely it is read. Pure asyncio: imported by perception and arm_mcp (always as `perception.latest`, so both share
one registry).

    sub = Latest.of(cli, "st3215/inference")
    e = await sub.get()               # newest entry; waits for the first one
    e = await sub.get(fresh=True)     # an entry that arrived after this call
"""
from __future__ import annotations

import asyncio
import weakref


class Latest:
    _subs: "weakref.WeakKeyDictionary[object, dict[str, Latest]]" = weakref.WeakKeyDictionary()

    def __init__(self, cli, queue: str):
        self.queue = queue
        self.entry = None
        self.seq = 0                          # entries received so far
        self.error = None
        self._new = asyncio.Event()
        self.errors = cli.follow(queue, self)  # station_py only ever calls `await target.put(entry)`

    @classmethod
    def of(cls, cli, queue: str) -> "Latest":
        subs = cls._subs.setdefault(cli, {})
        if queue not in subs:
            subs[queue] = cls(cli, queue)
        return subs[queue]

    async def put(self, entry) -> None:
        self.entry, self.seq = entry, self.seq + 1
        ev, self._new = self._new, asyncio.Event()
        ev.set()

    def _check(self) -> None:
        if self.error is None and self.errors is not None and not self.errors.empty():
            self.error = self.errors.get_nowait()
        if self.error is not None:
            raise RuntimeError(f"Station subscription to {self.queue} ended: {self.error!r}")

    async def get(self, fresh: bool = False, timeout: float = 5.0):
        """The newest entry. fresh=True (or nothing received yet): wait for one that arrives after this call."""
        self._check()
        if fresh or self.entry is None:
            try:
                await asyncio.wait_for(self._new.wait(), timeout)
            except asyncio.TimeoutError:
                self._check()
                raise TimeoutError(f"no entry on {self.queue} within {timeout} s") from None
        return self.entry
