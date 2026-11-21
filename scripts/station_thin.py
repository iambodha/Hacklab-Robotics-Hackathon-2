"""Thinned camera streams from Station.

client.follow() asks the server for EVERY entry (cameras run at 30 fps). All streams share one TCP connection with the joint-state
stream, and measured against a shared Station this cut the state rate by ~75% and caused multi-second state gaps. Here the SERVER only
sends every `step`-th entry (read_from_tail's step), which brought the state rate back to the state-only level.
"""
import asyncio


def follow_thin(client, queue_id, target, step=10, buf=8):
    """Like client.follow(queue_id, target) but only every `step`-th entry arrives. Returns the pump task (cancel it to stop)."""
    qr = client.read_from_tail(queue_id, b"\x00", 0, step, buf)

    async def pump():
        while True:
            e = await qr.data.get()
            if e is None:
                return
            await target.put(e)
    return asyncio.create_task(pump())
