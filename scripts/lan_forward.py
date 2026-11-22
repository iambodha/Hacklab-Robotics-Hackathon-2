"""Expose Station (which listens on 127.0.0.1 only) on this laptop's network address, without
restarting it. Forwards <lan-ip>:8888 (API) and <lan-ip>:8889 (web UI) to 127.0.0.1.

  python3 scripts/lan_forward.py            # forward 8888 and 8889 on the current Wi-Fi address
  python3 scripts/lan_forward.py 8889       # web UI only

WARNING: Station has NO authentication. Anyone on the same network can move the arm, change or
wipe calibration, start teleop and watch the cameras. Only run this on a network you trust
(e.g. your team's phone hotspot), and stop it (Ctrl-C / kill) when you're done.
If the laptop changes network, restart this script: it binds the address it found at startup.
"""
import asyncio
import socket
import sys


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))          # no packets are sent
        return s.getsockname()[0]
    finally:
        s.close()


async def pipe(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, OSError):
        pass
    finally:
        writer.close()


async def forward(lan: str, port: int):
    async def handle(cr, cw):
        peer = cw.get_extra_info("peername")
        try:
            sr, sw = await asyncio.open_connection("127.0.0.1", port)
        except OSError:
            cw.close()
            return
        print(f"{peer[0]} -> :{port}", flush=True)
        await asyncio.gather(pipe(cr, sw), pipe(sr, cw))

    server = await asyncio.start_server(handle, lan, port)
    print(f"forwarding {lan}:{port} -> 127.0.0.1:{port}", flush=True)
    await server.serve_forever()


async def main(ports):
    lan = lan_ip()
    if lan.startswith("127."):
        sys.exit("no network address found")
    print(f"Station web UI for others: http://{lan}:8889   API: {lan}:8888", flush=True)
    await asyncio.gather(*(forward(lan, p) for p in ports))


if __name__ == "__main__":
    asyncio.run(main([int(p) for p in sys.argv[1:]] or [8888, 8889]))
