"""Make Station's web UI usable behind an HTTPS tunnel (Cloudflare, ngrok). Local only: 127.0.0.1:8893.

Station's page opens its live-data WebSocket as `ws://<host>/api`. Browsers block plain ws:// on an https:// page,
so through an HTTPS tunnel the page loads but shows nothing. This relay sits in front of Station's web port and
changes that one address in the served JavaScript to use wss:// when the page itself is https. Everything else
(including the WebSocket) is passed through untouched. Station is not modified or restarted.

  python3 scripts/station_https_relay.py                 # 127.0.0.1:8893 -> Station web UI at 127.0.0.1:8889
  cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8893

Station's UI has no authentication and includes teleop and calibration: whoever has the tunnel URL can move the arm.
"""
import asyncio
import gzip
import re
import sys
import urllib.request

UP_HOST, UP_PORT = "127.0.0.1", int(sys.argv[2]) if len(sys.argv) > 2 else 8889
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8893
PATCH = (b"`ws://${", b'`${location.protocol==="https:"?"wss":"ws"}://${')


async def pipe(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, OSError):
        pass
    finally:
        writer.close()


def fetch_patched(path: str) -> bytes:
    req = urllib.request.Request(f"http://{UP_HOST}:{UP_PORT}{path}", headers={"Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=20) as r:
        body = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
    return body.replace(*PATCH)


async def handle(cr, cw):
    try:
        head = await asyncio.wait_for(cr.readuntil(b"\r\n\r\n"), 15)
    except Exception:
        cw.close()
        return
    first = head.split(b"\r\n", 1)[0].decode("latin1")
    parts = first.split(" ")
    path = parts[1] if len(parts) > 1 else "/"
    upgrade = re.search(rb"(?im)^upgrade:\s*websocket", head) is not None
    if not upgrade and parts[0] == "GET" and re.search(r"\.js(\?|$)", path):
        try:
            body = await asyncio.to_thread(fetch_patched, path)
            cw.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/javascript\r\nCache-Control: no-store\r\n"
                     b"Content-Length: " + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
            await cw.drain()
        except Exception as e:
            cw.write(b"HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\nContent-Length: 0\r\n\r\n")
            print("upstream error", path, e, flush=True)
        cw.close()
        return
    try:
        sr, sw = await asyncio.open_connection(UP_HOST, UP_PORT)
    except OSError:
        cw.close()
        return
    if not upgrade:      # one request per connection, so every request passes through this handler
        head = re.sub(rb"(?im)^connection:.*\r\n", b"", head).replace(b"\r\n\r\n", b"\r\nConnection: close\r\n\r\n", 1)
    sw.write(head)
    await sw.drain()
    await asyncio.gather(pipe(cr, sw), pipe(sr, cw))


async def main():
    server = await asyncio.start_server(handle, "127.0.0.1", PORT)
    print(f"Station HTTPS relay: http://127.0.0.1:{PORT} -> http://{UP_HOST}:{UP_PORT}", flush=True)
    await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
