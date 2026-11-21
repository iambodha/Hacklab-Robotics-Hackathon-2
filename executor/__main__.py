"""python -m executor serve [--backend fake|sim|real|replay:<recording>] [--port 8910]"""
import argparse


def main():
    ap = argparse.ArgumentParser(prog="executor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="executor + operator console on http://127.0.0.1:PORT")
    s.add_argument("--backend", default="fake", help="fake | sim | real | replay:<recording dir or id>")
    s.add_argument("--port", type=int, default=8910)
    a = ap.parse_args()
    if a.cmd == "serve":
        from .console.server import serve
        serve(a.backend, a.port)


if __name__ == "__main__":
    main()
