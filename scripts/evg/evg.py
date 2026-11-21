#!/usr/bin/env python3
"""One entry point for the whole flow.

  python3 scripts/evg/evg.py            console: task -> demo (use / record) -> model -> run with keys
  python3 scripts/evg/evg.py --web      same flow in the browser (http://127.0.0.1:8891, token in the printed URL)

Keys while recording: SPACE start/stop, then ENTER keep, SPACE re-record, A keep + record another, D discard.
Keys while running:   SPACE or X stop, P pause/resume, Q stop and quit.
"""
import argparse, os, select, sys, termios, time, tty
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from agent import providers
from session import Session

B, DIM, RED, GRN, YEL, RST = "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[33m", "\033[0m"


class Keys:
    """Single-key input. Use `with keys:` around key reads; line prompts use plain input()."""
    def __enter__(self):
        self.fd = sys.stdin.fileno(); self.old = termios.tcgetattr(self.fd); tty.setcbreak(self.fd); return self
    def __exit__(self, *a): termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
    def get(self, timeout=0.3):
        if select.select([sys.stdin], [], [], timeout)[0]:
            return sys.stdin.read(1)
        return None


keys = Keys()


def key(prompt, allowed, timeout=None):
    print(prompt, end="", flush=True)
    with keys:
        while True:
            k = keys.get(0.3)
            if k is not None and k.lower() in allowed:
                print(); return k.lower()
            if k in ("\x03",): raise KeyboardInterrupt


def ask(prompt, default=""):
    s = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip()
    return s or default


def step_task(s):
    cur, text = s.task(), ""
    if cur:
        print(f"\n{B}Task{RST}: {cur[:300]}{'…' if len(cur) > 300 else ''}")
        text = input("Enter = keep, or type a new description / /path/to/file.txt: ").strip()
        if not text:
            return
    while True:
        text = text or input(f"\n{B}Describe the task{RST} (text, or /path/to/file.txt): ").strip()
        try:
            t = s.set_task(text); print(f"{DIM}task set ({len(t)} chars){RST}"); return
        except (FileNotFoundError, ValueError) as e:
            print(f"{RED}{e}{RST}"); text = ""


def record_flow(s):
    while True:
        print(f"\n{B}Record a demo{RST}: use the leader arm to do the task once (e.g. put one item in its box).")
        print("The follower mirrors your leader arm. SPACE to start recording, Q to cancel.")
        if key("> ", {" ", "q"}) == "q": return
        try:
            name, tele = s.rec_start()
        except RuntimeError as e:
            print(f"{RED}{e}{RST}"); return
        print(f"{RED}● RECORDING{RST} {name}  ({tele}).  SPACE to stop.")
        with keys:
            while True:
                print(f"\r  {s.rec_elapsed():5.1f}s ", end="", flush=True)
                k = keys.get(0.2)
                if k == " ": break
                if s.rec["proc"].poll() is not None: break
        print()
        try:
            name, n = s.rec_stop()
        except RuntimeError as e:
            print(f"{RED}{e}{RST}"); return
        print(f"stopped: {n} samples.  {B}ENTER{RST} keep · {B}SPACE{RST} re-record · {B}A{RST} keep and record another · {B}D{RST} discard")
        k = key("> ", {"\n", "\r", " ", "a", "d"})
        if k == " ":
            s.rec_discard(); continue
        if k == "d":
            s.rec_discard(); return
        desc = ask("What does this demo show? (e.g. 'garbage -> black box')")
        name, n, kept = s.rec_keep(desc)
        print(f"{GRN}kept{RST} {name}: {n} samples -> {kept} after removing pauses")
        if k != "a": return


def step_demo(s):
    while True:
        ds = s.demos()
        print(f"\n{B}Demos{RST}")
        for i, d in enumerate(ds, 1):
            print(f" {'*' if d['selected'] else ' '} {i}. {d['name']}  {d['seconds']}s  {d['description'] or DIM + '(no description)' + RST}")
        if not ds: print(f" {DIM}none yet{RST}")
        print("Type numbers to use (e.g. 1,2), R = record a new one, Enter = continue with the * ones")
        a = input("> ").strip().lower()
        if a == "r": record_flow(s); continue
        if a:
            try: s.select_demos([ds[int(x) - 1]["name"] for x in a.replace(" ", "").split(",") if x])
            except (ValueError, IndexError): print(f"{RED}bad selection{RST}"); continue
        if s.cfg["demos"]: return
        print(f"{YEL}pick or record at least one demo{RST}")


def step_model(s):
    names = list(providers.PRESETS)
    print(f"\n{B}Model{RST}  (subscription = `claude -p`, no API key)")
    for i, n in enumerate(names, 1):
        p = providers.PRESETS[n]; print(f" {i}. {n:15} {p['provider']:10} {p['model']}")
    cur = s.cfg["preset"]
    a = ask("Choose number", str(names.index(cur) + 1) if cur in names else "1")
    try: preset = names[int(a) - 1]
    except (ValueError, IndexError): preset = cur
    cfg = providers.PRESETS[preset]; model = base = env = ""
    if cfg["provider"] == "openai":
        model = ask("Model name", s.cfg["model"] or cfg["model"]); base = ask("Base URL", s.cfg["base_url"] or cfg["base_url"])
        env = ask("Env var holding the API key (blank for local)", s.cfg["api_key_env"] or ("OPENAI_API_KEY" if "openai.com" in base else ""))
        if env and not os.environ.get(env): print(f"{YEL}warning: {env} is not set in this shell{RST}")
    elif cfg["provider"] == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print(f"{YEL}warning: ANTHROPIC_API_KEY is not set in this shell{RST}")
    ch = ask("Actions per model call (1 = look after every action; 2-4 = plan ahead, cheaper and faster, less closed-loop)", str(s.cfg.get("chunk", 1)))
    s.set_cfg(preset=preset, model=model, base_url=base, api_key_env=env, chunk=max(1, min(int(ch), 5)) if ch.isdigit() else 1)


def tail_log(pos):
    try:
        with open(C.LOG, "rb") as f:
            f.seek(pos); data = f.read(); return pos + len(data), data.decode(errors="replace")
    except FileNotFoundError:
        return pos, ""


def run_live(s, dry=False, replay=None):
    pos = C.LOG.stat().st_size if C.LOG.exists() else 0
    try:
        pid = s.run_start(dry_run=dry, replay=replay)
    except RuntimeError as e:
        print(f"{RED}{e}{RST}"); return False
    print(f"{GRN}started{RST} pid {pid} ({'dry-run' if dry else 'replay' if replay else 'LIVE'}).  "
          f"{B}SPACE/X{RST} stop · {B}P{RST} pause/resume · {B}Q{RST} stop+quit")
    quit_after = False
    with keys:
        while s.running():
            pos, txt = tail_log(pos)
            if txt: print(txt, end="", flush=True)
            k = keys.get(0.3)
            if k is None: continue
            k = k.lower()
            if k in (" ", "x", "q"):
                how, hold = s.run_stop(); print(f"{RED}STOPPED{RST} ({how}) {hold or ''}"); quit_after = k == "q"; break
            if k == "p": s.pause(not C.PAUSE.exists()); print("paused" if C.PAUSE.exists() else "resumed")
    pos, txt = tail_log(pos); print(txt, end="")
    print(f"{DIM}run ended: {s.status().get('phase')} {s.status().get('result') or ''}{RST}")
    if quit_after: raise SystemExit(0)
    return True


def console():
    s = Session()
    print(f"{B}evg: robot agent{RST}   Station: {GRN + 'up' + RST if s.station_up() else RED + 'DOWN' + RST}")
    if not s.station_up():
        if input("Start Station now? (y/n) ").strip().lower() == "y":
            print("starting..."); print("up" if s.start_station() else f"{RED}failed, see evg-run/station.log{RST}")
    step_task(s); step_demo(s); step_model(s)
    while True:
        print(f"\n{B}Ready{RST}: model {s.cfg['preset']}, demos {len(s.cfg['demos'])}")
        print(f" {B}ENTER{RST} start LIVE · {B}D{RST} dry-run (cameras only, never moves) · {B}R{RST} replay test (recorded demo as fake robot) · {B}T{RST} task · {B}M{RST} demos · {B}O{RST} model · {B}Q{RST} quit")
        k = key("> ", {"\n", "\r", "d", "r", "t", "m", "o", "q"})
        if k == "q": return
        if k == "t": step_task(s)
        elif k == "m": step_demo(s)
        elif k == "o": step_model(s)
        elif k == "d": run_live(s, dry=True)
        elif k == "r": run_live(s, replay=s.cfg["demos"][0])
        else:
            print(f"{YEL}The arm will move.{RST} Teleop is switched off first. Someone should be ready at the table. ENTER to confirm, any other key cancels.")
            if key("> ", {"\n", "\r", "n", " ", "c"}) in ("\n", "\r"): run_live(s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--web", action="store_true"); ap.add_argument("--port", type=int, default=8891)
    a = ap.parse_args()
    try:
        if a.web:
            import web; web.serve(Session(), a.port)
        else:
            console()
    except (KeyboardInterrupt, EOFError):
        print("\nbye (a running agent keeps running; use ctl.py stop)")
