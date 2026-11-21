"""Everything the console UI and the web UI both do: task, demos (record/keep/discard), provider choice, run control.
State lives in evg-run/ files, so either UI (or ctl.py) can see and drive the same session."""
import json, os, shutil, signal, socket, subprocess, sys, threading, time
import common as C
import demo_filter
from agent import mcp_client, providers

PY = sys.executable


class Session:
    def __init__(self):
        C.ensure_run(); C.apply_robot_env()
        self._lock = threading.RLock()
        self.busy = None           # label of the background operation in progress (UI shows it with a spinner)
        self.error = None          # last background error, shown until the next action
        self.cancel = False        # STOP pressed while a run was still starting
        self._up = (0.0, False)    # cached robot reachability (checked at most every 5 s)
        self.starting = False      # a recording is being started (teleop takes a few seconds); blocks a second start
        self.rec = None            # dict(name, proc, t0) while recording
        self.review = None         # name of a finished take waiting for keep/discard
        self.cfg = {"preset": "claude-haiku", "max_ticks": 80, "chunk": 1, "effort": "low", "demos": [], "base_url": "", "model": "", "api_key_env": ""}
        self.cfg.update(C.read_json(C.CONFIG, {}))
        self.cfg.pop("step_mode", None)                       # approval mode removed: every action runs
        self.cfg["demos"] = list(dict.fromkeys(self.cfg.get("demos", [])))

    # ---- background operations (the web UI must never block on a slow step)
    def bg(self, label, fn, *args, **kw):
        """Run fn in a thread; the UI sees `busy` until it ends and `error` if it fails. One at a time (STOP is separate)."""
        with self._lock:
            if self.busy: raise RuntimeError(f"still busy: {self.busy}")
            self.busy, self.error = label, None
        def work():
            try: fn(*args, **kw)
            except Exception as e: self.error = f"{label.rstrip('…')} failed: {e}"
            finally: self.busy = None
        threading.Thread(target=work, daemon=True).start()

    def step(self, label):
        """Update the progress text of the running background operation."""
        if self.busy: self.busy = label

    # ---- config / task
    def save(self): C.write_json(C.CONFIG, self.cfg)

    def set_cfg(self, **kw):
        self.cfg.update({k: v for k, v in kw.items() if v is not None}); self.save()

    def set_task(self, text_or_path):
        """Text, or a path (starts with / ~ or ./ and exists) to a text file."""
        s = text_or_path.strip()
        if s[:1] in "/~." and "\n" not in s:
            p = os.path.expanduser(s)
            if os.path.isfile(p):
                s = open(p).read().strip()
            else:
                raise FileNotFoundError(f"no such file: {p}")
        if not s:
            raise ValueError("task is empty")
        C.TASK.write_text(s); return s

    def task(self):
        try: return C.TASK.read_text()
        except FileNotFoundError: return ""

    # ---- station
    @staticmethod
    def station_up():
        try:
            with socket.create_connection((C.station_host(), 8888), timeout=1.0): return True
        except OSError:
            return False

    @staticmethod
    def backend(): return mcp_client.backend()

    def robot_up(self, max_age=5.0):
        """Is the arm reachable? MCP backend: the server's /health; Station backend: its TCP API. Cached so polling adds no load."""
        t, up = self._up
        if time.time() - t > max_age:
            up = mcp_client.health() if self.backend() == "mcp" else self.station_up()
            self._up = (time.time(), up)
        return up

    def start_station(self):
        if self.backend() == "mcp": raise RuntimeError("backend is MCP: there is no local Station to start (set EVG_BACKEND=station to use one)")
        if C.station_host() not in ("localhost", "127.0.0.1"): raise RuntimeError(f"Station runs on {C.station_host()}, not here; start it on that machine")
        if self.station_up(): return True
        subprocess.Popen(["/bin/zsh", str(C.SCRIPTS / "run-station.sh")], stdout=open(C.RUN / "station.log", "ab"), stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(40):
            time.sleep(0.5)
            if self.station_up(): return True
        return False

    def _script(self, name, *args, timeout=30):
        r = subprocess.run([PY, str(C.SCRIPTS / name), *args], capture_output=True, text=True, timeout=timeout, cwd=C.ROOT)
        return (r.stdout + r.stderr).strip()

    # ---- demos
    def demos(self):
        out = []
        if not C.DEMOS.is_dir(): return out
        for d in sorted(p for p in C.DEMOS.iterdir() if p.is_dir() and not p.name.startswith(".") and (p / "states.jsonl").exists()):
            meta = C.read_json(d / "meta.json", {})
            try: rows = [l for l in (d / "states.jsonl").read_text().splitlines() if l.strip()]; dur = json.loads(rows[-1])["t"] if rows else 0
            except (ValueError, OSError): rows, dur = [], 0
            out.append(dict(name=d.name, samples=len(rows), seconds=round(dur), description=meta.get("description", ""), selected=d.name in self.cfg["demos"]))
        return out

    def select_demos(self, names):
        self.set_cfg(demos=list(dict.fromkeys(n for n in names if (C.DEMOS / n / "states.jsonl").exists())))

    @staticmethod
    def record_procs():
        """[(pid, take_name)] of every record_demo.py process on this machine, tracked by us or not."""
        out = subprocess.run(["ps", "-eo", "pid,command"], capture_output=True, text=True).stdout
        procs = []
        for line in out.splitlines():
            if "record_demo.py" in line and "grep" not in line:
                parts = line.split()
                try: procs.append((int(parts[0]), parts[-1]))
                except (ValueError, IndexError): pass
        return procs

    def stop_orphans(self):
        """Stop recorders we are not tracking (e.g. left behind by a double click or a crashed UI). Returns their take names."""
        mine = self.rec["proc"].pid if self.rec else None
        found = [(pid, name) for pid, name in self.record_procs() if pid != mine]
        for pid, name in found:
            (C.DEMOS / name).mkdir(parents=True, exist_ok=True); (C.DEMOS / name / "STOP").touch()
        end = time.time() + 10
        while time.time() < end and any(pid in [p for p, _ in self.record_procs()] for pid, _ in found): time.sleep(0.3)
        for pid, name in found:
            if pid in [p for p, _ in self.record_procs()]:
                try: os.kill(pid, signal.SIGTERM)
                except ProcessLookupError: pass
            (C.DEMOS / name / "STOP").unlink(missing_ok=True)
        return [name for _, name in found]

    def rec_start(self):
        """Station + teleop on (the operator moves the leader arm; the follower mirrors; we record the follower)."""
        with self._lock:                     # check-and-set atomically: a second click during the slow teleop start must not start a second recorder
            if self.rec or self.starting: raise RuntimeError("already recording" if self.rec else "recording is already starting")
            if self.running(): raise RuntimeError("the agent is running; stop it first")
            self.starting = True
        try:
            if self.backend() == "mcp": raise RuntimeError("recording needs a leader arm on a local Station; not available over the MCP backend. Pick an existing demo.")
            if not self.station_up(): raise RuntimeError("Station is not running (scripts/run-station.sh)")
            stale = self.stop_orphans()
            out = self._script("teleop.py", "start")
            name = "take-" + time.strftime("%Y%m%d-%H%M%S")
            proc = subprocess.Popen([PY, str(C.SCRIPTS / "record_demo.py"), "900", name], cwd=C.ROOT,
                                    stdout=open(C.RUN / "record.log", "ab"), stderr=subprocess.STDOUT, start_new_session=True)
            with self._lock:
                self.rec = dict(name=name, proc=proc, t0=time.time()); self.review = None
            return name, (out.splitlines()[-1] if out else "") + (f"  (stopped stray recorder(s): {', '.join(stale)})" if stale else "")
        finally:
            self.starting = False

    def rec_stop(self):
        if self.starting: raise RuntimeError("recording is still starting; wait a few seconds")
        if not self.rec: raise RuntimeError("not recording")
        name, proc = self.rec["name"], self.rec["proc"]
        (C.DEMOS / name).mkdir(parents=True, exist_ok=True); (C.DEMOS / name / "STOP").touch()
        try: proc.wait(timeout=25)
        except subprocess.TimeoutExpired: proc.kill()
        (C.DEMOS / name / "STOP").unlink(missing_ok=True)
        self.rec, self.review = None, name
        f = C.DEMOS / name / "states.jsonl"
        n = sum(1 for _ in open(f)) if f.exists() else 0
        if n == 0:
            self.review = None
            raise RuntimeError(f"recording produced no samples; see {C.RUN / 'record.log'}")
        return name, n

    def rec_keep(self, description=""):
        name = self.review
        if not name: raise RuntimeError("nothing to keep")
        n, k, _ = demo_filter.run(name)
        C.write_json(C.DEMOS / name / "meta.json", dict(description=description.strip(), created=time.time(), samples=n, kept=k))
        self.cfg["demos"] = list(dict.fromkeys([*self.cfg["demos"], name])); self.save(); self.review = None
        return name, n, k

    def rec_discard(self):
        name = self.review
        if not name: return
        trash = C.DEMOS / ".trash"; trash.mkdir(exist_ok=True)
        shutil.move(str(C.DEMOS / name), str(trash / f"{name}-{int(time.time())}")); self.review = None

    def rec_elapsed(self): return time.time() - self.rec["t0"] if self.rec else 0

    # ---- run control
    @staticmethod
    def pid():
        try:
            pid = int(C.PID.read_text()); os.kill(pid, 0); return pid
        except (FileNotFoundError, ValueError, ProcessLookupError): return None
        except PermissionError: return int(C.PID.read_text())

    def running(self): return self.pid() is not None

    def run_start(self, dry_run=False, replay=None):
        if self.running(): raise RuntimeError("agent already running")
        if self.rec or self.starting: raise RuntimeError("a recording is in progress; stop it first")
        task = self.task()
        if not task: raise RuntimeError("no task set")
        if not self.cfg["demos"] and not (replay or dry_run): raise RuntimeError("no demo selected (a live run needs one: it sets the floor and joint limits)")
        self.cancel = False
        self.step("Starting: checking the arm is reachable…")
        if not replay and not self.robot_up(max_age=0): raise RuntimeError("the arm is not reachable" + (" (MCP server /health failed)" if self.backend() == "mcp" else f" (Station at {C.station_host()}:8888)"))
        if not (dry_run or replay) and self.backend() != "mcp":
            self.step("Starting: switching teleop off…"); self._script("teleop.py", "stop")          # mirroring would fight the agent
            if self.cancel: raise RuntimeError("cancelled by STOP")
            self.step("Starting: holding the arm…"); self._script("arm.py", "hold", timeout=40)
        if self.cancel: raise RuntimeError("cancelled by STOP")
        for f in (C.STOP, C.PAUSE, C.APPROVE, C.REJECT, C.STATUS): f.unlink(missing_ok=True)
        c = self.cfg
        args = [PY, str(C.EVG / "agent" / "run.py"), "--task", task, "--demo", ",".join(dict.fromkeys(c["demos"])),
                "--provider-preset", c["preset"], "--max-ticks", str(c["max_ticks"]), "--chunk", str(c.get("chunk", 1)), "--effort", c.get("effort", "low")]
        for flag, key in (("--model", "model"), ("--base-url", "base_url"), ("--api-key-env", "api_key_env")):
            if c.get(key): args += [flag, c[key]]
        if dry_run: args.append("--dry-run")
        if replay: args += ["--replay", replay]
        self.step("Starting: launching the agent…")
        p = subprocess.Popen(args, cwd=C.ROOT, stdout=open(C.LOG, "ab"), stderr=subprocess.STDOUT, start_new_session=True)
        C.PID.write_text(str(p.pid)); return p.pid

    def stop_now(self):
        """STOP button: immediate and never queued behind another operation. Cancels a run that is still starting,
        signals a running agent, then (in the background) waits for it and holds the arm."""
        self.cancel = True
        C.ensure_run(); C.STOP.touch()
        if self.rec:
            threading.Thread(target=lambda: self._safe(self.rec_stop), daemon=True).start()
        threading.Thread(target=lambda: self._safe(self.run_stop), daemon=True).start()

    def _safe(self, fn):
        try: fn()
        except Exception as e: self.error = f"stop: {e}"

    def run_stop(self):
        """STOP file -> wait -> SIGTERM -> SIGKILL; the agent holds the arm in its own finally; hold again here as a backstop."""
        C.STOP.touch(); pid = self.pid(); how = "stop file"
        if pid:
            def dead(t):
                end = time.time() + t
                while time.time() < end:
                    try: os.kill(pid, 0)
                    except ProcessLookupError: return True
                    time.sleep(0.1)
                return False
            if not dead(6):
                os.kill(pid, signal.SIGTERM); how = "SIGTERM"
                if not dead(4): os.kill(pid, signal.SIGKILL); how = "SIGKILL"; dead(2)
        C.PID.unlink(missing_ok=True); C.PAUSE.unlink(missing_ok=True)
        mode = (C.read_json(C.STATUS, {}) or {}).get("mode")
        hold = []
        if mode == "live":
            try:
                hold = [mcp_client.hold_arm()[:120]] if self.backend() == "mcp" else (self._script("arm.py", "hold", timeout=40).splitlines()[:1] if self.station_up() else [])
            except (mcp_client.McpError, subprocess.TimeoutExpired) as e:
                hold = [f"WARNING hold failed: {e}"]
        return how, hold

    def pause(self, on=True):
        if on: C.PAUSE.touch()
        else: C.PAUSE.unlink(missing_ok=True)

    def approve(self, yes=True):
        (C.APPROVE if yes else C.REJECT).touch()

    def status(self):
        st = C.read_json(C.STATUS, {}) or {}
        age = time.time() - st["ts"] if st.get("ts") else None
        return dict(st, running=self.running(), paused=C.PAUSE.exists(), age=age,
                    recording=bool(self.rec), starting=self.starting, stray=[n for p, n in self.record_procs() if not self.rec or p != self.rec['proc'].pid], rec_name=self.rec["name"] if self.rec else None, rec_seconds=round(self.rec_elapsed()),
                    review=self.review, station=self.robot_up(), backend=self.backend(), busy=self.busy, error=self.error, task=self.task(), cfg=self.cfg)

    def log_tail(self, n=25):
        try: return C.LOG.read_text(errors="replace").splitlines()[-n:]
        except FileNotFoundError: return []
