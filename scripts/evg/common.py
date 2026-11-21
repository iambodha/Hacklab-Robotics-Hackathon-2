"""Shared paths and small helpers for the evg tools."""
import json, os, time
from pathlib import Path

EVG = Path(__file__).resolve().parent
SCRIPTS = EVG.parent
ROOT = SCRIPTS.parent
RUN = ROOT / "evg-run"
DEMOS = ROOT / "demos"
STOP, PAUSE, APPROVE, REJECT = (RUN / n for n in ("STOP", "PAUSE", "APPROVE", "REJECT"))
PID, STATUS, LOG = RUN / "agent.pid", RUN / "status.json", RUN / "agent.log"
TASK, CONFIG = RUN / "task.txt", RUN / "config.json"


def ensure_run():
    RUN.mkdir(exist_ok=True)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, ValueError):
        return default


def write_json(path, obj):
    """Atomic write so readers never see a half-written file."""
    ensure_run()
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def now():
    return time.strftime("%H:%M:%S")


def station_host():
    """Host of the Station API: env STATION_HOST, else robot.json 'station_host', else localhost."""
    return os.environ.get("STATION_HOST") or (read_json(EVG / "robot.json", {}) or {}).get("station_host") or "localhost"


def apply_robot_env():
    """Export STATION_HOST so every script we spawn (arm.py, teleop.py, record_demo.py, the agent) talks to the same Station."""
    os.environ["STATION_HOST"] = station_host()
    return os.environ["STATION_HOST"]
