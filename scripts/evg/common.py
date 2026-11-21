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


# Camera stream ids seen on the original hackathon setup (used when nothing else is known).
DEFAULT_CAMERAS = {"over": "usbvideo/5cd55831e21692d4f05fcaec75c351a5", "grip": "usbvideo/e091eeda8dcda172482f039057691cdb"}


def detected_cameras():
    """Camera stream ids the LOCAL Station has created (one folder per camera in its data dir, plus any Station logs)."""
    import re
    found = {p.name for p in (ROOT / "station-run" / "station_data").glob("*/usbvideo/*") if p.is_dir()}
    for log in (ROOT / "station-run" / "station.log", ROOT / "station-run" / "stdout.log", RUN / "station.log"):
        try: found |= set(re.findall(r"usbvideo/([0-9a-f]{32})", log.read_text(errors="replace")))
        except OSError: pass
    return sorted(found)


def camera_queues():
    """{'over': 'usbvideo/<id>', 'grip': ...}. robot.json 'cameras' may hold full ids or prefixes (overview/gripper);
    prefixes are matched against the cameras the local Station has seen. Env EVG_CAM_OVERVIEW / EVG_CAM_GRIPPER override."""
    cfg = (read_json(EVG / "robot.json", {}) or {}).get("cameras", {})
    seen = detected_cameras()
    out = {}
    for name, key, env in (("over", "overview", "EVG_CAM_OVERVIEW"), ("grip", "gripper", "EVG_CAM_GRIPPER")):
        want = os.environ.get(env) or cfg.get(key) or ""
        want = want.removeprefix("usbvideo/")
        full = next((h for h in seen if want and h.startswith(want)), None)
        if not full and len(want) == 32: full = want
        out[name] = f"usbvideo/{full}" if full else DEFAULT_CAMERAS[name]
    return out


def apply_robot_env():
    """Export STATION_HOST and the camera ids so every script we spawn (arm.py, teleop.py, record_demo.py, the agent) agrees."""
    os.environ["STATION_HOST"] = station_host()
    cams = camera_queues()
    os.environ["EVG_CAM_OVERVIEW"], os.environ["EVG_CAM_GRIPPER"] = cams["over"], cams["grip"]
    return os.environ["STATION_HOST"]
