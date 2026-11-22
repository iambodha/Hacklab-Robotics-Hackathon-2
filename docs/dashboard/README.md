# Learning dashboard

`dashboard/` serves the learning inspection UI on :8920: attempt summaries, parameter/config history, phases, traces and optional operator/action logs. Source: `dashboard/__main__.py`, `server.py`, `data.py` and `static/`; tests cover its HTTP/data views. This is separate from the executor console at :8910.

For an offline, empty local store, first install the sim extra and set paths as in [setup](../release/QUICKSTART.md), then:

```sh
STATION_HOST=station-disabled-in-tests.invalid BOTTLE_LEARN_DIR="$PWD/.local/bottle" \
  ACTIONS_LOG="$PWD/.local/actions.jsonl" uv run --extra sim python -m dashboard
```

No real recordings are included. To inspect your own attempt data, point `BOTTLE_LEARN_DIR`, `ACTIONS_LOG`, optional `OPERATOR_LOG` and `SKILLCFG_DIR` at it. Some default paths are local-only; see [inventory](../release/LOCAL_PATHS.md). Optional Gemini judging/advice is not needed to view records. Approval of real changes belongs to the human CLI workflow in [profiles](../learning/PROFILES.md).
