# CLI and verification

Run from the repository root. Python commands use `uv run --no-project`. A shell helper keeps the same dependency set as the common test brief:

```sh
export STATION_HOST=station-disabled-in-tests.invalid
export DATA_ROOT='/Volumes/Dell HDD/Robotics-Hackathon'
export NORMA_CORE_REPO=/Users/bodha/Documents/b0dha/Robotics-Hackathon/repo/norma-core
export MUJOCO_GL=cgl  # macOS; use a supported MuJoCo backend on other systems
py() {
  uv run --no-project -q --with pytest --with numpy --with scipy --with pillow \
    --with mujoco --with opencv-contrib-python --with 'mcp>=1.12,<2' --with httpx \
    --with google-genai==3.3.0 python "$@"
}
py -m generaloid connectors
py -m generaloid describe --connector so101
py -m generaloid station                 # settings only; never connects
py -m generaloid modules
py -m generaloid tools --connector elrobot --mode sim --no-addons
py -m generaloid serve --connector elrobot --mode sim
py -m generaloid serve --connector so101 --mode sim
```

`serve` runs MCP stdio, suitable for any MCP client's server configuration. Configure its command as `uv`, arguments as the dependency options above followed by `python -m generaloid serve --connector elrobot --mode sim`, and cwd as the repository root. stdout is reserved for MCP; startup messages go to stderr. No model API key is required. Use `ROBOT_SIM_RECORD=0` to suppress the existing ElRobot session recorder during development.

`--layout home` or a layout file selects an ElRobot scene. SO-101 accepts a JSON file marked `"robot_model": "so101"`; otherwise it uses its bundled layout. Set `GENERALOID_GENERATED` before starting Python to relocate SO-101 generated scene assets. Set `DATA_ROOT` to the external disk for bulky recordings/reconstructions.

Real hardware deployment uses `serve --connector elrobot --mode real` with the configured Station host. **Do not run that during this validation task.** `tools --mode real` only imports and lists tools; tests mount real tools on `FakeStation`. SO-101 real calibration is incomplete. Existing `sim/run_real_server.sh`, `scripts_integrator/real_server_home.sh` and `real_operator/.devin/mcp_config.json` remain untouched and valid.

## Tests and scripted MCP client

```sh
py -m pytest -q -p no:cacheprovider generaloid/tests
nice -n 10 uv run --no-project -q --with numpy --with scipy --with pillow \
  --with mujoco --with opencv-contrib-python --with 'mcp>=1.12,<2' \
  --with httpx --with google-genai==3.3.0 python -m generaloid.checks pick \
  --connector elrobot --workdir "$DATA_ROOT/r5-generaloid/check-pick-elrobot"
# Repeat the client command with pick --connector so101, or twist --connector elrobot.
```

`generaloid.checks` starts the real CLI in a child process, initializes MCP, calls tools and judges the result from a separate truth log. It always selects sim mode, disables Station/relay hosts and keeps learning data isolated in its work directory. `tools` compares public schemas/descriptions, `pick` checks cube centre height > 5 cm, and `twist` checks the simulated cap's released-thread state after ten 90-degree strokes. A failed outcome returns a nonzero process status. A released thread is not proof that the cap was physically carried off the bottle.

The macOS execution sandbox in the takeover run denies CoreGraphics access (`invalid CoreGraphics connection`). For physics-only smoke checks, explicitly set `GENERALOID_CHECK_HEADLESS=1`. This selects `generaloid.tests.run_headless_server`, a sim-only test launcher replacing the renderer with an image labeled **RENDERING DISABLED / Physics-only test**. MCP, IK, physics and truth judging are unchanged. JSON results label the rendering substitution. This mode cannot validate camera appearance, visual perception or image-based judging. Normal `serve` never substitutes images. Run ordinary checks outside that graphics restriction before claiming rendered-camera validation.

The contract suite covers both connectors, FK/IK, metadata and tool support. Other tests cover ElRobot tool parity over stdio, real imports, fake Station reads/torque/bounded writes, Station refusal and host selection, cached connector isolation and optional comms behaviour. See the [journal](JOURNAL.md) and [validation report](VALIDATION.md) for exact counts and baseline failures.
