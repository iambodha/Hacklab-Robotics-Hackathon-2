# Install and run from a source checkout

Prerequisites: uv, network for the first dependency/model download, and space for a Python environment plus upstream robot assets. No Station process, USB device, LLM login or API key is required for the simulator demo.

```sh
bash quickstart.sh
bash quickstart.sh --connector so101
```

The wrapper disables Station/relay hosts, real motion, recording and cloud judging; it uses isolated learning paths and suppresses model-secret discovery. The default runs a scripted pick through MCP stdio with **physics-only diagnostic images**. Exit zero and `picked: true` mean the simulated cube rose above 50 mm; they do not validate cameras. Each invocation writes its result/truth/stderr into a fresh temporary workdir printed in the JSON. Delete that workdir when no longer needed.

For ordinary rendering, `bash quickstart.sh --rendered`. For an MCP client's server command, use `bash quickstart.sh --serve` with cwd set to the checkout; startup messages go to stderr. Add `--connector so101` for its generic twin. Close stdin / stop the MCP client to end serving. This wrapper only supports sim mode.

The normal server uses actual MuJoCo rendering and never silently replaces camera images. macOS uses CGL; Linux needs system OpenGL libraries and a supported `MUJOCO_GL` backend (OSMesa, EGL or GLFW). Choose the backend before Python starts. The current sandbox cannot open CoreGraphics, so rendered checks must be repeated on an unrestricted desktop. Windows/WSL and Linux are documented paths, not newly tested platforms.

## Python extras

`pyproject.toml` and `uv.lock` define a source-checkout environment. `tool.uv.package=false` is deliberate: legacy script imports and source-relative calibration/scene assets are not ready for standalone wheel distribution. Run commands from the repo root.

| Extra | Includes / purpose |
| --- | --- |
| core | MCP 1.x and httpx; CLI metadata and comms |
| `sim` | NumPy, SciPy, Pillow, MuJoCo, contrib OpenCV (ArUco) |
| `real` | Geometry/perception dependencies; Station runtime/client source is separate. Installing this extra does not enable motion. |
| `gemini` | Optional google-genai SDK; runtime GEMINI_API_KEY or GOOGLE_API_KEY |
| `dev` | pytest and reportlab for kitting print tests; combine with sim/gemini for the full suite |

```sh
uv sync --locked --extra sim --extra dev --extra gemini
export STATION_HOST=station-disabled-in-tests.invalid
export ROBOT_CAM_URL=http://camera-relay-disabled-in-tests.invalid:8890
export EXECUTOR_REAL_MOTION=0
export DATA_ROOT="$PWD/.local/data"
export SKILLCFG_DIR="$DATA_ROOT/skillcfg"
export BOTTLE_LEARN_DIR="$DATA_ROOT/bottle"
export BOTTLE_SIM_LEARN_DIR="$DATA_ROOT/bottle-sim"
export NORMA_CORE_REPO="$PWD/norma-core"
uv run --locked --extra sim --extra dev --extra gemini python -m pytest -q -p no:cacheprovider
```

The safety conftest also disables Station under pytest. Never set ROBOT_LIVE_TESTS during this validation. The full suite intentionally retains known failures: see [validation](VALIDATION.md) and [known issues](../KNOWN_ISSUES.md). For focused offline coverage: append `generaloid/tests generaloid/comms/tests tasks/tests/test_kitting.py learning/tests/test_learned_profile.py`.

## NormaCore is a separate checkout

The hackathon kept it at `repo/norma-core`, beside worktrees, rather than inside this repository. The robot models and meshes are required even for offline simulation; live Station access also needs its Python client. A Station binary/server is required only for hardware I/O. Nothing in quickstart starts that runtime.

Quickstart downloads [NormaCore source at 6b494f77](https://github.com/norma-core/norma-core/tree/6b494f77fac366a878ae71222458b196f8400dee), verifies the archive SHA256, and installs only hardware, SO-101 viewer assets, Station shared client source and upstream license/readme into the ignored `norma-core/` directory. Extraction accepts regular files/directories with Python's data filter; unrelated device OS images and symlinks are excluded. This is an asset snapshot, not a git checkout. Existing directories are never overwritten. Download is ~70 MB; extracted assets are larger. Review upstream licenses before redistributing models.

To maintain your own external checkout, obtain the same full revision from upstream and set `NORMA_CORE_REPO=/absolute/path/to/norma-core`. It must contain `hardware/elrobot/simulation/elrobot_follower.urdf`, meshes and SO-101 assets. A shallow/source-only copy missing hardware is insufficient. Quickstart accepts an existing full checkout without downloading; it does not verify its revision. [LOG.md](../../LOG.md) records historical Station installation; those instructions operate hardware and are separate from this offline quickstart.

Splat reconstruction additionally needs ffmpeg, pycolmap and a supported Brush build; these are not installed by the sim extra. See [the pipeline](../../sim/splat/README.md), set DATA_ROOT to your writable data volume, and do not check generated PLY/video into git. Matplotlib is only needed to regenerate kitting charts, not the simulator or suite.
