#!/usr/bin/env bash
# Video of the table -> Gaussian splat in the twin's frame.  See sim/splat/README.md.
#
#   sim/splat/prepare.sh <video> [name]
#
# Everything goes to $DATA_ROOT/splat/<name>/ (default name = video file name without extension):
#   video.<ext>            copy of the input
#   dataset/images/        ffmpeg frames (SPLAT_FPS per second, <= SPLAT_MAX_FRAMES, <= SPLAT_WIDTH px wide)
#   sfm/                   pycolmap database + unaligned global-SfM model (sfm/sparse/0), sfm.json
#   dataset/sparse/0/      the model moved into the twin frame by the ArUco markers (align.json has the transform)
#   brush/                 Brush training output (.ply exports)
# Stages whose output exists are skipped; delete the stage's folder to redo it.
# Env: DATA_ROOT, SPLAT_FPS (2), SPLAT_MAX_FRAMES (300), SPLAT_WIDTH (1600), SPLAT_STEPS (15000), SPLAT_LAYOUT (home),
#      BRUSH (brush_app path), SPLAT_SKIP_TRAIN=1 (stop after alignment).
set -euo pipefail

[ $# -ge 1 ] || { sed -n '2,15p' "$0"; exit 2; }
VIDEO=$1
NAME=${2:-$(basename "${VIDEO%.*}")}
DATA_ROOT=${DATA_ROOT:-"$(cd "$(dirname "$0")/../.." && pwd)/data"}   # set DATA_ROOT for an external drive
OUT="$DATA_ROOT/splat/$NAME"
FPS=${SPLAT_FPS:-2}
MAX_FRAMES=${SPLAT_MAX_FRAMES:-300}
WIDTH=${SPLAT_WIDTH:-1600}
STEPS=${SPLAT_STEPS:-15000}
LAYOUT=${SPLAT_LAYOUT:-home}
BRUSH=${BRUSH:-"$DATA_ROOT/splat/bin/brush-app-aarch64-apple-darwin/brush_app"}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=(uv run --no-project -q --with pycolmap==4.2.1 --with numpy --with scipy --with opencv-contrib-python --with mujoco --with pillow python)
export MUJOCO_GL=${MUJOCO_GL:-cgl}

mkdir -p "$OUT/dataset/images"
EXT=${VIDEO##*.}
[ -f "$OUT/video.$EXT" ] || cp "$VIDEO" "$OUT/video.$EXT"

echo "== 1/4 frames -> $OUT/dataset/images"
if [ -z "$(ls -A "$OUT/dataset/images")" ]; then
  ffmpeg -hide_banner -loglevel error -i "$OUT/video.$EXT" -vf "fps=$FPS,scale='min($WIDTH,iw)':-2" \
    -frames:v "$MAX_FRAMES" -q:v 2 "$OUT/dataset/images/f%04d.jpg"
fi
echo "   $(ls "$OUT/dataset/images" | wc -l | tr -d ' ') frames"

echo "== 2/4 poses (SIFT + matching + GLOMAP global mapper) -> $OUT/sfm"
[ -f "$OUT/sfm/sparse/0/images.bin" ] || "${PY[@]}" "$HERE/sfm.py" "$OUT"

echo "== 3/4 align to the twin frame with ArUco markers -> $OUT/dataset/sparse/0"
[ -f "$OUT/dataset/sparse/0/images.bin" ] || "${PY[@]}" "$HERE/align_markers.py" "$OUT" --layout "$LAYOUT"

[ "${SPLAT_SKIP_TRAIN:-0}" = 1 ] && { echo "SPLAT_SKIP_TRAIN=1: done before training"; exit 0; }
echo "== 4/4 train with Brush (Metal) -> $OUT/brush"
[ -x "$BRUSH" ] || { echo "Brush not found at $BRUSH (see README: install)"; exit 1; }
mkdir -p "$OUT/brush"
dot_clean -m "$OUT/dataset" 2>/dev/null || true   # exFAT: drop macOS ._* AppleDouble files Brush would try to load
[ -f "$OUT/brush/export_$STEPS.ply" ] || (cd "$OUT/brush" &&   # Brush writes a target/autotune cache into the cwd
  "$BRUSH" "$OUT/dataset" --total-steps "$STEPS" --export-every "$STEPS" --export-path "$OUT/brush")
PLY=$(ls -t "$OUT"/brush/*.ply | grep -v '/\._' | head -1)
echo "splat: $PLY"
"${PY[@]}" "$HERE/align_markers.py" "$OUT" --check-ply "$PLY"
