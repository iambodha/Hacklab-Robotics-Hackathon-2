# Camera video to a metric Gaussian splat

The [pipeline README](../../sim/splat/README.md) contains installation, filming, stage outputs, synthetic self-test and viewer instructions. It uses ffmpeg, COLMAP/GLOMAP through pycolmap, ArUco metric alignment and Brush training. No depth sensor is required.

Set `DATA_ROOT` to a writable data folder before invoking `sim/splat/prepare.sh <video> <name>`; the inherited default is a local external drive. Set `SPLAT_LAYOUT` to the matching table layout. Keep markers visible from multiple nondegenerate views and keep the scene static. Tools beyond the `sim` Python extra are separate installs; the documented Brush binary is macOS ARM64, so other systems need their own supported build.

The measured proof is synthetic: 90 registered frames, 0.46 mm mean marker residual, 4.6 mm mean camera-centre error at roughly 1 m, and 500 Brush steps as a smoke test. It does not establish accuracy for your video. A splat models appearance; collision/robot physics stay in MuJoCo. Side-by-side web and composited viewers described in the original README are proposals.
