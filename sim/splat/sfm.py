"""Camera poses for a splat capture: SIFT features -> matching -> global SfM (GLOMAP, built into COLMAP 4 / pycolmap).

  python sim/splat/sfm.py <DATA_ROOT/splat/<name>>     # reads dataset/images, writes sfm/{database.db,sparse/0}

Video frames are matched sequentially (each frame with its next `--overlap` frames, plus quadratic overlap); short
captures (<= --exhaustive-below frames) are matched exhaustively. One shared camera (a single phone lens), OPENCV
distortion model. The largest reconstruction is written to sfm/sparse/0 (COLMAP binary), unaligned (arbitrary
scale and frame); align_markers.py puts it into the twin's frame.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pycolmap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work", type=Path)
    ap.add_argument("--overlap", type=int, default=12)
    ap.add_argument("--exhaustive-below", type=int, default=120)
    ap.add_argument("--max-features", type=int, default=8192)
    ap.add_argument("--camera-model", default="OPENCV")
    a = ap.parse_args()
    images = a.work / "dataset" / "images"
    sfm = a.work / "sfm"
    names = sorted(p.name for p in images.glob("*.jpg") if not p.name.startswith("._"))   # exFAT AppleDouble files
    if len(names) < 10:
        raise SystemExit(f"only {len(names)} frames in {images}")
    sfm.mkdir(parents=True, exist_ok=True)
    db = sfm / "database.db"
    db.unlink(missing_ok=True)
    t0 = time.time()
    ext = pycolmap.FeatureExtractionOptions()
    ext.sift.max_num_features = a.max_features
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = a.camera_model
    pycolmap.extract_features(db, images, names, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
                              extraction_options=ext)
    t1 = time.time()
    if len(names) <= a.exhaustive_below:
        pycolmap.match_exhaustive(db)
    else:
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = a.overlap
        pairing.quadratic_overlap = True
        pycolmap.match_sequential(db, pairing_options=pairing)
    t2 = time.time()
    raw = sfm / "global"
    shutil.rmtree(raw, ignore_errors=True)
    raw.mkdir()
    opts = pycolmap.GlobalPipelineOptions()
    opts.image_names = names
    recs = pycolmap.global_mapping(db, images, raw, opts)
    t3 = time.time()
    if not recs:
        raise SystemExit("global mapping found no reconstruction")
    best = max(recs.values(), key=lambda r: r.num_reg_images())
    out = sfm / "sparse" / "0"
    out.mkdir(parents=True, exist_ok=True)
    best.write(out)
    shutil.rmtree(raw, ignore_errors=True)
    info = {"frames": len(names), "registered": best.num_reg_images(), "points3D": best.num_points3D(),
            "mean_reproj_px": round(best.compute_mean_reprojection_error(), 3), "models": len(recs),
            "camera": best.cameras[next(iter(best.cameras))].todict() if best.cameras else None,
            "seconds": {"features": round(t1 - t0, 1), "matching": round(t2 - t1, 1), "mapping": round(t3 - t2, 1)}}
    (sfm / "sfm.json").write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps({k: v for k, v in info.items() if k != "camera"}))


if __name__ == "__main__":
    main()
