"""Replay recorded Station history (an exported Parquet dataset) so perception/verify code can be
iterated without the arm. Needs pyarrow; run through uv so nothing is installed globally:

  uv run --no-project --with pyarrow python scripts/replay.py DATASET.parquet [--every N]
      -> per-frame table: episode, time, gripper opening/current, object blob in the gripper camera

In code:
  from replay import frames
  for f in frames("datasets/x.parquet"): f["grip_jpeg"], f["over_jpeg"], f["joints"][7]["current_ma"], ...

Export a dataset first (see LOG.md, "Dataset export from Station history").
Limit: replay is open-loop. It can test what the scripts SEE and JUDGE, not how the arm responds to commands.
"""
import sys
from pathlib import Path
import pyarrow.parquet as pq

GRIP_IDX, OVER_IDX = 0, 1     # image order in Station's normvla export (checked on 21 Nov: 0 = gripper camera)

def frames(path, every=1):
    t = pq.read_table(path, columns=["episode_start_ns", "timestamp_ns_since_episode_start", "joints", "images"])
    eps = {}
    for b in t.to_batches(max_chunksize=200):
        for i, r in enumerate(b.to_pylist()):
            ep = eps.setdefault(r["episode_start_ns"], len(eps) + 1)
            if i % every: continue
            imgs = r["images"]
            yield dict(episode=ep, start_ns=r["episode_start_ns"], t=r["timestamp_ns_since_episode_start"] / 1e9,
                       joints=r["joints"], grip_jpeg=imgs[GRIP_IDX]["jpeg"] if len(imgs) > GRIP_IDX else None,
                       over_jpeg=imgs[OVER_IDX]["jpeg"] if len(imgs) > OVER_IDX else None)

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import vision
    every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 5
    print("ep     t   grip  g_mA  blob(x,y,area)")
    for f in frames(sys.argv[1], every):
        g = f["joints"][7]
        w, h, lum = vision.decode(f["grip_jpeg"]); bs = vision.blobs(w, h, lum)
        b = min(bs, key=lambda b: abs(b["x"] - w / 2) + abs(b["y"] - h / 2) - 0.2 * b["area"]) if bs else None
        print(f"{f['episode']:2} {f['t']:6.1f}  {g['position_norm']:.2f} {g['current_ma']:5}  " +
              (f"({b['x']:.0f},{b['y']:.0f},{b['area']})" if b else "-"))
