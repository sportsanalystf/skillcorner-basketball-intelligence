"""Phase 9 — spatial features at every event moment + drive collapse profiles."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from src import config as C
from src.spatial import features as F

E = C.PROCESSED_DIR / "events"
D = C.DERIVED_DIR
rd = lambda n: pd.read_parquet(E / f"{n}.parquet")

SPECS = [
    ("shots",            "start_frame", "shooter_id"),
    ("drives",           "start_frame", "ballhandler_id"),
    ("drives_end",       "end_frame",   "ballhandler_id"),
    ("picks",            "frame",       "ballhandler_id"),
    ("closeouts",        "start_frame", "ballhandler_id"),
    ("off_ball_screens", "frame",       "cutter_id"),
    ("isolations",       "start_frame", "ballhandler_id"),
    ("posts",            "start_frame", "ballhandler_id"),
    ("handoffs",         "frame",       "receiver_id"),
]

for name, fcol, focal in SPECS:
    tbl = "drives" if name == "drives_end" else name
    ev = rd(tbl)
    t = time.time()
    sp = F.build_event_spatial(ev, fcol, focal)
    sp.to_parquet(D / f"spatial_{name}.parquet", index=False)
    print(f"  spatial_{name:18s} {sp.shape}  {time.time()-t:.1f}s")

t = time.time()
dc = F.drive_collapse_profile(rd("drives"))
dc.to_parquet(D / "drive_collapse.parquet", index=False)
print(f"  drive_collapse {dc.shape} {time.time()-t:.1f}s")

# ---- chance-level spacing time series (sample every 10 frames in frontcourt)
ch = rd("chances")
recs = []
t = time.time()
for gid, sub in ch.groupby("game_id"):
    sub = sub.dropna(subset=["frontcourt_frame", "end_frame", "off_team_id"])
    sub = sub[sub.end_frame > sub.frontcourt_frame]
    want = []
    for r in sub.itertuples(index=False):
        fs, fe = int(r.frontcourt_frame), int(min(r.end_frame, r.frontcourt_frame + 700))
        for f in range(fs, fe + 1, 10):
            want.append((r.id, f, r.off_team_id))
    if not want:
        continue
    q = pd.DataFrame(want, columns=["chance_id", "frame", "off_team_id"])
    q["key"] = q.chance_id + "@" + q.frame.astype(str)
    res = F.snapshot_features(int(gid), q)
    if res.empty:
        continue
    recs.append(res.merge(q[["key", "chance_id"]], on="key"))
cs = pd.concat(recs, ignore_index=True)
agg = cs.groupby("chance_id").agg(
    n_samples=("hull_area", "size"),
    hull_area_mean=("hull_area", "mean"), hull_area_min=("hull_area", "min"),
    mean_nn_dist_mean=("mean_nn_dist", "mean"),
    mean_pairwise_mean=("mean_pairwise_dist", "mean"),
    width_mean=("width", "mean"), depth_mean=("depth", "mean"),
    spacing_entropy_mean=("spacing_entropy", "mean"),
    n_in_paint_mean=("n_in_paint", "mean"),
    n_in_corners_mean=("n_in_corners", "mean"),
    n_beyond_arc_mean=("n_beyond_arc", "mean"),
    n_def_in_paint_mean=("n_def_in_paint", "mean"),
    n_off_tracked_mean=("n_off_tracked", "mean"),
).reset_index()
agg.to_parquet(D / "chance_spacing.parquet", index=False)
cs.to_parquet(D / "chance_spacing_samples.parquet", index=False)
print(f"  chance_spacing {agg.shape} from {len(cs):,} samples  {time.time()-t:.0f}s")
