"""Phase 4 — smooth tracking, compute kinematics, attach possession context."""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from src import config as C
from src.data import loaders
from src.tracking import engine, context
from src.preprocessing import court

OUT = C.PROCESSED_DIR / "tracking_enriched"
OUT.mkdir(parents=True, exist_ok=True)
VIZ = C.PROCESSED_DIR / "tracking_viz"
VIZ.mkdir(parents=True, exist_ok=True)

poss_all = pd.read_parquet(C.PROCESSED_DIR / "events/possessions.parquet")
chan_all = pd.read_parquet(C.PROCESSED_DIR / "events/chances.parquet")

# Columns persisted for analysis. vx/vy/ax/ay/heading/step_dist are dropped:
# each is recoverable from what is kept (vx_att/vy_att + left_hoop, accel_mag +
# accel_tangential), and dropping them cuts the tracking store from 617 MB to
# ~200 MB with no loss of information used downstream.
KEEP = ["game_id","frame_idx","wall_clock","game_clock","shot_clock","period",
        "team_id","is_home","player_id","x","y","x_s","y_s","speed_raw","speed",
        "accel_tangential","accel_mag","pred_error","is_detected","segment_id",
        "speed_band","flag_impossible_speed","flag_impossible_accel","flag_off_court",
        "possession_id","chance_id","off_team_id","left_hoop","is_offense",
        "chance_transition","x_att","y_att","vx_att","vy_att"]
ROUND2 = ["x","y","x_s","y_s","x_att","y_att","vx_att","vy_att","speed","speed_raw",
          "accel_tangential","accel_mag","pred_error"]
VIZ_STRIDE = 5   # 25 fps -> 5 fps for the animation layer

def _compact(d):
    d = d.copy()
    for c in ROUND2:
        if c in d.columns:
            d[c] = d[c].round(2).astype("float32")
    for c in ("game_clock", "shot_clock"):
        d[c] = d[c].round(1).astype("float32")
    for c in ("possession_id", "chance_id", "speed_band"):
        d[c] = d[c].astype("category")
    return d

qual_rows, phys_rows, smooth_rows = [], [], []
for gid in loaders.list_game_ids():
    t0 = time.time()
    raw = pd.read_parquet(C.PROCESSED_DIR / f"tracking/players/{gid}.parquet")
    smooth_rows.append({"game_id": gid, **engine.raw_vs_smoothed_report(raw)})
    df = engine.process_game_tracking(raw)
    df = context.attach_context(df, poss_all[poss_all.game_id == gid],
                                chan_all[chan_all.game_id == gid])
    df = context.add_attacking_coords(df)
    out = _compact(df[KEEP])
    out.to_parquet(OUT / f"{gid}.parquet", index=False, compression="zstd",
                   compression_level=9)
    VIZ.mkdir(parents=True, exist_ok=True)
    (out[out.frame_idx % VIZ_STRIDE == 0]
        [["game_id","frame_idx","game_clock","shot_clock","period","team_id","is_home",
          "player_id","x_s","y_s","speed","is_detected","possession_id","chance_id",
          "off_team_id","left_hoop"]]
        .to_parquet(VIZ / f"{gid}.parquet", index=False, compression="zstd",
                    compression_level=9))

    # ---- tracking quality per game
    qual_rows.append({
        "game_id": gid,
        "player_rows": len(df),
        "frames": int(df.frame_idx.nunique()),
        "pct_detected": float(df.is_detected.mean() * 100),
        "mean_pred_error": float(df.pred_error.mean()),
        "p95_pred_error": float(df.pred_error.quantile(0.95)),
        "mean_pred_error_detected": float(df.loc[df.is_detected, "pred_error"].mean()),
        "mean_pred_error_extrapolated": float(df.loc[~df.is_detected, "pred_error"].mean()),
        "pct_rows_in_possession": float(df.possession_id.notna().mean() * 100),
        "n_segments": int(df.segment_id.nunique()),
        "median_segment_len": float(df.groupby("segment_id").size().median()),
        "pct_impossible_speed": float(df.flag_impossible_speed.mean() * 100),
        "pct_impossible_accel": float(df.flag_impossible_accel.mean() * 100),
        "pct_off_court": float(df.flag_off_court.mean() * 100),
    })

    # ---- per player-game physical output (live-ball frames only)
    live = df[df.possession_id.notna() & df.speed.notna()]
    g = live.groupby("player_id")
    phys = pd.DataFrame({
        "game_id": gid,
        "frames_tracked": g.size(),
        "minutes_tracked": g.size() * C.DT / 60.0,
        "distance_ft": g.step_dist.sum(),
        "mean_speed": g.speed.mean(),
        "p95_speed": g.speed.quantile(0.95),
        "max_speed": g.speed.max(),
        "mean_abs_accel": g.accel_mag.mean(),
        "max_accel": g.accel_tangential.max(),
        "max_decel": g.accel_tangential.min(),
        "pct_detected": g.is_detected.mean() * 100,
    }).reset_index()
    band = (live.groupby(["player_id", "speed_band"]).step_dist.sum()
                .unstack(fill_value=0.0))
    band.columns = [f"dist_{c}" for c in band.columns]
    phys = phys.merge(band.reset_index(), on="player_id", how="left")
    phys["distance_per_min"] = phys.distance_ft / phys.minutes_tracked.replace(0, np.nan)
    phys_rows.append(phys)
    print(f"  {gid}: {len(df):,} rows  {time.time()-t0:.1f}s  "
          f"det={qual_rows[-1]['pct_detected']:.1f}%  inposs={qual_rows[-1]['pct_rows_in_possession']:.1f}%")

pd.DataFrame(qual_rows).to_parquet(C.DERIVED_DIR / "tracking_quality.parquet", index=False)
pd.DataFrame(smooth_rows).to_parquet(C.DERIVED_DIR / "smoothing_report.parquet", index=False)
pd.concat(phys_rows, ignore_index=True).to_parquet(C.DERIVED_DIR / "player_game_physical.parquet", index=False)
print("\nQUALITY SUMMARY"); print(pd.DataFrame(qual_rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
