"""Phases 7-8 — drive and pick-and-roll analytics."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from src import config as C
from src.drives import engine as DE
from src.picks import engine as PE

E = C.PROCESSED_DIR / "events"; D = C.DERIVED_DIR
T = {p.stem: pd.read_parquet(p) for p in E.glob("*.parquet")}
roster = pd.read_parquet(C.PROCESSED_DIR / "dim/roster.parquet")

dr = DE.build_drive_features(
    T, pd.read_parquet(D/"spatial_drives.parquet"),
    pd.read_parquet(D/"spatial_drives_end.parquet"),
    pd.read_parquet(D/"drive_collapse.parquet"))
dr.to_parquet(D/"drives_features.parquet", index=False)
print("drives:", dr.shape, "| PPD", round(dr.points_total.mean(),3),
      "| success", round(dr.successful.mean(),3))
print(dr.groupby("end_type").agg(n=("id","size"), ppd=("points_total","mean"),
      foul=("drew_foul","mean"), to=("turned_over","mean")).round(3).to_string())
print("\nby category:")
print(dr.groupby("category").agg(n=("id","size"), ppd=("points_total","mean"),
      blowby=("blowby","mean"), paint=("reached_paint","mean")).round(3).to_string())
print("\nhelp collapse: mean help defender distance closed =",
      round(dr.help_closed_ft.mean(),2),"ft; paint defenders added =",
      round(dr.paint_defenders_added.mean(),2))

pdp = DE.player_drive_profile(dr, roster)
pdp.to_parquet(D/"player_drive_profile.parquet", index=False)
print("\ntop drivers by PPD (min 8 drives):")
print(pdp.nlargest(8,"points_per_drive")[["player_name","team_name","drives","points_per_drive","blowby_rate","pass_out_rate"]].round(3).to_string(index=False))

t=time.time()
pk = PE.build_pick_features(T, pd.read_parquet(D/"spatial_picks.parquet"))
rp = PE.roll_pop_labels(T["picks"])
pk = pk.merge(rp, on="id", how="left")
pk.to_parquet(D/"picks_features.parquet", index=False)
print(f"\npicks: {pk.shape} ({time.time()-t:.0f}s)  direct rate {pk.direct.mean():.3f}")
print("screener action:", pk.screener_action.value_counts(dropna=False).to_dict())
print("\ncoverage effectiveness (ball-handler coverage):")
print(PE.coverage_effectiveness(pk).round(3).to_string(index=False))
print("\ncoverage effectiveness (screener coverage):")
print(PE.coverage_effectiveness(pk,"scr_def_type").round(3).to_string(index=False))

hp = PE.handler_profile(pk, roster); hp.to_parquet(D/"pick_handler_profile.parquet", index=False)
sp = PE.screener_profile(pk, roster); sp.to_parquet(D/"pick_screener_profile.parquet", index=False)
print("\nhandlers:", hp.shape, "screeners:", sp.shape)
