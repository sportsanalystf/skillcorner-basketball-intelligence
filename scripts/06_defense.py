"""Phase 10 — defensive analytics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from src import config as C
from src.defense import engine as DF

E = C.PROCESSED_DIR/"events"; D = C.DERIVED_DIR
T = {p.stem: pd.read_parquet(p) for p in E.glob("*.parquet")}
roster = pd.read_parquet(C.PROCESSED_DIR/"dim/roster.parquet")

co = DF.build_closeout_features(T, pd.read_parquet(D/"spatial_closeouts.parquet"))
co.to_parquet(D/"closeouts_features.parquet", index=False)
print("closeouts:", co.shape)
print(co[["start_distance","end_distance","distance_closed_ft","duration_s",
          "close_rate_fps","total_points_allowed"]].describe().round(2).to_string())
print("\nby ball-handler response:")
print(co.groupby("bhr_action").agg(n=("id","size"), start=("start_distance","mean"),
      end=("end_distance","mean"), rate=("close_rate_fps","mean"),
      pts=("total_points_allowed","mean"), stop=("stop","mean")).round(3).to_string())
print("\nby arrival tightness:")
print(co.groupby("arrived_tight").agg(n=("id","size"), pts=("total_points_allowed","mean"),
      stop=("stop","mean"), foul=("fouled","mean")).round(3).to_string())

cdp = DF.closeout_defender_profile(co, roster); cdp.to_parquet(D/"closeout_defender_profile.parquet", index=False)
print("\nfastest closers (min 8):")
print(cdp.nlargest(8,"mean_close_rate_fps")[["player_name","team_name","closeouts",
      "mean_close_rate_fps","mean_end_distance","stop_rate"]].round(2).to_string(index=False))

mw = DF.matchup_workload(T, roster); mw.to_parquet(D/"matchup_workload.parquet", index=False)
print("\nmatchup workload:", mw.shape)
print(mw.nlargest(6,"minutes_guarding")[["player_name","team_name","games","assignments",
      "minutes_guarding","points_allowed_per_36"]].round(2).to_string(index=False))

sh = pd.read_parquet(D/"shots_features.parquet")
sdp = DF.shot_defender_profile(sh, roster); sdp.to_parquet(D/"shot_defender_profile.parquet", index=False)
print("\nnearest-defender profiles:", sdp.shape)
print(sdp.nsmallest(6,"fg_vs_expected")[["player_name","shots_contested","mean_closest_dist",
      "opp_fg_pct","opp_xfg","fg_vs_expected"]].round(3).to_string(index=False))
