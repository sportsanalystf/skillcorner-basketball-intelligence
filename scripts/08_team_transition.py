"""Team profiles, transition, off-ball movement, event chains."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from src import config as C
from src.features import team_profiles as TP, transition as TR
from src.possessions import engine as PoE

E = C.PROCESSED_DIR/"events"; D = C.DERIVED_DIR
T = {p.stem: pd.read_parquet(p) for p in E.glob("*.parquet")}
sh = pd.read_parquet(D/"shots_features.parquet")
dr = pd.read_parquet(D/"drives_features.parquet")
pk = pd.read_parquet(D/"picks_features.parquet")
co = pd.read_parquet(D/"closeouts_features.parquet")
teams = pd.read_parquet(C.PROCESSED_DIR/"dim/teams.parquet")
tname = dict(zip(teams.team_id, teams.team_name))

off = TP.team_offense_profile(sh, dr, pk, T["chances"], co, T["off_ball_screens"])
dfp = TP.team_defense_profile(sh, dr, pk, T["chances"], co)
off["team_name"] = off.team_id.map(tname); dfp["team_name"] = dfp.team_id.map(tname)
off.to_parquet(D/"team_offense_profile.parquet", index=False)
dfp.to_parquet(D/"team_defense_profile.parquet", index=False)
print("team offense:", off.shape); print(off[["team_name","games","chances","pts_per_chance",
    "three_rate","picks_per_chance","drives_per_chance","transition_rate","efg"]].round(3).to_string(index=False))

notes = {int(t): TP.scouting_notes(t, off, dfp, dr, pk, sh, tname) for t in off.team_id}
Path(D/"scouting_notes.json").write_text(json.dumps(notes, indent=1))
print("\nexample scouting notes (", tname[off.team_id.iloc[0]], "):")
for n in notes[int(off.team_id.iloc[0])][:5]: print("  -", n["text"])

chn = TR.transition_summary(T["chances"], sh)
chn.to_parquet(D/"chances_transition.parquet", index=False)
ttp = TR.team_transition_profile(chn); ttp["team_name"] = ttp.team_id.map(tname)
ttp.to_parquet(D/"team_transition_profile.parquet", index=False)
print("\ntransition vs halfcourt (league):")
print(chn.groupby("transition").agg(n=("id","size"), ppc=("pts_scored","mean"),
      dur=("duration_s","mean"), tts=("time_to_shot_s","mean"), three=("shot_three","mean"),
      sq=("shot_quality","mean"), contested=("shot_contested","mean"), to=("outcome",lambda s:s.eq("TO").mean())
      ).round(3).to_string())

t=time.time()
ob = TR.offball_movement_before_shots(sh)
ob.to_parquet(D/"offball_before_shots.parquet", index=False)
print(f"\noff-ball movement: {ob.shape} ({time.time()-t:.0f}s)")
m = sh.merge(ob, on="id", how="left")
q = pd.qcut(m.offball_distance_ft, 4, labels=["Q1 least","Q2","Q3","Q4 most"])
print(m.groupby(q, observed=True).agg(n=("id","size"), fg=("outcome","mean"), xfg=("xfg","mean"),
      sq=("shot_quality","mean"), open_rate=("contested",lambda s: 1-s.mean()),
      sep=("separation_at_shot","mean")).round(3).to_string())

ac = PoE.build_action_chains(T)
ac.to_parquet(D/"action_chains.parquet", index=False)
top = (ac.groupby("full_chain").agg(n=("chance_id","size"), ppc=("pts_scored","mean"))
         .query("n >= 10").sort_values("ppc", ascending=False))
print("\nmost efficient chains (n>=10):"); print(top.head(12).round(3).to_string())
ce = PoE.chain_efficiency(ac); ce.to_parquet(D/"chain_efficiency.parquet", index=False)
print("\nmost efficient ACTION prefixes (outcome as dependent variable, n>=10):")
print(ce.head(12).round(3).to_string(index=False))
ct = PoE.chain_transitions(ac); ct.to_parquet(D/"chain_transitions.parquet", index=False)
print("\ntop action bigrams:"); print(ct.head(10).round(3).to_string(index=False))

cs = PoE.build_chance_summary(T); cs.to_parquet(D/"chance_summary.parquet", index=False)
print("\nchance summary:", cs.shape)
