"""Phases 11-12 — player profiles, similarity, role clustering."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from src import config as C
from src.features import player_profiles as PP
from src.models.similarity import StyleSpace, FEATURES

E = C.PROCESSED_DIR/"events"; D = C.DERIVED_DIR
T = {p.stem: pd.read_parquet(p) for p in E.glob("*.parquet")}
DV = {n: pd.read_parquet(D/f"{n}.parquet") for n in
      ["shots_features","drives_features","picks_features","closeouts_features","player_game_physical"]}
roster = pd.read_parquet(C.PROCESSED_DIR/"dim/roster.parquet")

gl = PP.game_level_profile(T, DV, roster)
gl.to_parquet(D/"player_game_profile.parquet", index=False)
print("game-level player profiles:", gl.shape)
print(gl.nlargest(6,"points")[["player_name","team_name","games","points","fga","fg_pct",
      "xfg","shot_value_added","drives","picks_used","distance_per_min"]].round(3).to_string(index=False))

agg = {k: pd.read_parquet(C.PROCESSED_DIR/f"aggregates/{k}_merged.parquet") for k in ["shots","drives","picks"]}
style = PP.build_season_style(agg)
style.to_parquet(D/"season_style.parquet", index=False)
print("\nseason style vectors:", style.shape,
      "| features present:", sum(f in style.columns for f in FEATURES),"/",len(FEATURES))

ss = StyleSpace(style)
print(f"PCA: {ss.n_components} components, {ss.explained:.1%} variance")
kt = ss.choose_k(); print(kt.round(4).to_string(index=False))
best_k = int(kt.loc[kt.silhouette.idxmax(),"k"])
ss.cluster(best_k)
print(f"\nsilhouette-optimal k = {best_k}  (peak {kt.silhouette.max():.3f})")
desc = ss.describe_clusters(); print(desc.to_string(index=False))
reps = ss.representatives(); print("\nrepresentatives:"); print(reps.round(2).to_string(index=False))
primary_labels = ss.labels.copy()

# exploratory finer partition — reported alongside, flagged as such
FINE_K = 6
ss.cluster(FINE_K)
desc6 = ss.describe_clusters(); reps6 = ss.representatives()
print(f"\nexploratory k={FINE_K} partition (silhouette {kt.set_index('k').loc[FINE_K,'silhouette']:.3f}):")
print(desc6.to_string(index=False))
fine_labels = ss.labels.copy()
desc6.to_parquet(D/"cluster_descriptions_k6.parquet", index=False)
reps6.to_parquet(D/"cluster_representatives_k6.parquet", index=False)
ss.cluster(best_k)

style_out = style.copy(); style_out["cluster"] = primary_labels; style_out["cluster_fine"] = fine_labels
for i in range(ss.n_components):
    style_out[f"pc{i+1}"] = ss.P[:,i]
style_out.to_parquet(D/"season_style_clustered.parquet", index=False)
sim = pd.DataFrame(ss.similarity_matrix(),
                   index=style.player_name+" ("+style.team_name+")",
                   columns=style.player_name+" ("+style.team_name+")")
sim.to_parquet(D/"player_similarity_matrix.parquet")
desc.to_parquet(D/"cluster_descriptions.parquet", index=False)
reps.to_parquet(D/"cluster_representatives.parquet", index=False)
kt.to_parquet(D/"cluster_k_selection.parquet", index=False)
pd.DataFrame({"feature": ss.features,
              **{f"pc{i+1}": ss.pca.components_[i] for i in range(ss.n_components)}}
             ).to_parquet(D/"style_pca_loadings.parquet", index=False)

i = int(style.index[style.player_name.str.contains("Ricky Rubio", na=False)][0]) \
    if style.player_name.str.contains("Ricky Rubio", na=False).any() else 0
print(f"\nmost similar to {style.loc[i,'player_name']}:")
print(ss.most_similar(i).round(3).to_string(index=False))
