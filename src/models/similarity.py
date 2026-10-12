"""Player similarity and role clustering on offensive style.

Method
------
1. Build the season style vector (:func:`src.features.player_profiles.build_season_style`)
   — shot mix, drive behaviour, pick-and-roll role. Volume columns are per game
   so that a starter and a rotation player with the same *style* land nearby.
2. Standardise (z-score) across the eligible population, median-impute the few
   missing entries, and reduce with PCA to the components explaining ~85% of
   variance. PCA is used rather than UMAP for the similarity metric because the
   mapping is linear and therefore explainable: we can say *which* features
   drive a given pair's similarity.
3. Similarity = cosine similarity in the PCA space. For every pair we also
   report the three standardised features that contribute most to the match, so
   a user sees why two players look alike.
4. Clustering: K-means over the same PCA space, k chosen by silhouette score
   over a candidate range. Cluster labels are *not* hard-coded — they are named
   afterwards from the features that separate each cluster from the population
   mean, and the raw feature table is shown alongside so a reader can disagree.

Limits: this describes offensive *style*, not quality or interchangeability, and
the aggregates are offence-only, so nothing about defence enters the model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "rim_attempt_rate", "midrange_attempt_rate", "three_attempt_rate",
    "corner3_share", "cns_share", "contested_share", "avg_shot_distance",
    "assisted_share", "shot_volume_per_game", "ft_rate",
    "drives_per_game", "end_type_kickout_rate", "end_type_pullup_rate",
    "end_type_shot_near_basket_rate", "blowby_rate", "assists_rate",
    "handler_picks_per_game", "screener_picks_per_game", "handler_shot_rate",
    "handler_pass_rate", "handler_assist_rate", "pick_role_ratio",
]


class StyleSpace:
    def __init__(self, df: pd.DataFrame, features: list[str] | None = None,
                 var_target: float = 0.85):
        feats = [f for f in (features or FEATURES) if f in df.columns]
        # drop features that are entirely missing in this population
        self.dropped_features = [f for f in feats if df[f].notna().sum() == 0]
        self.features = [f for f in feats if f not in self.dropped_features]
        self.df = df.reset_index(drop=True)
        X = self.df[self.features].to_numpy(dtype="float64")
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(X)
        Xi = self.imputer.transform(X)
        self.scaler = StandardScaler().fit(Xi)
        self.Z = self.scaler.transform(Xi)
        full = PCA().fit(self.Z)
        k = int(np.searchsorted(np.cumsum(full.explained_variance_ratio_), var_target) + 1)
        self.n_components = max(k, 2)
        self.pca = PCA(n_components=self.n_components).fit(self.Z)
        self.P = self.pca.transform(self.Z)
        self.explained = float(self.pca.explained_variance_ratio_.sum())

    # -- similarity ---------------------------------------------------------
    def similarity_matrix(self) -> np.ndarray:
        n = self.P / np.linalg.norm(self.P, axis=1, keepdims=True)
        return n @ n.T

    def most_similar(self, row_idx: int, k: int = 8) -> pd.DataFrame:
        sim = self.similarity_matrix()[row_idx]
        order = np.argsort(-sim)
        order = order[order != row_idx][:k]
        out = self.df.loc[order, ["player_name", "team_name", "games_played"]].copy()
        out["similarity"] = sim[order]
        drivers = []
        for j in order:
            d = np.abs(self.Z[row_idx] - self.Z[j])
            near = np.argsort(d)[:3]
            drivers.append(", ".join(self.features[i] for i in near))
        out["closest_on"] = drivers
        return out.reset_index(drop=True)

    def feature_contrast(self, row_idx: int, other_idx: int) -> pd.DataFrame:
        return (pd.DataFrame({
            "feature": self.features,
            "player_z": self.Z[row_idx], "other_z": self.Z[other_idx],
            "player": self.df.loc[row_idx, self.features].to_numpy(),
            "other": self.df.loc[other_idx, self.features].to_numpy(),
        }).assign(gap=lambda d: (d.player_z - d.other_z).abs())
          .sort_values("gap", ascending=False).reset_index(drop=True))

    # -- clustering ---------------------------------------------------------
    def choose_k(self, candidates=range(3, 10)) -> pd.DataFrame:
        rows = []
        for k in candidates:
            km = KMeans(n_clusters=k, n_init=25, random_state=0).fit(self.P)
            rows.append({"k": k, "inertia": km.inertia_,
                         "silhouette": silhouette_score(self.P, km.labels_)})
        return pd.DataFrame(rows)

    def cluster(self, k: int):
        km = KMeans(n_clusters=k, n_init=50, random_state=0).fit(self.P)
        self.labels = km.labels_
        self.kmeans = km
        return km.labels_

    def cluster_profile(self) -> pd.DataFrame:
        """Mean standardised feature value per cluster — how each differs from average."""
        z = pd.DataFrame(self.Z, columns=self.features)
        z["cluster"] = self.labels
        return z.groupby("cluster").mean().T

    def describe_clusters(self, top: int = 4) -> pd.DataFrame:
        prof = self.cluster_profile()
        rows = []
        for c_ in prof.columns:
            s = prof[c_].sort_values()
            rows.append({
                "cluster": c_,
                "n": int((self.labels == c_).sum()),
                "high": ", ".join(f"{i} (+{v:.2f})" for i, v in s.tail(top)[::-1].items()),
                "low": ", ".join(f"{i} ({v:.2f})" for i, v in s.head(top).items()),
            })
        return pd.DataFrame(rows)

    def representatives(self, k: int = 4) -> pd.DataFrame:
        """Players nearest each cluster centroid."""
        out = []
        for c_ in range(self.kmeans.n_clusters):
            idx = np.flatnonzero(self.labels == c_)
            d = np.linalg.norm(self.P[idx] - self.kmeans.cluster_centers_[c_], axis=1)
            for i in idx[np.argsort(d)[:k]]:
                out.append({"cluster": c_, "player_name": self.df.loc[i, "player_name"],
                            "team_name": self.df.loc[i, "team_name"],
                            "dist_to_centroid": float(np.linalg.norm(
                                self.P[i] - self.kmeans.cluster_centers_[c_]))})
        return pd.DataFrame(out)
