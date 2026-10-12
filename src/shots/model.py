"""Interpretable expected-field-goal (xFG) model.

The `shots` table already carries SkillCorner's own ``shotQuality`` (0-100).
That is *their* model and is reported as-is throughout the app. Separately, this
module fits **our own** transparent model so that (a) we can decompose which
factors drive shot quality in this dataset, and (b) we can score a shot on
features SkillCorner's index does not expose, such as offensive spacing.

Model
-----
``P(make) = logistic(f(distance) + g(defender distance) + shot-type effects +
             contest effects + context)``

Distance and defender distance enter through natural cubic splines so the fit
stays monotone-ish and readable; everything else is a one-hot effect. A gradient
boosting model is fitted alongside purely as an accuracy ceiling check — the
logistic model is what the app reports, because its coefficients are inspectable.

Evaluation is 5-fold **grouped by game** so that a game's own shots never appear
in both train and test. Metrics: log-loss, Brier score, AUC, and a calibration
table. With 1,316 box-score attempts across 10 games this is a small sample; the
model is presented as descriptive, not predictive of future games.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

NUM_SPLINE = ["dist_calc", "closest_def_dist"]
NUM_LINEAR = ["angle_deg", "shot_clock", "release_time", "dribbles_before"]
CAT = ["shot_type", "contest_level", "creation_type"]
BOOL = ["three", "catch_and_shoot", "created_from_paint", "chance_transition", "assist_opp"]
SPATIAL = ["hull_area", "mean_nn_dist", "n_def_in_paint", "n_in_corners",
           "mean_teammate_dist", "help_gap"]


def design_frame(sh: pd.DataFrame, with_spatial: bool = True) -> tuple[pd.DataFrame, list]:
    cols = NUM_SPLINE + NUM_LINEAR + CAT + BOOL + (SPATIAL if with_spatial else [])
    X = sh.reindex(columns=cols).copy()
    for c in BOOL:
        X[c] = X[c].astype("float64").fillna(0.0)
    for c in CAT:
        X[c] = X[c].astype("string").fillna("unknown")
    for c in NUM_SPLINE + NUM_LINEAR + (SPATIAL if with_spatial else []):
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median())
    return X, cols


def make_logistic(with_spatial: bool = True) -> Pipeline:
    num_lin = NUM_LINEAR + (SPATIAL if with_spatial else [])
    pre = ColumnTransformer([
        ("spl", Pipeline([("sc", StandardScaler()),
                          ("sp", SplineTransformer(n_knots=5, degree=3,
                                                   include_bias=False))]), NUM_SPLINE),
        ("num", StandardScaler(), num_lin),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first",
                              min_frequency=15), CAT),
        ("bin", "passthrough", BOOL),
    ])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(C=0.5, max_iter=4000))])


def make_gbm() -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", NUM_SPLINE + NUM_LINEAR + SPATIAL),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("bin", "passthrough", BOOL),
    ])
    return Pipeline([("pre", pre),
                     ("clf", HistGradientBoostingClassifier(
                         max_depth=3, max_iter=250, learning_rate=0.06,
                         min_samples_leaf=40, l2_regularization=1.0,
                         random_state=0))])


def cross_val_scores(sh: pd.DataFrame, model_fn, with_spatial=True, n_splits=5) -> dict:
    X, _ = design_frame(sh, with_spatial)
    y = sh["outcome"].astype(int).to_numpy()
    groups = sh["game_id"].to_numpy()
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    for tr, te in gkf.split(X, y, groups):
        m = model_fn()
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    base = np.full_like(oof, y.mean())
    return {
        "n": int(len(y)), "base_rate": float(y.mean()),
        "log_loss": float(log_loss(y, oof)),
        "log_loss_baseline": float(log_loss(y, base)),
        "brier": float(brier_score_loss(y, oof)),
        "brier_baseline": float(brier_score_loss(y, base)),
        "auc": float(roc_auc_score(y, oof)),
        "oof": oof,
    }


def fit_full(sh: pd.DataFrame, with_spatial: bool = True) -> Pipeline:
    X, _ = design_frame(sh, with_spatial)
    m = make_logistic(with_spatial)
    m.fit(X, sh["outcome"].astype(int).to_numpy())
    return m


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 8) -> pd.DataFrame:
    q = pd.qcut(p, bins, duplicates="drop")
    df = pd.DataFrame({"y": y, "p": p, "bin": q})
    return (df.groupby("bin", observed=True)
              .agg(n=("y", "size"), pred=("p", "mean"), actual=("y", "mean"))
              .reset_index().assign(gap=lambda d: d.actual - d.pred))


def coefficient_report(model: Pipeline) -> pd.DataFrame:
    pre = model.named_steps["pre"]
    names = pre.get_feature_names_out()
    coefs = model.named_steps["clf"].coef_[0]
    return (pd.DataFrame({"feature": names, "coef": coefs,
                          "odds_ratio": np.exp(coefs)})
            .sort_values("coef", key=np.abs, ascending=False).reset_index(drop=True))
