"""Statistical validation of the headline claims.

Each test asserts the *direction and significance* of a finding that is
reported in the README / CASE_STUDIES, so a pipeline change that silently
inverts a published result breaks the build.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src import config as C


def _boot_diff(a, b, n=2000, seed=0):
    """Bootstrap CI for mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    d = np.array([rng.choice(a, len(a), True).mean() - rng.choice(b, len(b), True).mean()
                  for _ in range(n)])
    return d.mean(), np.percentile(d, [2.5, 97.5])


def test_defender_distance_is_uninformative_marginally(derived):
    """Simpson's paradox, and one of the project's headline findings.

    Marginally, makes and misses have statistically indistinguishable defender
    distances — because rim shots (high FG%) are heavily contested while threes
    (low FG%) are open, and the two effects cancel. The naive comparison is not
    just weak, it points the wrong way.
    """
    sh = derived["shots_features"].dropna(subset=["closest_def_dist"])
    sh = sh[sh.is_box_score_fga]
    made = sh.loc[sh.outcome, "closest_def_dist"]
    miss = sh.loc[~sh.outcome, "closest_def_dist"]
    t, p = stats.ttest_ind(made, miss, equal_var=False)
    assert p > 0.05, f"the marginal comparison became significant (t={t:.2f} p={p:.3g})"


def test_defender_distance_predicts_makes_once_location_is_controlled(derived):
    """Conditional on shot distance and shot value, separation matters a lot."""
    import statsmodels.api as sm
    sh = derived["shots_features"]
    sh = sh[sh.is_box_score_fga].dropna(subset=["closest_def_dist", "dist_calc"])
    X = sm.add_constant(sh[["closest_def_dist", "dist_calc"]]
                        .assign(three=sh.three.astype(int)))
    fit = sm.Logit(sh.outcome.astype(int), X).fit(disp=0)
    coef = fit.params["closest_def_dist"]
    pval = fit.pvalues["closest_def_dist"]
    assert coef > 0 and pval < 0.001, f"coef={coef:.4f} p={pval:.3g}"
    # ~10% higher odds of a make per extra foot of separation
    assert 1.05 < np.exp(coef) < 1.20


def test_separation_effect_is_strongest_at_the_rim(derived):
    sh = derived["shots_features"]
    sh = sh[sh.is_box_score_fga].dropna(subset=["closest_def_dist"])
    rim = sh[sh.zone_family == "rim"]
    t, p = stats.ttest_ind(rim.loc[rim.outcome, "closest_def_dist"],
                           rim.loc[~rim.outcome, "closest_def_dist"], equal_var=False)
    assert t > 0 and p < 0.001, f"rim: t={t:.2f} p={p:.3g}"
    d, ci = _boot_diff(rim.loc[rim.outcome, "closest_def_dist"],
                       rim.loc[~rim.outcome, "closest_def_dist"])
    assert ci[0] > 0.3, f"rim separation gap CI {ci} — effect no longer clear"


def test_separation_does_not_predict_above_the_break_threes(derived):
    """The counterpart finding: on above-the-break threes, being open does not
    predict making it in this sample."""
    sh = derived["shots_features"]
    sh = sh[sh.is_box_score_fga & (sh.zone_family == "above_break_3")]
    sh = sh.dropna(subset=["closest_def_dist"])
    t, p = stats.ttest_ind(sh.loc[sh.outcome, "closest_def_dist"],
                           sh.loc[~sh.outcome, "closest_def_dist"], equal_var=False)
    assert p > 0.05, f"above-break threes now show an effect (t={t:.2f} p={p:.3g})"


def test_contest_level_is_monotone_in_fg_pct(derived):
    sh = derived["shots_features"]
    sh = sh[sh.is_box_score_fga & sh.contest_level.isin(["open", "average", "plus"])]
    tab = pd.crosstab(sh.contest_level, sh.outcome)
    chi2, p, *_ = stats.chi2_contingency(tab)
    assert p < 0.001
    fg = sh.groupby("contest_level").outcome.mean()
    assert fg["open"] > fg["average"] > fg["plus"]


def test_transition_produces_better_looks_but_not_more_points(derived):
    """Claim: transition raises shot quality and openness, not points per chance."""
    ch = derived["chances_transition"]
    tr, hc = ch[ch.transition], ch[~ch.transition]
    # shot quality is higher, significantly
    t, p = stats.ttest_ind(tr.shot_quality.dropna(), hc.shot_quality.dropna(),
                           equal_var=False)
    assert p < 0.001 and t > 0
    # points per chance is NOT significantly different
    t2, p2 = stats.ttest_ind(tr.pts_scored, hc.pts_scored, equal_var=False)
    assert p2 > 0.05, f"points per chance differ after all (p={p2:.3f})"


def test_offball_movement_buys_separation_not_conversion(derived):
    sh = derived["shots_features"].merge(derived["offball_before_shots"], on="id")
    sh = sh.dropna(subset=["offball_distance_ft", "separation_at_shot"])
    q = pd.qcut(sh.offball_distance_ft, 4, labels=False)
    lo, hi = sh[q == 0], sh[q == 3]
    # separation improves, significantly
    t, p = stats.ttest_ind(hi.separation_at_shot, lo.separation_at_shot, equal_var=False)
    assert p < 0.01 and t > 0
    # actual FG% does not
    tab = pd.crosstab(q == 3, sh.outcome)
    _, p2, *_ = stats.chi2_contingency(tab)
    assert p2 > 0.05, f"FG% does differ by movement quartile (p={p2:.3f})"


def test_help_defence_closes_significantly_during_drives(derived):
    dr = derived["drives_features"].dropna(subset=["help_at_start", "help_min"])
    t, p = stats.ttest_rel(dr.help_at_start, dr.help_min)
    assert p < 1e-20 and t > 0
    assert dr.help_closed_ft.mean() > 5.0


def test_going_under_the_screen_concedes_less_than_going_over(derived):
    pk = derived["picks_features"]
    over = pk.loc[pk.bhr_def_type == "over", "pick_points"]
    under = pk.loc[pk.bhr_def_type == "under", "pick_points"]
    assert under.mean() < over.mean()
    d, ci = _boot_diff(over, under)
    # honest reporting: the direction is stable, the CI may straddle zero
    assert d > 0
    assert len(under) > 200 and len(over) > 800


def test_tight_closeouts_concede_fewer_points_and_more_fouls(derived):
    co = derived["closeouts_features"]
    tight, loose = co[co.arrived_tight], co[~co.arrived_tight]
    assert tight.total_points_allowed.mean() < loose.total_points_allowed.mean()
    tab = pd.crosstab(co.arrived_tight, co.fouled)
    _, p, *_ = stats.chi2_contingency(tab)
    assert p < 0.05
    assert tight.fouled.mean() > loose.fouled.mean()


def test_spacing_adds_little_to_the_shot_model():
    import json
    m = json.loads((C.DERIVED_DIR / "xfg_metrics.json").read_text())
    delta = m["logistic+spatial"]["auc"] - m["logistic_no_spatial"]["auc"]
    assert abs(delta) < 0.02, f"spatial features moved AUC by {delta:+.3f}"


def test_chance_spacing_correlates_weakly_with_points(derived):
    sp = derived["chance_spacing"]
    ch = pd.read_parquet(C.PROCESSED_DIR / "events/chances.parquet").rename(
        columns={"id": "chance_id"})
    m = sp.merge(ch[["chance_id", "pts_scored"]], on="chance_id")
    r = m.hull_area_mean.corr(m.pts_scored)
    assert 0 < r < 0.15, f"r={r:.3f} — the reported 'weak positive' claim no longer holds"
