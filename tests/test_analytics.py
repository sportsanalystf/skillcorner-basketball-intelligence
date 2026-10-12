"""Basketball analytics: shots, drives, picks, spacing, possessions, defence."""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.spatial import geometry as G
from src.possessions import engine as PoE


# ------------------------------------------------------------------ shots
def test_box_score_fga_convention(derived):
    sh = derived["shots_features"]
    # FIBA: a miss while fouled is a free-throw trip, not a field goal attempt
    assert not sh.loc[sh.fouled & ~sh.outcome, "is_box_score_fga"].any()
    assert sh.loc[sh.fouled & sh.outcome, "is_box_score_fga"].all()


def test_points_follow_the_outcome(derived):
    sh = derived["shots_features"]
    assert (sh.loc[~sh.outcome, "points"] == 0).all()
    assert (sh.loc[sh.outcome & sh.three, "points"] == 3).all()
    assert (sh.loc[sh.outcome & ~sh.three, "points"] == 2).all()


def test_team_points_reconcile_with_official_scores(events, dims):
    sh, ft = events["shots"], events["free_throws"]
    pts = (sh[sh.outcome].assign(p=np.where(sh[sh.outcome].three, 3, 2))
             .groupby(["game_id", "off_team_id"]).p.sum()
             .add(ft[ft.outcome].groupby(["game_id", "off_team_id"]).size(), fill_value=0))
    for g in dims["games"].itertuples(index=False):
        assert int(pts.get((g.game_id, g.home_team_id), 0)) == g.home_score
        assert int(pts.get((g.game_id, g.away_team_id), 0)) == g.away_score


def test_xfg_is_a_probability_and_beats_the_base_rate(derived):
    import json
    sh = derived["shots_features"]
    assert sh.xfg.between(0, 1).all()
    m = json.loads((C.DERIVED_DIR / "xfg_metrics.json").read_text())["logistic+spatial"]
    assert m["log_loss"] < m["log_loss_baseline"]
    assert m["brier"] < m["brier_baseline"]
    assert m["auc"] > 0.60


def test_xfg_is_calibrated_within_tolerance():
    cal = pd.read_parquet(C.DERIVED_DIR / "xfg_calibration.parquet")
    assert cal.gap.abs().max() < 0.12
    assert abs((cal.pred * cal.n).sum() / cal.n.sum() -
               (cal.actual * cal.n).sum() / cal.n.sum()) < 0.02


def test_fg_pct_declines_monotonically_with_contest(derived):
    sh = derived["shots_features"]
    order = ["open", "light", "average", "plus"]
    fg = [sh.loc[sh.contest_level == c, "outcome"].mean() for c in order]
    assert fg[0] > fg[-1]
    assert sh.loc[sh.contest_level == "blocked", "outcome"].mean() == 0.0


def test_defender_distance_orders_the_contest_levels(derived):
    sh = derived["shots_features"]
    d = sh.groupby("contest_level").closest_def_dist.mean()
    assert d["open"] > d["average"] > d["plus"]


def test_creation_types_cover_every_shot(derived):
    sh = derived["shots_features"]
    assert sh.creation_type.notna().all()
    assert set(sh.creation_type) <= {"drive", "pick", "handoff", "off_ball_screen",
                                     "isolation", "post", "closeout", "unstructured"}


# ------------------------------------------------------------------ drives
def test_drive_outcome_tokens_are_consistent(derived):
    dr = derived["drives_features"]
    assert (dr.points_direct.isin([0, 2, 3])).all()
    assert (dr.points_created.isin([0, 2, 3])).all()
    assert (dr.loc[dr.turned_over, "points_total"] == 0).all()
    # a drive that ends in a turnover must be flagged as one
    assert dr.loc[dr.end_type == "turnover", "turned_over"].mean() > 0.9


def test_drive_geometry(derived):
    dr = derived["drives_features"].dropna(subset=["location_x", "end_loc_x"])
    assert (dr.duration_s > 0).mean() > 0.95
    # drives move toward the basket on average
    assert dr.drive_gain_ft.mean() > 0


def test_help_defence_closes_during_drives(derived):
    dr = derived["drives_features"].dropna(subset=["help_at_start", "help_min"])
    assert (dr.help_min <= dr.help_at_start + 1e-6).all()
    assert dr.help_closed_ft.mean() > 2.0


# ------------------------------------------------------------------- picks
def test_pick_coverage_vocabulary_matches_the_docs(derived):
    pk = derived["picks_features"]
    assert set(pk.bhr_def_type.dropna()) <= {"none", "ice", "switch", "over", "blitz", "under"}
    assert set(pk.scr_def_type.dropna()) <= {"none", "ice", "soft", "switch", "blitz", "show"}


def test_direct_pick_share_matches_the_documented_rate(derived):
    pk = derived["picks_features"]
    assert 0.45 < pk.direct.mean() < 0.60   # docs: ~52% league-wide


def test_pick_points_are_bounded(derived):
    pk = derived["picks_features"]
    assert pk.bhr_points.between(0, 3).all()
    assert pk.scr_points.between(0, 3).all()


def test_roll_pop_labels_are_exhaustive_where_tracking_exists(derived):
    pk = derived["picks_features"]
    lab = pk.screener_action.dropna()
    assert set(lab) == {"roll", "pop", "stay"}
    assert len(lab) / len(pk) > 0.95


# ----------------------------------------------------------------- spacing
def test_spacing_metrics_on_a_known_configuration():
    xy = np.array([[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]])
    m = G.spacing_metrics(xy)
    assert m["hull_area"] == pytest.approx(100.0)
    assert m["width"] == pytest.approx(10.0)
    assert m["depth"] == pytest.approx(10.0)
    assert m["mean_nn_dist"] == pytest.approx(10.0)


def test_spacing_metrics_degrade_gracefully():
    assert np.isnan(G.spacing_metrics(np.zeros((0, 2)))["hull_area"])
    assert np.isnan(G.spacing_metrics(np.array([[1.0, 2.0]]))["mean_pairwise_dist"])
    assert G.spacing_metrics(np.array([[0.0, 0.0], [3.0, 4.0]]))["mean_pairwise_dist"] == 5.0


def test_defensive_pressure_orders_distances():
    d = G.defensive_pressure(np.array([0.0, 0.0]),
                             np.array([[3.0, 0.0], [8.0, 0.0], [20.0, 0.0]]))
    assert d["nearest_def_dist"] == 3.0
    assert d["second_def_dist"] == 8.0
    assert d["help_gap"] == 5.0
    assert d["n_def_within_4ft"] == 1
    assert d["n_def_within_10ft"] == 2


def test_our_nearest_defender_reproduces_skillcorners_field(events, derived):
    sh = events["shots"].merge(derived["spatial_shots"], on="id", how="inner")
    ok = sh.dropna(subset=["closest_def_dist", "nearest_def_dist"])
    assert len(ok) > 1300
    assert (ok.closest_def_dist - ok.nearest_def_dist).abs().median() < 0.25
    assert (ok.closest_def_dist - ok.nearest_def_dist).abs().le(1.0).mean() > 0.95


def test_court_control_is_a_probability_and_symmetric():
    gx, gy = G.make_grid(20, 12)
    a = np.array([[-20.0, 0.0], [-25.0, 10.0], [-30.0, -10.0]])
    b = np.array([[-22.0, 0.0], [-27.0, 10.0], [-32.0, -10.0]])
    z = G.court_control(a, np.zeros_like(a), b, np.zeros_like(b), gx, gy)
    assert z.min() >= 0 and z.max() <= 1
    z_same = G.court_control(a, np.zeros_like(a), a, np.zeros_like(a), gx, gy)
    assert np.allclose(z_same, 0.5, atol=1e-9)


def test_court_value_surface_is_zero_in_the_backcourt():
    gx, gy = G.make_grid(30, 18)
    v = G.court_value_surface(gx, gy)
    assert (v[gx > 0] == 0).all()
    assert v.max() < 3.0 and v.min() >= 0


# -------------------------------------------------------------- possessions
def test_chance_summary_grain_and_counts(events, derived):
    cs = derived["chance_summary"]
    assert len(cs) == len(events["chances"])
    assert cs.chance_id.is_unique
    assert cs.n_shots.sum() == events["shots"].chance_id.notna().sum()


def test_a_possession_can_contain_several_chances(events):
    per = events["chances"].groupby("possession_id").size()
    assert per.max() > 1
    assert per.mean() > 1.0


def test_event_timeline_is_frame_sorted(events):
    tl = PoE.build_event_timeline(events, int(events["chances"].game_id.iloc[0]))
    assert tl.frame.is_monotonic_increasing
    assert len(tl) > 1000


def test_action_chains_cover_every_chance(events, derived):
    ac = derived["action_chains"]
    assert len(ac) == len(events["chances"])
    assert ac.action_chain.notna().all()


# ----------------------------------------------------------------- defence
def test_closeout_distance_checkpoints_are_ordered(derived):
    co = derived["closeouts_features"]
    # the defender is further away at the start of the closeout than at the end
    assert (co.start_distance > co.end_distance).mean() > 0.9
    assert co.duration_s.between(0, 12).mean() > 0.98


def test_closeout_stop_definition_is_consistent(derived):
    co = derived["closeouts_features"]
    assert (co.loc[co.stop, "total_points_allowed"] == 0).all()
    assert not co.loc[co.stop, "fouled"].any()


def test_matchup_intervals_are_valid(events):
    mu = events["matchups"]
    assert (mu.end_frame >= mu.start_frame).mean() > 0.99
    assert (mu.def_player_id != mu.off_player_id).all()
