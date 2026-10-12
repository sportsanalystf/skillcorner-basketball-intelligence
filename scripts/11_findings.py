"""Generate every number quoted in docs/CASE_STUDIES.md and the README.

Run this before editing the docs — nothing in them is typed by hand.
Output: reports/findings.json + a readable console dump.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
from src import config as C

D = C.DERIVED_DIR
P = C.PROCESSED_DIR
rd = lambda n: pd.read_parquet(D / f"{n}.parquet")
F = {}


def boot(a, b, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    d = np.array([rng.choice(a, len(a), True).mean() - rng.choice(b, len(b), True).mean()
                  for _ in range(n)])
    return float(d.mean()), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


sh = rd("shots_features"); fga = sh[sh.is_box_score_fga]
dr = rd("drives_features"); pk = rd("picks_features"); co = rd("closeouts_features")
ch = rd("chances_transition"); sp = rd("chance_spacing"); ob = rd("offball_before_shots")

# ---------------------------------------------------------------- FINDING 1
s1 = fga.dropna(subset=["closest_def_dist", "dist_calc"])
t, p = stats.ttest_ind(s1.loc[s1.outcome, "closest_def_dist"],
                       s1.loc[~s1.outcome, "closest_def_dist"], equal_var=False)
X = sm.add_constant(s1[["closest_def_dist", "dist_calc"]].assign(three=s1.three.astype(int)))
fit = sm.Logit(s1.outcome.astype(int), X).fit(disp=0)
strata = {}
for z, g in s1.groupby("zone_family"):
    if len(g) < 80: continue
    tt, pp = stats.ttest_ind(g.loc[g.outcome, "closest_def_dist"],
                             g.loc[~g.outcome, "closest_def_dist"], equal_var=False)
    strata[z] = {"n": len(g), "fg": float(g.outcome.mean()),
                 "made_dist": float(g.loc[g.outcome, "closest_def_dist"].mean()),
                 "miss_dist": float(g.loc[~g.outcome, "closest_def_dist"].mean()),
                 "t": float(tt), "p": float(pp)}
F["f1_defender_distance_simpson"] = {
    "n": len(s1),
    "marginal_made_ft": float(s1.loc[s1.outcome, "closest_def_dist"].mean()),
    "marginal_miss_ft": float(s1.loc[~s1.outcome, "closest_def_dist"].mean()),
    "marginal_t": float(t), "marginal_p": float(p),
    "logit_coef_def_dist": float(fit.params["closest_def_dist"]),
    "logit_odds_ratio": float(np.exp(fit.params["closest_def_dist"])),
    "logit_p": float(fit.pvalues["closest_def_dist"]),
    "logit_coef_shot_dist": float(fit.params["dist_calc"]),
    "by_zone": strata,
}

# ---------------------------------------------------------------- FINDING 2
tr, hc = ch[ch.transition], ch[~ch.transition]
t1, p1 = stats.ttest_ind(tr.shot_quality.dropna(), hc.shot_quality.dropna(), equal_var=False)
t2, p2 = stats.ttest_ind(tr.pts_scored, hc.pts_scored, equal_var=False)
F["f2_transition"] = {
    "n_transition": len(tr), "n_halfcourt": len(hc),
    "sq_transition": float(tr.shot_quality.mean()), "sq_halfcourt": float(hc.shot_quality.mean()),
    "sq_t": float(t1), "sq_p": float(p1),
    "contested_transition": float(tr.shot_contested.mean()),
    "contested_halfcourt": float(hc.shot_contested.mean()),
    "ppc_transition": float(tr.pts_scored.mean()), "ppc_halfcourt": float(hc.pts_scored.mean()),
    "ppc_t": float(t2), "ppc_p": float(p2),
    "to_transition": float(tr.outcome.eq("TO").mean()),
    "to_halfcourt": float(hc.outcome.eq("TO").mean()),
    "duration_transition": float(tr.duration_s.mean()),
    "duration_halfcourt": float(hc.duration_s.mean()),
    "time_to_shot_transition": float(tr.time_to_shot_s.mean()),
    "time_to_shot_halfcourt": float(hc.time_to_shot_s.mean()),
}

# ---------------------------------------------------------------- FINDING 3
m = sh.merge(ob, on="id").dropna(subset=["offball_distance_ft"])
q = pd.qcut(m.offball_distance_ft, 4, labels=["Q1", "Q2", "Q3", "Q4"])
g = m.groupby(q, observed=True).agg(
    n=("id", "size"), dist=("offball_distance_ft", "mean"), fg=("outcome", "mean"),
    xfg=("xfg", "mean"), sq=("shot_quality", "mean"),
    open_rate=("contested", lambda s: 1 - s.mean()), sep=("separation_at_shot", "mean"),
    sep_gain=("separation_gained", "mean"))
d_sep, ci_sep = boot(m.loc[q == "Q4", "separation_at_shot"], m.loc[q == "Q1", "separation_at_shot"])
d_fg, ci_fg = boot(m.loc[q == "Q4", "outcome"].astype(float),
                   m.loc[q == "Q1", "outcome"].astype(float))
F["f3_offball"] = {"quartiles": g.round(4).to_dict("index"),
                   "sep_gap_ft": d_sep, "sep_gap_ci": ci_sep,
                   "fg_gap": d_fg, "fg_gap_ci": ci_fg, "n": int(len(m))}

# ---------------------------------------------------------------- FINDING 4
d4 = dr.dropna(subset=["help_at_start", "help_min"])
t4, p4 = stats.ttest_rel(d4.help_at_start, d4.help_min)
col = rd("drive_collapse")
curve = col[col.t_since_start <= 2.0].groupby("t_since_start").agg(
    help=("second_def_dist", "mean"), near=("nearest_def_dist", "mean"),
    paint=("n_def_in_paint", "mean"), n=("drive_id", "size"))
by_end = dr.groupby("end_type").agg(n=("id", "size"), ppd=("points_total", "mean"),
    help_closed=("help_closed_ft", "mean"), paint_added=("paint_defenders_added", "mean"))
succ = dr[dr.successful]; fail = dr[~dr.successful]
d_help, ci_help = boot(succ.help_closed_ft, fail.help_closed_ft)
F["f4_help_collapse"] = {
    "n": len(d4), "help_start_ft": float(d4.help_at_start.mean()),
    "help_min_ft": float(d4.help_min.mean()),
    "help_closed_ft": float(d4.help_closed_ft.mean()),
    "t": float(t4), "p": float(p4),
    "paint_defenders_added": float(dr.paint_defenders_added.mean()),
    "curve": curve.round(3).to_dict("index"),
    "by_end_type": by_end.round(3).to_dict("index"),
    "succ_vs_fail_help_closed": d_help, "succ_vs_fail_ci": ci_help,
    "n_samples": int(len(col)),
}

# ---------------------------------------------------------------- FINDING 5
cov = pk.groupby("bhr_def_type").agg(
    n=("id", "size"), ppp=("pick_points", "mean"), handler_ppp=("bhr_points", "mean"),
    shot=("bhr_shot", "mean"), assist=("bhr_assist", "mean"),
    to=("bhr_turnover", "mean"), foul=("bhr_foul", "mean"), direct=("direct", "mean"))
d5, ci5 = boot(pk.loc[pk.bhr_def_type == "over", "pick_points"],
               pk.loc[pk.bhr_def_type == "under", "pick_points"])
scov = pk.groupby("scr_def_type").agg(n=("id", "size"), ppp=("pick_points", "mean"),
    scr_ppp=("scr_points", "mean"))
F["f5_pick_coverage"] = {"handler_coverage": cov.round(4).to_dict("index"),
                         "screener_coverage": scov.round(4).to_dict("index"),
                         "over_minus_under": d5, "ci": ci5,
                         "roll_rate": float(pk.screener_action.eq("roll").mean()),
                         "pop_rate": float(pk.screener_action.eq("pop").mean())}

# ---------------------------------------------------------------- FINDING 6
tight, loose = co[co.arrived_tight], co[~co.arrived_tight]
tab = pd.crosstab(co.arrived_tight, co.fouled)
chi2, pf, *_ = stats.chi2_contingency(tab)
d6, ci6 = boot(loose.total_points_allowed, tight.total_points_allowed)
by_act = co.groupby("bhr_action").agg(n=("id", "size"), start=("start_distance", "mean"),
    end=("end_distance", "mean"), rate=("close_rate_fps", "mean"),
    pts=("total_points_allowed", "mean"), stop=("stop", "mean"))
F["f6_closeouts"] = {
    "n": len(co), "tight_n": len(tight), "loose_n": len(loose),
    "pts_tight": float(tight.total_points_allowed.mean()),
    "pts_loose": float(loose.total_points_allowed.mean()),
    "pts_gap": d6, "pts_gap_ci": ci6,
    "foul_tight": float(tight.fouled.mean()), "foul_loose": float(loose.fouled.mean()),
    "foul_chi2_p": float(pf),
    "stop_tight": float(tight.stop.mean()), "stop_loose": float(loose.stop.mean()),
    "mean_start_ft": float(co.start_distance.mean()),
    "mean_end_ft": float(co.end_distance.mean()),
    "mean_rate_fps": float(co.close_rate_fps.mean()),
    "by_action": by_act.round(3).to_dict("index"),
}

# ---------------------------------------------------------------- FINDING 7
metrics = json.loads((D / "xfg_metrics.json").read_text())
F["f7_shot_model"] = metrics

# ---------------------------------------------------------------- FINDING 8
chs = pd.read_parquet(P / "events/chances.parquet").rename(columns={"id": "chance_id"})
ms = sp.merge(chs[["chance_id", "pts_scored", "outcome"]], on="chance_id").dropna(subset=["hull_area_mean", "pts_scored"])
qq = pd.qcut(ms.hull_area_mean, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
gg = ms.groupby(qq, observed=True).agg(n=("chance_id", "size"), hull=("hull_area_mean", "mean"),
    ppc=("pts_scored", "mean"), to=("outcome", lambda s: s.eq("TO").mean()))
r, pr = stats.pearsonr(ms.hull_area_mean, ms.pts_scored)
d8, ci8 = boot(ms.loc[qq == "Q5", "pts_scored"], ms.loc[qq == "Q1", "pts_scored"])
F["f8_spacing"] = {"quintiles": gg.round(4).to_dict("index"), "pearson_r": float(r),
                   "pearson_p": float(pr), "q5_minus_q1": d8, "ci": ci8,
                   "n": int(len(ms))}

# ---------------------------------------------------------------- FINDING 9
chain = rd("chain_efficiency")
F["f9_chains"] = {"top": chain.head(12).round(3).to_dict("records"),
                  "bigrams": rd("chain_transitions").head(12).round(3).to_dict("records")}

# --------------------------------------------------------------- FINDING 10
tq = rd("tracking_quality"); smr = rd("smoothing_report")
F["f10_tracking_quality"] = {
    "pct_detected_range": [float(tq.pct_detected.min()), float(tq.pct_detected.max())],
    "mean_pred_error_range": [float(tq.mean_pred_error.min()), float(tq.mean_pred_error.max())],
    "p95_pred_error_range": [float(tq.p95_pred_error.min()), float(tq.p95_pred_error.max())],
    "extrapolated_error_range": [float(tq.mean_pred_error_extrapolated.min()),
                                 float(tq.mean_pred_error_extrapolated.max())],
    "corr_raw_vs_delivered_speed": float(smr.corr_raw_with_delivered.mean()),
    "raw_max_speed": float(smr.raw_max_speed.max()),
    "smoothed_max_speed": float(smr.smoothed_max_speed.max()),
    "raw_mean_speed": float(smr.raw_mean_speed.mean()),
    "smoothed_mean_speed": float(smr.smoothed_mean_speed.mean()),
    "raw_p99": float(smr.raw_p99_speed.mean()), "smoothed_p99": float(smr.smoothed_p99_speed.mean()),
}

# --------------------------------------------------------------- FINDING 11
cr = sh.groupby("creation_type").agg(n=("id", "size"), fg=("outcome", "mean"),
    xfg=("xfg", "mean"), sq=("shot_quality", "mean"), pts=("points", "mean"),
    three=("three", "mean"), open_rate=("contested", lambda s: 1 - s.mean()),
    def_dist=("closest_def_dist", "mean"))
F["f11_creation"] = cr.round(4).to_dict("index")

# --------------------------------------------------------------- FINDING 12
ksel = rd("cluster_k_selection")
F["f12_clustering"] = {
    "silhouette_by_k": ksel.round(4).to_dict("records"),
    "best_k": int(ksel.loc[ksel.silhouette.idxmax(), "k"]),
    "best_silhouette": float(ksel.silhouette.max()),
    "n_players": int(len(rd("season_style_clustered"))),
    "clusters": rd("cluster_descriptions").to_dict("records"),
}

Path(C.REPORTS_DIR / "findings.json").write_text(json.dumps(F, indent=1, default=str))
print(json.dumps({k: (list(v.keys())[:6] if isinstance(v, dict) else v)
                  for k, v in F.items()}, indent=1)[:1200])
print("\n--- F1 defender distance ---")
print(json.dumps(F["f1_defender_distance_simpson"], indent=1, default=str))
print("\n--- F3 off-ball quartiles ---")
print(pd.DataFrame(F["f3_offball"]["quartiles"]).T.round(3).to_string())
print("sep gap", round(F["f3_offball"]["sep_gap_ft"], 3), F["f3_offball"]["sep_gap_ci"])
print("fg gap", round(F["f3_offball"]["fg_gap"], 4), F["f3_offball"]["fg_gap_ci"])
print("\n--- F8 spacing quintiles ---")
print(pd.DataFrame(F["f8_spacing"]["quintiles"]).T.round(3).to_string())
print("r =", round(F["f8_spacing"]["pearson_r"], 4), "p =", round(F["f8_spacing"]["pearson_p"], 5),
      "Q5-Q1", round(F["f8_spacing"]["q5_minus_q1"], 3), F["f8_spacing"]["ci"])
