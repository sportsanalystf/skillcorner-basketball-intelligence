"""Build and execute the analysis notebooks so they ship with real outputs."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"; NB.mkdir(exist_ok=True)

HEADER = '''import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "..")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from src import config as C
pd.set_option("display.width", 160, "display.max_columns", 60)
plt.rcParams.update({"figure.figsize": (9, 4.5), "axes.grid": True,
                     "grid.alpha": .25, "figure.dpi": 110})
E = C.PROCESSED_DIR / "events"; D = C.DERIVED_DIR
rd  = lambda n: pd.read_parquet(D / f"{n}.parquet")
rev = lambda n: pd.read_parquet(E / f"{n}.parquet")
print("project:", C.PROJECT_ROOT.name)'''

SPECS = {
"01_data_exploration": [
 ("md", "# 01 · Data exploration\n\nWhat is actually in the SkillCorner basketball open data, "
        "measured rather than assumed. Everything below reads the canonical Parquet layer built "
        "by `scripts/01_ingest.py`."),
 ("code", HEADER),
 ("md", "## Coverage"),
 ("code", '''games = pd.read_parquet(C.PROCESSED_DIR / "dim/games.parquet")
print(f"{len(games)} games, {games.home_team_id.nunique() + games.away_team_id.nunique()} team slots")
games[["game_id","date","home_team_name","home_score","away_score","away_team_name"]]'''),
 ("md", "## Event volume by family\n\nTwenty tables, four orders of magnitude apart in size."),
 ("code", '''rows = {p.stem: len(pd.read_parquet(p, columns=["id"])) for p in sorted(E.glob("*.parquet"))}
s = pd.Series(rows).sort_values()
print("total event rows:", f"{s.sum():,}")
ax = s.plot.barh(color="#FF6B35", logx=True); ax.set_xlabel("rows (log)"); plt.tight_layout(); plt.show()
s.to_frame("rows")'''),
 ("md", "## The chance is the analytic unit\n\nA possession can contain several chances "
        "(offensive rebounds). Per-chance is the right denominator for efficiency."),
 ("code", '''ch, po = rev("chances"), rev("possessions")
print(f"{len(po):,} possessions -> {len(ch):,} chances  ({len(ch)/len(po):.2f} chances per possession)")
print("\\npoints per chance:", round(ch.pts_scored.mean(), 3))
print("\\nchances per possession:"); print(ch.groupby("possession_id").size().value_counts().sort_index())
print("\\nhow chances start:"); print(ch.start_type.value_counts())
print("\\nhow chances end:"); print(ch.outcome.value_counts())'''),
 ("md", "## Score reconciliation\n\nThe first thing to check on any event feed: do the events "
        "reproduce the official score? All ten games, exactly."),
 ("code", '''sh, ft = rev("shots"), rev("free_throws")
pts = (sh[sh.outcome].assign(p=np.where(sh[sh.outcome].three, 3, 2))
         .groupby(["game_id","off_team_id"]).p.sum()
         .add(ft[ft.outcome].groupby(["game_id","off_team_id"]).size(), fill_value=0))
chk = [{"game": g.game_id,
        "home_calc": int(pts.get((g.game_id, g.home_team_id), 0)), "home_official": g.home_score,
        "away_calc": int(pts.get((g.game_id, g.away_team_id), 0)), "away_official": g.away_score}
       for g in games.itertuples(index=False)]
chk = pd.DataFrame(chk)
chk["match"] = (chk.home_calc == chk.home_official) & (chk.away_calc == chk.away_official)
print("all games reconcile:", chk.match.all()); chk'''),
 ("md", "## Enum vocabularies, as delivered\n\nNo coverage or contest label in this project is "
        "invented; these are the exact value sets in the data."),
 ("code", '''pk, dr = rev("picks"), rev("drives")
for lbl, s in [("shots.contest_level", sh.contest_level), ("shots.shot_type", sh.shot_type),
               ("picks.bhr_def_type", pk.bhr_def_type), ("picks.scr_def_type", pk.scr_def_type),
               ("drives.category", dr.category), ("drives.end_type", dr.end_type)]:
    print(f"{lbl:24s} {dict(s.value_counts())}")'''),
 ("md", "## Missingness\n\nWhere the feed is incomplete, and by how much."),
 ("code", '''miss = (sh[["location_x","closest_def_dist","shot_quality","release_time",
             "dribbles_before","passer_id","touch_id"]].isna().mean()*100).round(2)
print("shots — % null:"); print(miss)
print("\\nfouled shots (excluded from box-score FGA):", int(sh.fouled.sum()),
      "of", len(sh), f"({sh.fouled.mean():.1%})")'''),
],

"02_tracking_validation": [
 ("md", "# 02 · Tracking validation\n\nHow good is broadcast tracking, and what does smoothing "
        "actually change? Both questions answered from the raw frames."),
 ("code", HEADER),
 ("md", "## Detection and prediction error, per game\n\nThese reproduce SkillCorner's own "
        "published ranges (76–83 % detected, 1.5–1.9 ft mean error) from an independent computation."),
 ("code", '''tq = rd("tracking_quality")
print(f"detected: {tq.pct_detected.min():.1f}–{tq.pct_detected.max():.1f} %")
print(f"mean predError: {tq.mean_pred_error.min():.2f}–{tq.mean_pred_error.max():.2f} ft")
print(f"  detected {tq.mean_pred_error_detected.mean():.2f} ft | "
      f"extrapolated {tq.mean_pred_error_extrapolated.mean():.2f} ft")
tq[["game_id","frames","pct_detected","mean_pred_error","p95_pred_error",
    "mean_pred_error_detected","mean_pred_error_extrapolated","pct_rows_in_possession"]].round(2)'''),
 ("code", '''fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].bar(range(len(tq)), tq.pct_detected, color="#FF6B35")
ax[0].set_ylim(70, 88); ax[0].set_title("% positions detected"); ax[0].set_xticks(range(len(tq)))
ax[0].set_xticklabels(tq.game_id, rotation=60, fontsize=7)
w = .38; x = np.arange(len(tq))
ax[1].bar(x-w/2, tq.mean_pred_error_detected, w, label="detected", color="#3DDC97")
ax[1].bar(x+w/2, tq.mean_pred_error_extrapolated, w, label="extrapolated", color="#F2545B")
ax[1].set_title("mean predError (ft)"); ax[1].legend()
ax[1].set_xticks(x); ax[1].set_xticklabels(tq.game_id, rotation=60, fontsize=7)
plt.tight_layout(); plt.show()'''),
 ("md", "## Raw versus smoothed kinematics\n\nThe headline: the delivered positions are already "
        "temporally smooth. Our Savitzky–Golay filter clips the tail and changes nothing else."),
 ("code", '''sm = rd("smoothing_report")
print(sm[["game_id","raw_mean_speed","smoothed_mean_speed","raw_p99_speed","smoothed_p99_speed",
          "raw_max_speed","smoothed_max_speed","corr_raw_with_delivered"]].round(3).to_string(index=False))
print(f"\\ncorrelation of RAW central-difference speed with SkillCorner's own `speed` field: "
      f"{sm.corr_raw_with_delivered.mean():.4f}")
print(f"max speed {sm.raw_max_speed.max():.1f} -> {sm.smoothed_max_speed.max():.1f} ft/s")'''),
 ("md", "## One player, one possession: what the filter does"),
 ("code", '''from src.tracking import engine
gid = int(tq.game_id.iloc[0])
t = pd.read_parquet(C.PROCESSED_DIR / f"tracking_enriched/{gid}.parquet")
seg = t[t.speed.notna()].groupby(["player_id","segment_id"]).size().sort_values()
pid, sid = seg.index[len(seg)//2]
w = t[(t.player_id == pid) & (t.segment_id == sid)].head(300)
fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
ax[0].plot(w.frame_idx, w.x, lw=.8, alpha=.55, label="raw x", color="#F2545B")
ax[0].plot(w.frame_idx, w.x_s, lw=1.8, label="smoothed x", color="#FF6B35")
ax[0].set_ylabel("x (ft)"); ax[0].legend()
ax[1].plot(w.frame_idx, w.speed_raw, lw=.8, alpha=.55, label="delivered speed", color="#5B8DEF")
ax[1].plot(w.frame_idx, w.speed, lw=1.8, label="our smoothed speed", color="#3DDC97")
ax[1].set_ylabel("speed (ft/s)"); ax[1].set_xlabel("frame"); ax[1].legend()
plt.suptitle(f"player {pid}, segment {sid}"); plt.tight_layout(); plt.show()'''),
 ("md", "## Speed distribution and the movement bands"),
 ("code", '''sp = t.speed.dropna()
print(sp.describe(percentiles=[.5,.9,.99,.999]).round(2))
plt.hist(sp, bins=120, color="#FF6B35", alpha=.85)
for _, lo, hi in C.SPEED_BANDS[:-1]:
    plt.axvline(hi, color="#8A97AA", lw=.8, ls="--")
plt.xlim(0, 30); plt.xlabel("smoothed speed (ft/s)"); plt.ylabel("frames")
plt.title("speed distribution with our movement-intensity band edges"); plt.show()
print(t.speed_band.value_counts(normalize=True).round(4))'''),
 ("md", "## Frame continuity\n\nThe video frame index is continuous *within* a period and jumps "
        "at every period boundary — a property not stated in the source documentation."),
 ("code", '''b = pd.read_parquet(C.PROCESSED_DIR / f"tracking/ball/{gid}.parquet",
                    columns=["frame_idx","period","game_clock"]).sort_values("frame_idx")
f = b.frame_idx.to_numpy(); step = np.diff(f); per = b.period.to_numpy()
gaps = np.flatnonzero(step != 1)
print(f"{len(gaps)} gaps in {len(f):,} frames")
for i in gaps:
    print(f"  frame {f[i]:>7,} -> {f[i+1]:>7,}  (gap {step[i]:>5,})  period {per[i]} -> {per[i+1]}")
print("\\nmid-period gaps:", int(((step != 1) & (per[1:] == per[:-1])).sum()))'''),
],

"03_shot_analysis": [
 ("md", "# 03 · Shot analysis\n\nShot profile, the expected-FG model, and the Simpson's paradox "
        "in defender distance."),
 ("code", HEADER),
 ("code", '''sh = rd("shots_features"); fga = sh[sh.is_box_score_fga]
print(f"{len(sh):,} shot events, {len(fga):,} box-score FGA, FG% {fga.outcome.mean():.3f}")
sh.groupby("zone").agg(n=("id","size"), fg=("outcome","mean"), pts=("points","mean"),
                       xfg=("xfg","mean"), sq=("shot_quality","mean"),
                       def_dist=("closest_def_dist","mean")).round(3)'''),
 ("md", "## Contest level behaves monotonically"),
 ("code", '''order = ["open","light","average","plus","blocked"]
g = (fga.groupby("contest_level").agg(n=("id","size"), fg=("outcome","mean"),
     def_dist=("closest_def_dist","mean"), sq=("shot_quality","mean")).reindex(order))
print(g.round(3))
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].bar(g.index, g.fg, color="#FF6B35"); ax[0].set_title("FG% by contest level")
ax[1].bar(g.index, g.def_dist, color="#5B8DEF"); ax[1].set_title("mean defender distance (ft)")
plt.tight_layout(); plt.show()'''),
 ("md", "## The Simpson's paradox\n\nMarginally, defender distance says nothing about whether a "
        "shot goes in. The confound is that the best shots are the most contested."),
 ("code", '''from scipy import stats
s1 = fga.dropna(subset=["closest_def_dist","dist_calc"])
t, p = stats.ttest_ind(s1.loc[s1.outcome,"closest_def_dist"],
                       s1.loc[~s1.outcome,"closest_def_dist"], equal_var=False)
print(f"MARGINAL  made {s1.loc[s1.outcome,'closest_def_dist'].mean():.2f} ft  "
      f"missed {s1.loc[~s1.outcome,'closest_def_dist'].mean():.2f} ft   t={t:+.2f}  p={p:.3f}")
print("\\nthe confound:")
print(s1.groupby("zone_family").agg(n=("id","size"), fg=("outcome","mean"),
      def_dist=("closest_def_dist","mean")).round(3))
print("\\nWITHIN STRATA:")
for z, g_ in s1.groupby("zone_family"):
    if len(g_) < 80: continue
    tt, pp = stats.ttest_ind(g_.loc[g_.outcome,"closest_def_dist"],
                             g_.loc[~g_.outcome,"closest_def_dist"], equal_var=False)
    print(f"  {z:22s} n={len(g_):4d}  made {g_.loc[g_.outcome,'closest_def_dist'].mean():.2f}  "
          f"miss {g_.loc[~g_.outcome,'closest_def_dist'].mean():.2f}  t={tt:+.2f}  p={pp:.4f}")'''),
 ("code", '''import statsmodels.api as sm_
X = sm_.add_constant(s1[["closest_def_dist","dist_calc"]].assign(three=s1.three.astype(int)))
fit = sm_.Logit(s1.outcome.astype(int), X).fit(disp=0)
print(fit.summary2().tables[1].round(4))
print(f"\\nodds ratio per extra foot of separation: {np.exp(fit.params['closest_def_dist']):.3f}")'''),
 ("md", "## The xFG model: performance and calibration"),
 ("code", '''import json
m = json.loads((D / "xfg_metrics.json").read_text())
print(pd.DataFrame({k: v for k, v in m.items() if isinstance(v, dict)}).T.round(4))
print(f"\\nSkillCorner shotQuality AUC: {m['skillcorner_shotquality_auc']:.3f}"
      f"  (correlation with ours {m['skillcorner_vs_our_xfg_corr']:.3f})")
cal = rd("xfg_calibration")
plt.plot([0,1],[0,1],"--",color="#8A97AA",lw=1)
plt.plot(cal.pred, cal.actual, "o-", color="#FF6B35", ms=7)
plt.xlabel("predicted P(make)"); plt.ylabel("observed"); plt.title("xFG calibration (out-of-fold)")
plt.show(); cal.round(3)'''),
 ("md", "## Shot creation"),
 ("code", '''sh.groupby("creation_type").agg(n=("id","size"), fg=("outcome","mean"), xfg=("xfg","mean"),
   sq=("shot_quality","mean"), pts=("points","mean"), three=("three","mean"),
   open_rate=("contested", lambda s: 1-s.mean()),
   def_dist=("closest_def_dist","mean")).sort_values("n", ascending=False).round(3)'''),
],

"04_drive_analysis": [
 ("md", "# 04 · Drive analysis\n\nWhat drives produce, and how fast the defence collapses."),
 ("code", HEADER),
 ("code", '''dr = rd("drives_features")
print(f"{len(dr)} drives | {dr.points_total.mean():.3f} pts/drive | "
      f"{dr.successful.mean():.1%} successful | {dr.blowby.mean():.1%} blowby")
dr.groupby("end_type").agg(n=("id","size"), ppd=("points_total","mean"),
  foul=("drew_foul","mean"), to=("turned_over","mean"), gain=("drive_gain_ft","mean"),
  help_closed=("help_closed_ft","mean"), paint_added=("paint_defenders_added","mean")).round(3)'''),
 ("code", '''dr.groupby("category").agg(n=("id","size"), ppd=("points_total","mean"),
  blowby=("blowby","mean"), paint=("reached_paint","mean"),
  kickout=("passed_out","mean")).sort_values("n", ascending=False).round(3)'''),
 ("md", "## The defensive collapse curve\n\nSecond-nearest defender distance, sampled every "
        "0.2 s from drive start, split by how the drive ended."),
 ("code", '''col = rd("drive_collapse").merge(dr[["id","end_type","successful"]],
                                 left_on="drive_id", right_on="id")
cc = col[col.t_since_start <= 2.6]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
for k, g_ in cc.groupby("end_type"):
    if g_.drive_id.nunique() < 15: continue
    ax[0].plot(*g_.groupby("t_since_start").second_def_dist.mean().reset_index().values.T, lw=2, label=k)
    ax[1].plot(*g_.groupby("t_since_start").n_def_in_paint.mean().reset_index().values.T, lw=2, label=k)
ax[0].set_xlabel("s since drive start"); ax[0].set_ylabel("help defender distance (ft)"); ax[0].legend(fontsize=8)
ax[1].set_xlabel("s since drive start"); ax[1].set_ylabel("defenders in the paint"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()'''),
 ("code", '''from scipy import stats
d = dr.dropna(subset=["help_at_start","help_min"])
t, p = stats.ttest_rel(d.help_at_start, d.help_min)
print(f"help closes {d.help_closed_ft.mean():.2f} ft "
      f"({d.help_at_start.mean():.2f} -> {d.help_min.mean():.2f}), paired t={t:.1f}, p={p:.2g}")
print(f"paint defenders added: {dr.paint_defenders_added.mean():.2f}")
dr.groupby("successful").agg(n=("id","size"), gain=("drive_gain_ft","mean"),
  blowby=("blowby","mean"), help_closed=("help_closed_ft","mean"),
  paint_added=("paint_defenders_added","mean"), hull_min=("hull_min","mean")).round(3)'''),
],

"05_pick_roll_analysis": [
 ("md", "# 05 · Pick-and-roll analysis\n\nCoverage, outcome, and the tracking-derived roll/pop label."),
 ("code", HEADER),
 ("code", '''pk = rd("picks_features")
print(f"{len(pk)} picks | direct {pk.direct.mean():.1%} | {pk.pick_points.mean():.3f} pts/pick")
print("\\nscreener action (derived here from tracking):")
print(pk.screener_action.value_counts(normalize=True).round(3))
pk.groupby("bhr_def_type").agg(n=("id","size"), ppp=("pick_points","mean"),
  handler=("bhr_points","mean"), screener=("scr_points","mean"), shot=("bhr_shot","mean"),
  assist=("bhr_assist","mean"), to=("bhr_turnover","mean"), foul=("bhr_foul","mean"),
  direct=("direct","mean")).sort_values("n", ascending=False).round(4)'''),
 ("code", '''from scipy import stats
rng = np.random.default_rng(0)
a = pk.loc[pk.bhr_def_type=="over","pick_points"].to_numpy(float)
b = pk.loc[pk.bhr_def_type=="under","pick_points"].to_numpy(float)
dd = np.array([rng.choice(a,len(a),True).mean()-rng.choice(b,len(b),True).mean() for _ in range(4000)])
print(f"over − under = {dd.mean():+.4f} pts/pick, 95% CI [{np.percentile(dd,2.5):+.3f}, "
      f"{np.percentile(dd,97.5):+.3f}]  -> direction consistent, not separable")'''),
 ("md", "## Coverage-pair matrix"),
 ("code", '''piv = pk.pivot_table(index="bhr_def_type", columns="scr_def_type", values="pick_points", aggfunc="mean")
cnt = pk.pivot_table(index="bhr_def_type", columns="scr_def_type", values="id", aggfunc="size")
print("counts:"); print(cnt.fillna(0).astype(int))
print("\\npoints per pick (cells with n>=8):"); (piv.where(cnt>=8)).round(3)'''),
 ("md", "## Screen location"),
 ("code", '''pk.groupby("location_type").agg(n=("id","size"), ppp=("pick_points","mean"),
  handler=("bhr_points","mean"), screener=("scr_points","mean"),
  switch_forced=("scr_def_type", lambda s: s.eq("switch").mean()),
  direct=("direct","mean")).round(3)'''),
 ("code", '''rp = pk.dropna(subset=["screener_action"])
rp.groupby(["screener_action","scr_def_type"]).agg(n=("id","size"), ppp=("pick_points","mean"),
  scr_ppp=("scr_points","mean")).query("n >= 5").round(3)'''),
],

"06_spatial_analysis": [
 ("md", "# 06 · Spatial analysis\n\nSpacing metrics, their relationship to outcomes, and the "
        "court-control model."),
 ("code", HEADER),
 ("code", '''sp = rd("chance_spacing"); ch = rev("chances").rename(columns={"id":"chance_id"})
m = sp.merge(ch[["chance_id","pts_scored","outcome","transition"]], on="chance_id").dropna(subset=["pts_scored"])
print(f"{len(m):,} chances with spacing samples")
m[["hull_area_mean","mean_nn_dist_mean","width_mean","depth_mean","spacing_entropy_mean",
   "n_in_corners_mean","n_in_paint_mean","n_def_in_paint_mean"]].describe().round(2)'''),
 ("code", '''from scipy import stats
q = pd.qcut(m.hull_area_mean, 5, labels=["Q1 tightest","Q2","Q3","Q4","Q5 widest"])
g = m.groupby(q, observed=True).agg(n=("chance_id","size"), hull=("hull_area_mean","mean"),
    ppc=("pts_scored","mean"), to=("outcome", lambda s: s.eq("TO").mean()))
print(g.round(3))
r, p = stats.pearsonr(m.hull_area_mean, m.pts_scored)
print(f"\\nPearson r(hull, points) = {r:.4f}  p = {p:.3f}  -> no reliable relationship")
g.ppc.plot.bar(color="#FF6B35"); plt.ylabel("points per chance"); plt.title("by hull-area quintile")
plt.tight_layout(); plt.show()'''),
 ("md", "## Spacing does move the inputs to a good shot"),
 ("code", '''sh = rd("shots_features").merge(sp[["chance_id","hull_area_mean"]], on="chance_id").dropna(subset=["hull_area_mean"])
q2 = pd.qcut(sh.hull_area_mean, 5, labels=["Q1","Q2","Q3","Q4","Q5"])
sh.groupby(q2, observed=True).agg(n=("id","size"), fg=("outcome","mean"), xfg=("xfg","mean"),
  sq=("shot_quality","mean"), def_dist=("closest_def_dist","mean"),
  open_rate=("contested", lambda s: 1-s.mean())).round(3)'''),
 ("md", "## Court control at one instant"),
 ("code", '''from src.spatial import geometry as G
from src.preprocessing import court
gid = int(sh.game_id.iloc[0])
row = sh[(sh.game_id == gid) & sh.location_x.notna()].iloc[10]
t = pd.read_parquet(C.PROCESSED_DIR / f"tracking_enriched/{gid}.parquet",
                    columns=["frame_idx","team_id","x_att","y_att","vx_att","vy_att"],
                    filters=[("frame_idx","=",int(row.start_frame))])
off = t[t.team_id == row.off_team_id]; dfn = t[t.team_id != row.off_team_id]
oxy = off[["x_att","y_att"]].to_numpy(float); dxy = dfn[["x_att","y_att"]].to_numpy(float)
ov = np.nan_to_num(off[["vx_att","vy_att"]].to_numpy(float)); dv = np.nan_to_num(dfn[["vx_att","vy_att"]].to_numpy(float))
gx, gy = G.make_grid(90, 50)
ctrl = G.court_control(oxy, ov, dxy, dv, gx, gy)
val  = G.court_value_surface(gx, gy)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
for a, z, ttl, cm in [(ax[0], ctrl, "court control (1 = offence)", "coolwarm"),
                      (ax[1], val, "court value (expected pts/shot)", "viridis"),
                      (ax[2], (ctrl-.5)*2*val, "spatial advantage", "coolwarm")]:
    im = a.pcolormesh(gx, gy, z, cmap=cm, shading="auto")
    a.scatter(*oxy.T, c="#FF6B35", s=45, ec="k", zorder=3)
    a.scatter(*dxy.T, c="#5B8DEF", s=45, ec="k", zorder=3)
    a.scatter([row.location_x],[row.location_y], marker="*", s=200, c="gold", ec="k", zorder=4)
    a.set_title(ttl, fontsize=10); a.set_aspect("equal"); a.axis("off"); plt.colorbar(im, ax=a, shrink=.8)
plt.tight_layout(); plt.show()
print(f"offence controls {(ctrl>.5).mean():.0%} of the half court at this instant")'''),
],

"07_defensive_analysis": [
 ("md", "# 07 · Defensive analysis\n\nCloseouts, coverage and matchups — and what the data "
        "does not support."),
 ("code", HEADER),
 ("code", '''co = rd("closeouts_features")
print(f"{len(co)} closeouts | start {co.start_distance.mean():.1f} ft -> end "
      f"{co.end_distance.mean():.1f} ft | rate {co.close_rate_fps.mean():.1f} ft/s")
co.groupby("bhr_action").agg(n=("id","size"), start=("start_distance","mean"),
  end=("end_distance","mean"), rate=("close_rate_fps","mean"),
  pts=("total_points_allowed","mean"), stop=("stop","mean"), foul=("fouled","mean")).round(3)'''),
 ("code", '''from scipy import stats
tight, loose = co[co.arrived_tight], co[~co.arrived_tight]
print(f"tight (<=4 ft, n={len(tight)}): {tight.total_points_allowed.mean():.3f} pts, "
      f"{tight.fouled.mean():.1%} fouls, {tight.stop.mean():.1%} stops")
print(f"loose     (n={len(loose)}): {loose.total_points_allowed.mean():.3f} pts, "
      f"{loose.fouled.mean():.1%} fouls, {loose.stop.mean():.1%} stops")
chi2, p, *_ = stats.chi2_contingency(pd.crosstab(co.arrived_tight, co.fouled))
print(f"foul-rate difference: chi2 p = {p:.4f}")
b = pd.cut(co.end_distance, [0,2,3,4,5,6,8,30])
co.groupby(b, observed=True).agg(n=("id","size"), pts=("total_points_allowed","mean"),
  stop=("stop","mean"), foul=("fouled","mean")).round(3)'''),
 ("md", "## Coverage mix by defending team"),
 ("code", '''pk = rd("picks_features")
teams = dict(zip(*pd.read_parquet(C.PROCESSED_DIR/"dim/teams.parquet")[["team_id","team_name"]].values.T))
pk["def_team"] = pk.def_team_id.map(teams)
mix = pk.groupby("def_team").agg(picks=("id","size"), ppp_allowed=("pick_points","mean"),
  over=("bhr_def_type", lambda s: s.eq("over").mean()),
  under=("bhr_def_type", lambda s: s.eq("under").mean()),
  switch=("bhr_def_type", lambda s: s.eq("switch").mean())).sort_values("ppp_allowed")
mix.round(3)'''),
 ("md", "## Matchup workload\n\nRead as activity, not as a rating."),
 ("code", '''mw = rd("matchup_workload")
mw.nlargest(15, "minutes_guarding")[["player_name","team_name","games","assignments",
  "minutes_guarding","distinct_opponents","shots_allowed","points_allowed",
  "points_allowed_per_36"]].round(2)'''),
],

"08_player_profiles": [
 ("md", "# 08 · Player profiles, similarity and role clustering"),
 ("code", HEADER),
 ("code", '''gl = rd("player_game_profile")
print(f"{len(gl)} players with a tracked profile")
gl[gl.minutes_tracked >= 8].nlargest(12, "points")[["player_name","team_name","games",
  "minutes_tracked","points","fga","fg_pct","xfg","shot_value_added","drives","drive_ppd",
  "picks_used","pick_ppp","distance_per_min","p95_speed"]].round(3)'''),
 ("md", "## Style space: PCA on the season aggregates"),
 ("code", '''from src.models.similarity import StyleSpace
style = rd("season_style")
ss = StyleSpace(style)
print(f"{len(style)} player-team seasons, {len(ss.features)} features -> "
      f"{ss.n_components} components, {ss.explained:.1%} variance")
load = pd.DataFrame({"feature": ss.features,
                     **{f"pc{i+1}": ss.pca.components_[i] for i in range(3)}})
load.reindex(load.pc1.abs().sort_values(ascending=False).index).round(3)'''),
 ("md", "## Choosing k — and finding that there is no good k"),
 ("code", '''kt = ss.choose_k(); print(kt.round(4).to_string(index=False))
plt.plot(kt.k, kt.silhouette, "o-", color="#FF6B35")
plt.xlabel("k"); plt.ylabel("silhouette"); plt.title("no k separates cleanly — the space is a continuum")
plt.show()'''),
 ("code", '''ss.cluster(int(kt.loc[kt.silhouette.idxmax(),"k"]))
print(ss.describe_clusters().to_string(index=False))
print()
print(ss.representatives().round(2).to_string(index=False))'''),
 ("code", '''fig, ax = plt.subplots(figsize=(8.5, 6.5))
for c_ in sorted(set(ss.labels)):
    sel = ss.labels == c_
    ax.scatter(ss.P[sel,0], ss.P[sel,1], s=28, alpha=.8, label=f"cluster {c_}")
for i in np.argsort(-style.attempts.to_numpy())[:14]:
    ax.annotate(style.player_name.iloc[i], (ss.P[i,0], ss.P[i,1]), fontsize=7, alpha=.75)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(); ax.set_title("offensive style space")
plt.tight_layout(); plt.show()'''),
 ("md", "## Similarity"),
 ("code", '''i = int(style.index[style.player_name.str.contains("Rubio", na=False)][0]) \\
    if style.player_name.str.contains("Rubio", na=False).any() else int(style.attempts.idxmax())
print("most similar to", style.loc[i,"player_name"], "—", style.loc[i,"team_name"])
ss.most_similar(i, 10).round(3)'''),
],
}

def build(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
                for kind, src in cells]
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": sys.version.split()[0]}}
    return nb

ok = True
for name, cells in SPECS.items():
    t0 = time.time()
    nb = build(name, cells)
    try:
        NotebookClient(nb, timeout=900, kernel_name="python3",
                       resources={"metadata": {"path": str(NB)}}).execute()
        status = "ok"
    except Exception as exc:                      # noqa: BLE001
        status = f"FAILED: {type(exc).__name__}: {str(exc)[:300]}"
        ok = False
    nbf.write(nb, NB / f"{name}.ipynb")
    print(f"  {name:26s} {len(nb.cells):2d} cells  {time.time()-t0:5.1f}s  {status}")

raise SystemExit(0 if ok else 1)
