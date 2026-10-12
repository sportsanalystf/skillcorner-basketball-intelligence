"""Shot Intelligence — profile, quality model, and how shots get created."""
from __future__ import annotations
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from app.components import data as DATA, theme as TH, courtplot as CP

TH.apply("Shot Intelligence")
TH.header("Shot Intelligence", "Where, how, against what — and whether it went in",
          "Shot profile, an interpretable expected-FG model, and the action that "
          "created each attempt.")

sh = DATA.load("shots_features")
names = DATA.player_name_map()
teams = DATA.teams()
sh["player"] = sh.shooter_id_canonical.map(names)
sh["team"] = sh.off_team_id.map(teams)

f1, f2, f3, f4 = st.columns(4)
team = f1.selectbox("Team", ["All"] + sorted(sh.team.dropna().unique().tolist()))
d = sh if team == "All" else sh[sh.team == team]
plyr = f2.selectbox("Player", ["All"] + sorted(d.player.dropna().unique().tolist()))
d = d if plyr == "All" else d[d.player == plyr]
crea = f3.multiselect("Creation", sorted(sh.creation_type.dropna().unique()), default=[])
if crea: d = d[d.creation_type.isin(crea)]
cont = f4.multiselect("Contest", ["open", "light", "average", "plus", "blocked"], default=[])
if cont: d = d[d.contest_level.isin(cont)]

fga = d[d.is_box_score_fga]
TH.kpis([
    ("Attempts", f"{len(fga):,}", f"{len(d):,} shot events", "accent"),
    ("FG%", f"{fga.outcome.mean():.1%}" if len(fga) else "–", ""),
    ("eFG%", f"{((fga.outcome & fga.three).sum()*1.5 + (fga.outcome & ~fga.three).sum())/max(len(fga),1):.1%}", ""),
    ("Pts / shot", f"{d.points.mean():.2f}", ""),
    ("Expected FG%", f"{fga.xfg.mean():.1%}", "our model"),
    ("vs expected", f"{(fga.outcome.mean()-fga.xfg.mean())*100:+.1f} pp",
     "make rate − model", "good" if fga.outcome.mean() > fga.xfg.mean() else ""),
    ("SkillCorner SQ", f"{d.shot_quality.mean():.1f}", "0–100 index"),
    ("Mean def dist", f"{d.closest_def_dist.mean():.1f} ft", ""),
])

t1, t2, t3, t4, t5 = st.tabs(["Shot map", "Quality model", "Creation", "Contest & defender", "Players"])

with t1:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(CP.shot_chart(d, height=470, title="Attempts",
                        hover=["player", "region", "shot_type", "contest_level"]),
                        width='stretch', config={"displayModeBar": False})
    with c2:
        st.plotly_chart(CP.hexbin_court(fga, value_col="outcome", gridsize=3.4,
                        min_count=4, height=470, title="FG% surface",
                        cbar_title="FG%"),
                        width='stretch', config={"displayModeBar": False})
    z = (d.groupby("zone").agg(attempts=("id", "size"), fg=("outcome", "mean"),
                               pts=("points", "mean"), xfg=("xfg", "mean"),
                               sq=("shot_quality", "mean"),
                               def_dist=("closest_def_dist", "mean"))
           .reset_index().sort_values("attempts", ascending=False))
    z["vs_expected"] = z.fg - z.xfg
    st.dataframe(z.round(3), hide_index=True, width='stretch')

with t2:
    metrics = json.loads((DATA.D / "xfg_metrics.json").read_text())
    m = metrics["logistic+spatial"]
    TH.kpis([
        ("Out-of-fold AUC", f"{m['auc']:.3f}", "5-fold, grouped by game", "accent"),
        ("Log loss", f"{m['log_loss']:.4f}", f"baseline {m['log_loss_baseline']:.4f}"),
        ("Brier", f"{m['brier']:.4f}", f"baseline {m['brier_baseline']:.4f}"),
        ("SkillCorner SQ AUC", f"{metrics['skillcorner_shotquality_auc']:.3f}",
         "their model, same shots", "cool"),
        ("Correlation", f"{metrics['skillcorner_vs_our_xfg_corr']:.3f}", "ours vs theirs"),
        ("Spatial features add", f"{m['auc']-metrics['logistic_no_spatial']['auc']:+.3f}", "AUC"),
    ])
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### Calibration")
        cal = DATA.load("xfg_calibration")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(color=TH.LINE, dash="dash"), name="perfect"))
        fig.add_trace(go.Scatter(x=cal.pred, y=cal.actual, mode="markers+lines",
                                 marker=dict(size=10 + cal.n / 25, color=TH.ACCENT),
                                 line=dict(color=TH.ACCENT, width=2), name="observed",
                                 customdata=cal.n,
                                 hovertemplate="pred %{x:.3f}<br>actual %{y:.3f}"
                                               "<br>n=%{customdata}<extra></extra>"))
        fig.update_layout(height=380, xaxis_title="predicted P(make)",
                          yaxis_title="observed make rate")
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    with c2:
        st.markdown("### What drives the model")
        co = DATA.load("xfg_coefficients").head(18)
        fig = go.Figure(go.Bar(x=co.coef, y=co.feature, orientation="h",
                               marker_color=np.where(co.coef > 0, TH.ACCENT_2, TH.BAD)))
        fig.update_layout(height=380, xaxis_title="log-odds coefficient",
                          yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=20, b=30))
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    TH.caveat(
        "<b>This is our model, not SkillCorner's.</b> The <code>shotQuality</code> column "
        "on the shots table is SkillCorner's own 0–100 index and is shown separately "
        "everywhere. Our xFG is a logistic model with natural cubic splines on shot "
        "distance and defender distance, fitted on 1,322 box-score attempts and "
        "validated 5-fold grouped by game. On this sample SkillCorner's index is the "
        "more accurate of the two (AUC 0.711 vs 0.689); ours exists so the drivers "
        "are inspectable and so spacing features can be tested. Definitions in "
        "docs/METHODOLOGY.md.")

with t3:
    c1, c2 = st.columns([1, 1.25])
    with c1:
        cr = (d.groupby("creation_type").agg(
                shots=("id", "size"), fg=("outcome", "mean"), pts=("points", "mean"),
                xfg=("xfg", "mean"), sq=("shot_quality", "mean"),
                three=("three", "mean"), open_rate=("contested", lambda s: 1 - s.mean()),
                def_dist=("closest_def_dist", "mean"))
              .reset_index().sort_values("shots", ascending=False))
        cr["vs_expected"] = cr.fg - cr.xfg
        st.dataframe(cr.round(3), hide_index=True, width='stretch', height=330)
    with c2:
        fig = go.Figure()
        for i, (k, gdf) in enumerate(d.groupby("creation_type")):
            if len(gdf) < 15: continue
            fig.add_trace(go.Box(y=gdf.shot_quality, name=k, boxmean=True,
                                 marker_color=TH.CATEG[i % len(TH.CATEG)], width=.6))
        fig.update_layout(height=330, yaxis_title="SkillCorner shot quality",
                          showlegend=False)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    st.markdown("### Coverage faced when the shot came off a ball screen")
    pks = d[d.creation_type.eq("pick") | d.prev_pick_bhr_def.notna()]
    if len(pks) > 20:
        cov = (pks.groupby("prev_pick_bhr_def").agg(
                shots=("id", "size"), fg=("outcome", "mean"), xfg=("xfg", "mean"),
                three=("three", "mean"), sq=("shot_quality", "mean"))
               .reset_index().sort_values("shots", ascending=False))
        st.dataframe(cov.round(3), hide_index=True, width='stretch')

with t4:
    c1, c2 = st.columns(2)
    with c1:
        bins = pd.cut(d.closest_def_dist, [0, 2, 3, 4, 5, 6, 8, 12, 40])
        b = (d.groupby(bins, observed=True).agg(n=("id", "size"), fg=("outcome", "mean"),
             xfg=("xfg", "mean"), pts=("points", "mean")).reset_index())
        b["bin"] = b.closest_def_dist.astype(str)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=b.bin, y=b.fg, name="observed FG%",
                             marker_color=TH.ACCENT, opacity=.9))
        fig.add_trace(go.Scatter(x=b.bin, y=b.xfg, name="model xFG", mode="lines+markers",
                                 line=dict(color=TH.ACCENT_2, width=2)))
        fig.update_layout(height=380, xaxis_title="closest defender distance (ft)",
                          yaxis_title="FG%", xaxis_tickangle=-30)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    with c2:
        cl = (d.groupby("contest_level").agg(n=("id", "size"), fg=("outcome", "mean"),
              pts=("points", "mean"), def_dist=("closest_def_dist", "mean"),
              sq=("shot_quality", "mean"), xfg=("xfg", "mean")).reset_index())
        order = ["open", "light", "average", "plus", "blocked"]
        cl["contest_level"] = pd.Categorical(cl.contest_level, order, ordered=True)
        cl = cl.sort_values("contest_level")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=cl.contest_level.astype(str), y=cl.fg, name="FG%",
                             marker_color=TH.ACCENT))
        fig.add_trace(go.Scatter(x=cl.contest_level.astype(str), y=cl.def_dist / 20,
                                 name="def dist / 20", mode="lines+markers",
                                 line=dict(color=TH.ACCENT_3, width=2)))
        fig.update_layout(height=380, yaxis_title="FG%")
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.dataframe(cl.round(3), hide_index=True, width='stretch')

with t5:
    p = (sh[sh.is_box_score_fga].groupby("player").agg(
            team=("team", "first"), fga=("id", "size"), fg=("outcome", "mean"),
            pts=("points", "sum"), pps=("points", "mean"), xfg=("xfg", "mean"),
            sva=("shot_value_added", "sum"), three_rate=("three", "mean"),
            rim=("zone_family", lambda s: s.eq("rim").mean()),
            contested=("contested", "mean"), def_dist=("closest_def_dist", "mean"),
            sq=("shot_quality", "mean")).reset_index())
    p = p[p.fga >= 10]
    p["vs_expected"] = p.fg - p.xfg
    st.dataframe(p.sort_values("sva", ascending=False).round(3),
                 hide_index=True, width='stretch', height=430)
    fig = go.Figure(go.Scatter(x=p.xfg, y=p.fg, mode="markers",
                    marker=dict(size=6 + p.fga / 3, color=p.sva, colorscale=TH.DIV,
                                cmid=0, line=dict(width=.5, color="#0B0E14"),
                                colorbar=dict(title=dict(text="pts vs<br>expected",
                                                         font=dict(size=10)),
                                              thickness=10, len=.6)),
                    customdata=np.stack([p.player, p.team, p.fga], -1),
                    hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                                  "%{customdata[2]} FGA<br>xFG %{x:.3f} → FG %{y:.3f}<extra></extra>"))
    lo, hi = 0.2, 0.75
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                  line=dict(color=TH.LINE, dash="dash"))
    fig.update_layout(height=460, xaxis_title="expected FG% (model)",
                      yaxis_title="actual FG%",
                      title="Shot-making versus shot quality — min 10 attempts")
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    TH.caveat("With 10–30 attempts per player, the vertical spread here is mostly "
              "shooting variance. Treat it as a description of what happened in "
              "these ten games, not as a skill estimate.")
