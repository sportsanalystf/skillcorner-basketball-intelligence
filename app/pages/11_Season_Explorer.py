"""Season Explorer — the 293-game offensive aggregates as a searchable database."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from app.components import data as DATA, theme as TH, courtplot as CP

TH.apply("Season Explorer")
TH.header("Season Player Explorer", "ACB 2025/26 · 293 games · offence only",
          "Player-season aggregates with duplicate player ids merged through "
          "player_id_aliases.csv, so nobody is counted twice.")

sh = DATA.load("shots_merged")
dr = DATA.load("drives_merged")
pk = DATA.load("picks_merged")
style = DATA.load("season_style_clustered")

per_team = st.toggle("Per-team rows (exclude season-total rows for traded players)",
                     value=True,
                     help="A player who changed team mid-season has one row per team "
                          "plus a season-total row with team_name = 'total'. Mixing "
                          "them double-counts.")
S = sh[~sh.is_season_total_row] if per_team else sh[sh.is_season_total_row | ~sh.player_id_canonical.duplicated(keep=False)]
D = dr[~dr.is_season_total_row] if per_team else dr
K = pk[~pk.is_season_total_row] if per_team else pk

TH.kpis([
    ("Players", f"{S.player_id_canonical.nunique()}", f"{len(S)} player-team rows", "accent"),
    ("Teams", f"{S.team_name.nunique()}", ""),
    ("Attempts", f"{int(S.attempts.sum()):,}", "box-score FGA convention"),
    ("Points", f"{int(S.total_points.sum()):,}", "includes and-one FTs"),
    ("Drives", f"{int(D.total_drives.sum()):,}", ""),
    ("Direct picks", f"{int(K.handler_total_picks.sum()):,}", "handler role"),
])

t1, t2, t3 = st.tabs(["Player database", "Shooting profiles", "Style map"])

with t1:
    c1, c2, c3, c4 = st.columns(4)
    team = c1.selectbox("Team", ["All"] + sorted(S.team_name.dropna().unique().tolist()))
    mg = c2.slider("Min games", 0, 34, 10)
    ma = c3.slider("Min attempts", 0, 400, 50)
    clus = c4.multiselect("Style cluster", sorted(style.cluster.unique().tolist()), [])

    base = S[["player_id_canonical", "player_name", "team_name", "games_played",
              "possessions_played", "attempts", "mades", "total_points",
              "fg_percentage", "efg_percentage", "points_per_shot",
              "avg_attempts_distance", "three_attempts", "zone_restricted_area_attempts",
              "zone_mid_range_attempts", "zone_corner_3_attempts",
              "cns_attempts", "od_attempts", "contested_attempts",
              "assisted_mades", "fouled_fg_attempts"]].copy()
    base["three_rate"] = base.three_attempts / base.attempts.replace(0, np.nan)
    base["rim_rate"] = base.zone_restricted_area_attempts / base.attempts.replace(0, np.nan)
    base["cns_rate"] = base.cns_attempts / base.attempts.replace(0, np.nan)
    base["contested_rate"] = base.contested_attempts / base.attempts.replace(0, np.nan)
    base["pts_per_game"] = base.total_points / base.games_played.replace(0, np.nan)

    dcols = ["player_id_canonical", "team_name", "total_drives", "points_per_drive",
             "blowby_rate", "end_type_kickout_rate", "assists_rate", "fouls_rate"]
    kcols = ["player_id_canonical", "team_name", "handler_total_picks", "handler_ppp",
             "screener_total_picks", "screener_ppp"]
    tbl = (base.merge(D[[c for c in dcols if c in D.columns]],
                      on=["player_id_canonical", "team_name"], how="left")
                .merge(K[[c for c in kcols if c in K.columns]],
                       on=["player_id_canonical", "team_name"], how="left"))
    if team != "All": tbl = tbl[tbl.team_name == team]
    tbl = tbl[(tbl.games_played >= mg) & (tbl.attempts >= ma)]
    if clus:
        keep = style[style.cluster.isin(clus)].player_id_canonical
        tbl = tbl[tbl.player_id_canonical.isin(keep)]

    show = ["player_name", "team_name", "games_played", "attempts", "pts_per_game",
            "fg_percentage", "efg_percentage", "points_per_shot", "three_rate",
            "rim_rate", "cns_rate", "contested_rate", "avg_attempts_distance",
            "total_drives", "points_per_drive", "blowby_rate", "end_type_kickout_rate",
            "handler_total_picks", "handler_ppp", "screener_total_picks", "screener_ppp"]
    st.dataframe(tbl[[c for c in show if c in tbl.columns]]
                 .sort_values("pts_per_game", ascending=False).round(3),
                 hide_index=True, width='stretch', height=520)
    st.caption(f"{len(tbl)} rows. Rates are as delivered by SkillCorner where a rate "
               "column exists; where a player's two ids were merged, rates were "
               "recomputed from the summed counts (see src/data/aliases.py).")

with t2:
    c1, c2 = st.columns(2)
    q = S[(S.games_played >= 10) & (S.attempts >= 80)].copy()
    q["three_rate"] = q.three_attempts / q.attempts
    q["rim_rate"] = q.zone_restricted_area_attempts / q.attempts
    with c1:
        fig = go.Figure(go.Scatter(x=q.three_rate, y=q.efg_percentage, mode="markers",
                        marker=dict(size=4 + q.attempts / 30, color=q.rim_rate,
                                    colorscale=TH.SEQ, opacity=.85,
                                    line=dict(width=.4, color="#0B0E14"),
                                    colorbar=dict(title=dict(text="rim rate",
                                                             font=dict(size=10)),
                                                  thickness=10, len=.6)),
                        customdata=np.stack([q.player_name, q.team_name, q.attempts], -1),
                        hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                                      "%{customdata[2]} FGA<extra></extra>"))
        fig.update_layout(height=480, xaxis_title="three-point attempt rate",
                          yaxis_title="eFG%", title="Shot mix vs efficiency")
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    with c2:
        Kq = K[(K.games_played >= 10) & (K.handler_total_picks >= 60)]
        fig = go.Figure(go.Scatter(x=Kq.handler_total_picks / Kq.games_played,
                        y=Kq.handler_ppp, mode="markers",
                        marker=dict(size=7, color=TH.ACCENT, opacity=.8),
                        customdata=np.stack([Kq.player_name, Kq.team_name,
                                             Kq.handler_total_picks], -1),
                        hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                                      "%{customdata[2]} direct picks<extra></extra>"))
        fig.update_layout(height=480, xaxis_title="direct picks used per game",
                          yaxis_title="points per pick (handler)",
                          title="Pick-and-roll volume vs efficiency")
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    st.markdown("### Coverage splits — who handles what")
    cov_cols = [c for c in K.columns if c.startswith("handler_ppp_vs_")]
    if cov_cols:
        cq = K[(K.handler_total_picks >= 80)][["player_name", "team_name",
                                               "handler_total_picks"] + cov_cols]
        st.dataframe(cq.sort_values("handler_total_picks", ascending=False).round(3),
                     hide_index=True, width='stretch', height=400)
        lg = K[cov_cols].mul(K.handler_total_picks, axis=0).sum() / K.handler_total_picks.sum()
        st.caption("League volume-weighted mean: " +
                   ", ".join(f"{c.replace('handler_ppp_vs_','')} {v:.3f}"
                             for c, v in lg.items() if pd.notna(v)))

with t3:
    load = DATA.load("style_pca_loadings")
    fig = go.Figure()
    for c_ in sorted(style.cluster.unique()):
        dd = style[style.cluster == c_]
        fig.add_trace(go.Scatter(x=dd.pc1, y=dd.pc2, mode="markers",
                      name=f"cluster {c_}",
                      marker=dict(size=7 + dd.attempts / 40,
                                  color=TH.CATEG[int(c_) % len(TH.CATEG)], opacity=.8,
                                  line=dict(width=.4, color="#0B0E14")),
                      customdata=np.stack([dd.player_name, dd.team_name,
                                           dd.games_played, dd.attempts], -1),
                      hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                                    "%{customdata[2]} games, %{customdata[3]} FGA<extra></extra>"))
    fig.update_layout(height=560, xaxis_title="PC1", yaxis_title="PC2",
                      title="Offensive style space — 233 qualified player-team seasons")
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    st.dataframe(load.round(3), hide_index=True, width='stretch', height=340)
