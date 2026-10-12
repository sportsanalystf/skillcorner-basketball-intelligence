"""Pick & Roll Intelligence — handler, screener, coverage, outcome."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from app.components import data as DATA, theme as TH, courtplot as CP

TH.apply("Pick & Roll")
TH.header("Pick-and-Roll Intelligence", "On-ball screens with tracked coverage",
          "Every pick carries both defenders and the coverage each played — the "
          "thing a manual feed cannot give you.")

pk = DATA.load("picks_features")
names = DATA.player_name_map(); teams = DATA.teams()
pk["handler"] = pk.ballhandler_id_canonical.map(names)
pk["screener"] = pk.screener_id_canonical.map(names)
pk["team"] = pk.off_team_id.map(teams)
pk["def_team"] = pk.def_team_id.map(teams)

c1, c2, c3, c4 = st.columns(4)
team = c1.selectbox("Offence", ["All"] + sorted(pk.team.dropna().unique().tolist()))
d = pk if team == "All" else pk[pk.team == team]
cov = c2.multiselect("Handler coverage", sorted(pk.bhr_def_type.dropna().unique()), [])
if cov: d = d[d.bhr_def_type.isin(cov)]
loc = c3.multiselect("Screen location", sorted(pk.location_type.dropna().unique()), [])
if loc: d = d[d.location_type.isin(loc)]
only_direct = c4.toggle("Direct picks only", value=False,
                        help="`direct = true`: handler or screener shot, assisted or "
                             "was fouled straight off the pick. This is the subset the "
                             "official season aggregates count.")
if only_direct: d = d[d.direct]

TH.kpis([
    ("Picks", f"{len(d):,}", f"{d.direct.mean():.0%} direct", "accent"),
    ("Points / pick", f"{d.pick_points.mean():.3f}", "best of handler / screener"),
    ("Handler ppp", f"{d.bhr_points.mean():.3f}", ""),
    ("Screener ppp", f"{d.scr_points.mean():.3f}", ""),
    ("Handler shoots", f"{d.bhr_shot.mean():.1%}", ""),
    ("Handler assists", f"{d.bhr_assist.mean():.1%}", ""),
    ("Turnover", f"{d.bhr_turnover.mean():.1%}", ""),
    ("Roll rate", f"{d.screener_action.eq('roll').mean():.1%}", "our tracking label", "cool"),
])

t1, t2, t3, t4 = st.tabs(["Coverage", "Screen map", "Ball handlers", "Screeners"])

with t1:
    c1, c2 = st.columns(2)
    for col, by, title in ((c1, "bhr_def_type", "Coverage on the ball handler"),
                           (c2, "scr_def_type", "Coverage on the screener")):
        with col:
            g = (d.groupby(by).agg(picks=("id", "size"), ppp=("pick_points", "mean"),
                 handler_ppp=("bhr_points", "mean"), screener_ppp=("scr_points", "mean"),
                 shot=("bhr_shot", "mean"), assist=("bhr_assist", "mean"),
                 to=("bhr_turnover", "mean"), foul=("bhr_foul", "mean"),
                 direct=("direct", "mean")).reset_index().sort_values("picks", ascending=False))
            fig = go.Figure()
            fig.add_trace(go.Bar(x=g[by], y=g.ppp, marker_color=TH.ACCENT, name="pts/pick"))
            fig.add_trace(go.Scatter(x=g[by], y=g.direct, mode="markers", name="direct rate",
                                     yaxis="y2", marker=dict(size=11, color=TH.ACCENT_3)))
            fig.update_layout(height=330, title=title, yaxis_title="points per pick",
                              yaxis2=dict(overlaying="y", side="right", showgrid=False,
                                          range=[0, 1], title="direct rate",
                                          color=TH.TEXT_FAINT))
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
            st.dataframe(g.round(3), hide_index=True, width='stretch')
    st.markdown("### Coverage pair matrix")
    piv = d.pivot_table(index="bhr_def_type", columns="scr_def_type",
                        values="pick_points", aggfunc="mean")
    cnt = d.pivot_table(index="bhr_def_type", columns="scr_def_type",
                        values="id", aggfunc="size")
    piv = piv.where(cnt >= 8)
    fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index,
                               colorscale=TH.SEQ, text=cnt.values,
                               texttemplate="n=%{text}",
                               textfont=dict(size=9, color=TH.TEXT_DIM),
                               colorbar=dict(title=dict(text="pts/pick", font=dict(size=10)),
                                             thickness=10, len=.7)))
    fig.update_layout(height=320, xaxis_title="screener coverage",
                      yaxis_title="handler coverage",
                      title="Points per pick by coverage pair (cells with n ≥ 8)")
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    st.markdown("### Screener response: roll, pop or stay")
    TH.caveat("Roll / pop / stay is <b>our</b> label, derived from the screener's change "
              "in distance to the rim over the 1.5 s after the screen (≥4 ft closer = "
              "roll, ≥4 ft further = pop). SkillCorner does not ship a roll/pop field.")
    rp = (d.dropna(subset=["screener_action"]).groupby(["screener_action", "scr_def_type"])
            .agg(n=("id", "size"), ppp=("pick_points", "mean"),
                 scr_ppp=("scr_points", "mean")).reset_index())
    st.dataframe(rp[rp.n >= 5].round(3), hide_index=True, width='stretch')

with t2:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(CP.shot_chart(d, made_col="direct", color_by="location_type",
                        height=470, title="Screen locations",
                        hover=["handler", "screener", "bhr_def_type", "scr_def_type"]),
                        width='stretch', config={"displayModeBar": False})
    with c2:
        st.plotly_chart(CP.hexbin_court(d, value_col="pick_points", gridsize=4.0,
                        min_count=5, height=470, title="Points per pick by screen spot",
                        cbar_title="pts/pick"),
                        width='stretch', config={"displayModeBar": False})
    lt = (d.groupby("location_type").agg(n=("id", "size"), ppp=("pick_points", "mean"),
          handler=("bhr_points", "mean"), screener=("scr_points", "mean"),
          switch=("scr_def_type", lambda s: s.eq("switch").mean()),
          direct=("direct", "mean")).reset_index())
    st.dataframe(lt.round(3), hide_index=True, width='stretch')

with t3:
    h = DATA.load("pick_handler_profile")
    if team != "All": h = h[h.team_name == team]
    st.dataframe(h.round(3), hide_index=True, width='stretch', height=380)
    st.markdown("### How each handler is guarded, and what he does with it")
    who = st.selectbox("Ball handler", h.player_name.tolist(), key="pnr_h")
    hid = h[h.player_name == who].player_id_canonical.iloc[0]
    sub = pk[pk.ballhandler_id_canonical == hid]
    g = (sub.groupby("bhr_def_type").agg(picks=("id", "size"), ppp=("bhr_points", "mean"),
         shot=("bhr_shot", "mean"), assist=("bhr_assist", "mean"),
         to=("bhr_turnover", "mean"), foul=("bhr_foul", "mean"))
         .reset_index().sort_values("picks", ascending=False))
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(g.round(3), hide_index=True, width='stretch')
    with c2:
        lg = (pk.groupby("bhr_def_type").bhr_points.mean().rename("league").reset_index())
        m = g.merge(lg, on="bhr_def_type")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=m.bhr_def_type, y=m.ppp, name=who, marker_color=TH.ACCENT))
        fig.add_trace(go.Bar(x=m.bhr_def_type, y=m.league, name="league",
                             marker_color=TH.PANEL_2))
        fig.update_layout(height=300, yaxis_title="points per pick", barmode="group")
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    st.caption(f"{len(sub)} tracked picks for {who} across the ten games. "
               "Coverage splits below ~15 picks are indicative only.")

with t4:
    s = DATA.load("pick_screener_profile")
    if team != "All": s = s[s.team_name == team]
    st.dataframe(s.round(3), hide_index=True, width='stretch', height=380)
    q = s[s.picks_set >= 15]
    fig = go.Figure(go.Scatter(x=q.roll_rate, y=q.handler_ppp_when_screening, mode="markers",
                    marker=dict(size=6 + q.picks_set / 4, color=q.pct_switch_forced,
                                colorscale=TH.SEQ, line=dict(width=.5, color="#0B0E14"),
                                colorbar=dict(title=dict(text="switch<br>forced",
                                                         font=dict(size=10)),
                                              thickness=10, len=.6)),
                    customdata=np.stack([q.player_name, q.team_name, q.picks_set], -1),
                    hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                                  "%{customdata[2]} screens<extra></extra>"))
    fig.update_layout(height=460, xaxis_title="roll rate (tracking-derived)",
                      yaxis_title="handler points per pick when he screens",
                      title="Screeners: rolling versus what the ball handler produced")
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
