"""Scouting view — how a team attacks and how it defends."""
from __future__ import annotations
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from app.components import data as DATA, theme as TH, courtplot as CP

TH.apply("Scouting Report")
TH.header("Scouting Report", "How they attack · how they defend",
          "Automated notes, each backed by the number it came from and its sample size.")

off = DATA.load("team_offense_profile")
dfn = DATA.load("team_defense_profile")
notes = DATA.load_json("scouting_notes")
sh = DATA.load("shots_features"); dr = DATA.load("drives_features")
pk = DATA.load("picks_features"); co = DATA.load("closeouts_features")
tt = DATA.load("team_transition_profile")
teams = DATA.teams(); names = DATA.player_name_map()

team = st.selectbox("Team", off.sort_values("team_name").team_name.tolist())
tid = int(off[off.team_name == team].team_id.iloc[0])
o = off[off.team_id == tid].iloc[0]
d = dfn[dfn.team_id == tid].iloc[0] if (dfn.team_id == tid).any() else None

TH.kpis([
    ("Tracked games", f"{int(o.games)}", f"{int(o.chances)} chances", "accent"),
    ("Pts / chance", f"{o.pts_per_chance:.2f}",
     f"allowed {d.pts_allowed_per_chance:.2f}" if d is not None else ""),
    ("eFG%", f"{o.efg:.1%}", ""),
    ("3PA rate", f"{o.three_rate:.0%}", f"rim {o.rim_rate:.0%}"),
    ("Picks / chance", f"{o.picks_per_chance:.2f}", f"{o.pick_middle_rate:.0%} middle"),
    ("Drives / chance", f"{o.drives_per_chance:.2f}", f"{o.drive_kickout_rate:.0%} kick out"),
    ("Transition", f"{o.transition_rate:.0%}", ""),
    ("Turnovers", f"{o.to_rate:.0%}", "of chances"),
])

st.markdown("### Automated read")
c1, c2 = st.columns(2)
nn = notes.get(str(tid), notes.get(tid, []))
for i, n in enumerate(nn):
    tgt = c1 if i % 2 == 0 else c2
    pct = "" if n.get("league_pctile") is None else \
        f" · {n['league_pctile']*100:.0f}th pct of {n['rank_of']} teams"
    with tgt:
        TH.note(n["text"], f"{n['metric']} = {n['value']:.3f} · n = {n['n']}{pct}")

st.markdown("---")
t1, t2, t3 = st.tabs(["Offence", "Defence", "League comparison"])

with t1:
    s = sh[sh.off_team_id == tid]; dd = dr[dr.off_team_id == tid]; pp = pk[pk.off_team_id == tid]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(CP.hexbin_court(s, value_col="points", gridsize=4.0, min_count=2,
                        height=380, title="Shot locations", cbar_title="pts/shot"),
                        width='stretch', config={"displayModeBar": False})
    with c2:
        f = CP.blank_court(half=True, height=380, title="Drive origins and paths")
        CP.arrows(f, dd.location_x, dd.location_y, dd.end_loc_x, dd.end_loc_y,
                  color=TH.ACCENT, width=1.5, opacity=.5)
        f.add_trace(go.Scatter(x=dd.location_x, y=dd.location_y, mode="markers",
                    marker=dict(size=7, color=TH.ACCENT_2)))
        st.plotly_chart(f, width='stretch', config={"displayModeBar": False})
    with c3:
        f = CP.blank_court(half=True, height=380, title="Ball-screen locations")
        f.add_trace(go.Scatter(x=pp.location_x, y=pp.location_y, mode="markers",
                    marker=dict(size=8, color=pp.pick_points, colorscale=TH.SEQ,
                                line=dict(width=.4, color="#0B0E14"),
                                colorbar=dict(title=dict(text="pts", font=dict(size=9)),
                                              thickness=8, len=.5)),
                    customdata=np.stack([pp.ballhandler_id_canonical.map(names),
                                         pp.screener_id_canonical.map(names),
                                         pp.bhr_def_type], -1),
                    hovertemplate="<b>%{customdata[0]}</b> off %{customdata[1]}<br>"
                                  "%{customdata[2]}<extra></extra>"))
        st.plotly_chart(f, width='stretch', config={"displayModeBar": False})
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Who does what")
        who = (s.groupby(s.shooter_id_canonical.map(names)).agg(
                fga=("id", "size"), pts=("points", "sum"), fg=("outcome", "mean"),
                three=("three", "mean"), sq=("shot_quality", "mean")).reset_index()
               .rename(columns={"shooter_id_canonical": "player"})
               .sort_values("fga", ascending=False))
        st.dataframe(who.round(3), hide_index=True, width='stretch', height=320)
    with c2:
        st.markdown("#### Pick-and-roll pairs")
        pairs = (pp.groupby([pp.ballhandler_id_canonical.map(names),
                             pp.screener_id_canonical.map(names)])
                   .agg(picks=("id", "size"), ppp=("pick_points", "mean"),
                        switch=("scr_def_type", lambda x: x.eq("switch").mean()))
                   .reset_index())
        pairs.columns = ["handler", "screener", "picks", "ppp", "switch_rate"]
        st.dataframe(pairs.sort_values("picks", ascending=False).round(3),
                     hide_index=True, width='stretch', height=320)
    st.markdown("#### Action sequences this team runs")
    ac = DATA.load("action_chains")
    ac_t = ac[ac.off_team_id == tid]
    ce = (ac_t.groupby("action_chain").agg(n=("chance_id", "size"),
          ppc=("pts_scored", "mean")).reset_index())
    st.dataframe(ce[ce.n >= 3].sort_values("n", ascending=False).round(3),
                 hide_index=True, width='stretch', height=280)

with t2:
    if d is None:
        st.info("No defensive rows.")
    else:
        s = sh[sh.def_team_id == tid]; cc = co[co.def_team_id == tid]
        pp = pk[pk.def_team_id == tid]
        TH.kpis([
            ("Chances defended", f"{int(d.chances_defended)}", "", "cool"),
            ("Pts allowed / chance", f"{d.pts_allowed_per_chance:.2f}", ""),
            ("Opp 3PA rate", f"{d.opp_three_rate:.0%}", f"rim {d.opp_rim_rate:.0%}"),
            ("Opp contested", f"{d.opp_contested_rate:.0%}",
             f"{d.opp_mean_closest_def_dist:.1f} ft"),
            ("Shot quality allowed", f"{d.opp_shot_quality_allowed:.1f}", "SkillCorner index"),
            ("Closeout rate", f"{d.closeout_close_rate:.1f} ft/s",
             f"arrives {d.closeout_end_dist:.1f} ft"),
            ("Forced TO", f"{d.forced_to_rate:.0%}", ""),
            ("Blowby allowed", f"{d.drive_blowby_allowed:.0%}", ""),
        ])
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(CP.hexbin_court(s, value_col="points", gridsize=4.0,
                            min_count=2, height=400, title="Shots allowed",
                            cbar_title="pts/shot"),
                            width='stretch', config={"displayModeBar": False})
        with c2:
            mix = pp.bhr_def_type.value_counts(normalize=True)
            mix2 = pp.scr_def_type.value_counts(normalize=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=mix.index, y=mix.values, name="on the ball",
                                 marker_color=TH.ACCENT))
            fig.add_trace(go.Bar(x=mix2.index, y=mix2.values, name="on the screener",
                                 marker_color=TH.ACCENT_3))
            fig.update_layout(height=400, barmode="group",
                              yaxis_title="share of picks defended",
                              title="Ball-screen coverage mix")
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.markdown("#### Coverage results")
        r1 = (pp.groupby("bhr_def_type").agg(picks=("id", "size"),
              ppp_allowed=("pick_points", "mean"), shot=("bhr_shot", "mean"),
              assist=("bhr_assist", "mean"), to=("bhr_turnover", "mean")).reset_index())
        st.dataframe(r1.round(3), hide_index=True, width='stretch')
        st.markdown("#### Individual closeouts")
        cdp = DATA.load("closeout_defender_profile")
        st.dataframe(cdp[cdp.team_name == team].round(3), hide_index=True,
                     width='stretch')

with t3:
    metric = st.selectbox("Metric", [c for c in off.columns
                                     if off[c].dtype.kind == "f" and c != "team_id"])
    o2 = off.sort_values(metric)
    fig = go.Figure(go.Bar(x=o2[metric], y=o2.team_name, orientation="h",
                    marker_color=np.where(o2.team_id == tid, TH.ACCENT, TH.PANEL_2),
                    hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(height=520, xaxis_title=metric, margin=dict(l=10, r=10, t=20, b=30))
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    st.markdown("#### Transition vs half court")
    st.dataframe(tt[tt.team_id == tid].round(3), hide_index=True, width='stretch')
    st.dataframe(off.round(3), hide_index=True, width='stretch', height=400)
    TH.caveat("Each team here has one or two tracked games. Cross-team comparison on "
              "this page describes those specific games; it is not a season ranking. "
              "For season-scale player comparison use the Season Explorer.")
