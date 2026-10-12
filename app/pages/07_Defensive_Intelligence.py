"""Defensive Intelligence — closeouts, coverage, matchups, help."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from app.components import data as DATA, theme as TH, courtplot as CP

TH.apply("Defensive Intelligence")
TH.header("Defensive Intelligence", "Closeouts · coverage · matchups",
          "What the tracking actually supports on the defensive side — and, "
          "explicitly, what it does not.")

co = DATA.load("closeouts_features")
pk = DATA.load("picks_features")
dr = DATA.load("drives_features")
sh = DATA.load("shots_features")
names = DATA.player_name_map(); teams = DATA.teams()
co["defender"] = co.ballhandler_def_id_canonical.map(names)
co["def_team"] = co.def_team_id.map(teams)

TH.caveat("<b>The season aggregates are offence-only.</b> Everything on this page comes "
          "from the ten tracked games, so a defender has 8–25 closeouts and 15–35 "
          "minutes of tracked assignments. These are activity and geometry profiles, "
          "not defensive ratings — points allowed on a matchup confounds scheme with "
          "the individual, and we do not publish a single defensive score.")

t1, t2, t3, t4 = st.tabs(["Closeouts", "Pick coverage", "Matchups", "Contests"])

with t1:
    c1, c2, c3 = st.columns(3)
    tsel = c1.selectbox("Defending team", ["All"] + sorted(co.def_team.dropna().unique().tolist()))
    d = co if tsel == "All" else co[co.def_team == tsel]
    act = c2.multiselect("Ball-handler response", sorted(co.bhr_action.dropna().unique()), [])
    if act: d = d[d.bhr_action.isin(act)]
    zone = c3.multiselect("Zone", sorted(co.zone.dropna().unique()), [])
    if zone: d = d[d.zone.isin(zone)]

    TH.kpis([
        ("Closeouts", f"{len(d):,}", "", "accent"),
        ("Start distance", f"{d.start_distance.mean():.1f} ft", "when the closeout begins"),
        ("End distance", f"{d.end_distance.mean():.1f} ft", "at contest"),
        ("Distance closed", f"{d.distance_closed_ft.mean():.1f} ft", ""),
        ("Closing rate", f"{d.close_rate_fps.mean():.1f} ft/s", ""),
        ("Duration", f"{d.duration_s.mean():.2f} s", ""),
        ("Points allowed", f"{d.total_points_allowed.mean():.2f}", "per closeout"),
        ("Stop rate", f"{d.stop.mean():.1%}", "no points, no foul", "good"),
    ])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Does arriving tighter work?")
        bins = pd.cut(d.end_distance, [0, 2, 3, 4, 5, 6, 8, 30])
        b = (d.groupby(bins, observed=True).agg(n=("id", "size"),
             pts=("total_points_allowed", "mean"), stop=("stop", "mean"),
             foul=("fouled", "mean"), drive=("bhr_action", lambda s: s.eq("drive").mean()))
             .reset_index())
        b["bin"] = b.end_distance.astype(str)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=b.bin, y=b.pts, name="points allowed", marker_color=TH.ACCENT))
        fig.add_trace(go.Scatter(x=b.bin, y=b.foul, name="foul rate", mode="lines+markers",
                                 line=dict(color=TH.WARN, width=2), yaxis="y2"))
        fig.add_trace(go.Scatter(x=b.bin, y=b.drive, name="drove past", mode="lines+markers",
                                 line=dict(color=TH.ACCENT_3, width=2, dash="dot"), yaxis="y2"))
        fig.update_layout(height=380, xaxis_title="defender distance at the end of the closeout (ft)",
                          yaxis_title="points allowed",
                          yaxis2=dict(overlaying="y", side="right", showgrid=False,
                                      range=[0, .6], color=TH.TEXT_FAINT),
                          xaxis_tickangle=-25)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.dataframe(b[["bin", "n", "pts", "stop", "foul", "drive"]].round(3),
                     hide_index=True, width='stretch')
    with c2:
        st.markdown("### Closeout paths")
        f = CP.blank_court(half=True, height=380, title="")
        stop_ = d[d.stop]; nostop = d[~d.stop]
        CP.arrows(f, nostop.start_loc_x, nostop.start_loc_y, nostop.end_loc_x,
                  nostop.end_loc_y, color=TH.BAD, width=1.2, opacity=.35, name="scored on")
        CP.arrows(f, stop_.start_loc_x, stop_.start_loc_y, stop_.end_loc_x,
                  stop_.end_loc_y, color=TH.ACCENT_2, width=1.4, opacity=.5, name="stop")
        f.add_trace(go.Scatter(x=d.location_x, y=d.location_y, mode="markers",
                    marker=dict(size=5, color=TH.BALL, opacity=.7), name="ball handler",
                    customdata=np.stack([d.defender, d.bhr_action,
                                         d.total_points_allowed], -1),
                    hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}, "
                                  "%{customdata[2]} pts<extra></extra>"))
        f.update_layout(showlegend=True)
        st.plotly_chart(f, width='stretch', config={"displayModeBar": False})
    st.markdown("### Defender closeout profiles")
    cdp = DATA.load("closeout_defender_profile")
    if tsel != "All": cdp = cdp[cdp.team_name == tsel]
    st.dataframe(cdp.round(3), hide_index=True, width='stretch', height=380)
    q = cdp[cdp.closeouts >= 8]
    fig = go.Figure(go.Scatter(x=q.mean_close_rate_fps, y=q.mean_end_distance,
                    mode="markers", marker=dict(size=6 + q.closeouts,
                    color=q.points_allowed_per_closeout, colorscale=TH.SEQ,
                    reversescale=True, line=dict(width=.5, color="#0B0E14"),
                    colorbar=dict(title=dict(text="pts allowed<br>per closeout",
                                             font=dict(size=10)), thickness=10, len=.6)),
                    customdata=np.stack([q.player_name, q.team_name, q.closeouts], -1),
                    hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                                  "%{customdata[2]} closeouts<extra></extra>"))
    fig.update_layout(height=430, xaxis_title="closing rate (ft/s)",
                      yaxis_title="distance at contest (ft)",
                      title="Fast and tight is bottom-right — min 8 closeouts")
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

with t2:
    st.markdown("### How each team guards the pick-and-roll")
    pk2 = pk.copy(); pk2["def_team"] = pk2.def_team_id.map(teams)
    mix = (pk2.groupby(["def_team", "bhr_def_type"]).size().unstack(fill_value=0))
    mixp = mix.div(mix.sum(axis=1), axis=0)
    fig = go.Figure()
    for i, c_ in enumerate(mixp.columns):
        fig.add_trace(go.Bar(x=mixp.index, y=mixp[c_], name=str(c_),
                             marker_color=TH.CATEG[i % len(TH.CATEG)]))
    fig.update_layout(height=400, barmode="stack", yaxis_title="share of picks defended",
                      xaxis_tickangle=-35)
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    res = (pk2.groupby("def_team").agg(picks=("id", "size"),
           ppp_allowed=("pick_points", "mean"),
           over=("bhr_def_type", lambda s: s.eq("over").mean()),
           under=("bhr_def_type", lambda s: s.eq("under").mean()),
           switch=("bhr_def_type", lambda s: s.eq("switch").mean()),
           soft=("scr_def_type", lambda s: s.eq("soft").mean()),
           show=("scr_def_type", lambda s: s.eq("show").mean()),
           handler_shot=("bhr_shot", "mean"), forced_to=("bhr_turnover", "mean"))
           .reset_index().sort_values("ppp_allowed"))
    st.dataframe(res.round(3), hide_index=True, width='stretch', height=400)

with t3:
    mw = DATA.load("matchup_workload")
    st.markdown("### Tracked defensive assignments")
    st.caption("From the `matchups` table: assignment intervals within a chance. "
               "Minutes here count only frontcourt chances with a tracked rotation, "
               "so they are lower than box-score minutes.")
    st.dataframe(mw.sort_values("minutes_guarding", ascending=False).round(2),
                 hide_index=True, width='stretch', height=430)
    mu = DATA.events("matchups")
    mu["def"] = mu.def_player_id_canonical.map(names)
    mu["off"] = mu.off_player_id_canonical.map(names)
    mu["seconds"] = (mu.end_frame - mu.start_frame) / 25.0
    top_def = mw.nlargest(14, "minutes_guarding").player_name.tolist()
    sub = mu[mu["def"].isin(top_def)]
    piv = sub.pivot_table(index="def", columns="off", values="seconds", aggfunc="sum")
    keep = piv.sum().nlargest(16).index
    piv = piv[keep].fillna(0)
    fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index,
                    colorscale=TH.SEQ,
                    colorbar=dict(title=dict(text="seconds", font=dict(size=10)),
                                  thickness=10, len=.7),
                    hovertemplate="%{y} guarding %{x}<br>%{z:.0f}s<extra></extra>"))
    fig.update_layout(height=520, xaxis_tickangle=-45,
                      title="Who guarded whom (seconds) — busiest defenders vs busiest attackers",
                      margin=dict(l=10, r=10, t=50, b=120))
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

with t4:
    sdp = DATA.load("shot_defender_profile")
    st.markdown("### Nearest defender at release")
    TH.caveat("The closest defender is not always the assigned defender, and being "
              "nearest on a make is not evidence of poor defence. Minimum 15 contests; "
              "even so, a 15-shot sample has a standard error on FG% of about ±13 pp.")
    st.dataframe(sdp.round(3), hide_index=True, width='stretch', height=380)
    if len(sdp):
        fig = go.Figure(go.Bar(x=sdp.sort_values("fg_vs_expected").player_name,
                        y=sdp.sort_values("fg_vs_expected").fg_vs_expected,
                        marker_color=np.where(
                            sdp.sort_values("fg_vs_expected").fg_vs_expected < 0,
                            TH.ACCENT_2, TH.BAD),
                        customdata=sdp.sort_values("fg_vs_expected").shots_contested,
                        hovertemplate="%{x}<br>%{y:+.3f} vs expected<br>"
                                      "n=%{customdata}<extra></extra>"))
        fig.update_layout(height=380, yaxis_title="opponent FG% − model expectation",
                          xaxis_tickangle=-40, margin=dict(b=130))
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    st.markdown("### Team contest profile")
    sh2 = sh.copy(); sh2["def_team"] = sh2.def_team_id.map(teams)
    tc = (sh2.groupby("def_team").agg(shots=("id", "size"),
          def_dist=("closest_def_dist", "mean"), contested=("contested", "mean"),
          opp_fg=("outcome", "mean"), opp_xfg=("xfg", "mean"),
          opp_sq=("shot_quality", "mean"), three_allowed=("three", "mean"),
          rim_allowed=("zone_family", lambda s: s.eq("rim").mean())).reset_index())
    tc["fg_vs_expected"] = tc.opp_fg - tc.opp_xfg
    st.dataframe(tc.sort_values("opp_sq").round(3), hide_index=True,
                 width='stretch', height=430)
