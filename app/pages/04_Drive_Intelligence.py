"""Drive Intelligence — where drives start, what they produce, how the defence reacts."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from app.components import data as DATA, theme as TH, courtplot as CP
from src import config as C

TH.apply("Drive Intelligence")
TH.header("Drive Intelligence", "Advantage creation off the dribble",
          "Every drive with its origin, direction, defensive coverage and outcome — "
          "plus how quickly the help defence collapsed, measured from tracking.")

dr = DATA.load("drives_features")
col = DATA.load("drive_collapse")
names = DATA.player_name_map(); teams = DATA.teams()
dr["player"] = dr.ballhandler_id_canonical.map(names)
dr["team"] = dr.off_team_id.map(teams)

c1, c2, c3, c4 = st.columns(4)
team = c1.selectbox("Team", ["All"] + sorted(dr.team.dropna().unique().tolist()))
d = dr if team == "All" else dr[dr.team == team]
plyr = c2.selectbox("Ball handler", ["All"] + sorted(d.player.dropna().unique().tolist()))
d = d if plyr == "All" else d[d.player == plyr]
cat = c3.multiselect("Created by", sorted(dr.category.dropna().unique()), default=[])
if cat: d = d[d.category.isin(cat)]
end = c4.multiselect("Ended as", sorted(dr.end_type.dropna().unique()), default=[])
if end: d = d[d.end_type.isin(end)]

TH.kpis([
    ("Drives", f"{len(d):,}", "", "accent"),
    ("Points / drive", f"{d.points_total.mean():.3f}", "own + assisted, no FTs"),
    ("Success rate", f"{d.successful.mean():.1%}", "score, assist or foul drawn"),
    ("Blowby rate", f"{d.blowby.mean():.1%}", "beat the on-ball defender"),
    ("Reached paint", f"{d.reached_paint.mean():.1%}", ""),
    ("Kick-out rate", f"{d.passed_out.mean():.1%}", ""),
    ("Turnover rate", f"{d.turned_over.mean():.1%}", ""),
    ("Help closed", f"{d.help_closed_ft.mean():.1f} ft", "2nd defender, start→min", "cool"),
])

t1, t2, t3, t4 = st.tabs(["Drive map", "Outcomes", "Defensive collapse", "Players"])

with t1:
    c1, c2 = st.columns(2)
    with c1:
        f = CP.blank_court(half=True, height=470, title="Drive paths")
        succ = d[d.successful]; fail = d[~d.successful]
        CP.arrows(f, fail.location_x, fail.location_y, fail.end_loc_x, fail.end_loc_y,
                  color=TH.BAD, width=1.3, opacity=.35, name="unsuccessful")
        CP.arrows(f, succ.location_x, succ.location_y, succ.end_loc_x, succ.end_loc_y,
                  color=TH.ACCENT_2, width=1.8, opacity=.65, name="successful")
        f.add_trace(go.Scatter(x=d.location_x, y=d.location_y, mode="markers",
                    marker=dict(size=6, color=TH.ACCENT, opacity=.75),
                    customdata=np.stack([d.player, d.category, d.end_type,
                                         d.points_total], -1),
                    hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} → "
                                  "%{customdata[2]}<br>%{customdata[3]} pts<extra></extra>",
                    name="start"))
        f.update_layout(showlegend=True)
        st.plotly_chart(f, width='stretch', config={"displayModeBar": False})
    with c2:
        st.plotly_chart(CP.hexbin_court(d, value_col="points_total", gridsize=4.0,
                        min_count=3, height=470, title="Points per drive by origin",
                        cbar_title="pts/drive"),
                        width='stretch', config={"displayModeBar": False})
    z = (d.groupby("region").agg(n=("id", "size"), ppd=("points_total", "mean"),
         blowby=("blowby", "mean"), paint=("reached_paint", "mean"),
         kickout=("passed_out", "mean"), to=("turned_over", "mean"))
         .reset_index().sort_values("n", ascending=False))
    st.dataframe(z.round(3), hide_index=True, width='stretch')

with t2:
    c1, c2 = st.columns(2)
    with c1:
        e = (d.groupby("end_type").agg(n=("id", "size"), ppd=("points_total", "mean"),
             foul=("drew_foul", "mean"), to=("turned_over", "mean"),
             gain=("drive_gain_ft", "mean")).reset_index().sort_values("n", ascending=False))
        fig = go.Figure()
        fig.add_trace(go.Bar(x=e.end_type, y=e.n, name="drives", marker_color=TH.PANEL_2,
                             yaxis="y2", opacity=.8))
        fig.add_trace(go.Bar(x=e.end_type, y=e.ppd, name="pts / drive",
                             marker_color=TH.ACCENT))
        fig.update_layout(height=380, yaxis_title="points per drive",
                          yaxis2=dict(overlaying="y", side="right", showgrid=False,
                                      title="count", color=TH.TEXT_FAINT),
                          xaxis_tickangle=-25)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.dataframe(e.round(3), hide_index=True, width='stretch')
    with c2:
        ca = (d.groupby("category").agg(n=("id", "size"), ppd=("points_total", "mean"),
              blowby=("blowby", "mean"), paint=("reached_paint", "mean"),
              kickout=("passed_out", "mean"),
              help_closed=("help_closed_ft", "mean")).reset_index().sort_values("n", ascending=False))
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ca.category, y=ca.ppd, marker_color=TH.ACCENT_2,
                             name="pts / drive"))
        fig.update_layout(height=380, yaxis_title="points per drive", xaxis_tickangle=-25)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.dataframe(ca.round(3), hide_index=True, width='stretch')
    st.markdown("### Direction and dribble-through")
    dd = (d.groupby(["direction", "dribble_through"]).agg(
            n=("id", "size"), ppd=("points_total", "mean"), blowby=("blowby", "mean"),
            foul=("drew_foul", "mean")).reset_index())
    st.dataframe(dd.round(3), hide_index=True, width='stretch')

with t3:
    st.markdown("### How fast does the defence collapse?")
    st.caption("Second-nearest defender distance to the ball handler, sampled every "
               "5 frames (0.2 s) from drive start. Split by how the drive ended.")
    cc = col.merge(dr[["id", "end_type", "category", "points_total", "successful",
                       "off_team_id", "ballhandler_id_canonical"]],
                   left_on="drive_id", right_on="id", how="left")
    if team != "All":
        tid = [k for k, v in teams.items() if v == team]
        cc = cc[cc.off_team_id.isin(tid)]
    if plyr != "All":
        pid = [k for k, v in names.items() if v == plyr]
        cc = cc[cc.ballhandler_id_canonical.isin(pid)]
    cc = cc[cc.t_since_start <= 3.0]
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for i, (k, gg) in enumerate(cc.groupby("end_type")):
            if gg.drive_id.nunique() < 12: continue
            m = gg.groupby("t_since_start").second_def_dist.mean()
            fig.add_trace(go.Scatter(x=m.index, y=m.values, mode="lines", name=str(k),
                                     line=dict(width=2.2, color=TH.CATEG[i % len(TH.CATEG)])))
        fig.update_layout(height=380, xaxis_title="seconds since drive start",
                          yaxis_title="help defender distance (ft)")
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    with c2:
        fig = go.Figure()
        for i, (k, gg) in enumerate(cc.groupby("end_type")):
            if gg.drive_id.nunique() < 12: continue
            m = gg.groupby("t_since_start").n_def_in_paint.mean()
            fig.add_trace(go.Scatter(x=m.index, y=m.values, mode="lines", name=str(k),
                                     line=dict(width=2.2, color=TH.CATEG[i % len(TH.CATEG)])))
        fig.update_layout(height=380, xaxis_title="seconds since drive start",
                          yaxis_title="defenders in the paint")
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    st.markdown("### Successful vs unsuccessful drives")
    comp = (cc.groupby(["successful", "t_since_start"])
              .agg(help_dist=("second_def_dist", "mean"),
                   nearest=("nearest_def_dist", "mean"),
                   paint=("n_def_in_paint", "mean"),
                   hull=("hull_area", "mean")).reset_index())
    fig = go.Figure()
    for succ, colr, nm in ((True, TH.ACCENT_2, "successful"), (False, TH.BAD, "unsuccessful")):
        g = comp[comp.successful == succ]
        fig.add_trace(go.Scatter(x=g.t_since_start, y=g.help_dist, mode="lines",
                                 name=f"help dist — {nm}", line=dict(color=colr, width=2.4)))
        fig.add_trace(go.Scatter(x=g.t_since_start, y=g.nearest, mode="lines",
                                 name=f"on-ball dist — {nm}",
                                 line=dict(color=colr, width=1.4, dash="dot")))
    fig.update_layout(height=380, xaxis_title="seconds since drive start",
                      yaxis_title="distance (ft)")
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

with t4:
    p = DATA.load("player_drive_profile")
    if team != "All":
        p = p[p.team_name == team]
    st.dataframe(p.round(3), hide_index=True, width='stretch', height=430)
    q = p[p.drives >= 8]
    fig = go.Figure(go.Scatter(x=q.blowby_rate, y=q.points_per_drive, mode="markers",
                    marker=dict(size=7 + q.drives, color=q.pass_out_rate,
                                colorscale=TH.SEQ, line=dict(width=.5, color="#0B0E14"),
                                colorbar=dict(title=dict(text="kick-out<br>rate",
                                                         font=dict(size=10)),
                                              thickness=10, len=.6)),
                    customdata=np.stack([q.player_name, q.team_name, q.drives], -1),
                    hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                                  "%{customdata[2]} drives<extra></extra>"))
    fig.update_layout(height=460, xaxis_title="blowby rate", yaxis_title="points per drive",
                      title="Beating your man vs producing points — min 8 drives")
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    TH.caveat("8–27 drives per player over these ten games. Points per drive at that "
              "sample has a standard error of roughly ±0.25 — differences smaller "
              "than that are noise.")
