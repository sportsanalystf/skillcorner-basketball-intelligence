"""Game Explorer — synchronised tracking, timeline and possession reconstruction."""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components import data as DATA, theme as TH, courtplot as CP
from src import config as C
from src.preprocessing import court

TH.apply("Game Explorer")
TH.header("Game Explorer", "25 fps reconstruction",
          "Pick a possession and watch it. Tracking, events, shots, drives and "
          "screens are all on the same frame index, so nothing is approximated.")

g = DATA.games()
c1, c2, c3 = st.columns([2.2, 1, 1])
game_label = c1.selectbox("Game", g.label.tolist(), index=0)
grow = g[g.label == game_label].iloc[0]
gid = int(grow.game_id)

poss = DATA.events("possessions"); poss = poss[poss.game_id == gid]
chan = DATA.events("chances"); chan = chan[chan.game_id == gid]
teams = DATA.teams()
names = DATA.player_name_map()
roster = DATA.load("roster"); roster = roster[roster.game_id == gid]
jersey = dict(zip(roster.player_id, roster.jersey))

period = c2.selectbox("Period", sorted(chan.period.dropna().unique().astype(int)))
side = c3.selectbox("Offence", ["Both", grow.home_team_name, grow.away_team_name])

ch = chan[chan.period == period].copy()
if side != "Both":
    tid = grow.home_team_id if side == grow.home_team_name else grow.away_team_id
    ch = ch[ch.off_team_id == tid]
ch = ch.sort_values("start_frame")
ch["label"] = (ch.off_team_id.map(teams).fillna("?") + "  ·  " +
               ch.start_game_clock.map(DATA.fmt_clock) + "  ·  " +
               ch.outcome.fillna("?") + "  ·  " + ch.pts_scored.fillna(0).astype(int).astype(str) + " pts")

if ch.empty:
    st.warning("No chances match this filter."); st.stop()

sel = st.selectbox(f"Chance ({len(ch)} in this period)", ch.label.tolist(), index=0)
row = ch[ch.label == sel].iloc[0]
f0, f1 = int(row.start_frame), int(row.end_frame)
if f1 <= f0:
    f1 = f0 + 50

TH.kpis([
    ("Offence", teams.get(row.off_team_id, "?"), teams.get(row.def_team_id, "?") + " defending", "accent"),
    ("Duration", f"{max(row.start_game_clock - row.end_game_clock, 0):.1f} s",
     f"shot clock {row.start_shot_clock:.0f} → {row.end_shot_clock:.0f}"),
    ("Start", str(row.start_type), "chance start type"),
    ("Outcome", str(row.outcome), f"{int(row.pts_scored or 0)} points",
     "good" if (row.pts_scored or 0) > 0 else ""),
    ("Transition", "yes" if row.transition else "no", ""),
    ("Tracking quality", f"{int(row.quality_index)}/5",
     "usable" if row.usable else "flagged", "cool" if row.usable else ""),
])

tl_all = DATA.timeline(gid)
tl = tl_all[tl_all.chance_id == row.id].copy()

left, right = st.columns([1.45, 1])

with left:
    st.markdown("### Court")
    mode = st.radio("View", ["Animate", "Single frame", "Movement trails"],
                    horizontal=True, label_visibility="collapsed")

    tr = DATA.tracking_window(gid, f0, f1, viz=True)
    ball = DATA.ball_window(gid, f0, f1)
    if tr.empty:
        st.info("No tracking rows in this frame range.")
    else:
        att = bool(row.get("left_hoop", True)) if "left_hoop" in row else None
        lh = poss[poss.id == row.possession_id]
        sign = 1.0
        if len(lh) and not bool(lh.iloc[0].left_hoop):
            sign = -1.0
        tr = tr.copy()
        tr["px"] = tr.x_s * sign; tr["py"] = tr.y_s * sign
        tr["player_name"] = tr.player_id.map(names)
        bl = ball.copy()
        if not bl.empty:
            bl["px"] = bl.x * sign; bl["py"] = bl.y * sign

        frames = sorted(tr.frame_idx.unique())
        off_id = row.off_team_id

        if mode == "Single frame":
            fi = st.select_slider("Frame", options=frames, value=frames[len(frames)//2],
                                  format_func=lambda f: DATA.fmt_clock(
                                      tr[tr.frame_idx == f].game_clock.iloc[0]))
            d = tr[tr.frame_idx == fi]
            b = bl[bl.frame_idx == fi]
            fig = CP.blank_court(half=False, height=430)
            CP.snapshot(fig, d[d.team_id == off_id], d[d.team_id != off_id],
                        ball=(b.px.iloc[0], b.py.iloc[0]) if len(b) else None,
                        xcol="px", ycol="py", labels=jersey,
                        off_name=teams.get(off_id, "Off"),
                        def_name=teams.get(row.def_team_id, "Def"))
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

        elif mode == "Movement trails":
            fig = CP.blank_court(half=False, height=430)
            for pid, gdf in tr.groupby("player_id"):
                is_off = gdf.team_id.iloc[0] == off_id
                fig.add_trace(go.Scatter(
                    x=gdf.px, y=gdf.py, mode="lines",
                    line=dict(color=TH.OFFENSE if is_off else TH.DEFENSE, width=1.5),
                    opacity=.75, name=str(names.get(pid, pid)),
                    hovertemplate=f"{names.get(pid,pid)}<extra></extra>"))
            if not bl.empty:
                fig.add_trace(go.Scatter(x=bl.px, y=bl.py, mode="lines",
                                         line=dict(color=TH.BALL, width=2.4, dash="dot"),
                                         name="Ball", opacity=.9))
            last = tr[tr.frame_idx == frames[-1]]
            fig.add_trace(go.Scatter(x=last.px, y=last.py, mode="markers",
                                     marker=dict(size=9, color=[TH.OFFENSE if t == off_id
                                                                else TH.DEFENSE
                                                                for t in last.team_id],
                                                 line=dict(width=1, color="#0B0E14")),
                                     name="End", hoverinfo="skip"))
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

        else:  # Animate
            step = max(1, len(frames) // 90)
            keep = frames[::step]
            fr_objs = []
            for f in keep:
                d = tr[tr.frame_idx == f]
                b = bl[bl.frame_idx == f]
                o = d[d.team_id == off_id]; v = d[d.team_id != off_id]
                fr_objs.append(go.Frame(name=str(f), data=[
                    go.Scatter(x=o.px, y=o.py, mode="markers+text",
                               text=[jersey.get(p, "") for p in o.player_id],
                               textposition="middle center",
                               textfont=dict(size=8, color="#0B0E14"),
                               marker=dict(size=15, color=TH.OFFENSE,
                                           line=dict(width=1.3, color="rgba(255,255,255,.35)"))),
                    go.Scatter(x=v.px, y=v.py, mode="markers+text",
                               text=[jersey.get(p, "") for p in v.player_id],
                               textposition="middle center",
                               textfont=dict(size=8, color="#0B0E14"),
                               marker=dict(size=15, color=TH.DEFENSE,
                                           line=dict(width=1.3, color="rgba(255,255,255,.35)"))),
                    go.Scatter(x=b.px if len(b) else [], y=b.py if len(b) else [],
                               mode="markers",
                               marker=dict(size=10, color=TH.BALL,
                                           line=dict(width=1.2, color="#0B0E14")))]))
            fig = CP.blank_court(half=False, height=430)
            d0 = tr[tr.frame_idx == keep[0]]; b0 = bl[bl.frame_idx == keep[0]]
            o0 = d0[d0.team_id == off_id]; v0 = d0[d0.team_id != off_id]
            fig.add_trace(go.Scatter(x=o0.px, y=o0.py, mode="markers+text",
                                     text=[jersey.get(p, "") for p in o0.player_id],
                                     textposition="middle center",
                                     textfont=dict(size=8, color="#0B0E14"),
                                     marker=dict(size=15, color=TH.OFFENSE,
                                                 line=dict(width=1.3, color="rgba(255,255,255,.35)")),
                                     name=teams.get(off_id, "Off")))
            fig.add_trace(go.Scatter(x=v0.px, y=v0.py, mode="markers+text",
                                     text=[jersey.get(p, "") for p in v0.player_id],
                                     textposition="middle center",
                                     textfont=dict(size=8, color="#0B0E14"),
                                     marker=dict(size=15, color=TH.DEFENSE,
                                                 line=dict(width=1.3, color="rgba(255,255,255,.35)")),
                                     name=teams.get(row.def_team_id, "Def")))
            fig.add_trace(go.Scatter(x=b0.px if len(b0) else [], y=b0.py if len(b0) else [],
                                     mode="markers",
                                     marker=dict(size=10, color=TH.BALL,
                                                 line=dict(width=1.2, color="#0B0E14")),
                                     name="Ball"))
            fig.frames = fr_objs
            fig.update_layout(
                updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=1.06,
                    bgcolor=TH.PANEL_2, bordercolor=TH.LINE,
                    font=dict(color=TH.TEXT, size=11),
                    buttons=[dict(label="▶  Play", method="animate",
                                  args=[None, dict(frame=dict(duration=90, redraw=True),
                                                   fromcurrent=True, mode="immediate")]),
                             dict(label="❚❚  Pause", method="animate",
                                  args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                     mode="immediate")])])],
                sliders=[dict(active=0, y=-0.02, x=0.02, len=0.95,
                              bgcolor=TH.PANEL_2, bordercolor=TH.LINE,
                              currentvalue=dict(prefix="clock ", font=dict(size=11, color=TH.TEXT_DIM)),
                              font=dict(size=9, color=TH.TEXT_FAINT),
                              steps=[dict(method="animate", label=DATA.fmt_clock(
                                              tr[tr.frame_idx == f].game_clock.iloc[0]),
                                          args=[[str(f)], dict(mode="immediate",
                                                frame=dict(duration=0, redraw=True))])
                                     for f in keep])],
                showlegend=True)
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
            st.caption(f"{len(keep)} rendered frames from {len(frames)} available at 5 fps "
                       f"(downsampled from 25 fps for browser performance).")

with right:
    st.markdown("### Possession timeline")
    if tl.empty:
        st.info("No events tagged to this chance.")
    else:
        rows = []
        for r in tl.itertuples(index=False):
            t = (r.frame - f0) / C.FPS
            who = names.get(r.actor_id, "") if pd.notna(r.actor_id) else ""
            rows.append(f'<div><span class="t">{t:5.1f}s</span>  '
                        f'<span class="k">{r.event_table:<16s}</span> '
                        f'{who} <span class="t">{r.detail}</span></div>')
        st.markdown(f'<div class="tl">{"".join(rows)}</div>', unsafe_allow_html=True)

    st.markdown("### Spatial context")
    sp = DATA.load("chance_spacing")
    row_sp = sp[sp.chance_id == row.id]
    if row_sp.empty:
        st.caption("No spacing samples for this chance.")
    else:
        s = row_sp.iloc[0]
        TH.kpis([
            ("Hull area", f"{s.hull_area_mean:,.0f} ft²", "mean over chance"),
            ("Nearest-neighbour", f"{s.mean_nn_dist_mean:.1f} ft", "offence"),
            ("Width", f"{s.width_mean:.1f} ft", f"depth {s.depth_mean:.1f} ft"),
            ("In paint", f"{s.n_in_paint_mean:.1f}", f"def {s.n_def_in_paint_mean:.1f}"),
        ])
        samples = DATA.load("chance_spacing_samples")
        smp = samples[samples.chance_id == row.id]
        if len(smp) > 2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=smp.hull_area, mode="lines",
                                     line=dict(color=TH.ACCENT, width=2), name="Hull area"))
            fig.add_trace(go.Scatter(y=smp.n_def_in_paint * 150, mode="lines",
                                     line=dict(color=TH.ACCENT_3, width=1.6, dash="dot"),
                                     name="Def in paint ×150"))
            fig.update_layout(height=190, margin=dict(l=10, r=10, t=26, b=24),
                              xaxis_title="sample (0.4 s apart)", showlegend=True)
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

st.markdown("---")
st.markdown("### Everything that happened in this game")
tabs = st.tabs(["Shots", "Drives", "Picks", "Full event log"])
sh = DATA.load("shots_features"); sh = sh[sh.game_id == gid]
dr = DATA.load("drives_features"); dr = dr[dr.game_id == gid]
pk = DATA.load("picks_features"); pk = pk[pk.game_id == gid]

with tabs[0]:
    c = st.columns([1.2, 1])
    with c[0]:
        f = CP.shot_chart(sh, height=440, hover=["region", "shot_type", "contest_level"],
                          title="")
        st.plotly_chart(f, width='stretch', config={"displayModeBar": False})
    with c[1]:
        t = sh[["period", "start_game_clock", "shooter_id_canonical", "region", "shot_type",
                "contest_level", "closest_def_dist", "three", "outcome", "xfg", "shot_quality"]].copy()
        t["player"] = t.shooter_id_canonical.map(names)
        t["clock"] = t.start_game_clock.map(DATA.fmt_clock)
        st.dataframe(t[["period", "clock", "player", "region", "shot_type", "contest_level",
                        "closest_def_dist", "outcome", "xfg", "shot_quality"]].round(2),
                     hide_index=True, width='stretch', height=440)
with tabs[1]:
    f = CP.blank_court(half=True, height=440)
    CP.arrows(f, dr.location_x, dr.location_y, dr.end_loc_x, dr.end_loc_y,
              color=TH.ACCENT, width=1.8, opacity=.6)
    f.add_trace(go.Scatter(x=dr.location_x, y=dr.location_y, mode="markers",
                           marker=dict(size=7, color=TH.ACCENT_2), name="start",
                           customdata=np.stack([dr.ballhandler_id_canonical.map(names),
                                                dr.category, dr.end_type], -1),
                           hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} → "
                                         "%{customdata[2]}<extra></extra>"))
    st.plotly_chart(f, width='stretch', config={"displayModeBar": False})
with tabs[2]:
    f = CP.shot_chart(pk, made_col="direct", color_by="bhr_def_type", height=440,
                      hover=["location_type", "bhr_def_type", "scr_def_type"])
    st.plotly_chart(f, width='stretch', config={"displayModeBar": False})
with tabs[3]:
    t = tl_all.copy(); t["player"] = t.actor_id.map(names)
    t["min"] = (t.frame / C.FPS / 60).round(2)
    st.dataframe(t[["min", "period", "event_table", "player", "detail", "chance_id"]],
                 hide_index=True, width='stretch', height=420)
