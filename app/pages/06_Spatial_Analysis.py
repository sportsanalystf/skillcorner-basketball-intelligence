"""Spatial Analysis — spacing, court control, spatial advantage."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from app.components import data as DATA, theme as TH, courtplot as CP
from src import config as C
from src.spatial import geometry as G
from src.preprocessing import court

TH.apply("Spatial Analysis")
TH.header("Spatial Analysis", "Spacing · court control · spatial advantage",
          "Everything on this page is computed by this project from raw positions. "
          "None of it is a SkillCorner metric.")

sp = DATA.load("chance_spacing")
ch = DATA.events("chances").rename(columns={"id": "chance_id"})
sh = DATA.load("shots_features")
teams = DATA.teams(); names = DATA.player_name_map()
m = sp.merge(ch[["chance_id", "game_id", "off_team_id", "def_team_id", "pts_scored",
                 "outcome", "transition", "quality_index"]], on="chance_id", how="left")
m["team"] = m.off_team_id.map(teams)

t1, t2, t3, t4 = st.tabs(["Spacing & outcomes", "Team spacing", "Court control", "Spatial advantage"])

with t1:
    TH.kpis([
        ("Chances measured", f"{len(m):,}", "sampled every 0.4 s", "accent"),
        ("Mean hull area", f"{m.hull_area_mean.mean():,.0f} ft²", "5 offensive players"),
        ("Nearest-neighbour", f"{m.mean_nn_dist_mean.mean():.1f} ft", ""),
        ("Offensive width", f"{m.width_mean.mean():.1f} ft", f"depth {m.depth_mean.mean():.1f} ft"),
        ("Corner occupancy", f"{m.n_in_corners_mean.mean():.2f}", "players in a corner"),
        ("Paint (offence)", f"{m.n_in_paint_mean.mean():.2f}", f"defence {m.n_def_in_paint_mean.mean():.2f}"),
    ])
    st.markdown("### Does better spacing produce more points?")
    c1, c2 = st.columns(2)
    with c1:
        q = pd.qcut(m.hull_area_mean, 5, labels=["Q1 tightest", "Q2", "Q3", "Q4", "Q5 widest"])
        g = (m.groupby(q, observed=True).agg(n=("chance_id", "size"),
             ppc=("pts_scored", "mean"), to=("outcome", lambda s: s.eq("TO").mean()),
             hull=("hull_area_mean", "mean")).reset_index())
        g.columns = ["quintile", "n", "pts_per_chance", "turnover_rate", "hull_area"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=g.quintile.astype(str), y=g.pts_per_chance,
                             marker_color=TH.ACCENT, name="pts / chance"))
        fig.update_layout(height=340, yaxis_title="points per chance",
                          title="Convex-hull area quintile")
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.dataframe(g.round(3), hide_index=True, width='stretch')
    with c2:
        sm = sh.merge(sp[["chance_id", "hull_area_mean", "mean_nn_dist_mean",
                          "n_in_corners_mean", "n_def_in_paint_mean"]],
                      on="chance_id", how="left").dropna(subset=["hull_area_mean"])
        q2 = pd.qcut(sm.hull_area_mean, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        g2 = (sm.groupby(q2, observed=True).agg(n=("id", "size"), fg=("outcome", "mean"),
              xfg=("xfg", "mean"), sq=("shot_quality", "mean"),
              def_dist=("closest_def_dist", "mean"),
              open_rate=("contested", lambda s: 1 - s.mean())).reset_index())
        g2.columns = ["quintile", "n", "fg_pct", "xfg", "skc_shot_quality",
                      "def_dist", "open_rate"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=g2.quintile.astype(str), y=g2.skc_shot_quality,
                             marker_color=TH.ACCENT_2, name="shot quality"))
        fig.add_trace(go.Scatter(x=g2.quintile.astype(str), y=g2.def_dist * 8,
                                 mode="lines+markers", name="def dist × 8",
                                 line=dict(color=TH.ACCENT_3, width=2)))
        fig.update_layout(height=340, title="Shot quality by chance spacing quintile")
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.dataframe(g2.round(3), hide_index=True, width='stretch')
    corr = m[["hull_area_mean", "mean_nn_dist_mean", "width_mean", "spacing_entropy_mean",
              "n_in_corners_mean", "n_def_in_paint_mean", "pts_scored"]].corr()["pts_scored"]
    st.markdown("### Correlation of spacing metrics with points scored in the chance")
    cd = corr.drop("pts_scored").sort_values()
    fig = go.Figure(go.Bar(x=cd.values, y=cd.index, orientation="h",
                           marker_color=np.where(cd.values > 0, TH.ACCENT_2, TH.BAD)))
    fig.update_layout(height=280, xaxis_title="Pearson r with chance points",
                      margin=dict(l=10, r=10, t=20, b=30))
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    TH.caveat("All of these correlations are small (|r| < 0.1). Chance-level scoring is "
              "dominated by shot-making variance; spacing is a weak signal at this "
              "sample size and should not be read as causal.")

with t2:
    g = (m.groupby("team").agg(chances=("chance_id", "size"),
         hull=("hull_area_mean", "mean"), nn=("mean_nn_dist_mean", "mean"),
         width=("width_mean", "mean"), depth=("depth_mean", "mean"),
         entropy=("spacing_entropy_mean", "mean"), corners=("n_in_corners_mean", "mean"),
         paint=("n_in_paint_mean", "mean"), beyond_arc=("n_beyond_arc_mean", "mean"),
         ppc=("pts_scored", "mean")).reset_index().sort_values("hull", ascending=False))
    st.dataframe(g.round(3), hide_index=True, width='stretch', height=430)
    fig = go.Figure(go.Scatter(x=g.hull, y=g.ppc, mode="markers+text", text=g.team,
                    textposition="top center", textfont=dict(size=9, color=TH.TEXT_DIM),
                    marker=dict(size=10 + g.chances / 20, color=g.corners,
                                colorscale=TH.SEQ, line=dict(width=.5, color="#0B0E14"),
                                colorbar=dict(title=dict(text="corner<br>occupancy",
                                                         font=dict(size=10)),
                                              thickness=10, len=.6))))
    fig.update_layout(height=460, xaxis_title="mean convex hull area (ft²)",
                      yaxis_title="points per chance",
                      title="Team spacing vs offensive output")
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    st.caption("Each team has ~100–210 tracked chances (1–2 games). This is a "
               "description of these games, not a season ranking.")

with t3:
    st.markdown("### Time-to-arrive court control")
    st.markdown(
        "<div class='card'><p class='sub'>Each player is given a 0.20 s reaction "
        "delay, then travels in a straight line at 21 ft/s (the 99.9th percentile of "
        "smoothed player speed here). A cell's control is "
        "<code>σ(1.2 · (t_def − t_off))</code> — orange where the offence arrives first, "
        "blue where the defence does. This ignores turning cost and acceleration "
        "limits, so read it as relative accessibility, not a probability of winning "
        "a loose ball.</p></div>", unsafe_allow_html=True)
    gsel = DATA.games()
    lbl = st.selectbox("Game", gsel.label.tolist())
    grow = gsel[gsel.label == lbl].iloc[0]; gid = int(grow.game_id)
    shg = sh[sh.game_id == gid].dropna(subset=["location_x"]).copy()
    shg["player"] = shg.shooter_id_canonical.map(names)
    shg["lab"] = (shg.period.astype(str) + "Q " + shg.start_game_clock.map(DATA.fmt_clock)
                  + " · " + shg.player.fillna("?") + " · " + shg.region.fillna("")
                  + " · " + np.where(shg.outcome, "MADE", "miss"))
    ssel = st.selectbox("Shot", shg.lab.tolist())
    srow = shg[shg.lab == ssel].iloc[0]
    f = int(srow.start_frame)
    tr = DATA.tracking_window(gid, f - 2, f + 2, viz=False)
    tr = tr[tr.frame_idx == tr.frame_idx.iloc[(tr.frame_idx - f).abs().argmin()]] if len(tr) else tr
    if tr.empty:
        st.info("No tracking at that frame.")
    else:
        off = tr[tr.team_id == srow.off_team_id]
        dfn = tr[tr.team_id != srow.off_team_id]
        oxy = off[["x_att", "y_att"]].to_numpy("float64")
        ov = np.nan_to_num(off[["vx_att", "vy_att"]].to_numpy("float64"))
        dxy = dfn[["x_att", "y_att"]].to_numpy("float64")
        dv = np.nan_to_num(dfn[["vx_att", "vy_att"]].to_numpy("float64"))
        ok_o = np.isfinite(oxy).all(1); ok_d = np.isfinite(dxy).all(1)
        if ok_o.sum() < 2 or ok_d.sum() < 2:
            st.info("Not enough tracked players at this frame.")
        else:
            gx, gy = G.make_grid(80, 44)
            ctrl = G.court_control(oxy[ok_o], ov[ok_o], dxy[ok_d], dv[ok_d], gx, gy)
            c1, c2 = st.columns(2)
            with c1:
                fig = CP.blank_court(half=True, height=470, title="Court control")
                CP.heat_surface(fig, gx, gy, ctrl, colorscale=TH.DIV, zmid=0.5,
                                zmin=0, zmax=1, cbar_title="offence<br>control")
                CP.snapshot(fig, off[ok_o], dfn[ok_d], labels=None)
                fig.add_trace(go.Scatter(x=[srow.location_x], y=[srow.location_y],
                              mode="markers", name="shot",
                              marker=dict(size=15, symbol="star", color=TH.BALL,
                                          line=dict(width=1, color="#0B0E14"))))
                st.plotly_chart(fig, width='stretch',
                                config={"displayModeBar": False})
            with c2:
                val = G.court_value_surface(gx, gy)
                adv = (ctrl - 0.5) * 2 * val
                fig = CP.blank_court(half=True, height=470,
                                     title="Spatial advantage = control × court value")
                CP.heat_surface(fig, gx, gy, adv, colorscale=TH.DIV, zmid=0,
                                cbar_title="expected<br>points edge")
                CP.snapshot(fig, off[ok_o], dfn[ok_d], labels=None)
                st.plotly_chart(fig, width='stretch',
                                config={"displayModeBar": False})
            TH.kpis([
                ("Offence-controlled area", f"{(ctrl > .5).mean():.0%}", "of the half court"),
                ("Controlled value", f"{adv[adv > 0].sum() / max((adv != 0).sum(), 1):.3f}",
                 "mean positive advantage"),
                ("Shot quality", f"{srow.shot_quality:.0f}" if pd.notna(srow.shot_quality) else "–",
                 "SkillCorner index"),
                ("Model xFG", f"{srow.xfg:.1%}", ""),
                ("Defender distance", f"{srow.closest_def_dist:.1f} ft", str(srow.contest_level)),
            ])

with t4:
    st.markdown("### Spatial advantage through a possession")
    st.caption("Court value is the expected points per attempt at each spot; the "
               "advantage surface weights control by that value. Summed over the "
               "half court it gives one number per frame.")
    gsel = DATA.games()
    lbl2 = st.selectbox("Game", gsel.label.tolist(), key="adv_game")
    grow2 = gsel[gsel.label == lbl2].iloc[0]; gid2 = int(grow2.game_id)
    chg = ch[(ch.game_id == gid2)].copy()
    chg = chg[chg.end_frame > chg.start_frame + 40].sort_values("start_frame")
    chg["lab"] = (chg.period.astype(str) + "Q " + chg.start_game_clock.map(DATA.fmt_clock)
                  + " · " + chg.off_team_id.map(teams) + " · " + chg.outcome.fillna("?")
                  + " · " + chg.pts_scored.fillna(0).astype(int).astype(str) + " pts")
    csel = st.selectbox("Chance", chg.lab.tolist(), key="adv_ch")
    crow = chg[chg.lab == csel].iloc[0]
    tr = DATA.tracking_window(gid2, int(crow.start_frame), int(crow.end_frame), viz=True)
    if tr.empty:
        st.info("No tracking for this chance.")
    else:
        gx, gy = G.make_grid(40, 22)
        val = G.court_value_surface(gx, gy)
        series = []
        for f, d in tr.groupby("frame_idx"):
            off = d[d.team_id == crow.off_team_id]; dfn = d[d.team_id != crow.off_team_id]
            oxy = off[["x_s", "y_s"]].to_numpy("float64")
            dxy = dfn[["x_s", "y_s"]].to_numpy("float64")
            lh = off.left_hoop.iloc[0] if len(off) else np.nan
            if not np.isfinite(lh) or len(oxy) < 3 or len(dxy) < 3:
                continue
            s = 1.0 if lh > .5 else -1.0
            oxy = oxy * s; dxy = dxy * s
            oxy = oxy[np.isfinite(oxy).all(1)]; dxy = dxy[np.isfinite(dxy).all(1)]
            if len(oxy) < 3 or len(dxy) < 3:
                continue
            ctrl = G.court_control(oxy, np.zeros_like(oxy), dxy, np.zeros_like(dxy), gx, gy)
            adv = ((ctrl - .5) * 2 * val)
            series.append({"frame": f, "clock": d.game_clock.iloc[0],
                           "advantage": float(adv.sum() / 100.0),
                           "off_area": float((ctrl > .5).mean())})
        s = pd.DataFrame(series)
        if len(s) > 2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=(s.frame - s.frame.min()) / C.FPS, y=s.advantage,
                                     mode="lines", name="value-weighted advantage",
                                     line=dict(color=TH.ACCENT, width=2.4)))
            fig.add_trace(go.Scatter(x=(s.frame - s.frame.min()) / C.FPS, y=s.off_area * 5,
                                     mode="lines", name="offence-controlled area × 5",
                                     line=dict(color=TH.ACCENT_3, width=1.6, dash="dot")))
            fig.update_layout(height=340, xaxis_title="seconds into the chance",
                              yaxis_title="advantage index")
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
            TH.kpis([
                ("Peak advantage", f"{s.advantage.max():.2f}",
                 f"at {(s.advantage.idxmax()) * 0.2:.1f}s"),
                ("Mean advantage", f"{s.advantage.mean():.2f}", ""),
                ("Outcome", str(crow.outcome), f"{int(crow.pts_scored or 0)} points"),
            ])
            TH.caveat("The advantage index is defined by this project and has no "
                      "external validation. A rising line means the offence is gaining "
                      "access to more valuable space; it does not by itself predict points.")
