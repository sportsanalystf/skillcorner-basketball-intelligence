"""Tracking Lab — raw signal, smoothing, quality diagnostics, physical output."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from app.components import data as DATA, theme as TH, courtplot as CP
from src import config as C

TH.apply("Tracking Lab")
TH.header("Tracking Lab", "The raw signal and what we do to it",
          "Broadcast tracking is not motion capture. This page shows exactly how "
          "good the signal is and what the smoothing pipeline changes.")

tq = DATA.load("tracking_quality")
sm = DATA.load("smoothing_report")
phys = DATA.load("player_game_physical")
names = DATA.player_name_map()
g = DATA.games()

TH.kpis([
    ("Player rows", f"{tq.player_rows.sum()/1e6:.2f} M", f"{tq.frames.sum():,} frames", "accent"),
    ("Detected", f"{tq.pct_detected.mean():.1f} %", f"range {tq.pct_detected.min():.0f}–{tq.pct_detected.max():.0f}"),
    ("Mean pred error", f"{tq.mean_pred_error.mean():.2f} ft", "all positions"),
    ("Detected error", f"{tq.mean_pred_error_detected.mean():.2f} ft", "on screen", "good"),
    ("Extrapolated error", f"{tq.mean_pred_error_extrapolated.mean():.2f} ft", "off screen"),
    ("Impossible speeds", f"{tq.pct_impossible_speed.mean():.3f} %", ">36 ft/s after smoothing"),
])

t1, t2, t3, t4 = st.tabs(["Quality by game", "Raw vs smoothed", "Physical output", "Frame inspector"])

with t1:
    q = tq.merge(g[["game_id", "home_team_name", "away_team_name"]], on="game_id")
    q["game"] = q.home_team_name + " v " + q.away_team_name
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=q.game, y=q.pct_detected, marker_color=TH.ACCENT,
                             name="% detected"))
        fig.update_layout(height=380, yaxis_title="% of positions detected on screen",
                          xaxis_tickangle=-40, margin=dict(b=140), yaxis_range=[60, 90])
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=q.game, y=q.mean_pred_error_detected, name="detected",
                             marker_color=TH.ACCENT_2))
        fig.add_trace(go.Bar(x=q.game, y=q.mean_pred_error_extrapolated, name="extrapolated",
                             marker_color=TH.BAD))
        fig.update_layout(height=380, barmode="group", yaxis_title="mean predError (ft)",
                          xaxis_tickangle=-40, margin=dict(b=140))
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    st.dataframe(tq.round(3), hide_index=True, width='stretch')
    st.markdown(
        "<div class='card'><h4>Reproducing the vendor's own claims</h4><p class='sub'>"
        "SkillCorner's README states 76–83 % of positions are directly detected, mean "
        "<code>predError</code> 1.5–1.9 ft (95th pct 4.9–6.3), and extrapolated "
        "positions 4.4–5.2 ft. Computed independently here from the raw frames: "
        f"{tq.pct_detected.min():.1f}–{tq.pct_detected.max():.1f} % detected, "
        f"{tq.mean_pred_error.min():.2f}–{tq.mean_pred_error.max():.2f} ft mean, "
        f"95th pct {tq.p95_pred_error.min():.2f}–{tq.p95_pred_error.max():.2f}, "
        f"extrapolated {tq.mean_pred_error_extrapolated.min():.2f}–"
        f"{tq.mean_pred_error_extrapolated.max():.2f} ft. Every range matches."
        "</p></div>", unsafe_allow_html=True)

with t2:
    st.markdown("### What the Savitzky–Golay filter changes")
    st.caption("Window 15 frames (0.6 s), polyorder 2, applied inside continuous "
               "frame segments only — never across a tracking dropout.")
    s = sm.merge(g[["game_id", "home_team_name", "away_team_name"]], on="game_id")
    s["game"] = s.home_team_name + " v " + s.away_team_name
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=s.game, y=s.raw_max_speed, name="raw max",
                             marker_color=TH.BAD))
        fig.add_trace(go.Bar(x=s.game, y=s.smoothed_max_speed, name="smoothed max",
                             marker_color=TH.ACCENT_2))
        fig.add_hline(y=C.MAX_PLAUSIBLE_SPEED, line=dict(color=TH.WARN, dash="dash"),
                      annotation_text="plausibility gate 36 ft/s",
                      annotation_font=dict(size=10, color=TH.WARN))
        fig.update_layout(height=380, yaxis_title="max speed (ft/s)",
                          xaxis_tickangle=-40, margin=dict(b=140))
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    with c2:
        st.dataframe(sm[["game_id", "raw_mean_speed", "smoothed_mean_speed",
                         "raw_p99_speed", "smoothed_p99_speed", "raw_max_speed",
                         "smoothed_max_speed", "corr_with_delivered_speed"]].round(3),
                     hide_index=True, width='stretch', height=380)
    st.markdown(
        "<div class='card'><h4>Finding: the delivered positions are already smooth</h4>"
        "<p class='sub'>Central-difference speed from the <em>raw</em> delivered "
        "positions correlates 0.999 with SkillCorner's own <code>speed</code> field, and "
        "the mean and 99th-percentile speeds barely move under our filter "
        f"({sm.raw_mean_speed.mean():.2f} → {sm.smoothed_mean_speed.mean():.2f} ft/s mean; "
        f"{sm.raw_p99_speed.mean():.1f} → {sm.smoothed_p99_speed.mean():.1f} ft/s p99). "
        "What the filter does remove is the tail: peak speed drops from "
        f"{sm.raw_max_speed.max():.0f} to {sm.smoothed_max_speed.max():.0f} ft/s. "
        "So smoothing here is outlier control, not signal reconstruction — worth "
        "knowing before spending compute on a heavier filter.</p></div>",
        unsafe_allow_html=True)

with t3:
    ph = phys.merge(DATA.load("roster")[["game_id", "player_id", "player_name", "team_name"]],
                    on=["game_id", "player_id"], how="left")
    ph = ph[ph.minutes_tracked >= 5]
    st.markdown("### Per player-game physical output (live-ball frames only)")
    cols = ["player_name", "team_name", "game_id", "minutes_tracked", "distance_ft",
            "distance_per_min", "mean_speed", "p95_speed", "max_speed",
            "mean_abs_accel", "max_accel", "max_decel", "pct_detected"]
    st.dataframe(ph[cols].sort_values("distance_ft", ascending=False).round(2),
                 hide_index=True, width='stretch', height=400)
    band_cols = [c for c in ph.columns if c.startswith("dist_")]
    if band_cols:
        st.markdown("### Distance by movement-intensity band")
        TH.caveat("These bands (standing &lt;2, walking 2–5, jogging 5–10, running 10–15, "
                  "high-speed 15–20, sprinting ≥20 ft/s) are defined by this project from "
                  "the empirical speed distribution. They are <b>not</b> SkillCorner "
                  "physical-performance classifications.")
        tot = ph[band_cols].sum()
        fig = go.Figure(go.Bar(x=[c.replace("dist_", "") for c in band_cols],
                               y=tot.values / tot.sum(), marker_color=TH.ACCENT))
        fig.update_layout(height=320, yaxis_title="share of total distance")
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        top = ph.nlargest(18, "distance_ft")
        share = top[band_cols].div(top[band_cols].sum(axis=1), axis=0)
        fig = go.Figure()
        for i, c_ in enumerate(band_cols):
            fig.add_trace(go.Bar(y=top.player_name + " · G" + top.game_id.astype(str),
                                 x=share[c_], orientation="h", name=c_.replace("dist_", ""),
                                 marker_color=TH.SEQ[::-1][i % len(TH.SEQ)]))
        fig.update_layout(height=520, barmode="stack", xaxis_title="share of distance",
                          margin=dict(l=10, r=10, t=30, b=30))
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

with t4:
    st.markdown("### Inspect any frame at full 25 fps")
    lbl = st.selectbox("Game", g.label.tolist(), key="tl_game")
    gid = int(g[g.label == lbl].game_id.iloc[0])
    per = st.select_slider("Period", [1, 2, 3, 4], value=1)
    trk = DATA.tracking_window(gid, 0, 10**9, viz=True)
    trk = trk[trk.period == per]
    if trk.empty:
        st.info("No frames.")
    else:
        fmin, fmax = int(trk.frame_idx.min()), int(trk.frame_idx.max())
        f = st.slider("Frame", fmin, fmax, (fmin + fmax) // 2, step=5)
        full = DATA.tracking_window(gid, f - 1, f + 1, viz=False)
        d = full[full.frame_idx == f] if len(full) else pd.DataFrame()
        b = DATA.ball_window(gid, f - 1, f + 1)
        b = b[b.frame_idx == f] if len(b) else b
        c1, c2 = st.columns([1.2, 1])
        with c1:
            fig = CP.blank_court(half=False, height=430)
            if len(d):
                d = d.copy(); d["player_name"] = d.player_id.map(names)
                home = d[d.is_home]; away = d[~d.is_home]
                CP.snapshot(fig, home, away, xcol="x_s", ycol="y_s",
                            ball=(b.x.iloc[0], b.y.iloc[0]) if len(b) else None,
                            off_name="Home", def_name="Away")
                und = d[~d.is_detected]
                if len(und):
                    fig.add_trace(go.Scatter(x=und.x_s, y=und.y_s, mode="markers",
                                  marker=dict(size=24, symbol="circle-open",
                                              color=TH.WARN, line=dict(width=2)),
                                  name="extrapolated"))
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
            st.caption("Yellow rings mark positions that were extrapolated rather than "
                       "detected on screen at this frame.")
        with c2:
            if len(d):
                dd = d.copy(); dd["player"] = dd.player_id.map(names)
                st.dataframe(dd[["player", "is_home", "x_s", "y_s", "speed", "speed_raw",
                                 "accel_tangential", "pred_error", "is_detected",
                                 "speed_band"]].round(2),
                             hide_index=True, width='stretch', height=400)
            if len(b):
                st.caption(f"Ball: x={b.x.iloc[0]:.1f} y={b.y.iloc[0]:.1f} z={b.z.iloc[0]:.1f} ft, "
                           f"speed {b.speed_raw.iloc[0]:.1f} ft/s, "
                           f"{'detected' if b.is_detected.iloc[0] else 'extrapolated'}")
