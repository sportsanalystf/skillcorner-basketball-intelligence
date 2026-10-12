"""Possession reconstruction — why did this shot happen, and why did this drive work?"""
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

TH.apply("Why Did This Happen?")
TH.header("Why Did This Happen?", "Possession reconstruction",
          "Pick any shot or drive. The platform rebuilds the sequence that produced "
          "it — the action, the coverage, the spacing, the defensive response.")

sh = DATA.load("shots_features")
dr = DATA.load("drives_features")
col = DATA.load("drive_collapse")
sp = DATA.load("chance_spacing")
names = DATA.player_name_map(); teams = DATA.teams()
g = DATA.games()

mode = st.radio("Reconstruct", ["A shot", "A drive"], horizontal=True)
c1, c2 = st.columns([2, 1])
lbl = c1.selectbox("Game", g.label.tolist())
grow = g[g.label == lbl].iloc[0]; gid = int(grow.game_id)


def frame_snapshot(gid, frame, off_team):
    tr = DATA.tracking_window(gid, frame - 2, frame + 2, viz=False)
    if tr.empty:
        return None, None, None
    nearest = tr.frame_idx.iloc[(tr.frame_idx - frame).abs().argmin()]
    d = tr[tr.frame_idx == nearest].copy()
    d["player_name"] = d.player_id.map(names)
    return d[d.team_id == off_team], d[d.team_id != off_team], nearest


# ==========================================================================
if mode == "A shot":
    s = sh[(sh.game_id == gid) & sh.location_x.notna()].copy()
    s["player"] = s.shooter_id_canonical.map(names)
    s["lab"] = (s.period.astype(str) + "Q " + s.start_game_clock.map(DATA.fmt_clock) +
                " · " + s.player.fillna("?") + " · " + s.region.fillna("") + " " +
                np.where(s.three, "3PT", "2PT") + " · " +
                np.where(s.outcome, "MADE", "missed"))
    only_made = c2.toggle("Made shots only", value=False)
    if only_made:
        s = s[s.outcome]
    if s.empty:
        st.warning("No shots match."); st.stop()
    sel = st.selectbox(f"Shot ({len(s)} available)", s.lab.tolist())
    r = s[s.lab == sel].iloc[0]

    st.markdown("---")
    a, b, c_, d_ = st.columns(4)
    a.markdown(f"<div class='card'><h4>1 · The shot</h4><p class='sub'>"
               f"<b>{r.player}</b> · {'3PT' if r.three else '2PT'} from "
               f"<b>{r.region}</b>, {r.distance:.1f} ft.<br>"
               f"Type: {r.complex_shot_type}. Release {r.release_time:.1f} s after the "
               f"catch, {int(r.dribbles_before or 0)} dribbles.<br>"
               f"Result: <b>{'MADE' if r.outcome else 'MISSED'}</b>"
               f"{' · and-one' if r.fouled and r.outcome else ''}"
               f"{' · blocked' if r.blocked else ''}.</p></div>", unsafe_allow_html=True)
    dfd = names.get(r.closest_def_id_canonical, "unknown")
    b.markdown(f"<div class='card'><h4>2 · The defender</h4><p class='sub'>"
               f"Closest at release: <b>{dfd}</b>, {r.closest_def_dist:.1f} ft away.<br>"
               f"Contest level <b>{r.contest_level}</b>"
               f"{' (blocked)' if r.blocked else ''}.<br>"
               f"Defenders within 6 ft: {int(r.n_def_within_6ft) if pd.notna(r.get('n_def_within_6ft')) else '–'}"
               f" · in the paint: {int(r.n_def_in_paint) if pd.notna(r.get('n_def_in_paint')) else '–'}."
               f"</p></div>", unsafe_allow_html=True)
    crea_detail = ""
    if r.creation_type == "pick":
        crea_detail = (f"Ball screen {r.prev_pick_frames_before/C.FPS:.1f} s earlier by "
                       f"<b>{names.get(r.prev_pick_screener_id,'?')}</b> at the "
                       f"{r.prev_pick_loc_type}. Defence played <b>{r.prev_pick_bhr_def}</b> "
                       f"on the ball, <b>{r.prev_pick_scr_def}</b> on the screener.")
    elif r.creation_type == "drive":
        crea_detail = (f"Drive {r.prev_drive_frames_before/C.FPS:.1f} s earlier by "
                       f"<b>{names.get(r.prev_drive_bhr_id,'?')}</b> "
                       f"({r.prev_drive_category} → {r.prev_drive_end_type})"
                       f"{', blew by his man' if r.prev_drive_blowby else ''}.")
    elif r.creation_type == "closeout":
        crea_detail = (f"Closeout {r.prev_closeout_frames_before/C.FPS:.1f} s earlier: "
                       f"defender closed {r.prev_closeout_start_dist:.1f} → "
                       f"{r.prev_closeout_end_dist:.1f} ft; handler chose "
                       f"<b>{r.prev_closeout_action}</b>.")
    elif r.creation_type == "off_ball_screen":
        crea_detail = (f"Off-ball screen {r.prev_obscreen_frames_before/C.FPS:.1f} s "
                       f"earlier for <b>{names.get(r.prev_obs_cutter_id,'?')}</b>; "
                       f"cutter's man played <b>{r.prev_obs_cutter_def}</b>.")
    elif r.creation_type == "handoff":
        crea_detail = (f"Hand-off {r.prev_handoff_frames_before/C.FPS:.1f} s earlier; "
                       f"receiver's defender went <b>{r.prev_handoff_rec_def}</b>.")
    else:
        crea_detail = "No tracked action marking in the 4 s before release."
    c_.markdown(f"<div class='card'><h4>3 · How it was created</h4><p class='sub'>"
                f"<span class='pill on'>{r.creation_type}</span><br>{crea_detail}<br>"
                f"{'Assisted by <b>'+str(names.get(r.passer_id_canonical,'?'))+'</b>.' if r.assisted else 'Unassisted.'}"
                f"</p></div>", unsafe_allow_html=True)
    spc = sp[sp.chance_id == r.chance_id]
    sp_txt = "no spacing samples"
    if len(spc):
        z = spc.iloc[0]
        sp_txt = (f"Hull {z.hull_area_mean:,.0f} ft², nearest-neighbour "
                  f"{z.mean_nn_dist_mean:.1f} ft, {z.n_in_corners_mean:.1f} in the "
                  f"corners, {z.n_def_in_paint_mean:.1f} defenders in the paint.")
    d_.markdown(f"<div class='card'><h4>4 · Spacing & result</h4><p class='sub'>{sp_txt}<br>"
                f"Model xFG <b>{r.xfg:.1%}</b> · SkillCorner shot quality "
                f"<b>{r.shot_quality:.0f}</b>.<br>"
                f"Points: <b>{int(r.points)}</b> "
                f"({r.points - r.xpts:+.2f} vs the model's expectation).</p></div>",
                unsafe_allow_html=True)

    st.markdown("### The sequence on the court")
    c1, c2 = st.columns([1.1, 1])
    lookback = st.slider("Seconds before the shot", 1.0, 6.0, 3.0, 0.5)
    f1 = int(r.start_frame); f0 = f1 - int(lookback * C.FPS)
    tr = DATA.tracking_window(gid, f0, f1, viz=True)
    with c1:
        if tr.empty:
            st.info("No tracking.")
        else:
            sign = 1.0 if (tr.left_hoop.dropna().mean() or 1) > .5 else -1.0
            tr = tr.copy(); tr["px"] = tr.x_s * sign; tr["py"] = tr.y_s * sign
            fig = CP.blank_court(half=False, height=470, title="Movement into the shot")
            for pid, gg in tr.groupby("player_id"):
                is_off = gg.team_id.iloc[0] == r.off_team_id
                is_shooter = pid == r.shooter_id
                fig.add_trace(go.Scatter(
                    x=gg.px, y=gg.py, mode="lines",
                    line=dict(color=TH.BALL if is_shooter else
                              (TH.OFFENSE if is_off else TH.DEFENSE),
                              width=3 if is_shooter else 1.5),
                    opacity=1.0 if is_shooter else .6,
                    name=str(names.get(pid, pid)),
                    hovertemplate=f"{names.get(pid,pid)}<extra></extra>"))
            last = tr[tr.frame_idx == tr.frame_idx.max()]
            fig.add_trace(go.Scatter(x=last.px, y=last.py, mode="markers",
                          marker=dict(size=11, color=[TH.OFFENSE if t == r.off_team_id
                                                      else TH.DEFENSE for t in last.team_id],
                                      line=dict(width=1, color="#0B0E14")), name="at release"))
            fig.add_trace(go.Scatter(x=[r.location_x], y=[r.location_y], mode="markers",
                          marker=dict(size=17, symbol="star",
                                      color=TH.ACCENT_2 if r.outcome else TH.BAD,
                                      line=dict(width=1.2, color="#0B0E14")), name="shot"))
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    with c2:
        off, dfn, nf = frame_snapshot(gid, f1, r.off_team_id)
        if off is None or off.empty:
            st.info("No frame at release.")
        else:
            oxy = off[["x_att", "y_att"]].to_numpy("float64")
            dxy = dfn[["x_att", "y_att"]].to_numpy("float64")
            ov = np.nan_to_num(off[["vx_att", "vy_att"]].to_numpy("float64"))
            dv = np.nan_to_num(dfn[["vx_att", "vy_att"]].to_numpy("float64"))
            ko = np.isfinite(oxy).all(1); kd = np.isfinite(dxy).all(1)
            fig = CP.blank_court(half=True, height=470, title="Court control at release")
            if ko.sum() >= 2 and kd.sum() >= 2:
                gx, gy = G.make_grid(70, 40)
                ctrl = G.court_control(oxy[ko], ov[ko], dxy[kd], dv[kd], gx, gy)
                CP.heat_surface(fig, gx, gy, ctrl, colorscale=TH.DIV, zmid=.5, zmin=0,
                                zmax=1, opacity=.92, cbar_title="offence<br>control")
            CP.snapshot(fig, off[ko], dfn[kd])
            fig.add_trace(go.Scatter(x=[r.location_x], y=[r.location_y], mode="markers",
                          marker=dict(size=17, symbol="star",
                                      color=TH.ACCENT_2 if r.outcome else TH.BAD),
                          name="shot"))
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

    st.markdown("### Full chance timeline")
    tl = DATA.timeline(gid); tl = tl[tl.chance_id == r.chance_id]
    rows = []
    for e in tl.itertuples(index=False):
        t = (e.frame - r.start_frame) / C.FPS
        who = names.get(e.actor_id, "") if pd.notna(e.actor_id) else ""
        hi = "style='color:#FF6B35'" if e.event_table == "shot" else ""
        rows.append(f'<div {hi}><span class="t">{t:+6.1f}s</span>  '
                    f'<span class="k">{e.event_table:<16s}</span> {who} '
                    f'<span class="t">{e.detail}</span></div>')
    st.markdown(f'<div class="tl">{"".join(rows)}</div>', unsafe_allow_html=True)

# ==========================================================================
else:
    d = dr[(dr.game_id == gid) & dr.location_x.notna()].copy()
    d["player"] = d.ballhandler_id_canonical.map(names)
    d["lab"] = (d.period.astype(str) + "Q " + d.start_game_clock.map(DATA.fmt_clock) +
                " · " + d.player.fillna("?") + " · " + d.category.fillna("?") + " → " +
                d.end_type.fillna("?") + " · " + d.points_total.astype(int).astype(str) + " pts")
    only_succ = c2.selectbox("Filter", ["All drives", "Successful only", "Unsuccessful only"])
    if only_succ == "Successful only": d = d[d.successful]
    elif only_succ == "Unsuccessful only": d = d[~d.successful]
    if d.empty:
        st.warning("No drives match."); st.stop()
    sel = st.selectbox(f"Drive ({len(d)} available)", d.lab.tolist())
    r = d[d.lab == sel].iloc[0]

    st.markdown("---")
    TH.kpis([
        ("Ball handler", r.player, teams.get(r.off_team_id, ""), "accent"),
        ("Created by", str(r.category), f"direction {r.direction}"),
        ("Ended", str(r.end_type), "blowby" if r.blowby else "no blowby"),
        ("Ground gained", f"{r.drive_gain_ft:.1f} ft", f"{r.duration_s:.1f} s"),
        ("Help closed", f"{r.help_closed_ft:.1f} ft" if pd.notna(r.help_closed_ft) else "–",
         f"{r.help_at_start:.1f} → {r.help_min:.1f} ft" if pd.notna(r.help_at_start) else ""),
        ("Paint defenders", f"+{r.paint_defenders_added:.0f}" if pd.notna(r.paint_defenders_added) else "–",
         f"{r.n_def_paint_start:.0f} → {r.n_def_paint_max:.0f}" if pd.notna(r.n_def_paint_start) else ""),
        ("Points", f"{int(r.points_total)}", "own + assisted",
         "good" if r.points_total > 0 else ""),
    ])

    c1, c2 = st.columns([1.1, 1])
    with c1:
        f0, f1 = int(r.start_frame), int(r.end_frame)
        tr = DATA.tracking_window(gid, f0, f1 + 25, viz=True)
        if tr.empty:
            st.info("No tracking.")
        else:
            sign = 1.0 if (tr.left_hoop.dropna().mean() or 1) > .5 else -1.0
            tr = tr.copy(); tr["px"] = tr.x_s * sign; tr["py"] = tr.y_s * sign
            fig = CP.blank_court(half=True, height=480, title="The drive and the response")
            for pid, gg in tr.groupby("player_id"):
                is_off = gg.team_id.iloc[0] == r.off_team_id
                is_bh = pid == r.ballhandler_id
                is_pd = pid == r.ballhandler_def_id
                fig.add_trace(go.Scatter(
                    x=gg.px, y=gg.py, mode="lines",
                    line=dict(color=TH.BALL if is_bh else
                              (TH.WARN if is_pd else (TH.OFFENSE if is_off else TH.DEFENSE)),
                              width=3.2 if is_bh else (2.2 if is_pd else 1.4)),
                    opacity=1.0 if (is_bh or is_pd) else .5,
                    name=str(names.get(pid, pid)),
                    hovertemplate=f"{names.get(pid,pid)}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[r.location_x], y=[r.location_y], mode="markers",
                          marker=dict(size=13, color=TH.ACCENT_2, symbol="circle"),
                          name="drive start"))
            fig.add_trace(go.Scatter(x=[r.end_loc_x], y=[r.end_loc_y], mode="markers",
                          marker=dict(size=13, color=TH.ACCENT, symbol="square"),
                          name="drive end"))
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
            st.caption("Gold = ball handler, amber = his defender, orange = other "
                       "offensive players, blue = other defenders.")
    with c2:
        cc = col[col.drive_id == r.id].sort_values("t_since_start")
        if len(cc) > 2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cc.t_since_start, y=cc.nearest_def_dist,
                          mode="lines+markers", name="on-ball defender",
                          line=dict(color=TH.WARN, width=2.4)))
            fig.add_trace(go.Scatter(x=cc.t_since_start, y=cc.second_def_dist,
                          mode="lines+markers", name="help defender",
                          line=dict(color=TH.ACCENT_3, width=2.4)))
            fig.add_trace(go.Scatter(x=cc.t_since_start, y=cc.n_def_in_paint * 4,
                          mode="lines", name="defenders in paint × 4",
                          line=dict(color=TH.BAD, width=1.6, dash="dot")))
            fig.update_layout(height=290, xaxis_title="seconds into the drive",
                              yaxis_title="distance (ft)",
                              title="Defensive collapse")
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=cc.t_since_start, y=cc.hull_area, mode="lines",
                           name="offensive hull", line=dict(color=TH.ACCENT, width=2.2)))
            fig2.update_layout(height=190, xaxis_title="seconds into the drive",
                               yaxis_title="hull area (ft²)")
            st.plotly_chart(fig2, width='stretch', config={"displayModeBar": False})
        else:
            st.info("No collapse samples for this drive.")

    st.markdown("### How this drive compares")
    peers = dr[(dr.category == r.category) & (dr.end_type == r.end_type)]
    comp = pd.DataFrame({
        "metric": ["ground gained (ft)", "duration (s)", "help closed (ft)",
                   "paint defenders added", "points"],
        "this drive": [r.drive_gain_ft, r.duration_s, r.help_closed_ft,
                       r.paint_defenders_added, r.points_total],
        f"peer mean (n={len(peers)})": [peers.drive_gain_ft.mean(), peers.duration_s.mean(),
                                        peers.help_closed_ft.mean(),
                                        peers.paint_defenders_added.mean(),
                                        peers.points_total.mean()],
        "all drives": [dr.drive_gain_ft.mean(), dr.duration_s.mean(),
                       dr.help_closed_ft.mean(), dr.paint_defenders_added.mean(),
                       dr.points_total.mean()],
    })
    st.dataframe(comp.round(2), hide_index=True, width='stretch')
    st.markdown("### Successful vs unsuccessful drives, league-wide")
    cmp2 = (dr.groupby("successful").agg(
        n=("id", "size"), gain=("drive_gain_ft", "mean"), dur=("duration_s", "mean"),
        blowby=("blowby", "mean"), help_start=("help_at_start", "mean"),
        help_min=("help_min", "mean"), help_closed=("help_closed_ft", "mean"),
        paint_added=("paint_defenders_added", "mean"),
        hull_min=("hull_min", "mean")).reset_index())
    st.dataframe(cmp2.round(3), hide_index=True, width='stretch')
