"""Player Lab — multi-dimensional profiles, comparison, similarity, role clusters."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from app.components import data as DATA, theme as TH, courtplot as CP
from src import config as C

TH.apply("Player Lab")
TH.header("Player Development Lab", "Profiles · comparison · similarity",
          "Game-level tracking (10 games) and season-level offensive aggregates "
          "(293 games) side by side — never mixed into a single score.")

gl = DATA.load("player_game_profile")
style = DATA.load("season_style_clustered")
sh = DATA.load("shots_features")
dr = DATA.load("drives_features")
pk = DATA.load("picks_features")
names = DATA.player_name_map()

tab_p, tab_c, tab_s, tab_k = st.tabs(["Player card", "Compare", "Similarity", "Role clusters"])

# ---------------------------------------------------------------- player card
with tab_p:
    elig = gl[gl.minutes_tracked.fillna(0) >= 8].sort_values("points", ascending=False)
    who = st.selectbox("Player", elig.player_name.tolist(), index=0, key="pl_card")
    p = elig[elig.player_name == who].iloc[0]
    pid = p.player_id_canonical

    st.markdown(f"#### {p.player_name} · {p.team_name}")
    TH.kpis([
        ("Games tracked", f"{int(p.games or 0)}", f"{p.minutes_tracked:.0f} min", "accent"),
        ("Points", f"{int(p.points or 0)}", f"{int(p.fga or 0)} FGA"),
        ("FG%", f"{p.fg_pct:.1%}" if pd.notna(p.fg_pct) else "–",
         f"xFG {p.xfg:.1%}" if pd.notna(p.xfg) else ""),
        ("Shots vs expected", f"{p.shot_value_added:+.1f}" if pd.notna(p.shot_value_added) else "–",
         "points above model", "good" if (p.shot_value_added or 0) > 0 else ""),
        ("Drives", f"{int(p.drives or 0)}", f"{p.drive_ppd:.2f} pts/drive" if pd.notna(p.drive_ppd) else ""),
        ("Picks used", f"{int(p.picks_used or 0)}", f"{p.pick_ppp:.2f} ppp" if pd.notna(p.pick_ppp) else ""),
    ])
    TH.kpis([
        ("Distance", f"{p.distance_ft/5280:.2f} mi" if pd.notna(p.distance_ft) else "–",
         f"{p.distance_per_min:.0f} ft/min" if pd.notna(p.distance_per_min) else "", "cool"),
        ("Mean speed", f"{p.mean_speed:.1f} ft/s" if pd.notna(p.mean_speed) else "–",
         f"p95 {p.p95_speed:.1f}" if pd.notna(p.p95_speed) else "", "cool"),
        ("Top speed", f"{p.max_speed:.1f} ft/s" if pd.notna(p.max_speed) else "–", "", "cool"),
        ("Peak decel", f"{p.max_decel:.1f} ft/s²" if pd.notna(p.max_decel) else "–", "", "cool"),
        ("High-intensity", f"{p.high_intensity_share:.1%}" if pd.notna(p.high_intensity_share) else "–",
         "of distance ≥15 ft/s", "cool"),
        ("Touches / min", f"{p.usage_touches_per_min:.2f}" if pd.notna(p.usage_touches_per_min) else "–", ""),
    ])

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("### Shot map")
        s = sh[sh.shooter_id_canonical == pid]
        if len(s):
            st.plotly_chart(CP.shot_chart(s, height=420,
                            hover=["region", "shot_type", "contest_level", "closest_def_dist"]),
                            width='stretch', config={"displayModeBar": False})
        else:
            st.info("No tracked shots.")
    with c2:
        st.markdown("### Where he moves")
        path = C.PROCESSED_DIR / "tracking_viz"
        pos = []
        for gpath in sorted(path.glob("*.parquet")):
            d = pd.read_parquet(gpath, columns=["player_id", "x_s", "y_s", "left_hoop",
                                                "off_team_id", "team_id"],
                                filters=[("player_id", "=", int(pid))])
            if len(d):
                pos.append(d)
        if pos:
            pp = pd.concat(pos)
            pp = pp.dropna(subset=["left_hoop"])
            sign = np.where(pp.left_hoop > 0.5, 1.0, -1.0)
            pp["ax"] = pp.x_s * sign; pp["ay"] = pp.y_s * sign
            offside = pp[pp.team_id == pp.off_team_id]
            f = CP.blank_court(half=False, height=420)
            f.add_trace(go.Histogram2d(x=offside.ax, y=offside.ay, nbinsx=64, nbinsy=36,
                                       colorscale=TH.SEQ, opacity=.85, showscale=False,
                                       hovertemplate="x=%{x:.0f} y=%{y:.0f}<br>%{z}<extra></extra>"))
            f.data = tuple([f.data[-1]] + list(f.data[:-1]))
            st.plotly_chart(f, width='stretch', config={"displayModeBar": False})
            st.caption("Occupancy on offence, attacking frame (offensive hoop left). "
                       "5 fps samples across all tracked games.")
        else:
            st.info("No tracking rows for this player.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### Offensive profile")
        cols = ["shots", "fga", "fg_pct", "three_rate", "rim_rate", "mean_shot_dist",
                "contested_rate", "catch_and_shoot_rate", "assisted_rate",
                "xfg", "skc_shot_quality"]
        st.dataframe(pd.DataFrame({"metric": cols,
                                   "value": [round(p[c], 3) if pd.notna(p.get(c)) else None
                                             for c in cols]}),
                     hide_index=True, width='stretch', height=330)
    with c2:
        st.markdown("### Creation profile")
        cols = ["drives", "drive_ppd", "drive_blowby_rate", "drive_kickout_rate",
                "drive_to_rate", "picks_used", "pick_ppp", "pick_assist_rate",
                "picks_set", "screen_ppp", "roll_rate", "passes", "assist_opp_rate"]
        st.dataframe(pd.DataFrame({"metric": cols,
                                   "value": [round(p[c], 3) if pd.notna(p.get(c)) else None
                                             for c in cols]}),
                     hide_index=True, width='stretch', height=330)
    with c3:
        st.markdown("### Defensive activity")
        mw = DATA.load("matchup_workload")
        m = mw[mw.player_id_canonical == pid]
        cols = ["closeouts", "closeout_close_rate", "closeout_end_dist", "closeout_stop_rate"]
        rows = [(c, round(p[c], 3) if pd.notna(p.get(c)) else None) for c in cols]
        if len(m):
            mm = m.iloc[0]
            rows += [("minutes_guarding", round(mm.minutes_guarding, 1)),
                     ("assignments", int(mm.assignments)),
                     ("distinct_opponents", int(mm.distinct_opponents)),
                     ("shots_allowed", int(mm.shots_allowed)),
                     ("points_allowed", int(mm.points_allowed))]
        st.dataframe(pd.DataFrame(rows, columns=["metric", "value"]),
                     hide_index=True, width='stretch', height=330)
        TH.caveat("Defensive numbers here are <b>activity</b>, not a rating. "
                  "Points allowed on a matchup confound scheme with the individual.")

# ------------------------------------------------------------------- compare
with tab_c:
    elig = gl[gl.minutes_tracked.fillna(0) >= 8]
    pick = st.multiselect("Players (2–5)", elig.player_name.tolist(),
                          default=elig.nlargest(3, "points").player_name.tolist())
    if len(pick) >= 2:
        RADAR = ["shots", "fg_pct", "three_rate", "rim_rate", "drives", "drive_ppd",
                 "picks_used", "pick_assist_rate", "passes", "distance_per_min",
                 "p95_speed", "touches"]
        sub = elig[elig.player_name.isin(pick)].set_index("player_name")
        pop = elig[RADAR].astype(float)
        fig = go.Figure()
        for i, nm in enumerate(pick):
            vals = []
            for c in RADAR:
                v = sub.loc[nm, c]
                s = pop[c].dropna()
                vals.append(float((s < v).mean()) if pd.notna(v) and len(s) else 0.0)
            fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=RADAR + [RADAR[0]],
                                          fill="toself", name=nm, opacity=.45,
                                          line=dict(color=TH.CATEG[i % len(TH.CATEG)], width=2)))
        fig.update_layout(height=520, polar=dict(
            bgcolor=TH.PANEL, radialaxis=dict(range=[0, 1], showticklabels=False,
                                              gridcolor=TH.LINE),
            angularaxis=dict(gridcolor=TH.LINE, tickfont=dict(size=10, color=TH.TEXT_DIM))),
            showlegend=True)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.caption("Axes are percentile ranks within the tracked population "
                   f"(n={len(elig)} players with ≥8 tracked minutes), not raw values.")
        st.dataframe(sub[RADAR].round(3), width='stretch')

# ---------------------------------------------------------------- similarity
with tab_s:
    st.markdown("### Season offensive-style similarity")
    st.caption("Built on the 293-game aggregates, so the sample is a full season. "
               "Offence only — no defensive information enters this model.")
    lbl = style.player_name + " · " + style.team_name
    who = st.selectbox("Player", lbl.tolist(),
                       index=int(style.attempts.idxmax()), key="sim_pl")
    i = int(lbl[lbl == who].index[0])

    from src.models.similarity import StyleSpace
    @st.cache_resource(show_spinner=False)
    def _space():
        return StyleSpace(DATA.load("season_style"))
    ss = _space()
    sim = ss.most_similar(i, k=10)
    c1, c2 = st.columns([1, 1.15])
    with c1:
        st.dataframe(sim.round(3), hide_index=True, width='stretch', height=380)
    with c2:
        j = [k for k in np.argsort(-ss.similarity_matrix()[i]) if k != i][0]
        contrast = ss.feature_contrast(i, j).head(10)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=contrast.feature, x=contrast.player_z, orientation="h",
                             name=style.loc[i, "player_name"], marker_color=TH.ACCENT))
        fig.add_trace(go.Bar(y=contrast.feature, x=contrast.other_z, orientation="h",
                             name=style.loc[j, "player_name"], marker_color=TH.ACCENT_3))
        fig.update_layout(height=380, barmode="group", xaxis_title="z-score vs population",
                          margin=dict(l=10, r=10, t=30, b=30))
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    TH.caveat("Similarity describes <b>style</b>, not quality, and does not mean two "
              "players are interchangeable. The metric is cosine distance in a "
              f"{ss.n_components}-component PCA space capturing {ss.explained:.0%} of variance.")

# -------------------------------------------------------------- role clusters
with tab_k:
    ksel = DATA.load("cluster_k_selection")
    desc = DATA.load("cluster_descriptions")
    reps = DATA.load("cluster_representatives")
    c1, c2 = st.columns([1, 1.4])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ksel.k, y=ksel.silhouette, mode="lines+markers",
                                 line=dict(color=TH.ACCENT, width=2), name="silhouette"))
        fig.update_layout(height=250, xaxis_title="k", yaxis_title="silhouette",
                          margin=dict(l=10, r=10, t=30, b=30))
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.caption("Silhouette peaks at k=3 and stays low (0.17–0.27) throughout: "
                   "offensive style in this league is closer to a continuum than to "
                   "well-separated archetypes. That is itself the finding.")
    with c2:
        st.dataframe(desc, hide_index=True, width='stretch', height=250)

    show_fine = st.toggle("Show exploratory k=6 partition", value=False)
    if show_fine:
        st.dataframe(DATA.load("cluster_descriptions_k6"), hide_index=True,
                     width='stretch')
        st.dataframe(DATA.load("cluster_representatives_k6").round(2), hide_index=True,
                     width='stretch', height=260)
        colcl = "cluster_fine"
    else:
        st.dataframe(reps.round(2), hide_index=True, width='stretch', height=260)
        colcl = "cluster"

    fig = go.Figure()
    for c_ in sorted(style[colcl].unique()):
        d = style[style[colcl] == c_]
        fig.add_trace(go.Scatter(x=d.pc1, y=d.pc2, mode="markers", name=f"cluster {c_}",
                                 marker=dict(size=9, color=TH.CATEG[int(c_) % len(TH.CATEG)],
                                             opacity=.8, line=dict(width=.5, color="#0B0E14")),
                                 customdata=np.stack([d.player_name, d.team_name,
                                                      d.games_played], -1),
                                 hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}"
                                               "<br>%{customdata[2]} games<extra></extra>"))
    fig.update_layout(height=520, xaxis_title="PC1", yaxis_title="PC2", showlegend=True)
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    load = DATA.load("style_pca_loadings")
    st.markdown("### What the components mean")
    st.dataframe(load[["feature", "pc1", "pc2", "pc3"]].round(3)
                   .sort_values("pc1", key=abs, ascending=False),
                 hide_index=True, width='stretch', height=300)
