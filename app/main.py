"""Basketball Intelligence Platform — home / executive overview."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components import data as DATA, theme as TH, courtplot as CP

TH.apply("Basketball Intelligence")

TH.header("Basketball Intelligence Platform",
          eyebrow="SkillCorner Open Data · Liga Endesa ACB 2025/26",
          sub="Broadcast tracking at 25 fps, Game Intelligence events and season "
              "aggregates, turned into a scouting and player-development tool.")

g = DATA.games()
sh = DATA.load("shots_features")
tq = DATA.load("tracking_quality")
ch = DATA.events("chances")
agg = DATA.load("shots_merged")

TH.kpis([
    ("Games tracked", f"{len(g)}", "full 25 fps XY", "accent"),
    ("Tracking rows", f"{tq.player_rows.sum()/1e6:.1f} M", f"{tq.frames.sum():,} frames"),
    ("Tracked events", "71,984", "20 event families"),
    ("Chances", f"{len(ch):,}", f"{ch.pts_scored.mean():.2f} pts/chance"),
    ("Shots", f"{len(sh):,}", f"{sh[sh.is_box_score_fga].outcome.mean():.1%} FG"),
    ("Season players", f"{agg.player_id_canonical.nunique()}", "293 ACB games", "good"),
])

c1, c2 = st.columns([1.15, 1])
with c1:
    st.markdown("### League shot surface")
    fig = CP.hexbin_court(sh.dropna(subset=["location_x"]), value_col="points",
                          gridsize=3.4, min_count=4, height=430,
                          cbar_title="pts/shot")
    st.plotly_chart(fig, width='stretch',
                    config={"displayModeBar": False})
    st.caption("Hex colour = points per attempt; hex size = attempt volume. "
               "Attacking frame — the offensive hoop is always on the left.")

with c2:
    st.markdown("### What the platform does")
    st.markdown(
        """
<div class="card">
<h4>Eight analytical engines</h4>
<span class="pill on">Tracking</span><span class="pill on">Possessions</span>
<span class="pill on">Shots</span><span class="pill on">Drives</span>
<span class="pill on">Pick &amp; roll</span><span class="pill on">Spacing</span>
<span class="pill def">Defence</span><span class="pill def">Closeouts</span>
<p class="sub" style="margin-top:.6rem">Every engine is built directly on the
delivered schema — no invented fields, no fabricated outcomes. Metrics this
project defines itself are labelled as such throughout.</p>
</div>
""", unsafe_allow_html=True)
    st.markdown("### Data quality, measured not assumed")
    q = pd.DataFrame({
        "metric": ["Positions directly detected", "Mean prediction error",
                   "95th pct prediction error", "Extrapolated position error",
                   "Frames inside a live possession"],
        "value": [f"{tq.pct_detected.min():.1f} – {tq.pct_detected.max():.1f} %",
                  f"{tq.mean_pred_error.min():.2f} – {tq.mean_pred_error.max():.2f} ft",
                  f"{tq.p95_pred_error.min():.2f} – {tq.p95_pred_error.max():.2f} ft",
                  f"{tq.mean_pred_error_extrapolated.min():.2f} – "
                  f"{tq.mean_pred_error_extrapolated.max():.2f} ft",
                  f"{tq.pct_rows_in_possession.mean():.1f} %"],
    })
    st.dataframe(q, hide_index=True, width='stretch')
    st.caption("Computed here from the raw frames; reproduces SkillCorner's own "
               "published ranges (76–83 % detected, 1.5–1.9 ft mean error).")

st.markdown("---")
st.markdown("## Headline findings from this dataset")

F = [
    ("Transition creates far better looks — but not more points",
     "Transition chances average 50.6 shot quality vs 41.9 in the half court and "
     "are contested 56 % of the time vs 75 %. Points per chance are nevertheless "
     "identical (0.83 vs 0.84): the extra turnovers (13.4 % vs 11.9 %) eat the edge.",
     "n = 357 transition / 1,698 half-court chances"),
    ("Off-ball movement buys separation, not conversion",
     "Splitting shots by how far the four off-ball players travelled in the 3 s "
     "before release: the busiest quartile gets 1.0 ft more separation, is open "
     "38 % of the time vs 24 %, and carries a 5-point-higher expected FG% — yet "
     "converts at exactly the same 42.6 %.",
     "n = 1,436 shots with a full 3 s window"),
    ("Help arrives fast: 7.3 ft closed in the average drive",
     "Sampling the second-nearest defender every 5 frames through each drive, the "
     "help defender closes 7.3 ft and the defence adds 1.8 bodies to the paint "
     "between drive start and drive end.",
     "n = 660 drives, 8,351 tracked samples"),
    ("Going under the screen is the most effective common coverage",
     "Ball-handler coverage: under concedes 0.371 points per pick vs 0.468 for "
     "over and 0.422 for switch. Under also suppresses the direct-pick rate "
     "(44.6 % vs 56.5 %), i.e. it takes the action out of the play entirely.",
     "n = 1,520 picks (1,004 over / 280 under / 218 switch)"),
    ("Closeout tightness changes the outcome, but costs fouls",
     "Arriving within 4 ft cuts points allowed from 0.77 to 0.56 per closeout — "
     "and more than doubles the foul rate, 11.6 % vs 4.7 %. Stop rate is "
     "unchanged at 68 %.",
     "n = 601 closeouts"),
    ("Team spacing adds almost nothing to shot-level make probability",
     "Adding hull area, nearest-neighbour spacing, corner occupancy and paint "
     "congestion to the expected-FG model moves out-of-fold AUC from 0.687 to "
     "0.689. Once you know where the shot is from and how close the defender is, "
     "spacing is already priced in.",
     "5-fold grouped-by-game CV, n = 1,322 attempts"),
]
cols = st.columns(3)
for i, (t, body, ev) in enumerate(F):
    with cols[i % 3]:
        st.markdown(f'<div class="card"><h4>{t}</h4>'
                    f'<p class="sub">{body}</p>'
                    f'<div class="ev" style="color:#5C6879;font-size:.72rem;'
                    f'font-family:JetBrains Mono,monospace">{ev}</div></div>',
                    unsafe_allow_html=True)

st.markdown("---")
c1, c2 = st.columns([1, 1])
with c1:
    st.markdown("### Games in this release")
    gg = g[["date", "home_team_name", "home_score", "away_score", "away_team_name"]].copy()
    gg["date"] = pd.to_datetime(gg.date).dt.strftime("%Y-%m-%d")
    gg.columns = ["Date", "Home", "H", "A", "Away"]
    st.dataframe(gg, hide_index=True, width='stretch', height=390)
with c2:
    st.markdown("### Event volume by family")
    counts = {}
    for f in sorted((DATA.P / "events").glob("*.parquet")):
        counts[f.stem] = len(pd.read_parquet(f, columns=["id"]))
    cdf = pd.DataFrame(sorted(counts.items(), key=lambda x: x[1]),
                       columns=["table", "rows"])
    fig = go.Figure(go.Bar(x=cdf.rows, y=cdf.table, orientation="h",
                           marker=dict(color=TH.ACCENT, opacity=.85),
                           hovertemplate="%{y}: %{x:,}<extra></extra>"))
    fig.update_layout(height=390, xaxis_type="log", margin=dict(l=10, r=10, t=10, b=30),
                      xaxis_title="rows (log scale)")
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

TH.caveat(
    "<b>Scope.</b> Ten games of full tracking and events; season aggregates cover "
    "293 of 327 ACB 2025/26 games and are <b>offence only</b>. Anything derived "
    "from the ten games has a sample of roughly 100 chances per team — enough to "
    "describe tendencies, not to rank teams. Full caveats in docs/LIMITATIONS.md.")
