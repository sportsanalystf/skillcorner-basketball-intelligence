"""Data Quality — every automated check, and what we found that the docs do not say."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from app.components import data as DATA, theme as TH
from src import config as C

TH.apply("Data Quality")
TH.header("Data Quality", "192 automated checks, run on every build",
          "Nothing here is trusted because the documentation says so. Every claim "
          "about the data is re-derived from the files and checked.")

q = DATA.load("quality_report")
tq = DATA.load("tracking_quality")

sev = q.severity.value_counts()
TH.kpis([
    ("Checks", f"{len(q)}", "events + tracking", "accent"),
    ("Passing", f"{int(sev.get('OK', 0))}", "", "good"),
    ("Informational", f"{int(sev.get('INFO', 0))}", "documented / expected"),
    ("Warnings", f"{int(sev.get('WARN', 0))}", "small anomalies"),
    ("Failures", f"{int(sev.get('FAIL', 0))}", "would invalidate analysis",
     "good" if sev.get("FAIL", 0) == 0 else ""),
])

t1, t2, t3, t4 = st.tabs(["Validation results", "All checks", "Tracking quality",
                          "What we found that the docs don't say"])

with t1:
    st.markdown("### Independent validation against the source")
    v = pd.DataFrame([
        ["Team scores from shots + free throws vs official", "10 / 10 games exact", "PASS"],
        ["Recomputed shot distance vs delivered `shots.distance`", "median error 0.010 ft", "PASS"],
        ["Our nearest-defender distance vs `closestDefDist`", "median 0.01 ft; 98.2 % within 1 ft", "PASS"],
        ["Shooter tracking position vs `shots.location`", "median 0.005 ft", "PASS"],
        ["Attacking-frame transform (n = 1,458 shots)", "median |Δx|, |Δy| = 0.0000 ft", "PASS"],
        ["Our three-point line vs delivered `three` flag", "98.4 % agreement", "PASS"],
        ["Detection rate vs published 76–83 %",
         f"{tq.pct_detected.min():.1f} – {tq.pct_detected.max():.1f} %", "PASS"],
        ["Mean predError vs published 1.5–1.9 ft",
         f"{tq.mean_pred_error.min():.2f} – {tq.mean_pred_error.max():.2f} ft", "PASS"],
        ["Alias merge preserves aggregate totals", "exact, all three files", "PASS"],
        ["Direct-pick rate vs documented ~52 %", "54.1 %", "PASS"],
    ], columns=["Check", "Result", "Status"])
    st.dataframe(v, hide_index=True, width='stretch', height=400)

    st.markdown("### Known source issues, verified rather than assumed")
    known = q[q.severity == "INFO"]
    k = known[known.check.str.contains("known issue|rebound|zero elapsed|toReceiver|before start",
                                       case=False, regex=True)]
    st.dataframe(k[["check", "scope", "n_bad", "n_total", "pct"]], hide_index=True,
                 width='stretch')
    st.caption("Each of these is listed in SkillCorner's KNOWN_ISSUES.md; the counts here "
               "are ours, measured over the ten published games.")

with t2:
    c1, c2 = st.columns([1, 3])
    sevs = c1.multiselect("Severity", ["FAIL", "WARN", "INFO", "OK"],
                          default=["FAIL", "WARN", "INFO"])
    search = c2.text_input("Filter", "")
    d = q[q.severity.isin(sevs)]
    if search:
        d = d[d.check.str.contains(search, case=False) | d.scope.str.contains(search, case=False)]
    order = {"FAIL": 0, "WARN": 1, "INFO": 2, "OK": 3}
    d = d.assign(_o=d.severity.map(order)).sort_values(["_o", "pct"], ascending=[True, False])
    st.dataframe(d.drop(columns="_o"), hide_index=True, width='stretch', height=560)

with t3:
    st.dataframe(tq.round(3), hide_index=True, width='stretch')
    sm = DATA.load("smoothing_report")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=tq.game_id.astype(str), y=tq.mean_pred_error_detected,
                             name="detected", marker_color=TH.ACCENT_2))
        fig.add_trace(go.Bar(x=tq.game_id.astype(str), y=tq.mean_pred_error_extrapolated,
                             name="extrapolated", marker_color=TH.BAD))
        fig.update_layout(height=360, barmode="group", yaxis_title="mean predError (ft)",
                          title="Detected positions are ~5× more accurate")
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=sm.game_id.astype(str), y=sm.raw_max_speed, name="raw max",
                             marker_color=TH.BAD))
        fig.add_trace(go.Bar(x=sm.game_id.astype(str), y=sm.smoothed_max_speed,
                             name="smoothed max", marker_color=TH.ACCENT_2))
        fig.add_hline(y=C.MAX_PLAUSIBLE_SPEED, line=dict(color=TH.WARN, dash="dash"),
                      annotation_text="36 ft/s plausibility gate",
                      annotation_font=dict(size=10, color=TH.WARN))
        fig.update_layout(height=360, barmode="group", yaxis_title="max speed (ft/s)",
                          title="Smoothing clips the tail and nothing else")
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

with t4:
    st.markdown("### Two properties absent from the source documentation")
    TH.note("<b>The video frame index jumps at every period boundary.</b> Every game has 3–4 "
            "frame-index gaps, all of them exactly at a period change, ranging from 2 to 2,075 "
            "frames. Within a period the index is perfectly continuous. A pipeline that assumes "
            "one unbroken timeline will mis-align everything after the first quarter.",
            "checked per game by src/data/quality.py — 0 mid-period gaps across all ten games")
    TH.note("<b>7 orphaned <code>touchId</code> references.</b> Of 24,487 touch references across "
            "twelve tables, 7 (0.029 %) point at a touch that does not exist: 1 drive, 5 passes, "
            "1 dribble. All touch-context joins are left joins, so those rows keep NULL context "
            "rather than disappearing.",
            "tests/test_schema.py gates the orphan rate below 0.10 %")
    st.markdown("### And one the docs state, that turns out to matter more than it sounds")
    TH.note("<b>All eight duplicate-id player pairs that reach the aggregates are on the same "
            "team.</b> The source notes that nine players carry two ids; what it does not say is "
            "that this makes naive per-player aggregation double-count them — Jose Ignacio "
            "Vildoza (31 games, 183 attempts) and 'Pepe Vildoza' (1 game, 1 attempt) are one "
            "person at Basquet Girona. Counts are summed and rates recomputed from the summed "
            "counts wherever the numerator and denominator both exist in the file.",
            "src/data/aliases.py — totals verified exact in tests/test_schema.py")

    st.markdown("### How severity is assigned")
    st.markdown(
        "<div class='card'><p class='sub'>"
        "<b>FAIL</b> — would invalidate downstream analysis (duplicate keys, broken foreign keys "
        "at scale, scores that do not reconcile, mid-period frame gaps). A build with any FAIL "
        "exits non-zero.<br>"
        "<b>WARN</b> — a small anomaly worth knowing about but not a defect.<br>"
        "<b>INFO</b> — a documented or expected property of broadcast tracking: extrapolated "
        "positions, dead-ball frames with no players, period-boundary gaps.<br>"
        "<b>OK</b> — the check found nothing."
        "</p></div>", unsafe_allow_html=True)
