"""Visual system for the app: palette, CSS, and the shared Plotly template."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# --------------------------------------------------------------------------
INK        = "#0B0E14"   # page background
PANEL      = "#141A24"   # card background
PANEL_2    = "#1B2331"   # raised card
LINE       = "#26303F"   # hairlines
TEXT       = "#E6EBF2"
TEXT_DIM   = "#8A97AA"
TEXT_FAINT = "#5C6879"

ACCENT     = "#FF6B35"   # SkillCorner-ish orange — primary
ACCENT_2   = "#3DDC97"   # teal — secondary / positive
ACCENT_3   = "#5B8DEF"   # blue — defence
WARN       = "#FFC145"
BAD        = "#F2545B"

OFFENSE    = "#FF6B35"
DEFENSE    = "#5B8DEF"
BALL       = "#FFD166"

SEQ = ["#141A24", "#1F3446", "#22536A", "#1F7A87", "#2FA391", "#7CC97F", "#DDE96F", "#FFC145", "#FF6B35"]
# Diverging scale tuned for the dark court: the midpoint IS the court fill, so a
# neutral cell disappears into the floor instead of greying the whole surface out.
DIV = [[0.00, "#2B6CB8"], [0.18, "#2A5B93"], [0.38, "#1D2E42"],
       [0.50, "#101722"], [0.62, "#4A2A22"], [0.82, "#B4502A"], [1.00, "#FF6B35"]]
DIV_BAR = ["#5B8DEF", "#8FA9E8", "#C3C9DC", "#E8E8E8", "#F3C3B0", "#FF9D74", "#FF6B35"]
CATEG = ["#FF6B35", "#3DDC97", "#5B8DEF", "#FFC145", "#C77DFF", "#F2545B", "#4CC9F0", "#B5E48C"]

COURT_LINE = "rgba(150,163,182,0.45)"
COURT_FILL = "#101722"


def register_template() -> None:
    t = go.layout.Template()
    t.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                  size=12, color=TEXT),
        title=dict(font=dict(size=15, color=TEXT), x=0, xanchor="left", pad=dict(b=10)),
        margin=dict(l=48, r=24, t=52, b=44),
        xaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE,
                   tickfont=dict(color=TEXT_DIM, size=11),
                   title=dict(font=dict(color=TEXT_DIM, size=11))),
        yaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE,
                   tickfont=dict(color=TEXT_DIM, size=11),
                   title=dict(font=dict(color=TEXT_DIM, size=11))),
        colorway=CATEG,
        # right-aligned so it never collides with the left-aligned chart title
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=LINE, borderwidth=0,
                    font=dict(size=11, color=TEXT_DIM), orientation="h",
                    yanchor="bottom", y=1.01, xanchor="right", x=1.0),
        hoverlabel=dict(bgcolor=PANEL_2, bordercolor=LINE,
                        font=dict(color=TEXT, size=12)),
    )
    pio.templates["bball"] = t
    pio.templates.default = "bball"


CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"], .stApp, .stApp *:not(.tl):not(.tl *):not(.ev) {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI',
                   'Helvetica Neue', Arial, sans-serif !important; }}
  .tl, .tl *, .ev {{ font-family: 'JetBrains Mono', ui-monospace, monospace !important; }}
  .stApp {{ background: {INK}; color: {TEXT}; }}
  section[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {LINE}; }}
  section[data-testid="stSidebar"] * {{ color: {TEXT}; }}

  .block-container {{ padding-top: 2.0rem; padding-bottom: 3rem; max-width: 1500px; }}

  h1, h2, h3, h4 {{ color: {TEXT}; letter-spacing: -0.02em; font-weight: 650; }}
  h1 {{ font-size: 1.85rem; margin-bottom: .1rem; }}
  h2 {{ font-size: 1.20rem; margin-top: 1.6rem; }}
  h3 {{ font-size: 1.00rem; color: {TEXT_DIM}; text-transform: uppercase;
        letter-spacing: .08em; font-weight: 600; margin-top: 1.4rem; }}
  p, li, .stMarkdown {{ color: {TEXT}; }}

  .eyebrow {{ color:{ACCENT}; font-size:.70rem; font-weight:700;
              letter-spacing:.16em; text-transform:uppercase; margin-bottom:.25rem; }}
  .sub {{ color:{TEXT_DIM}; font-size:.90rem; margin-top:.15rem; }}

  .kpi-row {{ display:flex; gap:12px; flex-wrap:wrap; margin:.6rem 0 1.1rem 0; }}
  .kpi {{ flex:1 1 150px; background:{PANEL}; border:1px solid {LINE};
          border-radius:10px; padding:12px 14px; min-width:140px; }}
  .kpi .label {{ color:{TEXT_FAINT}; font-size:.66rem; font-weight:600;
                 letter-spacing:.1em; text-transform:uppercase; }}
  .kpi .value {{ color:{TEXT}; font-size:1.5rem; font-weight:680;
                 font-variant-numeric: tabular-nums; line-height:1.25; margin-top:2px; }}
  .kpi .delta {{ font-size:.74rem; color:{TEXT_DIM}; }}
  .kpi.accent {{ border-left:3px solid {ACCENT}; }}
  .kpi.good   {{ border-left:3px solid {ACCENT_2}; }}
  .kpi.cool   {{ border-left:3px solid {ACCENT_3}; }}

  .card {{ background:{PANEL}; border:1px solid {LINE}; border-radius:10px;
           padding:16px 18px; margin-bottom:12px; }}
  .card h4 {{ margin:0 0 .4rem 0; font-size:.95rem; }}

  .note {{ background:{PANEL}; border-left:3px solid {ACCENT_3}; border-radius:0 8px 8px 0;
           padding:10px 14px; margin:6px 0; font-size:.88rem; color:{TEXT}; }}
  .note .ev {{ color:{TEXT_FAINT}; font-size:.74rem; font-family:'JetBrains Mono',monospace; }}

  .caveat {{ background:rgba(255,193,69,.07); border-left:3px solid {WARN};
             border-radius:0 8px 8px 0; padding:10px 14px; margin:10px 0;
             font-size:.84rem; color:{TEXT_DIM}; }}

  .pill {{ display:inline-block; padding:2px 9px; border-radius:999px;
           font-size:.70rem; font-weight:600; letter-spacing:.04em;
           background:{PANEL_2}; color:{TEXT_DIM}; border:1px solid {LINE};
           margin-right:5px; }}
  .pill.on {{ background:rgba(255,107,53,.14); color:{ACCENT}; border-color:rgba(255,107,53,.35); }}
  .pill.def {{ background:rgba(91,141,239,.14); color:{ACCENT_3}; border-color:rgba(91,141,239,.35); }}

  .tl {{ font-family:'JetBrains Mono', monospace; font-size:.78rem; line-height:1.65; }}
  .tl .t {{ color:{TEXT_FAINT}; }}
  .tl .k {{ color:{ACCENT}; font-weight:600; }}

  div[data-testid="stMetricValue"] {{ font-size:1.4rem; color:{TEXT}; }}
  div[data-testid="stMetricLabel"] {{ color:{TEXT_FAINT}; }}

  .stTabs [data-baseweb="tab-list"] {{ gap:2px; border-bottom:1px solid {LINE}; }}
  .stTabs [data-baseweb="tab"] {{ background:transparent; color:{TEXT_DIM};
      padding:8px 16px; font-size:.86rem; font-weight:550; }}
  .stTabs [aria-selected="true"] {{ color:{ACCENT}; border-bottom:2px solid {ACCENT}; }}

  .stDataFrame {{ border:1px solid {LINE}; border-radius:8px; }}
  [data-testid="stDataFrameResizable"] {{ background:{PANEL}; }}

  .stSlider [data-baseweb="slider"] div[role="slider"] {{ background:{ACCENT}; }}
  .stSelectbox div[data-baseweb="select"] > div {{ background:{PANEL_2};
      border-color:{LINE}; color:{TEXT}; }}
  .stMultiSelect div[data-baseweb="select"] > div {{ background:{PANEL_2}; border-color:{LINE}; }}
  hr {{ border-color:{LINE}; }}
  footer, #MainMenu, [data-testid="stToolbar"], [data-testid="stDecoration"],
  [data-testid="stStatusWidget"] {{ display:none !important; }}
  [data-testid="stSidebarNav"] li a span {{ font-size:.88rem; }}
  [data-testid="stHeader"] {{ background:transparent; height:0; }}
</style>
"""


def apply(page_title: str = "Basketball Intelligence") -> None:
    st.set_page_config(page_title=page_title, page_icon="🏀", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    register_template()


def header(title: str, eyebrow: str = "", sub: str = "") -> None:
    if eyebrow:
        st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f"# {title}")
    if sub:
        st.markdown(f'<div class="sub">{sub}</div>', unsafe_allow_html=True)


def kpis(items: list[tuple], cls: str = "") -> None:
    """items: (label, value, optional delta/subtext, optional css class)"""
    html = ['<div class="kpi-row">']
    for it in items:
        label, value = it[0], it[1]
        delta = it[2] if len(it) > 2 else ""
        k = it[3] if len(it) > 3 else cls
        html.append(f'<div class="kpi {k}"><div class="label">{label}</div>'
                    f'<div class="value">{value}</div>'
                    f'<div class="delta">{delta}</div></div>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def caveat(text: str) -> None:
    st.markdown(f'<div class="caveat">{text}</div>', unsafe_allow_html=True)


def note(text: str, evidence: str = "") -> None:
    ev = f'<div class="ev">{evidence}</div>' if evidence else ""
    st.markdown(f'<div class="note">{text}{ev}</div>', unsafe_allow_html=True)
