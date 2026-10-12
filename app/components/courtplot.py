"""Court-based Plotly figures — the visual language of the whole app."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.components import theme as TH
from src import config as C
from src.preprocessing import court


def _base(half: bool, height: int, title: str = "", pad: float = 3.0) -> go.Figure:
    X, Y = C.COURT_X_MAX, C.COURT_Y_MAX
    x0 = -X - pad
    x1 = (0.0 + pad) if half else (X + pad)
    fig = go.Figure()
    fig.update_layout(
        title=title, height=height,
        shapes=court_shapes_cached(half),
        xaxis=dict(range=[x0, x1], showgrid=False, zeroline=False, visible=False,
                   constrain="domain"),
        yaxis=dict(range=[-Y - pad, Y + pad], showgrid=False, zeroline=False,
                   visible=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor=TH.COURT_FILL, paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=44 if title else 8, b=8),
        showlegend=False,
        legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="left", x=0,
                    font=dict(size=10, color=TH.TEXT_DIM), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


_SHAPE_CACHE: dict[bool, list] = {}


def court_shapes_cached(half: bool) -> list:
    if half not in _SHAPE_CACHE:
        _SHAPE_CACHE[half] = court.court_shapes(half=half, line_color=TH.COURT_LINE)
    return _SHAPE_CACHE[half]


def blank_court(half: bool = True, height: int = 520, title: str = "") -> go.Figure:
    return _base(half, height, title)


# --------------------------------------------------------------------------
def shot_chart(df: pd.DataFrame, xcol="location_x", ycol="location_y",
               made_col="outcome", color_by: str | None = None,
               size_col: str | None = None, height: int = 540,
               title: str = "", hover: list[str] | None = None) -> go.Figure:
    fig = _base(True, height, title)
    d = df.dropna(subset=[xcol, ycol])
    if d.empty:
        return fig
    hover = hover or []
    customdata = d[hover].to_numpy() if hover else None
    htmpl = "<br>".join([f"<b>%{{customdata[{i}]}}</b>" if i == 0 else
                         f"{h}: %{{customdata[{i}]}}" for i, h in enumerate(hover)])

    if color_by:
        for i, (k, g) in enumerate(d.groupby(color_by)):
            fig.add_trace(go.Scatter(
                x=g[xcol], y=g[ycol], mode="markers", name=str(k),
                marker=dict(size=8 if size_col is None else
                            6 + 10 * (g[size_col] - d[size_col].min()) /
                            max(d[size_col].max() - d[size_col].min(), 1e-9),
                            color=TH.CATEG[i % len(TH.CATEG)],
                            line=dict(width=0.7, color="rgba(0,0,0,.5)"), opacity=.85),
                customdata=g[hover].to_numpy() if hover else None,
                hovertemplate=(htmpl + "<extra></extra>") if hover else None))
        fig.update_layout(showlegend=True)
    else:
        made = d[made_col].astype(bool)
        for mask, name, sym, col in ((made, "Made", "circle", TH.ACCENT_2),
                                     (~made, "Missed", "x", TH.BAD)):
            g = d[mask]
            if g.empty:
                continue
            fig.add_trace(go.Scatter(
                x=g[xcol], y=g[ycol], mode="markers", name=name,
                marker=dict(size=9 if name == "Made" else 7, symbol=sym, color=col,
                            line=dict(width=1, color="rgba(0,0,0,.45)"),
                            opacity=.9 if name == "Made" else .7),
                customdata=g[hover].to_numpy() if hover else None,
                hovertemplate=(htmpl + "<extra></extra>") if hover else None))
        fig.update_layout(showlegend=True)
    return fig


def hexbin_court(df: pd.DataFrame, xcol="location_x", ycol="location_y",
                 value_col: str | None = None, gridsize: float = 3.2,
                 min_count: int = 3, height: int = 540, title: str = "",
                 colorscale=None, zmid=None, cbar_title: str = "") -> go.Figure:
    """Hex-binned court surface. Colour = mean of ``value_col`` (or attempt count)."""
    fig = _base(True, height, title)
    d = df.dropna(subset=[xcol, ycol]).copy()
    if d.empty:
        return fig
    dx = gridsize; dy = gridsize * np.sqrt(3) / 2
    row = np.round(d[ycol] / dy).astype(int)
    col = np.round((d[xcol] / dx) - (row % 2) * 0.5).astype(int)
    d["_cx"] = (col + (row % 2) * 0.5) * dx
    d["_cy"] = row * dy
    agg = d.groupby(["_cx", "_cy"]).agg(
        n=(xcol, "size"),
        v=(value_col, "mean") if value_col else (xcol, "size")).reset_index()
    agg = agg[agg.n >= min_count]
    if agg.empty:
        return fig
    smax = agg.n.max()
    fig.add_trace(go.Scatter(
        x=agg._cx, y=agg._cy, mode="markers",
        marker=dict(symbol="hexagon", size=8 + 22 * np.sqrt(agg.n / smax),
                    color=agg.v, colorscale=colorscale or TH.SEQ, cmid=zmid,
                    line=dict(width=0), opacity=.92,
                    colorbar=dict(title=dict(text=cbar_title, font=dict(size=10)),
                                  thickness=10, len=.55, x=1.0,
                                  tickfont=dict(size=9, color=TH.TEXT_DIM))),
        customdata=np.stack([agg.n, agg.v], axis=-1),
        hovertemplate="n=%{customdata[0]}<br>value=%{customdata[1]:.3f}<extra></extra>"))
    return fig


def arrows(fig: go.Figure, x0, y0, x1, y1, color=None, width=1.6, opacity=.55,
           name: str = "") -> go.Figure:
    """Draw many start→end segments efficiently as one trace."""
    x0 = np.asarray(x0); y0 = np.asarray(y0); x1 = np.asarray(x1); y1 = np.asarray(y1)
    ok = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(x1) & np.isfinite(y1)
    n = ok.sum()
    if n == 0:
        return fig
    xs = np.empty(n * 3); ys = np.empty(n * 3)
    xs[0::3] = x0[ok]; xs[1::3] = x1[ok]; xs[2::3] = np.nan
    ys[0::3] = y0[ok]; ys[1::3] = y1[ok]; ys[2::3] = np.nan
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=name,
                             line=dict(color=color or TH.ACCENT, width=width),
                             opacity=opacity, hoverinfo="skip"))
    return fig


def snapshot(fig: go.Figure, off: pd.DataFrame, dfn: pd.DataFrame,
             ball: tuple | None = None, xcol="x_att", ycol="y_att",
             labels: dict | None = None, off_name="Offence", def_name="Defence",
             size: int = 15) -> go.Figure:
    """Plot one frame: five offensive players, five defenders, the ball."""
    for d, col, nm, sym in ((off, TH.OFFENSE, off_name, "circle"),
                            (dfn, TH.DEFENSE, def_name, "circle")):
        if d is None or d.empty:
            continue
        txt = [str(labels.get(p, "")) if labels else "" for p in d.player_id]
        fig.add_trace(go.Scatter(
            x=d[xcol], y=d[ycol], mode="markers+text", name=nm,
            text=txt, textposition="middle center",
            textfont=dict(size=8, color="#0B0E14", family="JetBrains Mono"),
            marker=dict(size=size, symbol=sym, color=col,
                        line=dict(width=1.4, color="rgba(255,255,255,.35)")),
            customdata=np.stack([d.get("player_name", pd.Series([""] * len(d))),
                                 d.get("speed", pd.Series([np.nan] * len(d)))], -1),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:.1f} ft/s<extra></extra>"))
    if ball is not None and np.all(np.isfinite(ball[:2])):
        fig.add_trace(go.Scatter(x=[ball[0]], y=[ball[1]], mode="markers", name="Ball",
                                 marker=dict(size=10, color=TH.BALL, symbol="circle",
                                             line=dict(width=1.2, color="#0B0E14")),
                                 hoverinfo="skip"))
    fig.update_layout(showlegend=True)
    return fig


def heat_surface(fig: go.Figure, grid_x, grid_y, z, colorscale=None, zmid=None,
                 opacity=.95, cbar_title="", zmin=None, zmax=None) -> go.Figure:
    fig.add_trace(go.Heatmap(
        x=grid_x[0], y=grid_y[:, 0], z=z, colorscale=colorscale or TH.DIV,
        zmid=zmid, zmin=zmin, zmax=zmax, opacity=opacity, hoverongaps=False,
        colorbar=dict(title=dict(text=cbar_title, font=dict(size=10)), thickness=10,
                      len=.55, tickfont=dict(size=9, color=TH.TEXT_DIM)),
        hovertemplate="x=%{x:.0f} y=%{y:.0f}<br>%{z:.2f}<extra></extra>"))
    fig.data = tuple([fig.data[-1]] + list(fig.data[:-1]))  # keep markers on top
    fig.update_layout(shapes=court_shapes_cached(True))       # redraw lines over the surface
    return fig
