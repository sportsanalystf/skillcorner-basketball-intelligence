"""Spatial features from 25 fps tracking: spacing, defensive geometry, control.

All functions operate in the **attacking frame** (offensive hoop at
``(-HOOP_X, 0)``), so a "left corner" is always the offence's own left. Inputs
are per-frame snapshots of on-court players.

Every metric here is defined and motivated in ``docs/METHODOLOGY.md``. None of
them is a SkillCorner metric — they are computed by this project from the raw
positions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

from src import config as C
from src.preprocessing import court

OFF_HOOP = np.array(court.OFF_HOOP)


# --------------------------------------------------------------------------
# Team spacing (offence only)
# --------------------------------------------------------------------------
def spacing_metrics(xy: np.ndarray, ball_xy: np.ndarray | None = None) -> dict:
    """Spacing descriptors for one offensive team at one instant.

    ``xy`` is (n, 2) in the attacking frame. Returns NaNs when fewer than three
    players are tracked, which happens on frames where broadcast tracking has
    dropped someone.
    """
    n = len(xy)
    out = {
        "n_off_tracked": n, "hull_area": np.nan, "mean_pairwise_dist": np.nan,
        "min_pairwise_dist": np.nan, "mean_nn_dist": np.nan, "width": np.nan,
        "depth": np.nan, "spread_radius": np.nan, "spacing_entropy": np.nan,
        "n_in_paint": np.nan, "n_in_corners": np.nan, "n_beyond_arc": np.nan,
        "mean_dist_to_hoop": np.nan, "mean_dist_to_ball": np.nan,
        "centroid_x": np.nan, "centroid_y": np.nan,
    }
    if n < 2:
        return out

    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    iu = np.triu_indices(n, 1)
    pw = d[iu]
    out["mean_pairwise_dist"] = float(pw.mean())
    out["min_pairwise_dist"] = float(pw.min())
    np.fill_diagonal(d, np.inf)
    out["mean_nn_dist"] = float(d.min(axis=1).mean())

    out["width"] = float(xy[:, 1].max() - xy[:, 1].min())
    out["depth"] = float(xy[:, 0].max() - xy[:, 0].min())
    c = xy.mean(axis=0)
    out["centroid_x"], out["centroid_y"] = float(c[0]), float(c[1])
    out["spread_radius"] = float(np.hypot(*(xy - c).T).mean())

    if n >= 3:
        try:
            out["hull_area"] = float(ConvexHull(xy).volume)  # 2-D: volume == area
        except (QhullError, ValueError):
            out["hull_area"] = 0.0

    # angular dispersion around the hoop: how evenly the offence surrounds it
    ang = np.arctan2(xy[:, 1] - OFF_HOOP[1], xy[:, 0] - OFF_HOOP[0])
    hist, _ = np.histogram(ang, bins=8, range=(-np.pi, np.pi))
    p = hist / max(hist.sum(), 1)
    nz = p[p > 0]
    out["spacing_entropy"] = float(-(nz * np.log(nz)).sum() / np.log(8))

    out["n_in_paint"] = int(court.in_paint(xy[:, 0], xy[:, 1]).sum())
    out["n_in_corners"] = int(court.in_corner(xy[:, 0], xy[:, 1]).sum())
    out["n_beyond_arc"] = int(court.is_three(xy[:, 0], xy[:, 1]).sum())
    out["mean_dist_to_hoop"] = float(np.hypot(*(xy - OFF_HOOP).T).mean())
    if ball_xy is not None and np.all(np.isfinite(ball_xy)):
        out["mean_dist_to_ball"] = float(np.hypot(*(xy - np.asarray(ball_xy)).T).mean())
    return out


# --------------------------------------------------------------------------
# Defensive geometry around a focal offensive player
# --------------------------------------------------------------------------
def defensive_pressure(focal_xy: np.ndarray, def_xy: np.ndarray,
                       radii=(4.0, 6.0, 10.0)) -> dict:
    """Defender proximity around one offensive player (the ball handler / shooter)."""
    out = {"nearest_def_dist": np.nan, "second_def_dist": np.nan,
           "help_gap": np.nan, "mean_def_dist": np.nan}
    for r in radii:
        out[f"n_def_within_{int(r)}ft"] = np.nan
    if len(def_xy) == 0 or not np.all(np.isfinite(focal_xy)):
        return out
    d = np.sort(np.hypot(def_xy[:, 0] - focal_xy[0], def_xy[:, 1] - focal_xy[1]))
    out["nearest_def_dist"] = float(d[0])
    out["second_def_dist"] = float(d[1]) if len(d) > 1 else np.nan
    out["help_gap"] = float(d[1] - d[0]) if len(d) > 1 else np.nan
    out["mean_def_dist"] = float(d.mean())
    for r in radii:
        out[f"n_def_within_{int(r)}ft"] = int((d <= r).sum())
    return out


def paint_congestion(def_xy: np.ndarray) -> dict:
    """How many defenders are protecting the rim, and how close."""
    if len(def_xy) == 0:
        return {"n_def_in_paint": np.nan, "mean_def_dist_to_rim": np.nan,
                "nearest_def_to_rim": np.nan}
    dr = np.hypot(def_xy[:, 0] - OFF_HOOP[0], def_xy[:, 1] - OFF_HOOP[1])
    return {
        "n_def_in_paint": int(court.in_paint(def_xy[:, 0], def_xy[:, 1]).sum()),
        "mean_def_dist_to_rim": float(dr.mean()),
        "nearest_def_to_rim": float(dr.min()),
    }


# --------------------------------------------------------------------------
# Court control — a time-to-arrive pitch-control model
# --------------------------------------------------------------------------
def time_to_arrive(px: np.ndarray, py: np.ndarray, vx: np.ndarray, vy: np.ndarray,
                   gx: np.ndarray, gy: np.ndarray,
                   vmax: float = 21.0, reaction: float = 0.20) -> np.ndarray:
    """Time for each player to reach each grid cell, under a simple motion model.

    A player continues at their current velocity for ``reaction`` seconds, then
    travels the remaining straight-line distance at ``vmax`` ft/s (≈ 6.4 m/s,
    the 99.9th percentile of smoothed player speed in this dataset). This is the
    standard "time-to-intercept" simplification used in football pitch-control
    work; it ignores turning cost and acceleration limits, which is why the
    output is treated as a *relative* accessibility surface, not a probability
    of physically winning a ball.

    Returns an (n_players, n_cells) array of arrival times in seconds.
    """
    vx = np.nan_to_num(vx); vy = np.nan_to_num(vy)
    ax = px + vx * reaction
    ay = py + vy * reaction
    d = np.hypot(gx[None, :] - ax[:, None], gy[None, :] - ay[:, None])
    return reaction + d / vmax


def court_control(off_xy, off_v, def_xy, def_v, grid_x, grid_y, beta: float = 1.2):
    """Soft-max control surface in [0, 1]: 1 = offence controls the cell.

    ``control = sigmoid(beta * (t_def_min - t_off_min))`` — a cell is controlled
    by whichever team can get there sooner, softened so that near-ties are ~0.5.
    ``beta`` (1/seconds) sets how sharply a time advantage converts to control;
    1.2 makes a half-second edge ≈ 0.65 control.

    This is an analytical model built for this project, not a SkillCorner metric.
    """
    gx, gy = grid_x.ravel(), grid_y.ravel()
    t_off = time_to_arrive(off_xy[:, 0], off_xy[:, 1], off_v[:, 0], off_v[:, 1], gx, gy)
    t_def = time_to_arrive(def_xy[:, 0], def_xy[:, 1], def_v[:, 0], def_v[:, 1], gx, gy)
    adv = t_def.min(axis=0) - t_off.min(axis=0)
    return (1.0 / (1.0 + np.exp(-beta * adv))).reshape(grid_x.shape)


def court_value_surface(grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    """Expected points per shot attempt by location — the "value" of court space.

    Fitted from the observed shot outcomes in this dataset (see
    :func:`src.shots.model.fit_location_value`), but a smooth analytic fallback
    is used here so the surface is defined everywhere, including where nobody
    shoots. Value = points × P(make | distance), with P(make) a logistic decay
    calibrated to this dataset's rim / mid-range / three splits.
    """
    r = np.hypot(grid_x - OFF_HOOP[0], grid_y - OFF_HOOP[1])
    p = 1.0 / (1.0 + np.exp((r - 13.5) / 7.5)) * 0.55 + 0.10
    pts = np.where(court.is_three(grid_x, grid_y), 3.0, 2.0)
    val = p * pts
    val[grid_x > 0] = 0.0            # backcourt has no shooting value
    return val


def make_grid(nx: int = 60, ny: int = 34):
    """Half-court grid (attacking half only) in the attacking frame."""
    xs = np.linspace(-C.COURT_X_MAX, 0.0, nx)
    ys = np.linspace(-C.COURT_Y_MAX, C.COURT_Y_MAX, ny)
    return np.meshgrid(xs, ys)
