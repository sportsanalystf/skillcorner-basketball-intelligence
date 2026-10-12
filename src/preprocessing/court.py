"""Canonical court coordinate system, transforms and zoning.

Two coordinate frames exist in this dataset and they are NOT the same:

**Broadcast frame** — used by ``tracking_data``. x runs left→right from the
camera, origin at court centre. Which hoop a team attacks depends on the period
and on the team.

**Attacking frame** — used by every *event* table (`location`, `endLoc`,
`passerLoc`, ...). The offensive hoop is always at ``x = -HOOP_X``, the
defensive hoop at ``x = +HOOP_X``; the attacking team's left is ``+y``.

Empirically verified on all 1,459 shots in the 10 published games by joining
each shot to the shooter's tracking position at its ``startFrame``:

* ``possessions.leftHoop == True``  → attacking frame == broadcast frame
  (median |Δx| = 0.00 ft, |Δy| = 0.00 ft)
* ``possessions.leftHoop == False`` → attacking frame == broadcast frame
  rotated 180° about court centre, i.e. ``(x, y) → (-x, -y)``
  (median |Δx| = 0.00 ft, |Δy| = 0.00 ft)

Every transform in this module is explicit and reversible. Nothing in the
pipeline mutates coordinates in place — normalised values are always written to
new ``*_att`` columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C

OFF_HOOP = (-C.HOOP_X, 0.0)   # offensive hoop, attacking frame
DEF_HOOP = (+C.HOOP_X, 0.0)   # defensive hoop, attacking frame


# --------------------------------------------------------------------------
# Frame transforms
# --------------------------------------------------------------------------
def broadcast_to_attacking(x, y, left_hoop):
    """Rotate broadcast coordinates into the attacking frame.

    ``left_hoop`` is ``possessions.leftHoop`` / ``chances`` inherited: True when
    the offensive team attacks the left (negative-x) broadcast hoop.
    """
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    lh = np.asarray(left_hoop)
    sign = np.where(lh, 1.0, -1.0)
    return x * sign, y * sign


def attacking_to_broadcast(x, y, left_hoop):
    """Inverse of :func:`broadcast_to_attacking` (the rotation is an involution)."""
    return broadcast_to_attacking(x, y, left_hoop)


def add_attacking_frame(df: pd.DataFrame, xcol: str, ycol: str,
                        left_hoop_col: str = "left_hoop",
                        suffix: str = "_att") -> pd.DataFrame:
    """Append ``<xcol><suffix>`` / ``<ycol><suffix>`` columns. Non-destructive."""
    xa, ya = broadcast_to_attacking(df[xcol].to_numpy(), df[ycol].to_numpy(),
                                    df[left_hoop_col].to_numpy())
    out = df.copy()
    out[xcol + suffix] = xa.astype("float32")
    out[ycol + suffix] = ya.astype("float32")
    return out


# --------------------------------------------------------------------------
# Geometry helpers (attacking frame)
# --------------------------------------------------------------------------
def dist_to_off_hoop(x, y):
    return np.hypot(np.asarray(x) - OFF_HOOP[0], np.asarray(y) - OFF_HOOP[1])


def dist_to_def_hoop(x, y):
    return np.hypot(np.asarray(x) - DEF_HOOP[0], np.asarray(y) - DEF_HOOP[1])


def angle_from_baseline(x, y):
    """Angle in degrees of the shot line relative to the baseline (0 = corner,
    90 = straight on). Attacking frame."""
    dx = np.asarray(x) - OFF_HOOP[0]
    dy = np.asarray(y) - OFF_HOOP[1]
    return np.degrees(np.arctan2(np.abs(dx), np.abs(dy)))


def is_three(x, y) -> np.ndarray:
    """FIBA three-point line in the attacking frame (6.75 m arc, 6.60 m corners)."""
    x = np.asarray(x, dtype="float64"); y = np.asarray(y, dtype="float64")
    r = dist_to_off_hoop(x, y)
    corner = np.abs(y) >= C.THREE_CORNER_Y
    beyond_break = x <= -C.CORNER_LINE_X  # closer to baseline than the arc break
    return np.where(corner & beyond_break, True, r >= C.THREE_ARC_R)


def in_paint(x, y) -> np.ndarray:
    x = np.asarray(x); y = np.asarray(y)
    return (np.abs(y) <= C.PAINT_HALF_WIDTH) & (x <= -(C.COURT_X_MAX - C.PAINT_LENGTH))


def in_restricted_area(x, y) -> np.ndarray:
    return dist_to_off_hoop(x, y) <= C.RESTRICTED_AREA_R


def in_corner(x, y) -> np.ndarray:
    x = np.asarray(x); y = np.asarray(y)
    return (np.abs(y) >= C.THREE_CORNER_Y) & (x <= -C.CORNER_LINE_X)


def in_frontcourt(x) -> np.ndarray:
    """Attacking frame: the offensive half is x < 0."""
    return np.asarray(x) < 0.0


# --------------------------------------------------------------------------
# Shot zoning — our own vocabulary, mirroring the *aggregate* zone names
# (docs/aggregates_columns.md) so game-level and season-level views line up.
# --------------------------------------------------------------------------
ZONES = ["restricted_area", "paint_non_ra", "mid_range",
         "corner_3_left", "corner_3_right", "wing_3_left", "wing_3_right",
         "top_3", "backcourt", "far"]


def shot_zone(x, y) -> np.ndarray:
    """Assign each (x, y) in the attacking frame to an aggregate-compatible zone.

    Mirrors the zone vocabulary of ``acb_shotsaggregates`` (restricted_area,
    paint_non_ra, mid_range, corner_3 left/right, wing_3 left/right, top_3),
    plus ``backcourt`` and ``far``. Left/right follow the attacking team's own
    left (+y) and right (-y), matching the region strings on the event tables.
    """
    x = np.asarray(x, dtype="float64"); y = np.asarray(y, dtype="float64")
    r = dist_to_off_hoop(x, y)
    three = is_three(x, y)
    out = np.full(x.shape, "far", dtype=object)

    backcourt = x > 0
    far = (~backcourt) & three & (r > 30.0)

    # two-point zones
    ra = (~three) & (r <= C.RESTRICTED_AREA_R)
    paint = (~three) & (~ra) & in_paint(x, y)
    mid = (~three) & (~ra) & (~paint)

    # three-point zones (left = +y from the attacking team's perspective)
    corner = three & in_corner(x, y)
    ang = np.degrees(np.arctan2(np.abs(y), np.abs(x - OFF_HOOP[0]) + 1e-9))
    wing = three & (~corner) & (ang >= 30.0)
    top = three & (~corner) & (~wing)

    out[mid] = "mid_range"
    out[paint] = "paint_non_ra"
    out[ra] = "restricted_area"
    out[corner & (y > 0)] = "corner_3_left"
    out[corner & (y <= 0)] = "corner_3_right"
    out[wing & (y > 0)] = "wing_3_left"
    out[wing & (y <= 0)] = "wing_3_right"
    out[top] = "top_3"
    out[far] = "far"
    out[backcourt] = "backcourt"
    return out


ZONE_FAMILY = {
    "restricted_area": "rim", "paint_non_ra": "short_midrange_paint",
    "mid_range": "long_midrange", "corner_3_left": "corner_3",
    "corner_3_right": "corner_3", "wing_3_left": "above_break_3",
    "wing_3_right": "above_break_3", "top_3": "above_break_3",
    "backcourt": "other", "far": "other",
}


# --------------------------------------------------------------------------
# Court outline for plotting (attacking frame; full court)
# --------------------------------------------------------------------------
def court_shapes(half: bool = False, line_color: str = "rgba(160,170,185,0.55)",
                 width: float = 1.4) -> list[dict]:
    """Plotly shape dicts drawing a FIBA court in the attacking frame."""
    X, Y = C.COURT_X_MAX, C.COURT_Y_MAX
    hx = -C.HOOP_X
    s = []

    def line(x0, y0, x1, y1, w=width):
        return dict(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                    line=dict(color=line_color, width=w), layer="above")

    def rect(x0, y0, x1, y1, w=width):
        return dict(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                    line=dict(color=line_color, width=w), layer="above")

    def circle(cx, cy, r, w=width):
        return dict(type="circle", x0=cx - r, y0=cy - r, x1=cx + r, y1=cy + r,
                    line=dict(color=line_color, width=w), layer="above")

    def path(pts, w=width):
        d = "M " + " L ".join(f"{px},{py}" for px, py in pts)
        return dict(type="path", path=d, line=dict(color=line_color, width=w), layer="above")

    s.append(rect(-X, -Y, X, Y, width * 1.4))          # boundary
    s.append(line(0, -Y, 0, Y))                        # half-way line
    s.append(circle(0, 0, C.FT_CIRCLE_R))              # centre circle

    halves = [-1] if half else [-1, 1]
    for sgn in halves:
        bx = sgn * X                        # baseline x
        hoop_x = sgn * C.HOOP_X
        # key
        s.append(rect(bx, -C.PAINT_HALF_WIDTH,
                      bx - sgn * C.PAINT_LENGTH, C.PAINT_HALF_WIDTH))
        # free-throw circle
        s.append(circle(bx - sgn * C.PAINT_LENGTH, 0, C.FT_CIRCLE_R))
        # rim + backboard
        s.append(circle(hoop_x, 0, C.RIM_RADIUS, width * 1.2))
        bbx = bx - sgn * C.BACKBOARD_INSET
        s.append(line(bbx, -3.0, bbx, 3.0, width * 2.0))
        # restricted area arc
        th = np.linspace(-np.pi / 2, np.pi / 2, 40) * -sgn
        s.append(path([(hoop_x - sgn * C.RESTRICTED_AREA_R * np.cos(t),
                        C.RESTRICTED_AREA_R * np.sin(t)) for t in th]))
        # three-point line: corners + arc
        cx = C.CORNER_LINE_X
        s.append(line(bx, C.THREE_CORNER_Y, sgn * cx, C.THREE_CORNER_Y))
        s.append(line(bx, -C.THREE_CORNER_Y, sgn * cx, -C.THREE_CORNER_Y))
        a = np.arcsin(C.THREE_CORNER_Y / C.THREE_ARC_R)
        th = np.linspace(-a, a, 90)
        s.append(path([(hoop_x - sgn * C.THREE_ARC_R * np.cos(t),
                        C.THREE_ARC_R * np.sin(t)) for t in th]))
    return s
