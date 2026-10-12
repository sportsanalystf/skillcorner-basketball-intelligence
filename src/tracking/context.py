"""Attach possession / chance context to 25 fps tracking frames.

The tracking file has no notion of who is on offense. Every spatial metric in
this project (spacing, court control, help-defence distance) needs that, so we
interval-join the `possessions` and `chances` tables onto the frame index:

    possession.startFrame <= frame_idx <= possession.endFrame

Both tables are non-overlapping within a game, so a sorted `searchsorted` join
is exact and runs in O(n log m). Frames that fall in dead time between
possessions get NULL context and are excluded from possession-level analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _interval_labels(frame_idx: np.ndarray, starts: np.ndarray, ends: np.ndarray,
                     payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Map each frame to the interval containing it (or NaN/None)."""
    order = np.argsort(starts, kind="stable")
    s, e = starts[order], ends[order]
    pos = np.searchsorted(s, frame_idx, side="right") - 1
    valid = (pos >= 0) & (pos < len(s))
    pos_c = np.clip(pos, 0, max(len(s) - 1, 0))
    inside = valid & (frame_idx <= e[pos_c])
    out = {}
    for k, v in payload.items():
        vv = v[order]
        if vv.dtype == object or str(vv.dtype).startswith("<U"):
            col = np.full(len(frame_idx), None, dtype=object)
        else:
            col = np.full(len(frame_idx), np.nan, dtype="float64")
        col[inside] = vv[pos_c[inside]]
        out[k] = col
    return out


def attach_context(tracking: pd.DataFrame, possessions: pd.DataFrame,
                   chances: pd.DataFrame) -> pd.DataFrame:
    """Add possession_id / chance_id / off_team_id / left_hoop / transition columns."""
    fi = tracking["frame_idx"].to_numpy()

    p = possessions.dropna(subset=["start_frame", "end_frame"])
    pl = _interval_labels(
        fi, p["start_frame"].to_numpy(), p["end_frame"].to_numpy(),
        {"possession_id": p["id"].to_numpy(object),
         "off_team_id": p["off_team_id"].to_numpy("float64"),
         "left_hoop": p["left_hoop"].astype("float64").to_numpy()})

    c = chances.dropna(subset=["start_frame", "end_frame"])
    c = c[c["end_frame"] >= c["start_frame"]]
    cl = _interval_labels(
        fi, c["start_frame"].to_numpy(), c["end_frame"].to_numpy(),
        {"chance_id": c["id"].to_numpy(object),
         "chance_transition": c["transition"].astype("float64").to_numpy(),
         "chance_quality_index": c["quality_index"].astype("float64").to_numpy()})

    out = tracking.copy()
    out["possession_id"] = pl["possession_id"]
    out["off_team_id"] = pl["off_team_id"]
    out["left_hoop"] = pl["left_hoop"]
    out["chance_id"] = cl["chance_id"]
    out["chance_transition"] = cl["chance_transition"]
    out["chance_quality_index"] = cl["chance_quality_index"]
    out["is_offense"] = np.where(np.isnan(out["off_team_id"]), np.nan,
                                 (out["team_id"] == out["off_team_id"]).astype(float))
    return out


def add_attacking_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Rotate tracking x/y into the offence's attacking frame.

    Uses `left_hoop` from the possession in progress; NaN where no possession is
    live. The transform is the verified 180° rotation (see src.preprocessing.court).
    """
    lh = df["left_hoop"].to_numpy()
    sign = np.where(np.isnan(lh), np.nan, np.where(lh > 0.5, 1.0, -1.0))
    out = df.copy()
    for src, dst in (("x_s", "x_att"), ("y_s", "y_att")):
        out[dst] = (df[src].to_numpy() * sign).astype("float32")
    out["vx_att"] = (df["vx"].to_numpy() * sign).astype("float32")
    out["vy_att"] = (df["vy"].to_numpy() * sign).astype("float32")
    return out
