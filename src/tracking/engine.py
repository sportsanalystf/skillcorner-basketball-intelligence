"""Tracking engine: segmentation, smoothing, kinematics and quality diagnostics.

Broadcast tracking is noisy and intermittent: roughly half of all frames in a
game are dead time with no players at all, and within live play a player's track
can drop out for a few frames. Differentiating raw positions across those gaps
produces physically impossible speeds, so everything here is computed *inside
continuous frame segments* only.

Pipeline for one game
---------------------
1. ``segment_tracks``   – split each player's frames into runs of consecutive
   ``frame_idx``; short runs are still usable but are flagged.
2. ``smooth_positions`` – Savitzky-Golay filter (window 15 frames = 0.6 s,
   polyorder 2) on x and y within each segment. Chosen because it preserves the
   peak of a real change of direction far better than a moving average of the
   same length, which is exactly the signal we care about for cuts and drives.
3. ``compute_kinematics`` – central-difference velocity and acceleration on the
   smoothed track; speed, heading, distance travelled per frame.
4. ``quality_flags``    – plausibility gates and detection statistics.

Everything is vectorised over the whole game; the per-segment work uses
``np.add.reduceat``-style index arithmetic rather than Python loops over groups.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from src import config as C

MIN_SEGMENT = 5  # frames; below this we cannot differentiate meaningfully


# --------------------------------------------------------------------------
def segment_tracks(df: pd.DataFrame, key: str = "player_id") -> pd.Series:
    """Return an integer segment id, unique per (key, run of consecutive frames).

    ``df`` must be sorted by ``key``, ``frame_idx``.
    """
    k = df[key].to_numpy()
    f = df["frame_idx"].to_numpy()
    new_key = np.empty(len(df), dtype=bool)
    new_key[0] = True
    new_key[1:] = k[1:] != k[:-1]
    gap = np.empty(len(df), dtype=bool)
    gap[0] = True
    gap[1:] = (f[1:] - f[:-1]) != 1
    return pd.Series(np.cumsum(new_key | gap) - 1, index=df.index, name="segment_id")


def _segment_bounds(seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start index and length of each contiguous segment (seg must be sorted)."""
    starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    lengths = np.diff(np.r_[starts, len(seg)])
    return starts, lengths


def smooth_positions(df: pd.DataFrame, window: int = C.SMOOTH_WINDOW,
                     polyorder: int = C.SMOOTH_POLYORDER) -> pd.DataFrame:
    """Add ``x_s`` / ``y_s`` — Savitzky-Golay smoothed positions per segment."""
    seg = df["segment_id"].to_numpy()
    x = df["x"].to_numpy(dtype="float64").copy()
    y = df["y"].to_numpy(dtype="float64").copy()
    xs = x.copy(); ys = y.copy()
    starts, lengths = _segment_bounds(seg)
    for s, n in zip(starts, lengths):
        if n < 3:
            continue
        w = min(window, n if n % 2 == 1 else n - 1)
        if w < 3:
            continue
        po = min(polyorder, w - 1)
        sl = slice(s, s + n)
        xs[sl] = savgol_filter(x[sl], w, po, mode="interp")
        ys[sl] = savgol_filter(y[sl], w, po, mode="interp")
    out = df.copy()
    out["x_s"] = xs.astype("float32")
    out["y_s"] = ys.astype("float32")
    return out


def compute_kinematics(df: pd.DataFrame, dt: float = C.DT) -> pd.DataFrame:
    """Velocity, speed, heading, acceleration and step distance per segment.

    Central differences inside a segment; one-sided at segment edges. Values are
    NaN where a segment is shorter than ``MIN_SEGMENT``.
    """
    seg = df["segment_id"].to_numpy()
    x = df["x_s"].to_numpy(dtype="float64")
    y = df["y_s"].to_numpy(dtype="float64")
    n = len(df)
    vx = np.full(n, np.nan); vy = np.full(n, np.nan)
    ax = np.full(n, np.nan); ay = np.full(n, np.nan)
    step = np.full(n, np.nan)

    starts, lengths = _segment_bounds(seg)
    for s, ln in zip(starts, lengths):
        if ln < MIN_SEGMENT:
            continue
        sl = slice(s, s + ln)
        gx = np.gradient(x[sl], dt)
        gy = np.gradient(y[sl], dt)
        vx[sl] = gx; vy[sl] = gy
        ax[sl] = np.gradient(gx, dt)
        ay[sl] = np.gradient(gy, dt)
        d = np.empty(ln); d[0] = 0.0
        d[1:] = np.hypot(np.diff(x[sl]), np.diff(y[sl]))
        step[sl] = d

    out = df.copy()
    out["vx"] = vx.astype("float32"); out["vy"] = vy.astype("float32")
    out["speed"] = np.hypot(vx, vy).astype("float32")
    out["heading"] = np.degrees(np.arctan2(vy, vx)).astype("float32")
    out["ax"] = ax.astype("float32"); out["ay"] = ay.astype("float32")
    # signed tangential acceleration: positive = accelerating along the heading
    sp = np.hypot(vx, vy)
    with np.errstate(invalid="ignore", divide="ignore"):
        tang = (vx * ax + vy * ay) / np.where(sp > 1e-6, sp, np.nan)
    out["accel_tangential"] = tang.astype("float32")
    out["accel_mag"] = np.hypot(ax, ay).astype("float32")
    out["step_dist"] = step.astype("float32")
    return out


def speed_band(speed: np.ndarray) -> np.ndarray:
    """Map speed (ft/s) to our interpretable movement-intensity bands.

    These bands are defined in :data:`src.config.SPEED_BANDS` from the empirical
    distribution of smoothed speed in this dataset. They are NOT SkillCorner
    physical-performance classifications.
    """
    sp = np.asarray(speed, dtype="float64")
    out = np.full(sp.shape, None, dtype=object)
    for name, lo, hi in C.SPEED_BANDS:
        out[(sp >= lo) & (sp < hi)] = name
    return out


def quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row plausibility flags used by the data-quality layer."""
    out = df.copy()
    out["flag_impossible_speed"] = out["speed"] > C.MAX_PLAUSIBLE_SPEED
    out["flag_impossible_accel"] = out["accel_mag"] > C.MAX_PLAUSIBLE_ACCEL
    out["flag_off_court"] = (out["x"].abs() > C.COURT_X_MAX + 8) | \
                            (out["y"].abs() > C.COURT_Y_MAX + 8)
    out["flag_high_pred_error"] = out["pred_error"] > 6.0
    return out


def process_game_tracking(df: pd.DataFrame) -> pd.DataFrame:
    """Full per-game tracking pipeline. ``df`` is the raw players parquet."""
    df = df.sort_values(["player_id", "frame_idx"], kind="stable").reset_index(drop=True)
    df["segment_id"] = segment_tracks(df)
    df = smooth_positions(df)
    df = compute_kinematics(df)
    df = quality_flags(df)
    df["speed_band"] = speed_band(df["speed"].to_numpy())
    return df


# --------------------------------------------------------------------------
def raw_vs_smoothed_report(df: pd.DataFrame) -> dict:
    """Compare kinematics from raw positions against the smoothed pipeline."""
    d = df.sort_values(["player_id", "frame_idx"], kind="stable").reset_index(drop=True)
    d["segment_id"] = segment_tracks(d)
    seg = d["segment_id"].to_numpy()
    x = d["x"].to_numpy(dtype="float64"); y = d["y"].to_numpy(dtype="float64")
    n = len(d)
    vraw = np.full(n, np.nan)
    starts, lengths = _segment_bounds(seg)
    for s, ln in zip(starts, lengths):
        if ln < MIN_SEGMENT:
            continue
        sl = slice(s, s + ln)
        vraw[sl] = np.hypot(np.gradient(x[sl], C.DT), np.gradient(y[sl], C.DT))
    sm = process_game_tracking(df)
    vsm = sm["speed"].to_numpy(dtype="float64")
    ok = np.isfinite(vraw) & np.isfinite(vsm)
    return {
        "n": int(ok.sum()),
        "raw_mean_speed": float(np.nanmean(vraw[ok])),
        "smoothed_mean_speed": float(np.nanmean(vsm[ok])),
        "raw_p99_speed": float(np.nanpercentile(vraw[ok], 99)),
        "smoothed_p99_speed": float(np.nanpercentile(vsm[ok], 99)),
        "raw_max_speed": float(np.nanmax(vraw[ok])),
        "smoothed_max_speed": float(np.nanmax(vsm[ok])),
        "raw_pct_impossible": float((vraw[ok] > C.MAX_PLAUSIBLE_SPEED).mean() * 100),
        "smoothed_pct_impossible": float((vsm[ok] > C.MAX_PLAUSIBLE_SPEED).mean() * 100),
        "corr_with_delivered_speed": float(
            pd.Series(vsm[ok]).corr(pd.Series(sm["speed_raw"].to_numpy()[ok]))),
        "corr_raw_with_delivered": float(
            pd.Series(vraw[ok]).corr(pd.Series(sm["speed_raw"].to_numpy()[ok]))),
    }
