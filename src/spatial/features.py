"""Batch extraction of spatial features at event moments.

Given a list of (game_id, frame, focal_player_id) queries, pull the tracking
snapshot at each frame and compute spacing + defensive geometry. Done per game
so each enriched tracking file is read once, with column pruning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.spatial import geometry as G

TRACK_COLS = ["frame_idx", "player_id", "team_id", "x_att", "y_att",
              "vx_att", "vy_att", "speed", "off_team_id", "is_offense",
              "is_detected", "pred_error"]


def load_frames(game_id: int, frames: np.ndarray, cols=TRACK_COLS) -> pd.DataFrame:
    """Read only the requested frames of one game's enriched tracking."""
    path = C.PROCESSED_DIR / "tracking_enriched" / f"{game_id}.parquet"
    fr = np.unique(np.asarray(frames)[~pd.isna(frames)]).astype("int64")
    if len(fr) == 0:
        return pd.DataFrame(columns=cols)
    df = pd.read_parquet(path, columns=cols,
                         filters=[("frame_idx", ">=", int(fr.min())),
                                  ("frame_idx", "<=", int(fr.max()))])
    return df[df.frame_idx.isin(fr)]


def snapshot_features(game_id: int, queries: pd.DataFrame) -> pd.DataFrame:
    """Spacing + defensive features for each query row.

    ``queries`` needs: ``key`` (opaque id), ``frame``, ``off_team_id`` and
    optionally ``focal_player_id`` (the ball handler / shooter).
    """
    if queries.empty:
        return pd.DataFrame()
    tr = load_frames(game_id, queries["frame"].to_numpy())
    if tr.empty:
        return pd.DataFrame({"key": queries["key"]})

    by_frame = {f: d for f, d in tr.groupby("frame_idx", sort=False)}
    rows = []
    for q in queries.itertuples(index=False):
        d = by_frame.get(int(q.frame))
        rec = {"key": q.key}
        if d is None or d.empty:
            rows.append(rec); continue
        off = d[d.team_id == q.off_team_id]
        dfn = d[d.team_id != q.off_team_id]
        oxy = off[["x_att", "y_att"]].to_numpy(dtype="float64")
        dxy = dfn[["x_att", "y_att"]].to_numpy(dtype="float64")
        oxy = oxy[np.isfinite(oxy).all(axis=1)]
        dxy = dxy[np.isfinite(dxy).all(axis=1)]

        rec.update(G.spacing_metrics(oxy))
        rec["n_def_tracked"] = len(dxy)
        rec.update(G.paint_congestion(dxy))

        focal = getattr(q, "focal_player_id", np.nan)
        if pd.notna(focal):
            f = d[d.player_id == focal]
            if len(f):
                fxy = f[["x_att", "y_att"]].to_numpy(dtype="float64")[0]
                rec.update(G.defensive_pressure(fxy, dxy))
                rec["focal_x"] = float(fxy[0]); rec["focal_y"] = float(fxy[1])
                rec["focal_speed"] = float(f["speed"].to_numpy()[0])
                if len(oxy) > 1:
                    others = off[off.player_id != focal][["x_att", "y_att"]].to_numpy("float64")
                    others = others[np.isfinite(others).all(axis=1)]
                    if len(others):
                        rec["mean_teammate_dist"] = float(
                            np.hypot(*(others - fxy).T).mean())
                        rec["nearest_teammate_dist"] = float(
                            np.hypot(*(others - fxy).T).min())
        rows.append(rec)
    return pd.DataFrame(rows)


def build_event_spatial(events: pd.DataFrame, frame_col: str, focal_col: str | None,
                        key_col: str = "id") -> pd.DataFrame:
    """Run :func:`snapshot_features` across all games for one event table."""
    out = []
    for gid, sub in events.groupby("game_id"):
        q = pd.DataFrame({
            "key": sub[key_col].to_numpy(),
            "frame": sub[frame_col].to_numpy(),
            "off_team_id": sub["off_team_id"].to_numpy(),
        })
        if focal_col and focal_col in sub.columns:
            q["focal_player_id"] = sub[focal_col].to_numpy()
        q = q.dropna(subset=["frame", "off_team_id"])
        res = snapshot_features(int(gid), q)
        if not res.empty:
            out.append(res)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True).rename(columns={"key": key_col})


# --------------------------------------------------------------------------
def drive_collapse_profile(drives: pd.DataFrame, step: int = 5) -> pd.DataFrame:
    """Sample defensive geometry every ``step`` frames through each drive.

    Answers "how fast does the defence collapse on a drive?" by tracking the
    help defender's distance from the driver from drive start to drive end.
    """
    recs = []
    for gid, sub in drives.groupby("game_id"):
        sub = sub.dropna(subset=["start_frame", "end_frame", "ballhandler_id", "off_team_id"])
        if sub.empty:
            continue
        want = []
        for r in sub.itertuples(index=False):
            fs, fe = int(r.start_frame), int(r.end_frame)
            for f in range(fs, min(fe, fs + 125) + 1, step):
                want.append((r.id, f, f - fs, r.ballhandler_id, r.off_team_id))
        if not want:
            continue
        q = pd.DataFrame(want, columns=["drive_id", "frame", "frames_since_start",
                                        "focal_player_id", "off_team_id"])
        q["key"] = q.drive_id + "@" + q.frames_since_start.astype(str)
        res = snapshot_features(int(gid), q)
        if res.empty:
            continue
        res = res.merge(q[["key", "drive_id", "frames_since_start"]], on="key")
        res["t_since_start"] = res.frames_since_start / C.FPS
        recs.append(res)
    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()
