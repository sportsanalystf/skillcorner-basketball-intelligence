"""Transition and off-ball movement.

`chances.transition` is a SkillCorner flag; we use it as delivered rather than
re-deriving it. What tracking adds is *how fast* each team actually moves in
those chances, measured directly from positions.

Off-ball movement is measured from tracking as distance covered by the four
non-ball-handling offensive players in the seconds before a shot, together with
the separation the eventual shooter gained from their nearest defender.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.preprocessing import court
from src.spatial.features import load_frames


def transition_summary(chances: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    ch = chances.copy()
    ch["duration_s"] = ch.start_game_clock - ch.end_game_clock
    ch["frontcourt_delay_s"] = (ch.frontcourt_frame - ch.start_frame) / C.FPS
    s = shots.groupby("chance_id").agg(
        first_shot_frame=("start_frame", "min"),
        shot_dist=("distance", "first"), shot_three=("three", "first"),
        shot_made=("outcome", "first"), shot_quality=("shot_quality", "first"),
        shot_contested=("contested", "first")).reset_index()
    ch = ch.merge(s, left_on="id", right_on="chance_id", how="left")
    ch["time_to_shot_s"] = (ch.first_shot_frame - ch.start_frame) / C.FPS
    return ch


def team_transition_profile(ch: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tid, tr), g in ch.groupby(["off_team_id", "transition"]):
        rows.append({"team_id": tid, "transition": bool(tr), "chances": len(g),
                     "pts_per_chance": g.pts_scored.mean(),
                     "duration_s": g.duration_s.mean(),
                     "frontcourt_delay_s": g.frontcourt_delay_s.mean(),
                     "time_to_shot_s": g.time_to_shot_s.mean(),
                     "three_rate": g.shot_three.mean(),
                     "shot_quality": g.shot_quality.mean(),
                     "contested_rate": g.shot_contested.mean(),
                     "to_rate": g.outcome.eq("TO").mean()})
    return pd.DataFrame(rows)


def offball_movement_before_shots(shots: pd.DataFrame, window_s: float = 3.0,
                                  step: int = 5) -> pd.DataFrame:
    """Distance covered by off-ball offensive players in the window before a shot,
    plus the shooter's separation change over the same window."""
    wf = int(window_s * C.FPS)
    out = []
    for gid, sub in shots.groupby("game_id"):
        sub = sub.dropna(subset=["start_frame", "shooter_id", "off_team_id"])
        if sub.empty:
            continue
        frames = np.concatenate([np.arange(int(f) - wf, int(f) + 1, step)
                                 for f in sub.start_frame])
        tr = load_frames(int(gid), frames,
                         cols=["frame_idx", "player_id", "team_id", "x_att", "y_att", "speed"])
        if tr.empty:
            continue
        idx = {(f, p): (x, y, sp) for f, p, x, y, sp in
               zip(tr.frame_idx, tr.player_id, tr.x_att, tr.y_att, tr.speed)}
        by_frame = tr.groupby("frame_idx")
        fmap = {f: d for f, d in by_frame}
        for r in sub.itertuples(index=False):
            f1 = int(r.start_frame); f0 = f1 - wf
            fr = [f for f in range(f0, f1 + 1, step) if f in fmap]
            if len(fr) < 3:
                continue
            offball_dist = []; shooter_dist = 0.0
            prev = {}
            sep0 = sep1 = np.nan
            for f in fr:
                d = fmap[f]
                off = d[d.team_id == r.off_team_id]
                dfn = d[d.team_id != r.off_team_id]
                for p, x, y in zip(off.player_id, off.x_att, off.y_att):
                    if p in prev:
                        step_d = float(np.hypot(x - prev[p][0], y - prev[p][1]))
                        if p == r.shooter_id:
                            shooter_dist += step_d
                        else:
                            offball_dist.append(step_d)
                    prev[p] = (x, y)
                sh = off[off.player_id == r.shooter_id]
                if len(sh) and len(dfn):
                    sep = float(np.hypot(dfn.x_att - sh.x_att.iloc[0],
                                         dfn.y_att - sh.y_att.iloc[0]).min())
                    if f == fr[0]:
                        sep0 = sep
                    sep1 = sep
            out.append({"id": r.id,
                        "offball_distance_ft": float(np.sum(offball_dist)),
                        "shooter_distance_ft": shooter_dist,
                        "separation_start": sep0, "separation_at_shot": sep1,
                        "separation_gained": (sep1 - sep0) if np.isfinite(sep0) else np.nan,
                        "window_s": window_s})
    return pd.DataFrame(out)
