"""Defensive intelligence: closeouts, matchup workload, help arrival.

What the data supports
----------------------
* **Closeouts** are a first-class event with three distance checkpoints
  (`touchDistance` → `startDistance` → `endDistance`) and the ball handler's
  response (`bhrAction`). Closeout speed and distance are therefore directly
  measurable.
* **Matchups** give assignment intervals, so per-defender workload (who they
  guarded, for how long) and the offence produced against them is measurable.
* **Help arrival** is measurable from tracking as the second-nearest defender's
  distance over the course of a drive.

What the data does NOT support, and which we therefore do not compute
---------------------------------------------------------------------
* Season-level defensive rate stats — the season aggregates are offence-only.
* Individual "defensive rating" or any single defensive score. Points allowed on
  a matchup conflate team scheme with individual defence, and with 10 games the
  per-defender samples are 20-60 chances. We report volume, distances and
  outcomes, and let the user weigh them.
* Contest quality independent of the shooter's own decision-making.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.preprocessing import court

FOUL_TOKENS = {"FOUL", "FOU_S", "FOU_N", "FOU_B", "FOU_O"}


def build_closeout_features(tables: dict[str, pd.DataFrame],
                            spatial: pd.DataFrame) -> pd.DataFrame:
    co = tables["closeouts"].copy()
    toks = co["bhr_outcomes"].fillna("").map(lambda v: set(v.split("|")) - {""})

    co["duration_s"] = ((co.end_frame - co.start_frame) / C.FPS).astype("float32")
    co["closeout_distance_ft"] = np.hypot(co.end_loc_x - co.start_loc_x,
                                          co.end_loc_y - co.start_loc_y)
    co["distance_closed_ft"] = co.start_distance - co.end_distance
    co["closeout_speed_fps"] = co.closeout_distance_ft / co.duration_s.replace(0, np.nan)
    co["close_rate_fps"] = co.distance_closed_ft / co.duration_s.replace(0, np.nan)
    co["arrived_tight"] = co.end_distance <= 4.0

    co["points_allowed"] = [3 if "FGM3" in t else (2 if "FGM" in t else 0) for t in toks]
    co["points_created_allowed"] = [3 if "AST3" in t else (2 if "AST2" in t else 0) for t in toks]
    co["total_points_allowed"] = co.points_allowed + co.points_created_allowed
    co["forced_turnover"] = ["TO" in t for t in toks]
    co["fouled"] = [bool(t & FOUL_TOKENS) for t in toks]
    co["allowed_shot"] = [bool(t & {"FGM", "FGM3", "FGX", "FGX3"}) for t in toks]
    # SkillCorner's own framing: a closeout "worked" if the offence got nothing
    co["stop"] = (co.total_points_allowed == 0) & (~co.fouled)

    co["dist_hoop"] = court.dist_to_off_hoop(co.location_x, co.location_y)
    co["zone"] = court.shot_zone(co.location_x, co.location_y)
    co = co.merge(spatial, on="id", how="left")
    return co


def closeout_defender_profile(co: pd.DataFrame, roster: pd.DataFrame,
                              min_n: int = 8) -> pd.DataFrame:
    g = co.groupby("ballhandler_def_id_canonical")
    p = pd.DataFrame({
        "closeouts": g.size(),
        "mean_start_distance": g.start_distance.mean(),
        "mean_end_distance": g.end_distance.mean(),
        "mean_distance_closed": g.distance_closed_ft.mean(),
        "mean_close_rate_fps": g.close_rate_fps.mean(),
        "mean_duration_s": g.duration_s.mean(),
        "arrived_tight_rate": g.arrived_tight.mean(),
        "points_allowed_per_closeout": g.total_points_allowed.mean(),
        "stop_rate": g.stop.mean(),
        "foul_rate": g.fouled.mean(),
        "forced_to_rate": g.forced_turnover.mean(),
        "drive_allowed_rate": g.bhr_action.apply(lambda s: (s == "drive").mean()),
        "shot_allowed_rate": g.bhr_action.apply(lambda s: (s == "shot").mean()),
    }).reset_index().rename(columns={"ballhandler_def_id_canonical": "player_id_canonical"})
    p = p[p.closeouts >= min_n]
    names = roster.drop_duplicates("player_id_canonical")[
        ["player_id_canonical", "player_name", "team_name"]]
    return p.merge(names, on="player_id_canonical", how="left").sort_values("closeouts", ascending=False)


def matchup_workload(tables: dict[str, pd.DataFrame], roster: pd.DataFrame) -> pd.DataFrame:
    """Per-defender: who they guarded, for how long, and what that offence produced."""
    mu = tables["matchups"].copy()
    mu["seconds"] = (mu.end_frame - mu.start_frame) / C.FPS
    mu = mu[(mu.seconds > 0) & (mu.seconds < 60)]

    # points scored by the guarded player while this assignment was live
    sh = tables["shots"][["game_id", "shooter_id_canonical", "start_frame", "outcome", "three"]]
    sh = sh.assign(pts=np.where(sh.outcome, np.where(sh.three, 3, 2), 0))

    rows = []
    for (gid, dpid), sub in mu.groupby(["game_id", "def_player_id_canonical"]):
        s = sh[sh.game_id == gid]
        pts = 0; att = 0
        for r in sub.itertuples(index=False):
            m = s[(s.shooter_id_canonical == r.off_player_id_canonical) &
                  (s.start_frame >= r.start_frame) & (s.start_frame <= r.end_frame)]
            pts += int(m.pts.sum()); att += len(m)
        rows.append({"game_id": gid, "player_id_canonical": dpid,
                     "assignments": len(sub), "seconds_guarding": sub.seconds.sum(),
                     "distinct_opponents": sub.off_player_id_canonical.nunique(),
                     "shots_allowed": att, "points_allowed": pts})
    w = pd.DataFrame(rows)
    agg = w.groupby("player_id_canonical").agg(
        games=("game_id", "nunique"), assignments=("assignments", "sum"),
        minutes_guarding=("seconds_guarding", lambda s: s.sum() / 60),
        distinct_opponents=("distinct_opponents", "sum"),
        shots_allowed=("shots_allowed", "sum"),
        points_allowed=("points_allowed", "sum")).reset_index()
    agg["points_allowed_per_36"] = agg.points_allowed / agg.minutes_guarding.replace(0, np.nan) * 36
    agg["shots_allowed_per_36"] = agg.shots_allowed / agg.minutes_guarding.replace(0, np.nan) * 36
    names = roster.drop_duplicates("player_id_canonical")[
        ["player_id_canonical", "player_name", "team_name"]]
    return agg.merge(names, on="player_id_canonical", how="left")


def shot_defender_profile(shots: pd.DataFrame, roster: pd.DataFrame,
                          min_n: int = 15) -> pd.DataFrame:
    """Nearest-defender-at-release profile.

    The closest defender is not necessarily the assigned defender, and being
    nearest on a made shot is not proof of poor defence. This is a *volume and
    context* profile, not a rating.
    """
    s = shots.dropna(subset=["closest_def_id_canonical"])
    g = s.groupby("closest_def_id_canonical")
    p = pd.DataFrame({
        "shots_contested": g.size(),
        "mean_closest_dist": g.closest_def_dist.mean(),
        "contested_rate": g.contested.mean(),
        "opp_fg_pct": g.outcome.mean(),
        "opp_xfg": g.xfg.mean() if "xfg" in s.columns else np.nan,
        "opp_pts_per_shot": g.points.mean() if "points" in s.columns else np.nan,
        "three_share": g.three.mean(),
        "mean_shot_distance": g.distance.mean(),
    }).reset_index().rename(columns={"closest_def_id_canonical": "player_id_canonical"})
    if "xfg" in s.columns:
        p["fg_vs_expected"] = p.opp_fg_pct - p.opp_xfg
    p = p[p.shots_contested >= min_n]
    names = roster.drop_duplicates("player_id_canonical")[
        ["player_id_canonical", "player_name", "team_name"]]
    return p.merge(names, on="player_id_canonical", how="left").sort_values("shots_contested", ascending=False)
