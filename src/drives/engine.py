"""Drive intelligence.

The `drives` table gives how a drive started (`category`), which way it went
(`direction`), whether the handler beat his man (`blowby`) and how it ended
(`endType`). This module joins that to what the drive actually produced —
points, shots, fouls, turnovers, assists — and to the tracking-derived defensive
collapse around it.

Outcome accounting
------------------
Drive outcomes come from ``bhrOutcomes``, the same token vocabulary used by
`touches` (docs/data_dictionary/touches.md). We do not invent an outcome that
the token list does not state:

* ``points_direct``    – 2 or 3 from the handler's own make (FGM / FGM3)
* ``points_created``   – 2 or 3 credited to the handler's assist (AST2 / AST3)
* ``fouled``           – any FOUL / FOU_S / FOU_N / FOU_B token
* ``turnover``         – TO token
* ``pass_out``         – endType == 'kickout' or 'interiorPass'

``points_per_drive`` is (points_direct + points_created) / drives. It excludes
free-throw points, exactly as the season aggregate's ``points`` column does, so
the two are comparable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.preprocessing import court

FOUL_TOKENS = {"FOUL", "FOU_S", "FOU_N", "FOU_B", "FOU_O", "FOU_T", "FOU_F"}


def _tok(s: pd.Series) -> pd.Series:
    return s.fillna("").map(lambda v: set(v.split("|")) - {""})


def build_drive_features(tables: dict[str, pd.DataFrame],
                         spatial_start: pd.DataFrame,
                         spatial_end: pd.DataFrame,
                         collapse: pd.DataFrame) -> pd.DataFrame:
    dr = tables["drives"].copy()
    toks = _tok(dr["bhr_outcomes"])

    dr["points_direct"] = [3 if "FGM3" in t else (2 if "FGM" in t else 0) for t in toks]
    dr["points_created"] = [3 if "AST3" in t else (2 if "AST2" in t else 0) for t in toks]
    dr["points_total"] = dr.points_direct + dr.points_created
    dr["drew_foul"] = [bool(t & FOUL_TOKENS) for t in toks]
    dr["turned_over"] = ["TO" in t for t in toks]
    dr["took_shot"] = [bool(t & {"FGM", "FGM3", "FGX", "FGX3", "BLK"}) for t in toks]
    dr["assisted"] = [bool(t & {"AST", "AST2", "AST3"}) for t in toks]
    dr["assist_opportunity"] = [bool(t & {"AST", "AST2", "AST3", "AST0", "ASTF"}) for t in toks]
    dr["passed_out"] = dr.end_type.isin(["kickout", "interiorPass"])
    # SkillCorner's own aggregate definition of a successful drive
    dr["successful"] = dr.points_direct.gt(0) | dr.assisted | dr.drew_foul

    dr["start_dist_hoop"] = court.dist_to_off_hoop(dr.location_x, dr.location_y)
    dr["end_dist_hoop"] = court.dist_to_off_hoop(dr.end_loc_x, dr.end_loc_y)
    dr["drive_gain_ft"] = dr.start_dist_hoop - dr.end_dist_hoop
    dr["path_length_ft"] = np.hypot(dr.end_loc_x - dr.location_x, dr.end_loc_y - dr.location_y)
    dr["duration_s"] = ((dr.end_frame - dr.start_frame) / C.FPS).astype("float32")
    dr["speed_to_rim"] = dr.path_length_ft / dr.duration_s.replace(0, np.nan)
    dr["start_zone"] = court.shot_zone(dr.location_x, dr.location_y)
    dr["reached_paint"] = court.in_paint(dr.end_loc_x, dr.end_loc_y)

    ch = tables["chances"].rename(columns={"id": "chance_id"})[
        ["chance_id", "transition", "pts_scored", "quality_index"]].rename(
        columns={"transition": "chance_transition", "pts_scored": "chance_points",
                 "quality_index": "chance_quality_index"})
    dr = dr.merge(ch, on="chance_id", how="left")

    dr = dr.merge(spatial_start.add_prefix("start_").rename(columns={"start_id": "id"}),
                  on="id", how="left")
    dr = dr.merge(spatial_end.add_prefix("end_").rename(columns={"end_id": "id"}),
                  on="id", how="left")

    if not collapse.empty:
        c = collapse.copy()
        first = c[c.frames_since_start == 0].set_index("drive_id")
        piv = (c.groupby("drive_id")
                 .agg(help_min=("second_def_dist", "min"),
                      help_at_end=("second_def_dist", "last"),
                      n_def_paint_max=("n_def_in_paint", "max"),
                      nearest_min=("nearest_def_dist", "min"),
                      hull_min=("hull_area", "min"),
                      samples=("nearest_def_dist", "size")))
        piv["help_at_start"] = first["second_def_dist"]
        piv["nearest_at_start"] = first["nearest_def_dist"]
        piv["n_def_paint_start"] = first["n_def_in_paint"]
        piv["help_closed_ft"] = piv.help_at_start - piv.help_min
        piv["paint_defenders_added"] = piv.n_def_paint_max - piv.n_def_paint_start
        dr = dr.merge(piv.reset_index().rename(columns={"drive_id": "id"}), on="id", how="left")
    return dr


def player_drive_profile(dr: pd.DataFrame, roster: pd.DataFrame,
                         min_drives: int = 8) -> pd.DataFrame:
    g = dr.groupby("ballhandler_id_canonical")
    prof = pd.DataFrame({
        "drives": g.size(),
        "points_per_drive": g.points_total.mean(),
        "own_points_per_drive": g.points_direct.mean(),
        "created_points_per_drive": g.points_created.mean(),
        "shot_rate": g.took_shot.mean(),
        "foul_rate": g.drew_foul.mean(),
        "turnover_rate": g.turned_over.mean(),
        "pass_out_rate": g.passed_out.mean(),
        "blowby_rate": g.blowby.mean(),
        "success_rate": g.successful.mean(),
        "reached_paint_rate": g.reached_paint.mean(),
        "mean_gain_ft": g.drive_gain_ft.mean(),
        "mean_duration_s": g.duration_s.mean(),
        "left_share": g.direction.apply(lambda s: (s == "left").mean()),
        "help_closed_ft": g.help_closed_ft.mean(),
        "paint_defenders_added": g.paint_defenders_added.mean(),
    }).reset_index().rename(columns={"ballhandler_id_canonical": "player_id_canonical"})
    prof = prof[prof.drives >= min_drives]
    names = (roster.drop_duplicates("player_id_canonical")
                   [["player_id_canonical", "player_name", "team_name"]])
    return prof.merge(names, on="player_id_canonical", how="left").sort_values("drives", ascending=False)
