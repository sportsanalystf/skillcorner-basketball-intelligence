"""Shot feature assembly: event fields + tracking-derived spatial context +
the action that created the shot.

Shot *creation* is reconstructed by looking backwards inside the same chance for
the nearest preceding action marking (pick, handoff, off-ball screen, drive,
isolation, post, closeout) within a lookback window, plus the pass — if any —
that immediately preceded the shot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.preprocessing import court

CREATION_LOOKBACK_FRAMES = 100  # 4.0 s — long enough to cover a screen → catch → rise


def _nearest_preceding(shots: pd.DataFrame, actions: pd.DataFrame, frame_col: str,
                       label: str, extra: dict[str, str] | None = None,
                       lookback: int = CREATION_LOOKBACK_FRAMES) -> pd.DataFrame:
    """For each shot, the last action of this type in the same chance before it."""
    if actions.empty:
        return pd.DataFrame({"id": shots["id"], f"prev_{label}_frames_before": np.nan})
    a = actions[["chance_id", frame_col, "id"] + list((extra or {}).keys())].copy()
    a = a.rename(columns={frame_col: "afr", "id": "aid"})
    m = shots[["id", "chance_id", "start_frame"]].merge(a, on="chance_id", how="left")
    m = m[(m.afr <= m.start_frame) & (m.start_frame - m.afr <= lookback)]
    m = m.sort_values("afr").groupby("id").tail(1)
    out = pd.DataFrame({"id": m["id"],
                        f"prev_{label}_id": m["aid"],
                        f"prev_{label}_frames_before": (m.start_frame - m.afr).astype("float32")})
    for src, dst in (extra or {}).items():
        out[dst] = m[src].to_numpy()
    return shots[["id"]].merge(out, on="id", how="left")


def build_shot_features(tables: dict[str, pd.DataFrame],
                        spatial: pd.DataFrame) -> pd.DataFrame:
    sh = tables["shots"].copy()

    # --- geometry recomputed from the delivered location (attacking frame)
    sh["dist_calc"] = court.dist_to_off_hoop(sh.location_x, sh.location_y)
    sh["angle_deg"] = court.angle_from_baseline(sh.location_x, sh.location_y)
    sh["zone"] = court.shot_zone(sh.location_x, sh.location_y)
    sh["zone_family"] = sh["zone"].map(court.ZONE_FAMILY)
    sh["is_three_calc"] = court.is_three(sh.location_x, sh.location_y)
    sh["points_if_made"] = np.where(sh["three"], 3, 2)
    sh["points"] = np.where(sh["outcome"], sh["points_if_made"], 0)
    # FIBA box-score attempt convention: a miss while fouled is a FT trip, not a FGA
    sh["is_box_score_fga"] = ~(sh["fouled"] & ~sh["outcome"])

    # --- chance / possession context
    ch = tables["chances"].rename(columns={"id": "chance_id"})[
        ["chance_id", "transition", "start_type", "quality_index", "usable",
         "start_shot_clock", "possession_id"]].rename(
        columns={"transition": "chance_transition", "start_type": "chance_start_type",
                 "quality_index": "chance_quality_index", "usable": "chance_usable"})
    sh = sh.merge(ch, on=["chance_id", "possession_id"], how="left")

    # --- creation lineage
    lineage = [
        ("picks", "frame", "pick",
         {"bhr_def_type": "prev_pick_bhr_def", "scr_def_type": "prev_pick_scr_def",
          "screener_id": "prev_pick_screener_id", "ballhandler_id": "prev_pick_bhr_id",
          "location_type": "prev_pick_loc_type"}),
        ("drives", "start_frame", "drive",
         {"category": "prev_drive_category", "end_type": "prev_drive_end_type",
          "blowby": "prev_drive_blowby", "ballhandler_id": "prev_drive_bhr_id"}),
        ("handoffs", "frame", "handoff", {"receiver_def_type": "prev_handoff_rec_def"}),
        ("off_ball_screens", "frame", "obscreen",
         {"cutter_def_type": "prev_obs_cutter_def", "cutter_id": "prev_obs_cutter_id"}),
        ("isolations", "start_frame", "iso", {}),
        ("posts", "start_frame", "post", {}),
        ("closeouts", "start_frame", "closeout",
         {"bhr_action": "prev_closeout_action", "start_distance": "prev_closeout_start_dist",
          "end_distance": "prev_closeout_end_dist"}),
    ]
    for tbl, fcol, label, extra in lineage:
        sh = sh.merge(_nearest_preceding(sh, tables.get(tbl, pd.DataFrame()), fcol, label, extra),
                      on="id", how="left")

    # --- single "creation type" label, most specific action wins
    def creation(row):
        if pd.notna(row.get("prev_drive_frames_before")):
            return "drive"
        if pd.notna(row.get("prev_pick_frames_before")):
            return "pick"
        if pd.notna(row.get("prev_handoff_frames_before")):
            return "handoff"
        if pd.notna(row.get("prev_obscreen_frames_before")):
            return "off_ball_screen"
        if pd.notna(row.get("prev_iso_frames_before")):
            return "isolation"
        if pd.notna(row.get("prev_post_frames_before")):
            return "post"
        if pd.notna(row.get("prev_closeout_frames_before")):
            return "closeout"
        return "unstructured"
    sh["creation_type"] = sh.apply(creation, axis=1)

    # --- touch context
    tch = tables["touches"].rename(columns={"id": "touch_id"})[
        ["touch_id", "num_dribbles", "touch_time", "defender_id", "led_to_shot"]].rename(
        columns={"num_dribbles": "touch_dribbles", "touch_time": "touch_seconds",
                 "defender_id": "touch_defender_id", "led_to_shot": "touch_led_to_shot"})
    sh = sh.merge(tch, on="touch_id", how="left")

    # --- tracking-derived spacing / defensive geometry
    sh = sh.merge(spatial, on="id", how="left", suffixes=("", "_sp"))
    return sh
