"""Player profiles: game-level (tracking) and season-level (aggregates).

Two very different samples live side by side and are never mixed silently:

* **Game level** — the 10 published games. Rich (tracking, defence, spacing) but
  tiny: 20-40 minutes per player. Used for spatial and defensive description.
* **Season level** — the offence-only aggregates over 293 games. Large enough
  for player comparison and clustering, but offence only and with no tracking
  context beyond what the aggregates already pivot on.

Both carry an explicit sample-size column so the UI can gate on it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C

FOUL_TOKENS = {"FOUL", "FOU_S", "FOU_N", "FOU_B", "FOU_O"}


def _rate(num, den):
    den = pd.Series(den).replace(0, np.nan)
    return (pd.Series(num).to_numpy() / den.to_numpy())


def game_level_profile(tables: dict[str, pd.DataFrame], derived: dict[str, pd.DataFrame],
                       roster: pd.DataFrame) -> pd.DataFrame:
    sh = derived["shots_features"]
    dr = derived["drives_features"]
    pk = derived["picks_features"]
    co = derived["closeouts_features"]
    tc = tables["touches"]
    ps = tables["passes"]
    cp = tables["chance_players"]
    phys = derived["player_game_physical"]

    key = "player_id_canonical"

    # on-court volume
    onc = (cp.groupby("player_id_canonical")
             .agg(chances_on_court=("chance_id", "nunique"),
                  games=("game_id", "nunique"),
                  off_chances=("offense", "sum")).reset_index())
    onc["def_chances"] = onc.chances_on_court - onc.off_chances

    g = sh.groupby("shooter_id_canonical")
    shot = pd.DataFrame({
        "shots": g.size(), "fga": g.is_box_score_fga.sum(),
        "fg_pct": g.apply(lambda d: d.loc[d.is_box_score_fga, "outcome"].mean(), include_groups=False),
        "points": g.points.sum(),
        "three_rate": g.three.mean(), "rim_rate": g.zone_family.apply(lambda s: (s == "rim").mean()),
        "mean_shot_dist": g.distance.mean(),
        "contested_rate": g.contested.mean(),
        "mean_closest_def_dist": g.closest_def_dist.mean(),
        "catch_and_shoot_rate": g.catch_and_shoot.mean(),
        "assisted_rate": g.assisted.mean(),
        "xfg": g.xfg.mean(), "xpts_per_shot": g.xpts.mean(),
        "shot_value_added": g.shot_value_added.sum(),
        "skc_shot_quality": g.shot_quality.mean(),
    }).reset_index().rename(columns={"shooter_id_canonical": key})

    gd = dr.groupby("ballhandler_id_canonical")
    drive = pd.DataFrame({
        "drives": gd.size(), "drive_ppd": gd.points_total.mean(),
        "drive_blowby_rate": gd.blowby.mean(), "drive_kickout_rate": gd.passed_out.mean(),
        "drive_to_rate": gd.turned_over.mean(), "drive_foul_rate": gd.drew_foul.mean(),
        "drive_help_closed": gd.help_closed_ft.mean(),
    }).reset_index().rename(columns={"ballhandler_id_canonical": key})

    gh = pk.groupby("ballhandler_id_canonical")
    hand = pd.DataFrame({
        "picks_used": gh.size(), "pick_ppp": gh.bhr_points.mean(),
        "pick_assist_rate": gh.bhr_assist.mean(), "pick_to_rate": gh.bhr_turnover.mean(),
        "pick_shot_rate": gh.bhr_shot.mean(),
        "pct_vs_switch": gh.bhr_def_type.apply(lambda s: (s == "switch").mean()),
        "pct_vs_under": gh.bhr_def_type.apply(lambda s: (s == "under").mean()),
    }).reset_index().rename(columns={"ballhandler_id_canonical": key})

    gs = pk.groupby("screener_id_canonical")
    scr = pd.DataFrame({
        "picks_set": gs.size(), "screen_ppp": gs.scr_points.mean(),
        "roll_rate": gs.screener_action.apply(lambda s: (s == "roll").mean()),
        "screen_forces_switch": gs.scr_def_type.apply(lambda s: (s == "switch").mean()),
    }).reset_index().rename(columns={"screener_id_canonical": key})

    gt = tc.groupby("player_id_canonical")
    touch = pd.DataFrame({
        "touches": gt.size(), "touch_seconds": gt.touch_time.sum(),
        "mean_touch_time": gt.touch_time.mean(), "mean_dribbles": gt.num_dribbles.mean(),
        "touch_to_shot_rate": gt.led_to_shot.mean(),
    }).reset_index().rename(columns={"player_id_canonical": key})

    gp = ps.groupby("passer_id_canonical")
    pas = pd.DataFrame({
        "passes": gp.size(), "pass_complete_rate": gp.complete.mean(),
        "pass_to_shot_rate": gp.led_to_shot.mean(), "assist_opp_rate": gp.assist_opp.mean(),
        "mean_pass_distance": gp.distance.mean(),
    }).reset_index().rename(columns={"passer_id_canonical": key})

    gc = co.groupby("ballhandler_def_id_canonical")
    dfn = pd.DataFrame({
        "closeouts": gc.size(), "closeout_close_rate": gc.close_rate_fps.mean(),
        "closeout_end_dist": gc.end_distance.mean(),
        "closeout_stop_rate": gc.stop.mean(),
    }).reset_index().rename(columns={"ballhandler_def_id_canonical": key})

    ph = (phys.merge(roster[["game_id", "player_id", "player_id_canonical"]],
                     on=["game_id", "player_id"], how="left")
              .groupby(key).agg(
                  minutes_tracked=("minutes_tracked", "sum"),
                  distance_ft=("distance_ft", "sum"),
                  mean_speed=("mean_speed", "mean"), p95_speed=("p95_speed", "mean"),
                  max_speed=("max_speed", "max"), mean_abs_accel=("mean_abs_accel", "mean"),
                  max_decel=("max_decel", "min"),
                  dist_sprinting=("dist_sprinting", "sum") if "dist_sprinting" in phys.columns else ("distance_ft", "sum"),
                  dist_high_speed=("dist_high_speed", "sum") if "dist_high_speed" in phys.columns else ("distance_ft", "sum"),
              ).reset_index())
    ph["distance_per_min"] = ph.distance_ft / ph.minutes_tracked.replace(0, np.nan)
    ph["high_intensity_share"] = (ph.dist_sprinting + ph.dist_high_speed) / ph.distance_ft.replace(0, np.nan)

    names = roster.drop_duplicates(key)[[key, "player_name", "team_id", "team_name", "jersey"]]
    out = names
    for d in (onc, ph, shot, drive, hand, scr, touch, pas, dfn):
        out = out.merge(d, on=key, how="left")
    out["pts_per_chance_on_court"] = out.points / out.off_chances.replace(0, np.nan)
    out["usage_touches_per_min"] = out.touches / out.minutes_tracked.replace(0, np.nan)
    return out


# --------------------------------------------------------------------------
# Season-level style vector (offence only, from the aggregates)
# --------------------------------------------------------------------------
STYLE_FEATURES = {
    # shot profile — what kind of shots and from where
    "rim_attempt_rate": ("shots", "rim_attempt_rate"),
    "midrange_attempt_rate": ("shots", "long_midrange_attempt_rate"),
    "three_attempt_rate": ("shots", "zone_three_attempt_rate"),
    "corner3_share": ("shots", None),
    "cns_share": ("shots", None),
    "contested_share": ("shots", None),
    "avg_shot_distance": ("shots", "avg_attempts_distance"),
    "assisted_share": ("shots", None),
    "shot_volume_per_game": ("shots", None),
    # driving
    "drives_per_game": ("drives", None),
    "drive_kickout_rate": ("drives", "end_type_kickout_rate"),
    "drive_pullup_rate": ("drives", "end_type_pullup_rate"),
    "drive_rim_rate": ("drives", "end_type_shot_near_basket_rate"),
    "drive_blowby_rate": ("drives", "blowby_rate"),
    "drive_assist_rate": ("drives", "assists_rate"),
    # pick-and-roll role
    "handler_picks_per_game": ("picks", None),
    "screener_picks_per_game": ("picks", None),
    "handler_shot_rate": ("picks", "handler_shot_taken_rate"),
    "handler_pass_rate": ("picks", None),
    "screener_score_rate": ("picks", None),
}


def build_season_style(agg: dict[str, pd.DataFrame], min_games: int = 8) -> pd.DataFrame:
    """One offence-style vector per player-team-season row (per-team rows only)."""
    s = agg["shots"][~agg["shots"].is_season_total_row].copy()
    d = agg["drives"][~agg["drives"].is_season_total_row].copy()
    p = agg["picks"][~agg["picks"].is_season_total_row].copy()

    key = ["player_id_canonical", "team_id"]
    base = s[key + ["player_name", "team_name", "games_played", "possessions_played",
                    "attempts", "total_points", "efg_percentage", "points_per_shot"]].copy()

    def gc(df, col, default=np.nan):
        return df[col] if col in df.columns else pd.Series(default, index=df.index)

    base["rim_attempt_rate"] = gc(s, "rim_attempt_rate")
    base["midrange_attempt_rate"] = gc(s, "long_midrange_attempt_rate")
    base["three_attempt_rate"] = gc(s, "zone_three_attempt_rate")
    base["corner3_share"] = _rate(gc(s, "zone_corner_3_attempts"), s.attempts)
    base["cns_share"] = _rate(gc(s, "cns_attempts"), s.attempts)
    base["contested_share"] = _rate(gc(s, "contested_attempts"), s.attempts)
    base["avg_shot_distance"] = gc(s, "avg_attempts_distance")
    base["assisted_share"] = _rate(gc(s, "assisted_mades"), gc(s, "mades"))
    base["shot_volume_per_game"] = _rate(s.attempts, s.games_played)
    base["ft_rate"] = _rate(gc(s, "fouled_fg_attempts"), s.attempts)

    dd = d.set_index(key)
    base = base.set_index(key)
    base["drives_per_game"] = dd.total_drives / dd.games_played.replace(0, np.nan)
    for c_ in ["end_type_kickout_rate", "end_type_pullup_rate",
               "end_type_shot_near_basket_rate", "blowby_rate", "assists_rate",
               "fouls_rate", "points_per_drive", "category_pick_rate", "category_iso_rate"]:
        base[c_] = dd[c_] if c_ in dd.columns else np.nan

    pp = p.set_index(key)
    base["handler_picks_per_game"] = pp.handler_total_picks / pp.games_played.replace(0, np.nan)
    base["screener_picks_per_game"] = pp.screener_total_picks / pp.games_played.replace(0, np.nan)
    for c_, src in [("handler_shot_rate", "handler_shot_taken_rate"),
                    ("handler_ppp", "handler_ppp"), ("screener_ppp", "screener_ppp"),
                    ("handler_assist_rate", "handler_assist_rate"),
                    ("handler_turnover_rate", "handler_turnover_rate"),
                    ("screener_score_rate", "screener_score_rate")]:
        base[c_] = pp[src] if src in pp.columns else np.nan
    base["handler_pass_rate"] = (pp["handler_pass"] / pp["handler_total_picks"].replace(0, np.nan)
                                 if "handler_pass" in pp.columns else np.nan)
    base["pick_role_ratio"] = (base.handler_picks_per_game /
                               (base.handler_picks_per_game + base.screener_picks_per_game).replace(0, np.nan))

    base = base.reset_index()
    return base[(base.games_played >= min_games) & (base.attempts >= 40)].reset_index(drop=True)
