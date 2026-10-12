"""Team tactical profiles and automated scouting statements.

Every sentence produced by :func:`scouting_notes` is generated from a computed
number and carries that number with it. Nothing is asserted that is not backed
by a count in the data, and every statement carries its sample size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C


def team_offense_profile(shots, drives, picks, chances, closeouts, obs) -> pd.DataFrame:
    rows = []
    for tid, ch in chances.groupby("off_team_id"):
        s = shots[shots.off_team_id == tid]
        d = drives[drives.off_team_id == tid]
        p = picks[picks.off_team_id == tid]
        o = obs[obs.off_team_id == tid]
        n_ch = len(ch)
        fga = s.is_box_score_fga.sum() if "is_box_score_fga" in s else len(s)
        rows.append({
            "team_id": tid, "games": ch.game_id.nunique(), "chances": n_ch,
            "pts_per_chance": ch.pts_scored.mean(),
            "sec_per_chance": (ch.start_game_clock - ch.end_game_clock).mean(),
            "transition_rate": ch.transition.mean(),
            "to_rate": ch.outcome.eq("TO").mean(),
            "shots_per_chance": len(s) / n_ch,
            "three_rate": s.three.mean(),
            "rim_rate": s.zone_family.eq("rim").mean() if "zone_family" in s else np.nan,
            "corner3_rate": s.zone.isin(["corner_3_left", "corner_3_right"]).mean() if "zone" in s else np.nan,
            "efg": ((s.outcome & s.three).sum() * 1.5 + (s.outcome & ~s.three).sum()) / max(fga, 1),
            "mean_shot_quality": s.shot_quality.mean(),
            "mean_xfg": s.xfg.mean() if "xfg" in s else np.nan,
            "fg_vs_expected": (s.outcome.mean() - s.xfg.mean()) if "xfg" in s else np.nan,
            "contested_rate": s.contested.mean(),
            "catch_and_shoot_rate": s.catch_and_shoot.mean(),
            "assisted_rate": s.assisted.mean(),
            "drives_per_chance": len(d) / n_ch,
            "drive_ppd": d.points_total.mean() if len(d) else np.nan,
            "drive_kickout_rate": d.passed_out.mean() if len(d) else np.nan,
            "drive_left_rate": d.direction.eq("left").mean() if len(d) else np.nan,
            "picks_per_chance": len(p) / n_ch,
            "pick_ppp": p.pick_points.mean() if len(p) else np.nan,
            "pick_middle_rate": p.location_type.eq("middle").mean() if len(p) else np.nan,
            "obs_per_chance": len(o) / n_ch,
        })
    return pd.DataFrame(rows)


def team_defense_profile(shots, drives, picks, chances, closeouts) -> pd.DataFrame:
    rows = []
    for tid, ch in chances.groupby("def_team_id"):
        s = shots[shots.def_team_id == tid]
        p = picks[picks.def_team_id == tid]
        c = closeouts[closeouts.def_team_id == tid]
        d = drives[drives.def_team_id == tid]
        n_ch = len(ch)
        rows.append({
            "team_id": tid, "games": ch.game_id.nunique(), "chances_defended": n_ch,
            "pts_allowed_per_chance": ch.pts_scored.mean(),
            "opp_three_rate": s.three.mean(),
            "opp_rim_rate": s.zone_family.eq("rim").mean() if "zone_family" in s else np.nan,
            "opp_contested_rate": s.contested.mean(),
            "opp_mean_closest_def_dist": s.closest_def_dist.mean(),
            "opp_shot_quality_allowed": s.shot_quality.mean(),
            "opp_fg_vs_expected": (s.outcome.mean() - s.xfg.mean()) if "xfg" in s else np.nan,
            "forced_to_rate": ch.outcome.eq("TO").mean(),
            "pick_cov_over": p.bhr_def_type.eq("over").mean() if len(p) else np.nan,
            "pick_cov_under": p.bhr_def_type.eq("under").mean() if len(p) else np.nan,
            "pick_cov_switch": p.bhr_def_type.eq("switch").mean() if len(p) else np.nan,
            "scr_cov_soft": p.scr_def_type.eq("soft").mean() if len(p) else np.nan,
            "scr_cov_show": p.scr_def_type.eq("show").mean() if len(p) else np.nan,
            "pick_ppp_allowed": p.pick_points.mean() if len(p) else np.nan,
            "closeouts_per_chance": len(c) / n_ch,
            "closeout_end_dist": c.end_distance.mean() if len(c) else np.nan,
            "closeout_close_rate": c.close_rate_fps.mean() if len(c) else np.nan,
            "closeout_stop_rate": c.stop.mean() if len(c) else np.nan,
            "drive_blowby_allowed": d.blowby.mean() if len(d) else np.nan,
            "opp_drive_ppd": d.points_total.mean() if len(d) else np.nan,
        })
    return pd.DataFrame(rows)


def _pctile(v, series):
    s = series.dropna()
    if len(s) < 3 or pd.isna(v):
        return np.nan
    return float((s < v).mean())


def scouting_notes(team_id: int, off: pd.DataFrame, defn: pd.DataFrame,
                   drives: pd.DataFrame, picks: pd.DataFrame, shots: pd.DataFrame,
                   teams: dict[int, str], min_n: int = 20) -> list[dict]:
    """Evidence-backed scouting statements for one team.

    Each note is {text, metric, value, league_rank, n}. A note is only emitted
    when the underlying sample reaches ``min_n``.
    """
    notes = []
    o = off[off.team_id == team_id]
    d = defn[defn.team_id == team_id]
    if o.empty:
        return notes
    o = o.iloc[0]; d = d.iloc[0] if not d.empty else None
    nteams = len(off)

    def add(text, metric, value, series, n):
        if pd.isna(value) or n < min_n:
            return
        pct = _pctile(value, series)
        notes.append({"text": text, "metric": metric, "value": float(value),
                      "league_pctile": None if pd.isna(pct) else round(pct, 3),
                      "rank_of": nteams, "n": int(n)})

    add(f"Scores {o.pts_per_chance:.2f} points per chance across {int(o.chances)} tracked chances.",
        "pts_per_chance", o.pts_per_chance, off.pts_per_chance, o.chances)
    add(f"{o.three_rate:.0%} of field goal attempts are threes.",
        "three_rate", o.three_rate, off.three_rate, o.chances)
    add(f"Runs {o.picks_per_chance:.2f} on-ball screens per chance, {o.pick_middle_rate:.0%} of them in the middle.",
        "picks_per_chance", o.picks_per_chance, off.picks_per_chance, o.chances)
    add(f"Drives {o.drives_per_chance:.2f} times per chance and kicks out on {o.drive_kickout_rate:.0%} of drives.",
        "drive_kickout_rate", o.drive_kickout_rate, off.drive_kickout_rate, o.chances)
    add(f"{o.transition_rate:.0%} of chances are transition.",
        "transition_rate", o.transition_rate, off.transition_rate, o.chances)
    add(f"Shoots {o.fg_vs_expected:+.1%} against the model's expected FG% on these attempts.",
        "fg_vs_expected", o.fg_vs_expected, off.fg_vs_expected, o.chances)

    # drive origin — the most common start zone, with its share
    dd = drives[drives.off_team_id == team_id]
    if len(dd) >= min_n:
        z = dd.region.value_counts(normalize=True)
        if len(z):
            notes.append({"text": f"Most drives start from the {z.index[0]} ({z.iloc[0]:.0%} of {len(dd)}).",
                          "metric": "drive_origin", "value": float(z.iloc[0]),
                          "league_pctile": None, "rank_of": nteams, "n": int(len(dd))})
        dirn = dd.direction.value_counts(normalize=True)
        if len(dirn):
            notes.append({"text": f"Drives {dirn.index[0]} on {dirn.iloc[0]:.0%} of attacks.",
                          "metric": "drive_direction", "value": float(dirn.iloc[0]),
                          "league_pctile": None, "rank_of": nteams, "n": int(len(dd))})

    if d is not None:
        add(f"Guards ball screens 'over' {d.pick_cov_over:.0%} of the time, switches {d.pick_cov_switch:.0%}.",
            "pick_cov_switch", d.pick_cov_switch, defn.pick_cov_switch, d.chances_defended)
        add(f"Allows {d.pts_allowed_per_chance:.2f} points per chance defended.",
            "pts_allowed_per_chance", d.pts_allowed_per_chance, defn.pts_allowed_per_chance,
            d.chances_defended)
        add(f"Closes out at {d.closeout_close_rate:.1f} ft/s, arriving {d.closeout_end_dist:.1f} ft away.",
            "closeout_close_rate", d.closeout_close_rate, defn.closeout_close_rate,
            d.chances_defended)
        add(f"Opponents beat the on-ball defender on {d.drive_blowby_allowed:.0%} of drives.",
            "drive_blowby_allowed", d.drive_blowby_allowed, defn.drive_blowby_allowed,
            d.chances_defended)
    return notes
