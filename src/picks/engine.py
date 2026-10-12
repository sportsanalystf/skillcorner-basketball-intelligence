"""Pick-and-roll intelligence.

Every pick carries the handler, screener, both defenders and — crucially — the
coverage played on each (`bhrDefType`, `scrDefType`). Coverage vocabularies are
taken verbatim from the data; nothing is invented:

    bhrDefType : over, under, switch, ice, blitz, none
    scrDefType : soft, show, switch, ice, blitz, none

Outcomes are decoded from the ``bhrOutcomes`` / ``scrOutcomes`` token lists using
the same rules as drives. "Direct" picks (``direct = true``) are the subset the
official season aggregates count, so any comparison against
``acb_picksaggregates`` filters to those first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.preprocessing import court

FOUL_TOKENS = {"FOUL", "FOU_S", "FOU_N", "FOU_B", "FOU_O"}


def _decode(tokens: pd.Series, prefix: str) -> pd.DataFrame:
    t = tokens.fillna("").map(lambda v: set(v.split("|")) - {""})
    return pd.DataFrame({
        f"{prefix}_points_direct": [3 if "FGM3" in s else (2 if "FGM" in s else 0) for s in t],
        f"{prefix}_points_created": [3 if "AST3" in s else (2 if "AST2" in s else 0) for s in t],
        f"{prefix}_shot": [bool(s & {"FGM", "FGM3", "FGX", "FGX3", "BLK"}) for s in t],
        f"{prefix}_made": [bool(s & {"FGM", "FGM3"}) for s in t],
        f"{prefix}_three_att": [bool(s & {"FGM3", "FGX3"}) for s in t],
        f"{prefix}_assist": [bool(s & {"AST", "AST2", "AST3"}) for s in t],
        f"{prefix}_assist_opp": [bool(s & {"AST", "AST2", "AST3", "AST0", "ASTF"}) for s in t],
        f"{prefix}_turnover": ["TO" in s for s in t],
        f"{prefix}_foul": [bool(s & FOUL_TOKENS) for s in t],
        f"{prefix}_pass": [bool(s & {"PASS", "PASS_SCR"}) for s in t],
        f"{prefix}_pass_to_screener": ["PASS_SCR" in s for s in t],
        f"{prefix}_blocked": ["BLK" in s for s in t],
    }, index=tokens.index)


def build_pick_features(tables: dict[str, pd.DataFrame],
                        spatial: pd.DataFrame) -> pd.DataFrame:
    pk = tables["picks"].copy()
    pk = pd.concat([pk, _decode(pk["bhr_outcomes"], "bhr"),
                    _decode(pk["scr_outcomes"], "scr")], axis=1)

    pk["bhr_points"] = pk.bhr_points_direct + pk.bhr_points_created
    pk["scr_points"] = pk.scr_points_direct + pk.scr_points_created
    pk["pick_points"] = pk[["bhr_points", "scr_points"]].max(axis=1)
    pk["bhr_successful"] = pk.bhr_points_direct.gt(0) | pk.bhr_assist | pk.bhr_foul
    pk["scr_successful"] = pk.scr_points_direct.gt(0) | pk.scr_assist | pk.scr_foul

    pk["dist_hoop"] = court.dist_to_off_hoop(pk.location_x, pk.location_y)
    pk["zone"] = court.shot_zone(pk.location_x, pk.location_y)
    pk["coverage"] = pk.bhr_def_type.fillna("none") + " / " + pk.scr_def_type.fillna("none")

    ch = tables["chances"].rename(columns={"id": "chance_id"})[
        ["chance_id", "transition", "pts_scored", "quality_index"]].rename(
        columns={"transition": "chance_transition", "pts_scored": "chance_points",
                 "quality_index": "chance_quality_index"})
    pk = pk.merge(ch, on="chance_id", how="left")
    pk = pk.merge(spatial, on="id", how="left")

    # did the screener roll or pop? derived from the screener's own movement in
    # the 1.5 s after the screen — computed separately in roll_pop_labels().
    return pk


def roll_pop_labels(picks: pd.DataFrame, window_frames: int = 38) -> pd.DataFrame:
    """Classify the screener's action after the screen as roll / pop / slip / stay.

    Derived from tracking: we compare the screener's distance to the offensive
    hoop at the screen frame and ``window_frames`` (1.5 s) later.

        roll  – moves >= 4 ft closer to the rim
        pop   – moves >= 4 ft further from the rim
        stay  – net movement < 4 ft

    This is OUR label, not a SkillCorner field; the dataset has no roll/pop flag.
    """
    from src.spatial.features import load_frames

    out = []
    for gid, sub in picks.groupby("game_id"):
        sub = sub.dropna(subset=["frame", "screener_id"])
        if sub.empty:
            continue
        f0 = sub.frame.to_numpy().astype(int)
        f1 = f0 + window_frames
        tr = load_frames(int(gid), np.concatenate([f0, f1]),
                         cols=["frame_idx", "player_id", "x_att", "y_att", "speed"])
        key = tr.set_index(["frame_idx", "player_id"])
        for r, a, b in zip(sub.itertuples(index=False), f0, f1):
            try:
                p0 = key.loc[(a, r.screener_id)]
                p1 = key.loc[(b, r.screener_id)]
            except KeyError:
                out.append({"id": r.id, "screener_action": None,
                            "screener_rim_gain_ft": np.nan, "screener_speed_after": np.nan})
                continue
            p0 = p0.iloc[0] if isinstance(p0, pd.DataFrame) else p0
            p1 = p1.iloc[0] if isinstance(p1, pd.DataFrame) else p1
            d0 = court.dist_to_off_hoop(p0.x_att, p0.y_att)
            d1 = court.dist_to_off_hoop(p1.x_att, p1.y_att)
            gain = float(d0 - d1)
            lab = "roll" if gain >= 4 else ("pop" if gain <= -4 else "stay")
            out.append({"id": r.id, "screener_action": lab,
                        "screener_rim_gain_ft": gain,
                        "screener_speed_after": float(p1.speed)})
    return pd.DataFrame(out)


def handler_profile(pk: pd.DataFrame, roster: pd.DataFrame, min_picks: int = 15) -> pd.DataFrame:
    g = pk.groupby("ballhandler_id_canonical")
    prof = pd.DataFrame({
        "picks": g.size(),
        "direct_rate": g.direct.mean(),
        "ppp": g.bhr_points.mean(),
        "shot_rate": g.bhr_shot.mean(),
        "assist_rate": g.bhr_assist.mean(),
        "assist_opp_rate": g.bhr_assist_opp.mean(),
        "pass_to_screener_rate": g.bhr_pass_to_screener.mean(),
        "turnover_rate": g.bhr_turnover.mean(),
        "foul_drawn_rate": g.bhr_foul.mean(),
        "three_rate": g.bhr_three_att.mean(),
        "success_rate": g.bhr_successful.mean(),
        "pct_middle": g.location_type.apply(lambda s: (s == "middle").mean()),
        "pct_wing": g.location_type.apply(lambda s: (s == "wing").mean()),
        "pct_stepup": g.location_type.apply(lambda s: (s == "stepUp").mean()),
    }).reset_index().rename(columns={"ballhandler_id_canonical": "player_id_canonical"})
    prof = prof[prof.picks >= min_picks]
    names = roster.drop_duplicates("player_id_canonical")[
        ["player_id_canonical", "player_name", "team_name"]]
    return prof.merge(names, on="player_id_canonical", how="left").sort_values("picks", ascending=False)


def screener_profile(pk: pd.DataFrame, roster: pd.DataFrame, min_picks: int = 15) -> pd.DataFrame:
    g = pk.groupby("screener_id_canonical")
    prof = pd.DataFrame({
        "picks_set": g.size(),
        "ppp": g.scr_points.mean(),
        "shot_rate": g.scr_shot.mean(),
        "assist_rate": g.scr_assist.mean(),
        "turnover_rate": g.scr_turnover.mean(),
        "foul_drawn_rate": g.scr_foul.mean(),
        "success_rate": g.scr_successful.mean(),
        "handler_ppp_when_screening": g.bhr_points.mean(),
        "pct_switch_forced": g.scr_def_type.apply(lambda s: (s == "switch").mean()),
    }).reset_index().rename(columns={"screener_id_canonical": "player_id_canonical"})
    if "screener_action" in pk.columns:
        r = pk.groupby("screener_id_canonical").screener_action.apply(
            lambda s: (s == "roll").mean()).rename("roll_rate").reset_index().rename(
            columns={"screener_id_canonical": "player_id_canonical"})
        prof = prof.merge(r, on="player_id_canonical", how="left")
    prof = prof[prof.picks_set >= min_picks]
    names = roster.drop_duplicates("player_id_canonical")[
        ["player_id_canonical", "player_name", "team_name"]]
    return prof.merge(names, on="player_id_canonical", how="left").sort_values("picks_set", ascending=False)


def coverage_effectiveness(pk: pd.DataFrame, by: str = "bhr_def_type") -> pd.DataFrame:
    g = pk.groupby(by)
    out = pd.DataFrame({
        "picks": g.size(),
        "ppp": g.pick_points.mean(),
        "handler_ppp": g.bhr_points.mean(),
        "screener_ppp": g.scr_points.mean(),
        "handler_shot_rate": g.bhr_shot.mean(),
        "handler_assist_rate": g.bhr_assist.mean(),
        "turnover_rate": g.bhr_turnover.mean(),
        "foul_rate": g.bhr_foul.mean(),
        "direct_rate": g.direct.mean(),
    }).reset_index()
    return out.sort_values("picks", ascending=False)
