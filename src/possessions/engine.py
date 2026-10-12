"""Possession → chance → event hierarchy.

`possessions` is one span of team control; `chances` is one scoring opportunity
inside it (an offensive rebound keeps the possession but starts a new chance).
Per docs/PRIMER.md, per-chance efficiency is the right denominator for
points-per-possession style numbers, so **chances** are the analytic unit here
and possessions are the parent grouping.

This module produces:

* :func:`build_chance_summary` – one wide row per chance with duration, start /
  end context, the actions that occurred, ball-movement counts and the outcome.
* :func:`build_event_timeline` – every event in a game as one long, sorted table
  (frame, table, actor, description), which drives the possession timeline UI
  and the event-chain analysis.
* :func:`build_action_chains` – the ordered sequence of *named actions* in each
  chance, e.g. ``pick > drive > kickout > shot``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.preprocessing import court

# Tables that carry a chance_id and describe something that happened.
TIMELINE_SPEC = {
    #  table            frame col        actor col            label
    "touches":          ("start_frame", "player_id",        "touch"),
    "passes":           ("start_frame", "passer_id",        "pass"),
    "dribbles":         ("frame",       "ballhandler_id",   "dribble"),
    "picks":            ("frame",       "ballhandler_id",   "pick"),
    "handoffs":         ("frame",       "receiver_id",      "handoff"),
    "off_ball_screens": ("frame",       "cutter_id",        "off_ball_screen"),
    "drives":           ("start_frame", "ballhandler_id",   "drive"),
    "isolations":       ("start_frame", "ballhandler_id",   "isolation"),
    "posts":            ("start_frame", "ballhandler_id",   "post"),
    "closeouts":        ("start_frame", "ballhandler_def_id", "closeout"),
    "shots":            ("start_frame", "shooter_id",       "shot"),
    "free_throws":      ("frame",       "shooter_id",       "free_throw"),
    "rebounds":         ("frame",       "rebounder_id",     "rebound"),
    "turnovers":        ("frame",       "turned_over_id",   "turnover"),
    "fouls":            ("frame",       "fouler_id",        "foul"),
    "timeouts":         ("frame",       None,               "timeout"),
}

ACTION_TABLES = ["picks", "handoffs", "off_ball_screens", "drives", "isolations",
                 "posts", "closeouts"]


def _count_by_chance(df: pd.DataFrame, name: str) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="int64", name=name)
    return df.groupby("chance_id").size().rename(name)


def build_chance_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per chance, enriched with everything that happened inside it."""
    ch = tables["chances"].copy()
    ch = ch.rename(columns={"id": "chance_id"})

    ch["duration_s"] = (ch["start_game_clock"] - ch["end_game_clock"]).astype("float32")
    ch["duration_wall_s"] = ((ch["end_wall_clock"] - ch["start_wall_clock"]) / 1000.0).astype("float32")
    ch["n_frames"] = (ch["end_frame"] - ch["start_frame"]).astype("float32")
    ch["shot_clock_used"] = (ch["start_shot_clock"] - ch["end_shot_clock"]).astype("float32")
    ch["frontcourt_delay_s"] = ((ch["frontcourt_frame"] - ch["start_frame"]) / C.FPS).astype("float32")

    # possession context
    po = tables["possessions"].rename(columns={"id": "possession_id"})[
        ["possession_id", "game_id", "left_hoop", "start_type", "pts_scored"]].rename(
        columns={"start_type": "possession_start_type", "pts_scored": "possession_pts"})
    ch = ch.merge(po, on=["possession_id", "game_id"], how="left")

    ch["n_off_players"] = ch["off_player_ids"].fillna("").map(lambda s: len([x for x in s.split("|") if x]))
    ch["n_def_players"] = ch["def_player_ids"].fillna("").map(lambda s: len([x for x in s.split("|") if x]))

    # counts of everything inside the chance
    for tbl, col in [("touches", "n_touches"), ("passes", "n_passes"),
                     ("dribbles", "n_dribbles"), ("picks", "n_picks"),
                     ("handoffs", "n_handoffs"), ("off_ball_screens", "n_off_ball_screens"),
                     ("drives", "n_drives"), ("isolations", "n_isolations"),
                     ("posts", "n_posts"), ("closeouts", "n_closeouts"),
                     ("shots", "n_shots"), ("free_throws", "n_free_throws"),
                     ("turnovers", "n_turnovers"), ("fouls", "n_fouls"),
                     ("rebounds", "n_rebounds")]:
        ch = ch.merge(_count_by_chance(tables.get(tbl), col), left_on="chance_id",
                      right_index=True, how="left")
        ch[col] = ch[col].fillna(0).astype("int16")

    # terminal shot of the chance
    sh = tables["shots"]
    if not sh.empty:
        last = (sh.sort_values("start_frame").groupby("chance_id").tail(1)
                  [["chance_id", "shooter_id", "shooter_id_canonical", "three", "outcome",
                    "contest_level", "closest_def_dist", "distance", "shot_type",
                    "complex_shot_type", "catch_and_shoot", "region", "shot_quality",
                    "location_x", "location_y", "assisted", "fouled", "blocked"]]
                .rename(columns={c: f"shot_{c}" for c in
                                 ["shooter_id", "shooter_id_canonical", "three", "outcome",
                                  "contest_level", "closest_def_dist", "distance",
                                  "shot_type", "complex_shot_type", "catch_and_shoot",
                                  "region", "shot_quality", "location_x", "location_y",
                                  "assisted", "fouled", "blocked"]}))
        ch = ch.merge(last, on="chance_id", how="left")

    ch["ended_in_shot"] = ch["n_shots"] > 0
    ch["ended_in_turnover"] = ch["outcome"].eq("TO")
    ch["points"] = ch["pts_scored"].fillna(0).astype("int16")
    return ch


def build_event_timeline(tables: dict[str, pd.DataFrame], game_id: int) -> pd.DataFrame:
    """Long, frame-sorted table of every event in one game."""
    rows = []
    for tbl, (fcol, acol, label) in TIMELINE_SPEC.items():
        df = tables.get(tbl)
        if df is None or df.empty:
            continue
        d = df[df.game_id == game_id]
        if d.empty or fcol not in d.columns:
            continue
        r = pd.DataFrame({
            "frame": d[fcol].to_numpy(),
            "event_table": label,
            "event_id": d["id"].to_numpy(),
            "chance_id": d["chance_id"].to_numpy() if "chance_id" in d else None,
            "possession_id": d["possession_id"].to_numpy() if "possession_id" in d else None,
            "period": d["period"].to_numpy() if "period" in d else np.nan,
            "actor_id": d[acol].to_numpy() if acol and acol in d.columns else np.nan,
        })
        # a compact human-readable detail string per family.
        # Every field is coerced to str first: a column that happens to be all-NaN
        # for one game arrives as float dtype and would otherwise break concatenation.
        def S(col, default=""):
            if col not in d.columns:
                return np.full(len(d), default, dtype=object)
            return d[col].astype("object").where(d[col].notna(), default).astype(str).to_numpy()

        def B(col):
            if col not in d.columns:
                return np.zeros(len(d), dtype=bool)
            return d[col].fillna(False).astype(bool).to_numpy()

        def N(col, nd=1):
            if col not in d.columns:
                return np.full(len(d), "?", dtype=object)
            v = pd.to_numeric(d[col], errors="coerce").round(nd)
            return v.astype("object").where(v.notna(), "?").astype(str).to_numpy()

        if label == "shot":
            r["detail"] = (np.where(B("outcome"), "MADE ", "MISS ") +
                           np.where(B("three"), "3PT ", "2PT ") + S("region") +
                           " (" + S("contest_level", "?") + ")")
        elif label == "pick":
            r["detail"] = ("bhr def " + S("bhr_def_type", "none") + " / scr def " +
                           S("scr_def_type", "none") + " @ " + S("location_type", "?"))
        elif label == "drive":
            r["detail"] = (S("category", "?") + " \u2192 " + S("end_type", "?") +
                           np.where(B("blowby"), " [blowby]", ""))
        elif label == "pass":
            r["detail"] = np.where(B("complete"), "complete", "TURNOVER")
        elif label == "closeout":
            r["detail"] = ("\u2192 " + S("bhr_action", "?") + ", " +
                           N("start_distance") + "\u2192" + N("end_distance") + " ft")
        elif label == "rebound":
            r["detail"] = np.where(B("defensive"), "defensive", "OFFENSIVE")
        elif label == "foul":
            r["detail"] = S("foul_type")
        elif label == "free_throw":
            r["detail"] = np.where(B("outcome"), "made", "missed")
        elif label == "off_ball_screen":
            r["detail"] = ("cutter def " + S("cutter_def_type", "none") +
                           " / scr def " + S("screener_def_type", "none"))
        elif label == "touch":
            r["detail"] = (N("num_dribbles", 0) + " dr, " + N("touch_time") + "s")
        else:
            r["detail"] = ""
        rows.append(r)
    tl = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if tl.empty:
        return tl
    return tl.sort_values(["frame", "event_table"], kind="stable").reset_index(drop=True)


def build_action_chains(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Ordered sequence of named actions per chance, plus the terminal outcome.

    Produces one row per chance: ``chain`` is a ``>``-joined token string such as
    ``pick > drive > shot``, built from the action markings sorted by frame and
    terminated by the chance outcome. This is the substrate for the event-chain
    explorer.
    """
    frames = []
    for tbl in ACTION_TABLES:
        d = tables.get(tbl)
        if d is None or d.empty:
            continue
        fcol = "frame" if "frame" in d.columns else "start_frame"
        tok = {"picks": "pick", "handoffs": "handoff", "off_ball_screens": "obscreen",
               "drives": "drive", "isolations": "iso", "posts": "post",
               "closeouts": "closeout"}[tbl]
        sub = pd.DataFrame({"game_id": d["game_id"], "chance_id": d["chance_id"],
                            "frame": d[fcol], "token": tok})
        if tbl == "drives":
            sub["token"] = "drive:" + d["end_type"].fillna("?").to_numpy()
        if tbl == "picks":
            sub["token"] = "pick:" + d["bhr_def_type"].fillna("none").to_numpy()
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    acts = pd.concat(frames, ignore_index=True).sort_values(["chance_id", "frame"])

    chain = (acts.groupby(["game_id", "chance_id"])["token"]
                 .apply(lambda s: " > ".join(s)).rename("action_chain").reset_index())
    n_act = acts.groupby(["game_id", "chance_id"]).size().rename("n_actions").reset_index()

    ch = tables["chances"].rename(columns={"id": "chance_id"})
    out = ch[["game_id", "chance_id", "off_team_id", "def_team_id", "outcome",
              "pts_scored", "transition", "quality_index", "usable"]].merge(
        chain, on=["game_id", "chance_id"], how="left").merge(
        n_act, on=["game_id", "chance_id"], how="left")
    out["action_chain"] = out["action_chain"].fillna("(no tracked action)")
    out["n_actions"] = out["n_actions"].fillna(0).astype("int16")
    out["terminal"] = out["outcome"].map(
        lambda o: {"FGM": "made 2", "FGM3": "made 3", "FGX": "missed 2", "FGX3": "missed 3",
                   "TO": "turnover", "FOU": "foul", "VIO": "violation",
                   "Out of Bounds": "out of bounds", "EPD": "end of period",
                   "JMP": "jump ball"}.get(o, str(o)))
    out["full_chain"] = out["action_chain"] + " > " + out["terminal"]
    return out


def chain_efficiency(chains: pd.DataFrame, min_n: int = 10) -> pd.DataFrame:
    """Efficiency of each *action* sequence, with the outcome as the dependent variable.

    Grouping on the full chain including the terminal is circular (a chain that
    ends "made 3" always scores 3). This groups on the action prefix only.
    """
    g = chains.groupby("action_chain")
    out = pd.DataFrame({
        "chances": g.size(),
        "pts_per_chance": g.pts_scored.mean(),
        "shot_rate": g.outcome.apply(lambda s: s.isin(["FGM", "FGM3", "FGX", "FGX3"]).mean()),
        "three_rate": g.outcome.apply(lambda s: s.isin(["FGM3", "FGX3"]).mean()),
        "make_rate": g.outcome.apply(lambda s: s.isin(["FGM", "FGM3"]).mean()),
        "turnover_rate": g.outcome.apply(lambda s: s.eq("TO").mean()),
        "foul_rate": g.outcome.apply(lambda s: s.eq("FOU").mean()),
        "transition_rate": g.transition.mean(),
    }).reset_index()
    return out[out.chances >= min_n].sort_values("pts_per_chance", ascending=False)


def chain_transitions(chains: pd.DataFrame) -> pd.DataFrame:
    """Bigram counts over the action sequence: what tends to follow what."""
    rows = []
    for c_, pts in zip(chains.action_chain, chains.pts_scored):
        toks = [t for t in c_.split(" > ") if t and not t.startswith("(")]
        for a, b in zip(toks, toks[1:]):
            rows.append({"from": a, "to": b, "pts": pts})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return (df.groupby(["from", "to"]).agg(n=("pts", "size"), pts_per_chance=("pts", "mean"))
              .reset_index().sort_values("n", ascending=False))
