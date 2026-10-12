"""Cached data access for the app.

Rules that keep the app responsive:
* Derived Parquet tables are small (< 12 MB total) and are loaded whole, cached.
* The 8.3 M-row tracking table is NEVER loaded whole. Frame windows are read
  through PyArrow row-group filters; the animation layer reads the 5 fps
  ``tracking_viz`` copy instead of the full-rate store.
* DuckDB is used for anything that needs a join across the big tables.
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402

P = C.PROCESSED_DIR
D = C.DERIVED_DIR


@st.cache_data(show_spinner=False)
def load(name: str) -> pd.DataFrame:
    """Load a derived / processed table by short name."""
    for cand in (D / f"{name}.parquet", P / f"{name}.parquet",
                 P / "events" / f"{name}.parquet", P / "dim" / f"{name}.parquet",
                 P / "aggregates" / f"{name}.parquet"):
        if cand.exists():
            return pd.read_parquet(cand)
    raise FileNotFoundError(name)


@st.cache_data(show_spinner=False)
def load_json(name: str):
    return json.loads((D / f"{name}.json").read_text())


@st.cache_resource(show_spinner=False)
def duck():
    import duckdb
    return duckdb.connect(str(C.DUCKDB_PATH), read_only=True)


@st.cache_data(show_spinner=False)
def games() -> pd.DataFrame:
    g = load("games").copy()
    g["label"] = (pd.to_datetime(g.date).dt.strftime("%Y-%m-%d") + "  ·  " +
                  g.home_team_name + " " + g.home_score.astype(str) + "–" +
                  g.away_score.astype(str) + " " + g.away_team_name)
    return g.sort_values("date")


@st.cache_data(show_spinner=False)
def teams() -> dict[int, str]:
    t = load("teams")
    return dict(zip(t.team_id, t.team_name))


@st.cache_data(show_spinner=False)
def players() -> pd.DataFrame:
    r = load("roster")
    return (r.sort_values("game_id")
             .drop_duplicates("player_id_canonical", keep="last")
             [["player_id_canonical", "player_name", "team_id", "team_name", "jersey"]])


@st.cache_data(show_spinner=False)
def player_name_map() -> dict:
    p = players()
    return dict(zip(p.player_id_canonical, p.player_name))


@st.cache_data(show_spinner=False)
def tracking_window(game_id: int, f0: int, f1: int, viz: bool = True) -> pd.DataFrame:
    """Player tracking between two frames. ``viz=True`` reads the 5 fps copy."""
    sub = "tracking_viz" if viz else "tracking_enriched"
    path = P / sub / f"{game_id}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path, filters=[("frame_idx", ">=", int(f0)),
                                        ("frame_idx", "<=", int(f1))])
    return df.sort_values(["frame_idx", "team_id", "player_id"])


@st.cache_data(show_spinner=False)
def ball_window(game_id: int, f0: int, f1: int) -> pd.DataFrame:
    path = P / "tracking" / "ball" / f"{game_id}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path, columns=["frame_idx", "x", "y", "z", "speed_raw",
                                          "is_detected", "has_ball_track"],
                           filters=[("frame_idx", ">=", int(f0)),
                                    ("frame_idx", "<=", int(f1))])


@st.cache_data(show_spinner=False)
def timeline(game_id: int) -> pd.DataFrame:
    from src.possessions import engine as PoE
    T = {p.stem: pd.read_parquet(p) for p in (P / "events").glob("*.parquet")}
    return PoE.build_event_timeline(T, int(game_id))


@st.cache_data(show_spinner=False)
def events(name: str) -> pd.DataFrame:
    return pd.read_parquet(P / "events" / f"{name}.parquet")


def fmt_clock(seconds: float) -> str:
    if pd.isna(seconds):
        return "--:--"
    s = max(float(seconds), 0.0)
    return f"{int(s // 60)}:{int(s % 60):02d}"
