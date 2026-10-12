"""Schema, identity and referential-integrity checks on the canonical model."""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.data import loaders
from src.data.ingest import to_snake


def test_all_ten_games_present(dims):
    assert len(dims["games"]) == 10
    assert dims["games"].game_id.nunique() == 10


def test_every_event_table_materialised(events):
    missing = set(C.EVENT_TABLES) - set(events)
    assert not missing, f"missing event tables: {missing}"


@pytest.mark.parametrize("tbl", C.EVENT_TABLES)
def test_primary_key_unique(events, tbl):
    df = events[tbl]
    assert df.id.is_unique, f"{tbl}.id is not unique"
    assert df.id.notna().all()


@pytest.mark.parametrize("tbl", ["shots", "drives", "picks", "touches", "passes",
                                 "closeouts", "rebounds", "turnovers", "fouls",
                                 "chance_players", "matchups"])
def test_chance_foreign_key_resolves(events, tbl):
    ids = set(events["chances"].id)
    child = events[tbl].chance_id.dropna()
    orphans = set(child) - ids
    assert not orphans, f"{tbl}: {len(orphans)} chance_ids with no parent chance"


def test_possession_foreign_key_resolves(events):
    ids = set(events["possessions"].id)
    orphans = set(events["chances"].possession_id.dropna()) - ids
    assert not orphans


def test_touch_foreign_key_orphan_rate_within_tolerance(events):
    """`touchId` orphans are real but rare marking noise, not a pipeline bug.

    Measured over the 10 published games: 7 orphaned references out of 24,487
    (0.029 %) — 1 drive, 5 passes, 1 dribble. Documented in docs/LIMITATIONS.md;
    the touch-context joins are left joins, so these rows keep NULL touch context
    rather than being dropped. The gate below fails if that rate ever grows.
    """
    ids = set(events["touches"].id)
    total = bad = 0
    per_table = {}
    for tbl in ("shots", "drives", "picks", "closeouts", "isolations", "posts",
                "handoffs", "passes", "dribbles", "rebounds", "turnovers", "fouls"):
        child = events[tbl].touch_id.dropna()
        o = len(set(child) - ids)
        per_table[tbl] = o
        bad += o
        total += child.nunique()
    rate = 100 * bad / total
    assert rate < 0.10, f"touch_id orphan rate {rate:.3f}% too high: {per_table}"


def test_player_ids_join_to_roster(events, dims):
    roster = set(dims["roster"].player_id)
    for tbl, col in [("shots", "shooter_id"), ("drives", "ballhandler_id"),
                     ("picks", "screener_id"), ("touches", "player_id"),
                     ("matchups", "def_player_id")]:
        vals = events[tbl][col].dropna().astype("int64")
        assert set(vals) <= roster, f"{tbl}.{col} has ids not on any roster"


def test_alias_canonicalisation_applied(events, dims):
    al = loaders.load_player_aliases()
    dupes = set(al.loc[al.player_id != al.canonical_player_id, "player_id"])
    sh = events["shots"]
    # any raw id that has an alias must map to its canonical id
    mapped = sh.loc[sh.shooter_id.isin(dupes), ["shooter_id", "shooter_id_canonical"]]
    if len(mapped):
        m = dict(zip(al.player_id, al.canonical_player_id))
        assert (mapped.shooter_id.map(m) == mapped.shooter_id_canonical).all()


def test_no_duplicate_players_in_merged_aggregates():
    for kind in ("shots", "drives", "picks"):
        df = pd.read_parquet(C.PROCESSED_DIR / f"aggregates/{kind}_merged.parquet")
        per_team = df[~df.is_season_total_row]
        dup = per_team.groupby(["player_id_canonical", "team_id"]).size()
        assert (dup <= 1).all(), f"{kind}: duplicate player-team rows remain"


def test_alias_merge_preserves_totals():
    for kind, col in (("shots", "attempts"), ("drives", "total_drives"),
                      ("picks", "handler_total_picks")):
        raw = pd.read_parquet(C.PROCESSED_DIR / f"aggregates/{kind}.parquet")
        mrg = pd.read_parquet(C.PROCESSED_DIR / f"aggregates/{kind}_merged.parquet")
        a = raw[~raw.is_season_total_row][col].sum()
        b = mrg[~mrg.is_season_total_row][col].sum()
        assert a == b, f"{kind}.{col}: {a} != {b} after alias merge"


def test_snake_case_conversion():
    assert to_snake("ballhandlerDefId") == "ballhandler_def_id"
    assert to_snake("gameId") == "game_id"
    assert to_snake("id") == "id"
    assert to_snake("closestDefDist") == "closest_def_dist"


def test_location_lists_exploded(events):
    for tbl, flds in C.LOCATION_FIELDS.items():
        if tbl not in events:
            continue
        for f in flds:
            base = to_snake(f)
            assert f"{base}_x" in events[tbl].columns, f"{tbl}.{base}_x missing"
            assert f"{base}_y" in events[tbl].columns


def test_known_type_quirk_touch_wall_clock_is_numeric(events):
    col = events["closeouts"]["touch_wall_clock"]
    assert pd.api.types.is_numeric_dtype(col)
