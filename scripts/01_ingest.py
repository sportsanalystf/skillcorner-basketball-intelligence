"""Phase 2 — ingest raw SkillCorner data into canonical Parquet."""
import sys, json, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import pandas as pd
from src import config as C
from src.data import ingest, loaders
from src.data.aliases import collapse_aliases, APPROXIMATED_RATES

t0 = time.time()
print("== dimensions ==")
dims = ingest.build_dimensions()
(C.PROCESSED_DIR / "dim").mkdir(parents=True, exist_ok=True)
for k, v in dims.items():
    v.to_parquet(C.PROCESSED_DIR / "dim" / f"{k}.parquet", index=False)
    print(f"  {k:8s} {v.shape}")

print("== aggregates ==")
(C.PROCESSED_DIR / "aggregates").mkdir(parents=True, exist_ok=True)
alias = ingest.build_alias_map()
for kind in ("shots", "drives", "picks"):
    df = loaders.load_aggregate(kind)
    df["player_id_canonical"] = df["player_id"].map(lambda v: alias.get(v, v))
    df["is_season_total_row"] = df["team_name"].eq("total")
    df.to_parquet(C.PROCESSED_DIR / "aggregates" / f"{kind}.parquet", index=False)
    merged = collapse_aliases(df, kind)
    merged.to_parquet(C.PROCESSED_DIR / "aggregates" / f"{kind}_merged.parquet", index=False)
    print(f"  agg_{kind:7s} {df.shape} -> alias-merged {merged.shape} "
          f"({len(APPROXIMATED_RATES.get(kind, []))} rates weighted-averaged)")

print("== events ==")
ev = ingest.ingest_events()
(C.PROCESSED_DIR / "events").mkdir(parents=True, exist_ok=True)
for tbl, df in ev.items():
    df.to_parquet(C.PROCESSED_DIR / "events" / f"{tbl}.parquet", index=False)
print("  wrote", len(ev), "event tables:",
      {k: len(v) for k, v in sorted(ev.items(), key=lambda x: -len(x[1]))})

print("== tracking ==")
stats = []
for gid in loaders.list_game_ids():
    s = ingest.ingest_tracking_game(gid)
    stats.append(s)
    print(f"  {gid}: {s['frames']:,} frames, {s['player_rows']:,} player rows, "
          f"{s['players_bytes']/1e6:.1f} MB, {s['seconds']}s")
pd.DataFrame(stats).to_parquet(C.PROCESSED_DIR / "tracking_ingest_stats.parquet", index=False)
print(f"TOTAL {time.time()-t0:.0f}s")
