"""Phase 15 — run the full data-quality suite and persist the report."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src import config as C
from src.data import loaders
from src.data.quality import check_events, check_tracking

E = C.PROCESSED_DIR / "events"
events = {p.stem: pd.read_parquet(p) for p in E.glob("*.parquet")}
dims = {p.stem: pd.read_parquet(p) for p in (C.PROCESSED_DIR / "dim").glob("*.parquet")}

rep = check_events(events, dims)
tr_reps = []
for gid in loaders.list_game_ids():
    t = pd.read_parquet(C.PROCESSED_DIR / f"tracking_enriched/{gid}.parquet")
    tr_reps.append(check_tracking(t, gid))
rep = pd.concat([rep] + tr_reps, ignore_index=True)
rep.to_parquet(C.DERIVED_DIR / "quality_report.parquet", index=False)

print(rep.severity.value_counts().to_string())
print("\nAnything not OK:")
print(rep[rep.severity != "OK"].sort_values(["severity", "pct"], ascending=[True, False])
        .to_string(index=False, max_colwidth=52))
fails = rep[rep.severity == "FAIL"]
print(f"\nFAILs: {len(fails)}")
sys.exit(1 if len(fails) else 0)
