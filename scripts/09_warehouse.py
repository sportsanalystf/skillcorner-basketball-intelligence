import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
from src.data import warehouse
con, made = warehouse.build()
print(f"built {C.DUCKDB_PATH} with {len(made)} objects")
for q, label in [
    ("SELECT count(*) FROM tracking_players", "tracking rows"),
    ("SELECT count(*) FROM ev_shots", "shots"),
    ("SELECT count(*) FROM d_shots_features", "shot features"),
]:
    t = time.time(); n = con.execute(q).fetchone()[0]
    print(f"  {label:16s} {n:>12,}  ({time.time()-t:.2f}s)")
t = time.time()
r = con.execute("""
  SELECT period, count(*) n, avg(speed) sp FROM tracking_players
  WHERE game_id = 114243 AND speed IS NOT NULL GROUP BY 1 ORDER BY 1""").fetchdf()
print(f"  filtered aggregate query {time.time()-t:.2f}s\n{r.round(3).to_string(index=False)}")
print("\nobjects:", ", ".join(made))
