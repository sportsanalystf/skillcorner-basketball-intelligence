"""Reconnaissance: profile the real SkillCorner open-data schemas (no assumptions)."""
import json, gzip, os, sys, collections
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("reports")
OUT.mkdir(parents=True, exist_ok=True)

games = sorted(p.name for p in (RAW / "matches").iterdir() if p.is_dir())
profile = {"games": games, "tables": {}}

def typename(v):
    if v is None: return "null"
    if isinstance(v, bool): return "bool"
    if isinstance(v, int): return "int"
    if isinstance(v, float): return "float"
    if isinstance(v, str): return "str"
    if isinstance(v, list): return "list"
    if isinstance(v, dict): return "dict"
    return type(v).__name__

agg = collections.defaultdict(lambda: {"rows": 0, "fields": collections.defaultdict(lambda: {"types": collections.Counter(), "nulls": 0, "examples": []})})
enum_vals = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))

for g in games:
    de = json.loads((RAW / "matches" / g / f"{g}_dynamic_events.json").read_text())
    for tbl, rows in de.items():
        A = agg[tbl]
        A["rows"] += len(rows)
        for r in rows:
            for k, v in r.items():
                F = A["fields"][k]
                F["types"][typename(v)] += 1
                if v is None: F["nulls"] += 1
                elif len(F["examples"]) < 3 and typename(v) in ("str", "list", "dict"):
                    F["examples"].append(v)
                if isinstance(v, str) and len(v) < 30 and not v.startswith(("2025", "2026")):
                    enum_vals[tbl][k][v] += 1
                if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                    for x in v: enum_vals[tbl][k][x] += 1

for tbl, A in sorted(agg.items()):
    profile["tables"][tbl] = {
        "rows_10_games": A["rows"],
        "fields": {k: {"types": dict(v["types"]), "null_pct": round(100*v["nulls"]/max(A["rows"],1), 2),
                       "examples": v["examples"][:2]} for k, v in A["fields"].items()},
    }
    # enums: keep low-cardinality string fields
    en = {}
    for k, c in enum_vals[tbl].items():
        if 0 < len(c) <= 40:
            en[k] = dict(c.most_common(40))
    profile["tables"][tbl]["enums"] = en

(OUT / "schema_profile.json").write_text(json.dumps(profile, indent=1))
print("TABLES / rows over 10 games")
for t, d in sorted(profile["tables"].items(), key=lambda x: -x[1]["rows_10_games"]):
    print(f"  {t:18s} {d['rows_10_games']:8d}  fields={len(d['fields'])}")
