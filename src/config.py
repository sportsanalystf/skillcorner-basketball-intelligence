"""Central configuration: paths, court geometry, and pipeline constants.

Every geometric constant here is either (a) taken from the SkillCorner
documentation, or (b) a FIBA court dimension used to draw/segment the court.
Nothing here is inferred from the data itself.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path(os.environ.get("SKC_RAW_DIR", PROJECT_ROOT / "data" / "raw" / "opendata"))
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DERIVED_DIR = PROJECT_ROOT / "data" / "derived"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DUCKDB_PATH = PROCESSED_DIR / "basketball.duckdb"

for _p in (PROCESSED_DIR, DERIVED_DIR, REPORTS_DIR, FIGURES_DIR):
    _p.mkdir(parents=True, exist_ok=True)

MATCHES_DIR = RAW_DIR / "matches"
AGGREGATES_DIR = RAW_DIR / "aggregates"
ALIASES_CSV = RAW_DIR / "player_id_aliases.csv"
MATCHES_JSON = RAW_DIR / "matches.json"

# --------------------------------------------------------------------------
# Tracking / clock constants (from docs/PRIMER.md)
# --------------------------------------------------------------------------
FPS = 25
DT = 1.0 / FPS  # seconds per frame

# --------------------------------------------------------------------------
# Court geometry — FIBA, in FEET, origin at court centre (SkillCorner convention)
#   x runs along the length of the court, y across the width.
#   FIBA court: 28 m x 15 m  ->  91.86 ft x 49.21 ft
# --------------------------------------------------------------------------
M_TO_FT = 3.280839895013123

COURT_LENGTH = 28.0 * M_TO_FT          # 91.86 ft
COURT_WIDTH = 15.0 * M_TO_FT           # 49.21 ft
COURT_X_MAX = COURT_LENGTH / 2.0       # +/- 45.93
COURT_Y_MAX = COURT_WIDTH / 2.0        # +/- 24.61

# Hoop: FIBA backboard 1.20 m from the endline, ring centre 1.575 m from endline
HOOP_INSET = 1.575 * M_TO_FT           # 5.17 ft from the baseline
HOOP_X = COURT_X_MAX - HOOP_INSET      # 40.76 ft (magnitude)
RIM_RADIUS = 0.225 * M_TO_FT           # 0.74 ft

# Three-point line: 6.75 m arc, 6.60 m in the corners, corner line 0.90 m from sideline
THREE_ARC_R = 6.75 * M_TO_FT           # 22.15 ft
THREE_CORNER_Y = COURT_Y_MAX - 0.90 * M_TO_FT   # 21.66 ft
CORNER_LINE_X = HOOP_X - ((THREE_ARC_R ** 2 - THREE_CORNER_Y ** 2) ** 0.5)

# Paint / key: FIBA rectangular key 4.90 m wide, 5.80 m from endline
PAINT_HALF_WIDTH = (4.90 / 2.0) * M_TO_FT       # 8.04 ft
PAINT_LENGTH = 5.80 * M_TO_FT                   # 19.03 ft from the baseline
FT_CIRCLE_R = 1.80 * M_TO_FT                    # 5.91 ft
RESTRICTED_AREA_R = 1.25 * M_TO_FT              # 4.10 ft
BACKBOARD_INSET = 1.20 * M_TO_FT

# --------------------------------------------------------------------------
# Analysis constants
# --------------------------------------------------------------------------
# Savitzky-Golay smoothing window for tracking derivatives (frames).
# 15 frames = 0.6 s: long enough to suppress per-frame jitter from broadcast
# tracking, short enough to preserve a real change of direction.
SMOOTH_WINDOW = 15
SMOOTH_POLYORDER = 2

# Physiological plausibility gates used by the data-quality layer (ft/s, ft/s^2).
# Usain Bolt peaks at ~40 ft/s; elite basketball players rarely exceed ~33 ft/s.
MAX_PLAUSIBLE_SPEED = 36.0
MAX_PLAUSIBLE_ACCEL = 40.0

# Movement-intensity bands. These are OUR interpretable bands, derived from the
# empirical distribution of smoothed player speed in this dataset (see
# docs/METHODOLOGY.md). They are NOT SkillCorner physical-performance categories.
SPEED_BANDS = [
    ("standing", 0.0, 2.0),
    ("walking", 2.0, 5.0),
    ("jogging", 5.0, 10.0),
    ("running", 10.0, 15.0),
    ("high_speed", 15.0, 20.0),
    ("sprinting", 20.0, 1e9),
]

REGIONS = [
    "left corner three", "left corner two", "left wing three", "left wing two",
    "backcourt", "far", "middle three", "middle two", "key", "ra",
    "right wing three", "right wing two", "right corner three", "right corner two",
]

EVENT_TABLES = [
    "possessions", "chances", "chance_players", "matchups", "shots", "free_throws",
    "rebounds", "turnovers", "fouls", "timeouts", "passes", "touches", "dribbles",
    "picks", "handoffs", "off_ball_screens", "drives", "isolations", "posts",
    "closeouts",
]

# Fields holding an [x, y] (or [x, y, z]) coordinate pair, per table. Used by the
# ingest layer to explode list columns into scalar _x / _y columns.
LOCATION_FIELDS = {
    "possessions": ["endLoc"],
    "chances": ["endLoc"],
    "chance_players": ["shotLoc", "rimLoc", "reboundLoc"],
    "shots": ["location"],
    "rebounds": ["location"],
    "turnovers": ["location"],
    "fouls": ["location"],
    "passes": ["passerLoc", "receiverLoc"],
    "touches": ["location", "endLoc", "closestDefLoc", "endClosestDefLoc"],
    "dribbles": ["location"],
    "picks": ["location"],
    "handoffs": ["location"],
    "off_ball_screens": ["location"],
    "drives": ["location", "endLoc"],
    "isolations": ["location", "endLoc"],
    "posts": ["location", "endLoc"],
    "closeouts": ["location", "startLoc", "endLoc", "touchLoc"],
}

# List[str] columns kept as a pipe-joined string for columnar storage.
LIST_STR_FIELDS = {
    "possessions": ["outcome", "offPlayerIds", "defPlayerIds"],
    "chances": ["offPlayerIds", "defPlayerIds"],
    "touches": ["outcomes", "regionsIn"],
    "picks": ["bhrOutcomes", "scrOutcomes"],
    "handoffs": ["bhrOutcomes"],
    "off_ball_screens": ["outcomes", "scrOutcomes"],
    "drives": ["bhrOutcomes"],
    "isolations": ["bhrOutcomes"],
    "posts": ["bhrOutcomes"],
    "closeouts": ["bhrOutcomes"],
}

# Outcome-token vocabulary (docs/data_dictionary/touches.md)
OUTCOME_TOKENS = {
    "PASS": "Pass", "PASS_SCR": "Pass to screener", "FGM": "Field goal made",
    "FGM3": "Made 3", "FGX": "Missed field goal", "FGX3": "Missed 3",
    "BLK": "Blocked shot", "FOUL": "Any foul", "AST": "Assist",
    "AST3": "Assist for 3", "AST2": "Assist for 2", "AST0": "Assist opportunity missed",
    "ASTF": "Assist opportunity, shooter fouled", "TO": "Turnover",
    "FOU_S": "Shooting foul", "FOU_T": "Technical", "FOU_F": "Flagrant",
    "FOU_B": "Foul in the bonus", "FOU_N": "Personal, non-shooting",
    "FOU_O": "Offensive foul", "TMO": "Timeout",
}
