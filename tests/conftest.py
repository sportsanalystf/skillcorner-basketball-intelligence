import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402


@pytest.fixture(scope="session")
def events():
    return {p.stem: pd.read_parquet(p)
            for p in (C.PROCESSED_DIR / "events").glob("*.parquet")}


@pytest.fixture(scope="session")
def derived():
    return {p.stem: pd.read_parquet(p) for p in C.DERIVED_DIR.glob("*.parquet")}


@pytest.fixture(scope="session")
def dims():
    return {p.stem: pd.read_parquet(p) for p in (C.PROCESSED_DIR / "dim").glob("*.parquet")}


@pytest.fixture(scope="session")
def tracking_sample():
    p = sorted((C.PROCESSED_DIR / "tracking_enriched").glob("*.parquet"))[0]
    return pd.read_parquet(p)
