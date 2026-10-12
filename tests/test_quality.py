"""The data-quality system must run clean of FAILs on the shipped data."""
import pandas as pd
import pytest

from src import config as C
from src.data.quality import check_events, check_tracking


@pytest.fixture(scope="module")
def report(events, dims):
    return check_events(events, dims)


def test_no_failing_event_checks(report):
    fails = report[report.severity == "FAIL"]
    assert fails.empty, fails.to_string(index=False)


def test_report_covers_every_event_table(report, events):
    scoped = set(report.scope)
    for t in events:
        assert any(s == t or s.startswith(t + ".") for s in scoped), t


def test_tracking_quality_has_no_failures(tracking_sample):
    r = check_tracking(tracking_sample, int(tracking_sample.game_id.iloc[0]))
    fails = r[r.severity == "FAIL"]
    assert fails.empty, fails.to_string(index=False)


def test_detection_rate_matches_the_published_range():
    tq = pd.read_parquet(C.DERIVED_DIR / "tracking_quality.parquet")
    assert tq.pct_detected.between(70, 90).all()
    assert tq.mean_pred_error.between(1.0, 2.5).all()
    assert tq.mean_pred_error_detected.between(0.5, 1.5).all()
    assert tq.mean_pred_error_extrapolated.between(3.5, 6.5).all()


def test_frames_are_almost_entirely_inside_possessions():
    tq = pd.read_parquet(C.DERIVED_DIR / "tracking_quality.parquet")
    assert (tq.pct_rows_in_possession > 95).all()


def test_no_duplicate_player_rows_per_frame(tracking_sample):
    assert not tracking_sample.duplicated(["frame_idx", "player_id"]).any()
