"""Court geometry, coordinate transforms and zoning."""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.preprocessing import court


def test_court_dimensions_are_fiba():
    assert C.COURT_LENGTH == pytest.approx(91.86, abs=0.01)
    assert C.COURT_WIDTH == pytest.approx(49.21, abs=0.01)
    assert C.HOOP_X == pytest.approx(40.76, abs=0.01)
    assert C.THREE_ARC_R == pytest.approx(22.15, abs=0.01)


def test_broadcast_to_attacking_is_an_involution():
    x = np.array([10.0, -22.5, 0.0, 44.0])
    y = np.array([-3.0, 17.2, 0.0, -20.0])
    lh = np.array([True, False, False, True])
    ax, ay = court.broadcast_to_attacking(x, y, lh)
    bx, by = court.attacking_to_broadcast(ax, ay, lh)
    assert np.allclose(bx, x) and np.allclose(by, y)


def test_left_hoop_true_is_identity_false_is_rotation():
    ax, ay = court.broadcast_to_attacking([10.0], [4.0], [True])
    assert ax[0] == pytest.approx(10.0) and ay[0] == pytest.approx(4.0)
    ax, ay = court.broadcast_to_attacking([10.0], [4.0], [False])
    assert ax[0] == pytest.approx(-10.0) and ay[0] == pytest.approx(-4.0)


def test_transform_matches_the_delivered_event_coordinates(events):
    """The empirical basis for the whole spatial layer, re-verified as a test.

    Join every shot to the shooter's tracking position at its startFrame and
    check the rotation reproduces the delivered event location.
    """
    from src import config as C2
    sh = events["shots"].dropna(subset=["location_x", "start_frame"])
    poss = events["possessions"][["id", "game_id", "left_hoop"]].rename(
        columns={"id": "possession_id"})
    sh = sh.merge(poss, on=["game_id", "possession_id"], how="left")
    checked = 0
    for gid, sub in sh.groupby("game_id"):
        tr = pd.read_parquet(C2.PROCESSED_DIR / f"tracking/players/{gid}.parquet",
                             columns=["frame_idx", "player_id", "x", "y"])
        m = sub.merge(tr, left_on=["start_frame", "shooter_id"],
                      right_on=["frame_idx", "player_id"], how="inner")
        if m.empty:
            continue
        ax, ay = court.broadcast_to_attacking(m.x.to_numpy(), m.y.to_numpy(),
                                              m.left_hoop.to_numpy())
        assert np.nanmedian(np.abs(ax - m.location_x)) < 0.05
        assert np.nanmedian(np.abs(ay - m.location_y)) < 0.05
        checked += len(m)
    assert checked > 1000, f"only {checked} shots verified"


def test_reported_shot_distance_matches_geometry(events):
    sh = events["shots"].dropna(subset=["location_x", "distance"])
    d = court.dist_to_off_hoop(sh.location_x, sh.location_y)
    assert np.nanmedian(np.abs(d - sh.distance)) < 0.05


def test_three_point_classification_agrees_with_delivered_flag(events):
    sh = events["shots"].dropna(subset=["location_x"])
    calc = court.is_three(sh.location_x, sh.location_y)
    agree = (calc == sh.three.to_numpy()).mean()
    assert agree > 0.98, f"only {agree:.3%} agreement on the three-point line"


def test_corner_three_is_a_three_and_close_to_the_baseline():
    x, y = -40.0, -22.5
    assert court.is_three(x, y)
    assert court.in_corner(x, y)
    assert court.shot_zone(np.array([x]), np.array([y]))[0] == "corner_3_right"


def test_restricted_area_and_paint():
    assert court.in_restricted_area(-C.HOOP_X + 2.0, 0.0)
    assert not court.in_restricted_area(-C.HOOP_X + 8.0, 0.0)
    assert court.in_paint(-40.0, 0.0)
    assert not court.in_paint(-40.0, 15.0)


def test_zones_cover_every_shot(events):
    sh = events["shots"].dropna(subset=["location_x"])
    z = court.shot_zone(sh.location_x, sh.location_y)
    assert pd.Series(z).notna().all()
    assert set(pd.unique(z)) <= set(court.ZONES)


def test_zone_families_map_completely():
    assert set(court.ZONES) <= set(court.ZONE_FAMILY)


def test_court_shapes_render():
    s = court.court_shapes(half=False)
    assert len(s) > 15
    assert all("type" in d for d in s)
