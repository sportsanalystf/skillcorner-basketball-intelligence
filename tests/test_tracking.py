"""Tracking engine: segmentation, smoothing, kinematics."""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.tracking import engine


def _synthetic(n=200, gap_at=None, speed=10.0):
    """A player moving in +x at constant speed, optionally with a frame gap."""
    f = np.arange(n)
    if gap_at is not None:
        f = np.r_[f[:gap_at], f[gap_at:] + 40]
    t = np.arange(len(f)) * C.DT
    return pd.DataFrame({
        "player_id": 1, "frame_idx": f, "x": speed * t, "y": np.zeros(len(f)),
        "speed_raw": speed, "pred_error": 1.0, "is_detected": True,
    })


def test_segments_split_on_frame_gaps():
    df = _synthetic(100, gap_at=50)
    seg = engine.segment_tracks(df)
    assert seg.nunique() == 2
    assert (seg.values[:50] == 0).all() and (seg.values[50:] == 1).all()


def test_segments_split_between_players():
    a = _synthetic(20); b = _synthetic(20); b["player_id"] = 2
    df = pd.concat([a, b], ignore_index=True)
    assert engine.segment_tracks(df).nunique() == 2


def test_constant_velocity_recovered_exactly():
    df = _synthetic(200, speed=12.0)
    df["segment_id"] = engine.segment_tracks(df)
    df = engine.smooth_positions(df)
    out = engine.compute_kinematics(df)
    mid = out.iloc[20:-20]
    assert np.allclose(mid.speed, 12.0, atol=0.05)
    assert np.allclose(mid.accel_tangential, 0.0, atol=0.5)


def test_kinematics_never_cross_a_segment_boundary():
    df = _synthetic(100, gap_at=50)
    df["segment_id"] = engine.segment_tracks(df)
    out = engine.compute_kinematics(engine.smooth_positions(df))
    # the 40-frame jump would give a huge speed if differenced across the gap
    assert out.speed.max() < 20.0


def test_short_segments_yield_nan_not_garbage():
    df = _synthetic(3)
    df["segment_id"] = engine.segment_tracks(df)
    out = engine.compute_kinematics(engine.smooth_positions(df))
    assert out.speed.isna().all()


def test_smoothing_reduces_noise_but_preserves_the_mean():
    rng = np.random.default_rng(0)
    df = _synthetic(400, speed=8.0)
    df["x"] = df.x + rng.normal(0, 0.35, len(df))
    df["y"] = df.y + rng.normal(0, 0.35, len(df))
    df["segment_id"] = engine.segment_tracks(df)
    raw = engine.compute_kinematics(df.assign(x_s=df.x, y_s=df.y))
    sm = engine.compute_kinematics(engine.smooth_positions(df))
    assert sm.speed.std() < raw.speed.std() / 2
    assert abs(sm.speed.mean() - 8.0) < 1.0


def test_speed_bands_partition_the_range():
    sp = np.array([0.0, 1.9, 2.0, 4.9, 5.0, 9.9, 10.0, 14.9, 15.0, 19.9, 20.0, 40.0])
    b = engine.speed_band(sp)
    assert None not in list(b)
    assert b[0] == "standing" and b[-1] == "sprinting"


def test_real_game_kinematics_are_physically_plausible(tracking_sample):
    t = tracking_sample
    sp = t.speed.dropna()
    assert sp.median() < 12
    assert (sp > C.MAX_PLAUSIBLE_SPEED).mean() < 0.001
    assert sp.quantile(0.999) < 30


def test_our_speed_tracks_the_delivered_speed_field(tracking_sample):
    t = tracking_sample.dropna(subset=["speed", "speed_raw"])
    assert t.speed.corr(t.speed_raw) > 0.95


def test_distance_is_the_integral_of_speed(tracking_sample):
    """Per-player distance from step sums must match speed x time within 10%."""
    t = tracking_sample.dropna(subset=["speed"])
    phys = pd.read_parquet(C.DERIVED_DIR / "player_game_physical.parquet")
    p = phys[phys.game_id == int(t.game_id.iloc[0])]
    live = t[t.possession_id.notna()]
    for pid, g in live.groupby("player_id"):
        row = p[p.player_id == pid]
        if row.empty or len(g) < 2000:
            continue
        implied = g.speed.mean() * len(g) * C.DT
        assert abs(implied - row.distance_ft.iloc[0]) / implied < 0.10
        break
