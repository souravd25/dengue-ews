"""Verify that weather lags are computed per ward and never leak across wards or into the future."""
import pandas as pd
from src.features import load_raw, build_features, add_targets


def _df():
    return add_targets(build_features(load_raw()))


def test_rain_lag_matches_manual_shift_for_one_ward():
    df = _df()
    w = df[df["ward_id"] == "CCC_Ward_01"].reset_index(drop=True)
    for k in (1, 2, 4, 8):
        expected = w["rainfall_mm"].shift(k)
        pd.testing.assert_series_equal(w[f"rain_lag_{k}w"], expected, check_names=False)


def test_lags_do_not_leak_across_wards():
    df = _df()
    first_rows = df.groupby("ward_id").head(8)  # first 8 weeks of every ward
    assert first_rows["rain_lag_8w"].isna().sum() >= df["ward_id"].nunique()
    first_week = df.groupby("ward_id").head(1)
    assert first_week["rain_lag_1w"].isna().all()  # no value borrowed from the previous ward


def test_rolling_sum_uses_only_past_and_current_weeks():
    df = _df()
    w = df[df["ward_id"] == "DNCC_Ward_01"].reset_index(drop=True)
    i = 20
    expected = w.loc[i - 3:i, "rainfall_mm"].sum()
    assert abs(w.loc[i, "rain_sum_4w"] - expected) < 1e-9


def test_target_is_future_cases_only():
    df = _df()
    w = df[df["ward_id"] == "CCC_Ward_01"].reset_index(drop=True)
    assert w.loc[10, "target_3w"] == w.loc[13, "dengue_cases"]


def test_each_ward_has_weekly_spacing():
    df = load_raw()
    gaps = df.groupby("ward_id")["date"].apply(lambda s: s.diff().dt.days.dropna().unique().tolist())
    assert all(g == [7] for g in gaps)
