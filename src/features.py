"""Feature pipeline: per-ward multi-week weather lags + static ward vulnerability.

Mosquito breeding clock: rain -> eggs/larvae (7-14 d) -> viral incubation -> cases,
so we expose rainfall lags of 1..8 weeks and rolling sums, computed PER WARD.
"""
import pandas as pd
import numpy as np

HORIZONS = (2, 3, 4)  # weeks ahead
STATIC = ["population_density_sqkm", "water_logging_risk_index", "sanitation_score"]
WEATHER = ["avg_temp_c", "rainfall_mm", "relative_humidity_pct"]
RAIN_LAGS = list(range(1, 9))


def load_raw(path="data/dengue.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["ward_id", "date"]).reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ward_id", "date"]).copy()
    g = df.groupby("ward_id", sort=False)
    for k in RAIN_LAGS:
        df[f"rain_lag_{k}w"] = g["rainfall_mm"].shift(k)
    for k in (2, 4):
        df[f"temp_lag_{k}w"] = g["avg_temp_c"].shift(k)
        df[f"humidity_lag_{k}w"] = g["relative_humidity_pct"].shift(k)
    # rolling sums (include current week) -> accumulated water in drains/containers
    df["rain_sum_4w"] = g["rainfall_mm"].transform(lambda s: s.rolling(4, min_periods=4).sum())
    df["rain_sum_8w"] = g["rainfall_mm"].transform(lambda s: s.rolling(8, min_periods=8).sum())
    df["temp_mean_4w"] = g["avg_temp_c"].transform(lambda s: s.rolling(4, min_periods=4).mean())
    df["humidity_mean_4w"] = g["relative_humidity_pct"].transform(lambda s: s.rolling(4, min_periods=4).mean())
    # ward vulnerability x water interaction
    df["waterlog_x_rain4w"] = df["water_logging_risk_index"] * df["rain_sum_4w"]
    # seasonality
    woy = df["date"].dt.isocalendar().week.astype(int)
    df["season_sin"] = np.sin(2 * np.pi * woy / 52)
    df["season_cos"] = np.cos(2 * np.pi * woy / 52)
    return df


FEATURES = (
    STATIC + WEATHER
    + [f"rain_lag_{k}w" for k in RAIN_LAGS]
    + ["temp_lag_2w", "temp_lag_4w", "humidity_lag_2w", "humidity_lag_4w"]
    + ["rain_sum_4w", "rain_sum_8w", "temp_mean_4w", "humidity_mean_4w", "waterlog_x_rain4w",
       "season_sin", "season_cos"]
)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("ward_id", sort=False)["dengue_cases"]
    for h in HORIZONS:
        df[f"target_{h}w"] = g.shift(-h)
    return df


def driver_group(name: str) -> str:
    """Human-friendly group for the dashboard driver chart."""
    if name.startswith("rain_lag") or name in ("rain_sum_4w", "rain_sum_8w", "rainfall_mm"):
        return "Rainfall (multi-week lags)"
    if name in ("avg_temp_c", "temp_mean_4w") or name.startswith("temp_lag"):
        return "Temperature"
    if name in ("relative_humidity_pct", "humidity_mean_4w") or name.startswith("humidity_lag"):
        return "Humidity"
    if name in ("water_logging_risk_index", "waterlog_x_rain4w"):
        return "Ward drainage / water-logging"
    if name == "population_density_sqkm":
        return "Population density"
    if name == "sanitation_score":
        return "Sanitation"
    return "Seasonality"
