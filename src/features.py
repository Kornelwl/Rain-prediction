from pathlib import Path

import pandas as pd

from config import CLEAN_DATA_PATH, FEATURES_DATA_PATH


def create_features(df: pd.DataFrame):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    grouped = df.groupby("station_id", group_keys=False)

    df["humidity_roll_mean_3"] = grouped["humidity"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    df["cloud_cover_roll_mean_3"] = grouped["cloud_cover"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    df["wind_speed_roll_mean_3"] = grouped["wind_speed"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    df["rain_last_3_sum"] = grouped["rain_mm"].rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)

    df["pressure_previous"] = grouped["pressure"].shift(1)
    df["pressure_change"] = df["pressure"] - df["pressure_previous"]

    df["rain_previous"] = grouped["rain_mm"].shift(1).fillna(0)

    df["rain_next"] = grouped["rain_mm"].shift(-1)
    df["will_rain_next"] = (df["rain_next"] > 0).astype(int)

    df = df.dropna(subset=["pressure_previous", "rain_next"]).reset_index(drop=True)

    return df


def run_feature_engineering():
    print("Creating features...")
    df = pd.read_csv(CLEAN_DATA_PATH)
    features_df = create_features(df)

    Path(FEATURES_DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(FEATURES_DATA_PATH, index=False)

    print(f"Feature data saved to {FEATURES_DATA_PATH}")
    print(f"Rows: {len(features_df)}")


if __name__ == "__main__":
    run_feature_engineering()