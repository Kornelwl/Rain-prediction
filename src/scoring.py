from pathlib import Path

import numpy as np
import pandas as pd

from config import FEATURES_DATA_PATH, SCORED_DATA_PATH


def normalize_0_100(series):
    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series([50] * len(series), index=series.index)

    return 100 * (series - min_value) / (max_value - min_value)


def calculate_rain_risk_score(df: pd.DataFrame):
    df = df.copy()

    df["humidity_score"] = df["humidity_roll_mean_3"].clip(0, 100)

    df["cloud_score"] = df["cloud_cover_roll_mean_3"].clip(0, 100)

    pressure_drop = -df["pressure_change"]
    pressure_drop = pressure_drop.clip(lower=0)
    df["pressure_drop_score"] = normalize_0_100(pressure_drop)

    df["previous_rain_score"] = np.where(df["rain_previous"] > 0, 100, 0)

    df["wind_score"] = normalize_0_100(df["wind_speed_roll_mean_3"])

    df["rain_risk_score"] = (
        0.30 * df["humidity_score"]
        + 0.25 * df["cloud_score"]
        + 0.20 * df["pressure_drop_score"]
        + 0.15 * df["previous_rain_score"]
        + 0.10 * df["wind_score"]
    )

    df["rain_risk_score"] = df["rain_risk_score"].round(2)

    df["risk_class"] = pd.cut(
        df["rain_risk_score"],
        bins=[-1, 33, 66, 100],
        labels=["low", "medium", "high"]
    )

    df["predicted_rain_rule"] = (df["rain_risk_score"] >= 60).astype(int)

    return df


def run_scoring():
    print("Calculating rain risk score...")
    df = pd.read_csv(FEATURES_DATA_PATH)
    scored_df = calculate_rain_risk_score(df)

    Path(SCORED_DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(SCORED_DATA_PATH, index=False)

    print(f"Scored data saved to {SCORED_DATA_PATH}")


if __name__ == "__main__":
    run_scoring()