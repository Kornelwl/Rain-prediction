import json
from pathlib import Path

import pandas as pd

from config import RAW_DATA_PATH, CLEAN_DATA_PATH


def unwrap_api_response(data):
    """
    Handles different possible API response formats:
    - list of measurements
    - {"items": [...]}
    - {"data": [...]}
    - {"measurements": [...]}
    - {"body": "[...]"} from AWS API Gateway
    - {"body": {"items": [...]}}
    """

    if isinstance(data, dict) and "body" in data:
        body = data["body"]

        if isinstance(body, str):
            try:
                body = json.loads(body)
            except json.JSONDecodeError:
                raise ValueError("API response has 'body', but body is not valid JSON.")

        data = body

    if isinstance(data, dict):
        possible_keys = [
            "items",
            "data",
            "measurements",
            "records",
            "results",
            "weather",
        ]

        for key in possible_keys:
            if key in data:
                return data[key]

        # If still dict, maybe it is a single measurement
        return [data]

    if isinstance(data, list):
        return data

    raise ValueError(f"Unsupported API response format: {type(data)}")


def load_raw_data(path: str = RAW_DATA_PATH):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return unwrap_api_response(data)


def normalize_column_names(df: pd.DataFrame):
    """
    Makes column names safer in case API returns slightly different naming.
    """

    df = df.copy()

    rename_map = {
        "time": "timestamp",
        "datetime": "timestamp",
        "date_time": "timestamp",
        "station": "station_id",
        "stationId": "station_id",
        "temperature_c": "temperature",
        "temp": "temperature",
        "hum": "humidity",
        "press": "pressure",
        "windSpeed": "wind_speed",
        "windDirection": "wind_direction",
        "rain": "rain_mm",
        "rainfall": "rain_mm",
        "cloudCover": "cloud_cover",
        "clouds": "cloud_cover",
    }

    df = df.rename(columns=rename_map)

    return df


def clean_weather_data(data):
    df = pd.DataFrame(data)
    df = normalize_column_names(df)

    print("Columns received from API:")
    print(list(df.columns))

    print("\nFirst rows received from API:")
    print(df.head())

    required_columns = [
        "timestamp",
        "station_id",
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "wind_direction",
        "rain_mm",
        "cloud_cover",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns from API response: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
            f"Check data/raw/weather_raw.json to see the exact API format."
        )

    df = df[required_columns].copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    numeric_columns = [
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "wind_direction",
        "rain_mm",
        "cloud_cover",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "station_id"])

    df = df.drop_duplicates(subset=["timestamp", "station_id"])

    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    df["humidity"] = df["humidity"].clip(lower=0, upper=100)
    df["cloud_cover"] = df["cloud_cover"].clip(lower=0, upper=100)
    df["rain_mm"] = df["rain_mm"].clip(lower=0)
    df["wind_speed"] = df["wind_speed"].clip(lower=0)

    return df


def save_clean_data(df, path: str = CLEAN_DATA_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_processing():
    print("Processing raw weather data...")
    data = load_raw_data()
    df = clean_weather_data(data)
    save_clean_data(df)
    print(f"Clean data saved to {CLEAN_DATA_PATH}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    run_processing()