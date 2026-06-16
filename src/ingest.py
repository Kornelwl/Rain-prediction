import json
import requests
from pathlib import Path

from config import BASE_URL, API_TOKEN, DEFAULT_STATION_ID, DEFAULT_LIMIT, RAW_DATA_PATH


def fetch_weather_batch(station_id: str = DEFAULT_STATION_ID, limit: int = DEFAULT_LIMIT):
    url = f"{BASE_URL}/weather/batch"
    headers = {
        "Authorization": f"Bearer {API_TOKEN}"
    }
    params = {
        "station_id": station_id,
        "limit": limit
    }

    response = requests.get(url, headers=headers, params=params, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"API request failed. Status: {response.status_code}, Body: {response.text}"
        )

    return response.json()


def save_raw_data(data, path: str = RAW_DATA_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def run_ingestion():
    print("Fetching data from Weather REST API...")
    data = fetch_weather_batch()
    save_raw_data(data)
    print(f"Raw data saved to {RAW_DATA_PATH}")


if __name__ == "__main__":
    run_ingestion()