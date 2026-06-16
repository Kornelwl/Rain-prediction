import os

BASE_URL = "https://e6uw49pbah.execute-api.us-east-1.amazonaws.com/dev"
API_TOKEN = os.getenv("WEATHER_API_TOKEN", "STUDENT_TOKEN_2026")

DEFAULT_STATION_ID = "GDN_01"
DEFAULT_LIMIT = 100

RAW_DATA_PATH = "data/raw/weather_raw.json"
CLEAN_DATA_PATH = "data/processed/weather_clean.csv"
FEATURES_DATA_PATH = "data/processed/weather_features.csv"
SCORED_DATA_PATH = "data/output/rain_risk_scored.csv"
EVALUATION_PATH = "data/output/evaluation_metrics.txt"
PLOT_PATH = "data/output/rain_risk_chart.png"
CONFUSION_MATRIX_PATH = "data/output/confusion_matrix.png"