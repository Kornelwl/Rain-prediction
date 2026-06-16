from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from config import (
    SCORED_DATA_PATH,
    EVALUATION_PATH,
    PLOT_PATH,
    CONFUSION_MATRIX_PATH,
)


def evaluate_predictions(df: pd.DataFrame):
    y_true = df["will_rain_next"]
    y_pred = df["predicted_rain_rule"]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    return metrics


def save_metrics(metrics, path: str = EVALUATION_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        file.write("Rain Risk Scoring Evaluation\n")
        file.write("============================\n\n")

        for name, value in metrics.items():
            file.write(f"{name}: {value:.4f}\n")


def create_risk_chart(df: pd.DataFrame):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    plt.figure(figsize=(12, 6))
    plt.plot(df["timestamp"], df["rain_risk_score"], marker="o", label="Rain risk score")
    plt.axhline(60, linestyle="--", label="Rain prediction threshold")
    plt.title("Rain Risk Score Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Rain Risk Score 0-100")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()


def create_confusion_matrix_chart(df: pd.DataFrame):
    y_true = df["will_rain_next"]
    y_pred = df["predicted_rain_rule"]

    matrix = confusion_matrix(y_true, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["No rain", "Rain"]
    )

    display.plot()
    plt.title("Confusion Matrix - Rain Prediction")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()


def run_evaluation():
    print("Evaluating predictions...")
    df = pd.read_csv(SCORED_DATA_PATH)

    metrics = evaluate_predictions(df)
    save_metrics(metrics)
    create_risk_chart(df)
    create_confusion_matrix_chart(df)

    print(f"Evaluation saved to {EVALUATION_PATH}")
    print(f"Chart saved to {PLOT_PATH}")
    print(f"Confusion matrix saved to {CONFUSION_MATRIX_PATH}")

    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    run_evaluation()