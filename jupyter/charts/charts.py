import pandas as pd
import matplotlib.pyplot as plt


# Wczytanie danych

scored = pd.read_csv("rain_risk_scored.csv")
metrics = pd.read_csv("evaluation_metrics.csv")

scored["timestamp"] = pd.to_datetime(scored["timestamp"])


# Evaluation metrics chart

metrics_plot = metrics[
    metrics["metric"].isin(["Accuracy", "Precision", "Recall", "F1-score"])
].copy()

metrics_plot["value"] = metrics_plot["value"].astype(float)

plt.figure(figsize=(8, 5))
plt.bar(metrics_plot["metric"], metrics_plot["value"])
plt.ylim(0, 1)
plt.title("Evaluation Metrics")
plt.xlabel("Metric")
plt.ylabel("Value")
plt.grid(axis="y", alpha=0.3)

for i, value in enumerate(metrics_plot["value"]):
    plt.text(i, value + 0.02, f"{value:.2f}", ha="center")

plt.tight_layout()
plt.savefig("evaluation_metrics_chart.png", dpi=300)
plt.show()


#Rain risk score over time

plot_df = scored.sort_values("timestamp")

plt.figure(figsize=(12, 5))
plt.plot(plot_df["timestamp"], plot_df["rain_risk_score"], marker="o")
plt.axhline(y=60, linestyle="--", label="Prediction threshold = 60")

plt.title("Rain Risk Score Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Rain Risk Score")
plt.xticks(rotation=45)
plt.ylim(0, 100)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("rain_risk_chart.png", dpi=300)
plt.show()


#Confusion matrix

tp = int(metrics.loc[metrics["metric"] == "TP", "value"].iloc[0])
tn = int(metrics.loc[metrics["metric"] == "TN", "value"].iloc[0])
fp = int(metrics.loc[metrics["metric"] == "FP", "value"].iloc[0])
fn = int(metrics.loc[metrics["metric"] == "FN", "value"].iloc[0])

confusion_matrix = [
    [tn, fp],
    [fn, tp]
]

plt.figure(figsize=(6, 5))
plt.imshow(confusion_matrix)

plt.title("Confusion Matrix")
plt.xticks([0, 1], ["Predicted no rain", "Predicted rain"])
plt.yticks([0, 1], ["Actual no rain", "Actual rain"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, confusion_matrix[i][j], ha="center", va="center", fontsize=14)

plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.colorbar()
plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=300)
plt.show()


#Risk class distribution

risk_counts = scored["risk_class"].value_counts().reset_index()
risk_counts.columns = ["risk_class", "count"]

plt.figure(figsize=(7, 5))
plt.bar(risk_counts["risk_class"].astype(str), risk_counts["count"])

plt.title("Risk Class Distribution")
plt.xlabel("Risk class")
plt.ylabel("Number of records")
plt.grid(axis="y", alpha=0.3)

for i, value in enumerate(risk_counts["count"]):
    plt.text(i, value + 0.5, str(value), ha="center")

plt.tight_layout()
plt.savefig("risk_class_distribution.png", dpi=300)
plt.show()