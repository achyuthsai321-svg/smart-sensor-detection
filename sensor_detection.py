"""
Smart Sensor Detection
======================
Detects anomalies in IoT sensor data using Isolation Forest.
Author: Achyuth Sai V
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_STATE  = 42
CONTAMINATION = 0.07   # Expected anomaly fraction (~7%)
N_SAMPLES     = 1000
OUTPUT_DIR    = "outputs"
MODEL_DIR     = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)


# ── 1. Data Generation (simulates real IoT sensor stream) ─────────────────────
def generate_sensor_data(n_samples: int = N_SAMPLES, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate synthetic multi-sensor time-series data with injected anomalies."""
    rng = np.random.default_rng(seed)

    timestamps  = pd.date_range("2024-01-01", periods=n_samples, freq="1min")
    temperature = 25 + 5 * np.sin(np.linspace(0, 4 * np.pi, n_samples)) + rng.normal(0, 0.8, n_samples)
    humidity    = 60 + 10 * np.cos(np.linspace(0, 4 * np.pi, n_samples)) + rng.normal(0, 1.2, n_samples)
    pressure    = 1013 + 3 * np.sin(np.linspace(0, 2 * np.pi, n_samples)) + rng.normal(0, 0.5, n_samples)
    vibration   = rng.exponential(0.5, n_samples)

    # Inject anomalies
    n_anomalies = int(n_samples * CONTAMINATION)
    anomaly_idx = rng.choice(n_samples, n_anomalies, replace=False)
    temperature[anomaly_idx] += rng.choice([-1, 1], n_anomalies) * rng.uniform(15, 30, n_anomalies)
    humidity[anomaly_idx]    += rng.choice([-1, 1], n_anomalies) * rng.uniform(20, 40, n_anomalies)
    pressure[anomaly_idx]    += rng.choice([-1, 1], n_anomalies) * rng.uniform(10, 25, n_anomalies)
    vibration[anomaly_idx]   *= rng.uniform(5, 15, n_anomalies)

    labels = np.zeros(n_samples, dtype=int)
    labels[anomaly_idx] = 1   # 1 = anomaly, 0 = normal

    df = pd.DataFrame({
        "timestamp":   timestamps,
        "temperature": np.round(temperature, 2),
        "humidity":    np.round(humidity, 2),
        "pressure":    np.round(pressure, 2),
        "vibration":   np.round(vibration, 4),
        "true_label":  labels
    })
    df.to_csv("data/sensor_data.csv", index=False)
    print(f"[✓] Dataset generated: {n_samples} samples, {n_anomalies} anomalies injected.")
    return df


# ── 2. Feature Engineering ─────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling statistics and derived features from raw sensor readings."""
    df = df.copy()
    window = 10

    for col in ["temperature", "humidity", "pressure", "vibration"]:
        df[f"{col}_rolling_mean"] = df[col].rolling(window, min_periods=1).mean()
        df[f"{col}_rolling_std"]  = df[col].rolling(window, min_periods=1).std().fillna(0)
        df[f"{col}_diff"]         = df[col].diff().fillna(0)

    df["temp_humidity_ratio"] = df["temperature"] / (df["humidity"] + 1e-5)
    df["pressure_vibration"]  = df["pressure"] * df["vibration"]

    print(f"[✓] Feature engineering done: {df.shape[1]} total features.")
    return df


# ── 3. Model Training ──────────────────────────────────────────────────────────
def train_model(df: pd.DataFrame):
    """Scale features and train an Isolation Forest anomaly detector."""
    feature_cols = [c for c in df.columns if c not in ["timestamp", "true_label"]]
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_scaled)

    # Isolation Forest: -1 = anomaly → remap to 1; 1 = normal → remap to 0
    raw_preds            = model.predict(X_scaled)
    df["predicted_label"] = np.where(raw_preds == -1, 1, 0)
    df["anomaly_score"]   = -model.score_samples(X_scaled)   # higher = more anomalous

    print("[✓] Model trained successfully.")
    return model, scaler, df, feature_cols


# ── 4. Evaluation ──────────────────────────────────────────────────────────────
def evaluate(df: pd.DataFrame):
    """Print classification report and save confusion matrix."""
    y_true = df["true_label"]
    y_pred = df["predicted_label"]

    print("\n── Classification Report ──────────────────────────")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"], ax=ax)
    ax.set_title("Confusion Matrix — Smart Sensor Detection", fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
    plt.close()
    print(f"[✓] Confusion matrix saved → {OUTPUT_DIR}/confusion_matrix.png")


# ── 5. Visualisation ───────────────────────────────────────────────────────────
def visualise(df: pd.DataFrame):
    """Plot sensor readings with anomalies highlighted."""
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Smart Sensor Detection — Anomaly Analysis", fontsize=14, fontweight="bold", y=1.01)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    sensors = ["temperature", "humidity", "pressure", "vibration"]
    units   = ["°C", "%RH", "hPa", "g"]
    colors  = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    normal  = df[df["predicted_label"] == 0]
    anomaly = df[df["predicted_label"] == 1]

    for i, (sensor, unit, color) in enumerate(zip(sensors, units, colors)):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.plot(df["timestamp"], df[sensor], color=color, alpha=0.5, linewidth=0.8, label="Sensor reading")
        ax.scatter(anomaly["timestamp"], anomaly[sensor],
                   color="red", s=18, zorder=5, label="Anomaly detected", alpha=0.85)
        ax.set_title(f"{sensor.capitalize()} ({unit})", fontweight="bold")
        ax.set_xlabel("Time"); ax.set_ylabel(unit)
        ax.legend(fontsize=7)
        ax.tick_params(axis="x", rotation=30, labelsize=7)

    plt.savefig(f"{OUTPUT_DIR}/sensor_anomalies.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Sensor plot saved → {OUTPUT_DIR}/sensor_anomalies.png")

    # Anomaly score distribution
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.hist(df[df["predicted_label"]==0]["anomaly_score"], bins=40, alpha=0.7, color="#2196F3", label="Normal")
    ax2.hist(df[df["predicted_label"]==1]["anomaly_score"], bins=40, alpha=0.7, color="#F44336", label="Anomaly")
    ax2.axvline(df["anomaly_score"].quantile(1 - CONTAMINATION), color="black", linestyle="--", label="Threshold")
    ax2.set_title("Anomaly Score Distribution", fontweight="bold")
    ax2.set_xlabel("Anomaly Score"); ax2.set_ylabel("Count")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/score_distribution.png", dpi=150)
    plt.close()
    print(f"[✓] Score distribution saved → {OUTPUT_DIR}/score_distribution.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   Smart Sensor Detection — Anomaly Detection System")
    print("=" * 55)

    df                          = generate_sensor_data()
    df                          = engineer_features(df)
    model, scaler, df, feat_cols = train_model(df)
    evaluate(df)
    visualise(df)

    # Save results
    df[["timestamp", "temperature", "humidity", "pressure",
        "vibration", "anomaly_score", "predicted_label"]].to_csv(
        f"{OUTPUT_DIR}/detection_results.csv", index=False)

    print(f"\n[✓] Results saved → {OUTPUT_DIR}/detection_results.csv")
    print("\nDone! All outputs in the 'outputs/' folder.")
