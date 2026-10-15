# Smart Sensor Detection 🔍

An IoT anomaly detection system that identifies faulty sensor readings using **Isolation Forest** machine learning on multi-sensor time-series data.

## 📌 Overview

This project simulates a real-world IoT sensor stream (temperature, humidity, pressure, vibration) and detects anomalous readings that could indicate sensor failure or abnormal environmental conditions.

## 🧠 Approach

- **Data**: Synthetic multi-sensor time-series with injected anomalies (~7% contamination)
- **Feature Engineering**: Rolling mean, rolling std, diff, and cross-sensor interaction features
- **Model**: Isolation Forest (unsupervised anomaly detection)
- **Evaluation**: Classification report, confusion matrix, anomaly score distribution

## 📁 Project Structure

```
smart-sensor-detection/
│
├── sensor_detection.py     # Main script — run this
├── requirements.txt        # Dependencies
├── data/
│   └── sensor_data.csv     # Auto-generated on first run
└── outputs/
    ├── detection_results.csv     # Predictions with anomaly scores
    ├── sensor_anomalies.png      # Sensor readings with anomalies highlighted
    ├── confusion_matrix.png      # Model evaluation
    └── score_distribution.png   # Anomaly score histogram
```

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/achyuthsai321-svg/smart-sensor-detection.git
cd smart-sensor-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the detection system
python sensor_detection.py
```

## 📊 Sample Output

| Metric    | Normal | Anomaly |
|-----------|--------|---------|
| Precision | ~0.98  | ~0.91   |
| Recall    | ~0.99  | ~0.85   |
| F1-Score  | ~0.98  | ~0.88   |

## 🛠️ Tech Stack

- **Python 3.10+**
- **scikit-learn** — Isolation Forest model
- **Pandas / NumPy** — Data processing & feature engineering
- **Matplotlib / Seaborn** — Visualisation

## 👤 Author

**Achyuth Sai V**  
B.Tech CSE-CYBER SECURITY — SRM Institute of Science and Technology  
📧 achyuthsai321@gmail.com  
🔗 [GitHub](https://github.com/achyuthsai321-svg)
