"""
anomaly_detection.py
---------------------
Detects unusual or abnormal log patterns using Isolation Forest.

Input:
    data/combined_dataset.csv

Output:
    data/outputs/anomaly_reports/anomalies_detected.csv
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
INPUT_FILE = "data/failure_patterns_labeled_human.csv"
OUTPUT_DIR = "data/outputs/anomaly_reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print("Loading combined dataset...")
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# Select features for anomaly detection
text_col = "error_msg"

# Numeric metadata features
num_features = [
    "execution_duration",
    "failure_freq_suite",
    "failure_freq_dut",
    "time_since_last_failure"
]

available_num_features = [f for f in num_features if f in df.columns]
print(f"Numeric features used: {available_num_features}")

# Text Vectorization (TF-IDF)
print("Vectorizing log messages (TF-IDF)...")
vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words='english',
    ngram_range=(1, 2)
)
X_text = vectorizer.fit_transform(df[text_col].astype(str))

# Combine text and numeric features
X_num = df[available_num_features].fillna(0)
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

# Combine text + numeric into one feature matrix
from scipy.sparse import hstack
X_combined = hstack([X_text, X_num_scaled])

print(f"Combined feature matrix shape: {X_combined.shape}")

# Isolation Forest Model
print("\nRunning Isolation Forest...")
iso = IsolationForest(
    contamination=0.05,   
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
df["anomaly_score"] = iso.fit_predict(X_combined)
df["anomaly_score"] = iso.decision_function(X_combined)

# Label anomalies
df["is_anomaly"] = (df["anomaly_score"] < np.percentile(df["anomaly_score"], 5)).astype(int)

# Save detected anomalies
anomalies = df[df["is_anomaly"] == 1]
output_csv = os.path.join(OUTPUT_DIR, "anomalies_detected.csv")
anomalies.to_csv(output_csv, index=False)

print(f"Saved {len(anomalies)} anomalies → {output_csv}")

# Visualization
plt.figure(figsize=(8, 5))
sns.histplot(df["anomaly_score"], bins=40, kde=True)
plt.title("Anomaly Score Distribution (Isolation Forest)")
plt.xlabel("Anomaly Score")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "anomaly_score_distribution.png"))
plt.close()

print("Anomaly detection completed successfully.")
print(f"Histogram saved → {OUTPUT_DIR}/anomaly_score_distribution.png")

# Show top anomalies
if len(anomalies) > 0:
    print("\nTop 5 anomalies:")
    print(anomalies[["filename", "status", "error_msg", "anomaly_score"]].head())
else:
    print("No anomalies detected.")
