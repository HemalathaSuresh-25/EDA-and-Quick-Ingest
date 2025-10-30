"""
anomaly_detection.py
---------------------
Detects unusual or abnormal log patterns using Isolation Forest.

Input:
    data/failure_patterns_labeled_human.csv

Output:
    data/outputs/anomaly_reports/anomalies_detected.csv
    data/outputs/anomaly_reports/anomaly_score_distribution.png
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack

# Configuration
INPUT_FILE = "data/failure_patterns_labeled_human.csv"
OUTPUT_DIR = "data/outputs/anomaly_reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print("📂 Loading combined dataset...")
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# Feature setup
text_col = "error_msg"
num_features = [
    "execution_duration",
    "failure_freq_suite",
    "failure_freq_dut",
    "time_since_last_failure"
]
available_num_features = [f for f in num_features if f in df.columns]
print(f"📊 Numeric features used: {available_num_features}")

# TF-IDF Vectorization
print("🧠 Vectorizing log messages (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
X_text = vectorizer.fit_transform(df[text_col].astype(str))

# Scale numeric features
X_num = df[available_num_features].fillna(0)
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

# Combine text + numeric
X_combined = hstack([X_text, X_num_scaled])
print(f"✅ Combined feature matrix shape: {X_combined.shape}")

# Isolation Forest
print("\n🚀 Running Isolation Forest...")
iso = IsolationForest(
    contamination=0.05,
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
iso.fit(X_combined)

# Get anomaly scores and predictions
df["anomaly_score"] = iso.decision_function(X_combined)
df["is_anomaly"] = (df["anomaly_score"] < np.percentile(df["anomaly_score"], 5)).astype(int)

# Create output summary
anomalies = df[df["is_anomaly"] == 1].copy()

# Create readable fields
anomalies["LogID"] = anomalies.index + 1
anomalies["MessageSnippet"] = anomalies[text_col].astype(str).str.slice(0, 120) + "..."
output_csv = os.path.join(OUTPUT_DIR, "anomalies_detected.csv")

# Save only relevant columns
anomalies[["LogID", "MessageSnippet", "anomaly_score"]].to_csv(output_csv, index=False)

print(f"\n💾 Saved {len(anomalies)} anomalies → {output_csv}")

# Visualization
plt.figure(figsize=(8, 5))
sns.histplot(df["anomaly_score"], bins=40, kde=True)
plt.title("Anomaly Score Distribution (Isolation Forest)")
plt.xlabel("Anomaly Score")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "anomaly_score_distribution.png"))
plt.close()

print("✅ Anomaly detection completed successfully.")
print(f"📊 Histogram saved → {OUTPUT_DIR}/anomaly_score_distribution.png")

# Show top anomalies
if len(anomalies) > 0:
    print("\n🔎 Top 5 anomalies:")
    print(anomalies[["LogID", "MessageSnippet", "anomaly_score"]].head())
else:
    print("No anomalies detected.")
