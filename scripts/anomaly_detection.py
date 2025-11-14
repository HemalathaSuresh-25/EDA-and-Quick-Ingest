"""
anomaly_detection.py
---------------------
Detects unusual or abnormal log patterns using Isolation Forest
+ PCA / t-SNE visualization
+ Metadata highlighting (TestCase, DUT, Suite)
+ Saves model & vectorizers
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack

INPUT_FILE = "data/failure_patterns_labeled_human.csv"
OUTPUT_DIR = "data/outputs/anomaly_reports"
MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("STARTING ANOMALY DETECTION (Isolation Forest)")


# LOAD DATA
print("Loading dataset...")
df = pd.read_csv(INPUT_FILE)
print(f"✔ Loaded {df.shape[0]} rows, {df.shape[1]} columns\n")

# SELECT COLUMNS
TEXT_COL = "error_msg"

NUMERIC_FEATURES = [
    "execution_duration",
    "failure_freq_suite",
    "failure_freq_dut",
    "time_since_last_failure"
]

available_num = [c for c in NUMERIC_FEATURES if c in df.columns]
print(f" Using numeric features: {available_num}")

META_COLS = [c for c in ["TestCase", "DUT", "Suite"] if c in df.columns]
print(f" Metadata columns found: {META_COLS}\n")

print(" Vectorizing log messages (TF-IDF)...")

vectorizer = TfidfVectorizer(
    max_features=2000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_text = vectorizer.fit_transform(df[TEXT_COL].astype(str))
print(f" TF-IDF shape: {X_text.shape}")

X_num = df[available_num].fillna(0)
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

# Combine features
X = hstack([X_text, X_num_scaled])
print(f" Combined feature matrix: {X.shape}\n")


print(" Running Isolation Forest...")

iso = IsolationForest(
    contamination=0.05,
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
iso.fit(X)

df["anomaly_score"] = iso.decision_function(X)
threshold = np.percentile(df["anomaly_score"], 5)
df["is_anomaly"] = (df["anomaly_score"] < threshold).astype(int)

print(f" Anomaly threshold score: {threshold:.5f}")
print(f" Total anomalies found: {df['is_anomaly'].sum()}/{len(df)}\n")

print(" Saving model + vectorizer + scaler...")

joblib.dump(iso, os.path.join(MODEL_DIR, "isolation_forest.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_anomaly.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_anomaly.pkl"))

print(f" Saved in folder: {MODEL_DIR}\n")

df["LogID"] = df.index + 1
df["MessageSnippet"] = df[TEXT_COL].str[:150] + "..."

full_out = os.path.join(OUTPUT_DIR, "anomaly_full_scores.csv")
anom_out = os.path.join(OUTPUT_DIR, "anomalies_detected.csv")

df.to_csv(full_out, index=False)

columns_to_save = ["LogID", "MessageSnippet", "anomaly_score", "is_anomaly"] + META_COLS
df[df["is_anomaly"] == 1][columns_to_save].to_csv(anom_out, index=False)

print(f" Saved ALL scores → {full_out}")
print(f" Saved ANOMALIES ONLY → {anom_out}\n")

plt.figure(figsize=(8,5))
sns.histplot(df["anomaly_score"], bins=50, kde=True)
plt.title("Anomaly Score Distribution")
plt.xlabel("Anomaly Score")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "anomaly_score_distribution.png"))
plt.close()

print(" Saved: anomaly_score_distribution.png")

print("\n Generating PCA scatter plot...")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

df["pca_x"], df["pca_y"] = X_pca[:, 0], X_pca[:, 1]

plt.figure(figsize=(10,6))
sns.scatterplot(
    data=df,
    x="pca_x", y="pca_y",
    hue="is_anomaly",
    palette={0: "blue", 1: "red"},
    alpha=0.6
)

plt.title("PCA Scatter Plot (Anomalies Highlighted)")
plt.savefig(os.path.join(OUTPUT_DIR, "pca_anomaly_scatter.png"))
plt.close()

print(" Saved: pca_anomaly_scatter.png")

print("Running t-SNE (this may take time)...")

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
X_tsne = tsne.fit_transform(X.toarray())

df["tsne_x"], df["tsne_y"] = X_tsne[:, 0], X_tsne[:, 1]

plt.figure(figsize=(10,6))
sns.scatterplot(
    data=df,
    x="tsne_x", y="tsne_y",
    hue="is_anomaly",
    palette={0: "blue", 1: "red"},
    alpha=0.6
)

plt.title("t-SNE Scatter Plot (Anomalies Highlighted)")
plt.savefig(os.path.join(OUTPUT_DIR, "tsne_anomaly_scatter.png"))
plt.close()

print("Saved: tsne_anomaly_scatter.png")
print(" Anomaly Detection Completed Successfully!")

