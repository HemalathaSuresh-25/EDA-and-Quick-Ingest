#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Visualization Dashboard
-------------------------------
Generates:

1. Confusion Matrix
2. Anomaly Score Histogram
3. TF-IDF + PCA Scatter
4. TF-IDF + t-SNE Scatter
5. PCA Anomaly Plot with DUT + Suite labels

Input Files:
- data/outputs/model_rf/classified_logs.csv
- data/outputs/anomaly_reports/anomaly_full_scores.csv

Outputs:
- data/outputs/visuals/*.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer


# ------------------------------------------------------
# Paths
# ------------------------------------------------------
CLASSIFIED = "data/outputs/model_rf/classified_logs.csv"
ANOMALY = "data/outputs/anomaly_reports/anomaly_full_scores.csv"
OUTPUT_DIR = "data/outputs/visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------
# Load Data
# ------------------------------------------------------
df_class = pd.read_csv(CLASSIFIED)
df_anom = pd.read_csv(ANOMALY)

print("Loaded classified logs:", df_class.shape)
print("Loaded anomaly scores:", df_anom.shape)

# Auto-Detect Column Names (uppercase/lowercase)
# ------------------------------------------------------
def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of the expected columns found. Available columns: {df.columns.tolist()}")

actual_candidates = [
    "ActualStatus", "actualstatus", "actual", "Actual",
    "GroundTruth", "Label", "ActualLabel", "actual_label"
]

pred_candidates = [
    "PredictedLabel", "predictedlabel", "predicted",
    "Prediction", "PredictedClass", "predicted_label"
]

ACTUAL_COL = find_column(df_class, actual_candidates)
PRED_COL = find_column(df_class, pred_candidates)

print("Detected Actual Column:", ACTUAL_COL)
print("Detected Predicted Column:", PRED_COL)


# ------------------------------------------------------
# 1. Confusion Matrix
# ------------------------------------------------------
plt.figure(figsize=(7, 6))
cm = confusion_matrix(df_class[ACTUAL_COL], df_class[PRED_COL])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Classification Results")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=300)
plt.close()

print("Saved: confusion_matrix.png")


# ------------------------------------------------------
# 2. Anomaly Score Histogram
# ------------------------------------------------------
score_col = None
for col in ["anomaly_score", "score", "anomalyScore"]:
    if col in df_anom.columns:
        score_col = col
        break

if score_col is None:
    raise ValueError("Anomaly score column not found!")

plt.figure(figsize=(8, 5))
sns.histplot(df_anom[score_col], bins=40, kde=True)
plt.title("Anomaly Score Distribution")
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/anomaly_score_histogram.png", dpi=300)
plt.close()

print("Saved: anomaly_score_histogram.png")


# ------------------------------------------------------
# 3. TF-IDF Vectorization
# ------------------------------------------------------
TEXT_COLUMN = None
for col in ["MessageSnippet", "log_message", "message", "msg"]:
    if col in df_anom.columns:
        TEXT_COLUMN = col
        break

if TEXT_COLUMN is None:
    raise ValueError("MessageSnippet column not found in anomaly file!")

print("Generating TF-IDF embeddings from:", TEXT_COLUMN)

vectorizer = TfidfVectorizer(max_features=300, stop_words="english")
X = vectorizer.fit_transform(df_anom[TEXT_COLUMN].fillna("")).toarray()

print("TF-IDF shape:", X.shape)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ------------------------------------------------------
# 4. PCA Scatter
# ------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(X_scaled)
df_anom["pca_1"] = pca_result[:, 0]
df_anom["pca_2"] = pca_result[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_anom,
    x="pca_1",
    y="pca_2",
    hue="is_anomaly",
    palette={0: "blue", 1: "red"},
    alpha=0.7
)
plt.title("PCA Scatter Plot Highlighting Anomalies")
plt.savefig(f"{OUTPUT_DIR}/pca_scatter.png", dpi=300)
plt.close()

print("Saved: pca_scatter.png")


# ------------------------------------------------------
# 5. t-SNE Scatter
# ------------------------------------------------------
print("Running t-SNE... (may take 1–2 minutes)")

tsne = TSNE(
    n_components=2,
    perplexity=35,
    learning_rate=200,
    init="random",
    random_state=42
)
tsne_result = tsne.fit_transform(X_scaled)

df_anom["tsne_1"] = tsne_result[:, 0]
df_anom["tsne_2"] = tsne_result[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_anom,
    x="tsne_1",
    y="tsne_2",
    hue="is_anomaly",
    palette={0: "blue", 1: "red"},
    alpha=0.7
)
plt.title("t-SNE Scatter Plot Highlighting Anomalies")
plt.savefig(f"{OUTPUT_DIR}/tsne_scatter.png", dpi=300)
plt.close()

print("Saved: tsne_scatter.png")


# ------------------------------------------------------
# 6. PCA Anomaly Metadata Labels
# ------------------------------------------------------
if not all(col in df_anom.columns for col in ["dut", "suite"]):
    print("Warning: DUT/Suite not found. Skipping metadata label plot.")
else:
    df_anoms = df_anom[df_anom["is_anomaly"] == 1]

    plt.figure(figsize=(10, 8))
    plt.scatter(df_anom["pca_1"], df_anom["pca_2"], alpha=0.3, label="Normal")
    plt.scatter(df_anoms["pca_1"], df_anoms["pca_2"], color="red", label="Anomaly")

    for _, row in df_anoms.iterrows():
        label = f"{row['dut']} | {row['suite']}"
        plt.text(row["pca_1"] + 0.02, row["pca_2"] + 0.02, label, fontsize=7)

    plt.title("PCA Anomaly Plot with Metadata Labels")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/pca_anomaly_metadata.png", dpi=300)
    plt.close()

    print("Saved: pca_anomaly_metadata.png")


# ------------------------------------------------------
# Done
# ------------------------------------------------------
print("\nAll visualizations saved to:", OUTPUT_DIR)
