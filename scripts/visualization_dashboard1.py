"""
visualization_reports.py
------------------------
Generates visual insights for log classification and anomaly detection.

Inputs:
    data/outputs/classified_logs.csv
    data/outputs/anomaly_reports/anomalies_detected.csv

Outputs:
    data/outputs/visualizations/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
CLASSIFIED_LOGS = "data/outputs/model_rf/classified_logs.csv"
ANOMALY_FILE = "data/outputs/anomaly_reports/anomalies_detected.csv"
OUTPUT_DIR = "data/outputs/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------------
# 1️⃣ Confusion Matrix Visualization
# --------------------------------------------------------------------
if os.path.exists(CLASSIFIED_LOGS):
    print("📊 Loading classified logs for confusion matrix...")
    df_class = pd.read_csv(CLASSIFIED_LOGS)

    if {"PredictedLabel", "ActualStatus"}.issubset(df_class.columns):
        labels = sorted(df_class["ActualStatus"].unique())
        cm = confusion_matrix(df_class["ActualStatus"], df_class["PredictedLabel"], labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

        plt.figure(figsize=(6, 5))
        disp.plot(cmap="Blues", colorbar=True)
        plt.title("Confusion Matrix – Log Classification")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
        plt.close()
        print("✅ Confusion matrix saved →", os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    else:
        print("⚠️ Columns PredictedLabel or ActualStatus not found in classified_logs.csv.")
else:
    print("⚠️ classified_logs.csv not found, skipping confusion matrix.")

# --------------------------------------------------------------------
# 2️⃣ Load Anomaly Data
# --------------------------------------------------------------------
if not os.path.exists(ANOMALY_FILE):
    print("⚠️ anomalies_detected.csv not found. Skipping anomaly visualizations.")
    exit()

print("\n📊 Loading anomaly data for visualization...")
df_anom = pd.read_csv(ANOMALY_FILE)
print(f"✅ Loaded {len(df_anom)} anomaly records")

# Try to ensure anomaly_score column exists
possible_cols = [c for c in df_anom.columns if "score" in c.lower()]
if "anomaly_score" not in df_anom.columns and possible_cols:
    df_anom.rename(columns={possible_cols[0]: "anomaly_score"}, inplace=True)

if "anomaly_score" not in df_anom.columns:
    raise ValueError("❌ anomaly_score column not found even after renaming check!")

# --------------------------------------------------------------------
# 3️⃣ Anomaly Score Distribution
# --------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df_anom["anomaly_score"], bins=40, kde=True, color="orange")
plt.title("Anomaly Score Distribution (Isolation Forest)")
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "anomaly_score_distribution.png"))
plt.close()
print("✅ Anomaly score distribution plot saved.")

# --------------------------------------------------------------------
# 4️⃣ Prepare Numeric Data for PCA/t-SNE
# --------------------------------------------------------------------
num_data = df_anom.select_dtypes(include=[np.number]).fillna(0)
scaler = StandardScaler()
scaled = scaler.fit_transform(num_data)

# --------------------------------------------------------------------
# 5️⃣ PCA Scatter Plot
# --------------------------------------------------------------------
print("\n🧭 Generating PCA scatter plot...")
pca = PCA(n_components=2, random_state=42)
reduced = pca.fit_transform(scaled)
df_anom["pca1"], df_anom["pca2"] = reduced[:, 0], reduced[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_anom, x="pca1", y="pca2",
    hue=(df_anom["anomaly_score"] < df_anom["anomaly_score"].quantile(0.1)),
    palette={True: "red", False: "blue"}, alpha=0.7
)
plt.title("PCA Scatter Plot – Highlighting Anomalies")
plt.legend(title="Anomaly", labels=["Normal", "Anomaly"])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "anomaly_scatter_pca.png"))
plt.close()
print("✅ PCA scatter with anomalies saved.")

# --------------------------------------------------------------------
# 6️⃣ t-SNE Scatter Plot
# --------------------------------------------------------------------
print("\n🎯 Generating t-SNE scatter plot...")
try:
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
    tsne_results = tsne.fit_transform(scaled)
    df_anom["tsne1"], df_anom["tsne2"] = tsne_results[:, 0], tsne_results[:, 1]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_anom, x="tsne1", y="tsne2",
        hue=(df_anom["anomaly_score"] < df_anom["anomaly_score"].quantile(0.1)),
        palette={True: "red", False: "blue"}, alpha=0.7
    )
    plt.title("t-SNE Scatter Plot – Highlighting Anomalies")
    plt.legend(title="Anomaly", labels=["Normal", "Anomaly"])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "anomaly_scatter_tsne.png"))
    plt.close()
    print("✅ t-SNE scatter with anomalies saved.")
except Exception as e:
    print("⚠️ Skipping t-SNE due to:", e)

# --------------------------------------------------------------------
# 7️⃣ Highlight anomalies with metadata
# --------------------------------------------------------------------
meta_cols = [c for c in ["TestCase", "DUT", "suite"] if c in df_anom.columns]
if meta_cols:
    summary_file = os.path.join(OUTPUT_DIR, "anomaly_metadata_summary.csv")
    df_anom[["anomaly_score"] + meta_cols].to_csv(summary_file, index=False)
    print(f"💾 Metadata summary saved → {summary_file}")

print("\n🎉 Visualization pipeline completed successfully.")
