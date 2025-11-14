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

# ---------------------------------------------------------------
# File Paths
# ---------------------------------------------------------------
CLASSIFIED = "data/outputs/model_rf/classified_logs.csv"
ANOMALY = "data/outputs/anomaly_reports/anomaly_full_scores.csv"
OUTPUT_DIR = "data/outputs/visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------
df_class = pd.read_csv(CLASSIFIED)
df_anom = pd.read_csv(ANOMALY)

# ---------------------------------------------------------------
# 1️⃣ CONFUSION MATRIX
# ---------------------------------------------------------------
plt.figure(figsize=(7, 6))
cm = confusion_matrix(df_class["ActualStatus"], df_class["PredictedLabel"])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Classification Results")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=300)
plt.show()

# ---------------------------------------------------------------
# 2️⃣ ANOMALY SCORE HISTOGRAM
# ---------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df_anom["anomaly_score"], bins=40, kde=True)
plt.title("Anomaly Score Distribution")
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/anomaly_score_histogram.png", dpi=300)
plt.show()

# ---------------------------------------------------------------
# 3️⃣ TF-IDF EMBEDDINGS (MessageSnippet)
# ---------------------------------------------------------------
TEXT_COLUMN = "MessageSnippet"

if TEXT_COLUMN not in df_anom.columns:
    raise ValueError(f"❌ Column `{TEXT_COLUMN}` not found in anomaly file!")

print("Generating TF-IDF embeddings from:", TEXT_COLUMN)

vectorizer = TfidfVectorizer(
    max_features=300,
    stop_words="english"
)

X = vectorizer.fit_transform(df_anom[TEXT_COLUMN].fillna("")).toarray()

print("TF-IDF shape:", X.shape)

# Scale the embeddings
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------
# 4️⃣ PCA VISUALIZATION
# ---------------------------------------------------------------
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
plt.show()

# ---------------------------------------------------------------
# 5️⃣ t-SNE VISUALIZATION
# ---------------------------------------------------------------
print("Running t-SNE... (takes 1–2 minutes)")

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
plt.show()

# ---------------------------------------------------------------
# 6️⃣ PCA WITH METADATA LABELS
# ---------------------------------------------------------------
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
plt.show()


