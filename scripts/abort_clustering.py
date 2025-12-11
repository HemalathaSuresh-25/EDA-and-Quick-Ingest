#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# =========================
# 1. LOAD YOUR DATA
# =========================
DF_FILE = "C:/Users/hemalatha/Desktop/attest-eda/data/feature_engineered_testcases.csv"

df = pd.read_csv(DF_FILE)
df_abort = df[df["AB"] == "ABORT"].copy()

print("Total abort logs:", len(df_abort))

# =========================
# 2. CLEAN ABORT MESSAGE
# =========================
def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"\d{2}:\d{2}:\d{2}\.\d+", " ", t)  # remove timestamps
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

df_abort["clean_msg"] = df_abort["Y"].apply(clean_text)

# =========================
# 3. EMBEDDINGS (MiniLM)
# =========================
print("Generating embeddings...")
model = SentenceTransformer("all-mpnet-base-v2")
embeddings = model.encode(df_abort["clean_msg"].tolist(), show_progress_bar=True)

# =========================
# 4. FIND BEST NUMBER OF CLUSTERS (KNEE PLOT)
# =========================
print("Finding best number of clusters...")

distortions = []
K_range = range(2, 12)

for K in K_range:
    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(embeddings)
    distortions.append(kmeans.inertia_)

plt.plot(K_range, distortions, marker="o")
plt.title("KMeans Elbow Curve")
plt.xlabel("K (clusters)")
plt.ylabel("Inertia")
plt.grid()
plt.savefig("abort_elbow_curve.png")
print("Elbow curve saved as abort_elbow_curve.png")

# =========================
# 5. CHOOSE K (OR MANUALLY DECIDE)
# =========================
BEST_K = 6         # adjust after seeing elbow curve!

kmeans_final = KMeans(n_clusters=BEST_K, random_state=42)
df_abort["cluster"] = kmeans_final.fit_predict(embeddings)

# =========================
# 6. EXPORT FOR MANUAL LABELING
# =========================
OUT = "abort_cluster_output.csv"

df_abort[["filename", "Y", "clean_msg", "cluster"]].to_csv(OUT, index=False)

print("\n==============================")
print("CLUSTERING COMPLETED!")
print(f"Output saved to: {OUT}")
print("==============================\n")
