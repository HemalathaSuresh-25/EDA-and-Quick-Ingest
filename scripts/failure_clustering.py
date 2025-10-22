import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

INPUT_FILE = "C:/Users/hemalatha/Desktop/attest-eda/data/features/failure_features.csv"
OUTPUT_DIR = "C:/Users/hemalatha/Desktop/attest-eda/data/cluster"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "failure_clusters.csv")

TOP_KEYWORDS = 5
KMEANS_CLUSTERS = 10
BERT_MODEL = "all-MiniLM-L6-v2"


#Clustering Function 
def cluster_failures_bert():
    print("🔹 Loading dataset...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Validate presence of necessary columns
    required_cols = ["status", "error_msg"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset.")

    # Filter only FAIL logs
    df["status"] = df["status"].astype(str).str.strip().str.upper()
    df_failures = df[df["status"] == "FAIL"].copy()

    if df_failures.empty:
        raise ValueError("No 'FAIL' logs found for clustering!")

    # Clean error messages
    df_failures["error_msg"] = df_failures["error_msg"].fillna("").astype(str)

    # Encode using BERT embeddings
    print(f"Encoding {df_failures.shape[0]} failure messages using BERT model: {BERT_MODEL} ...")
    model = SentenceTransformer(BERT_MODEL)
    embeddings = model.encode(
        df_failures["error_msg"].tolist(),
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Apply KMeans clustering
    print(f"Clustering embeddings with KMeans (k={KMEANS_CLUSTERS})...")
    kmeans = KMeans(n_clusters=KMEANS_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Assign clusters back to full dataframe
    df["cluster"] = -1  # default for non-fail logs
    df.loc[df_failures.index, "cluster"] = cluster_labels

    # Extract top keywords per cluster using TF-IDF
    print("Extracting top keywords per cluster using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
    X_tfidf = vectorizer.fit_transform(df_failures["error_msg"])
    feature_names = np.array(vectorizer.get_feature_names_out())

    top_keywords_per_cluster = {}
    for cluster_num in range(KMEANS_CLUSTERS):
        cluster_indices = np.where(cluster_labels == cluster_num)[0]
        if len(cluster_indices) == 0:
            top_keywords_per_cluster[cluster_num] = []
            continue
        cluster_tfidf = X_tfidf[cluster_indices].mean(axis=0)
        top_indices = np.asarray(cluster_tfidf).flatten().argsort()[::-1][:TOP_KEYWORDS]
        top_keywords = feature_names[top_indices].tolist()
        top_keywords_per_cluster[cluster_num] = top_keywords
        print(f"Cluster {cluster_num}: {', '.join(top_keywords)}")

    # Save results
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n Failure clusters (BERT) saved → {OUTPUT_FILE}")

    return df, top_keywords_per_cluster

if __name__ == "__main__":
    cluster_failures_bert()
