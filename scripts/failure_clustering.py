import os
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer


INPUT_FILE = "C:/Users/hemalatha/Desktop/attest-eda/data/features/failure_features.csv"
OUTPUT_DIR = "C:/Users/hemalatha/Desktop/attest-eda/data/cluster"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "failure_clusters.csv")


TOP_KEYWORDS = 5
KMEANS_CLUSTERS = 10
BERT_MODEL = "all-MiniLM-L6-v2"


# Clustering Function 
def cluster_failures_bert():
    print("Loading dataset...")
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
    print(f"Found {len(df_failures)} FAIL logs for clustering")

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

    # Calculate silhouette score for validation
    sil_score = silhouette_score(embeddings, cluster_labels)
    print(f"Silhouette score: {sil_score:.3f} (Higher is better, >0.4 is good)")

    # Assign clusters back to full dataframe
    df["fail_cluster"] = -1  # default for non-fail logs
    df.loc[df_failures.index, "fail_cluster"] = cluster_labels
    
    # Extract top keywords per cluster using TF-IDF
    print("Extracting top keywords per cluster using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
    X_tfidf = vectorizer.fit_transform(df_failures["error_msg"])
    feature_names = np.array(vectorizer.get_feature_names_out())

    top_keywords_per_cluster = {}
    cluster_sizes = {}
    
    print("\n=== CLUSTER SUMMARY ===")
    for cluster_num in range(KMEANS_CLUSTERS):
        cluster_indices = np.where(cluster_labels == cluster_num)[0]
        cluster_sizes[cluster_num] = len(cluster_indices)
        
        if len(cluster_indices) == 0:
            top_keywords_per_cluster[cluster_num] = []
            print(f"Cluster {cluster_num}: 0 logs (empty)")
            continue
            
        cluster_tfidf = X_tfidf[cluster_indices].mean(axis=0)
        top_indices = np.asarray(cluster_tfidf).flatten().argsort()[::-1][:TOP_KEYWORDS]
        top_keywords = feature_names[top_indices].tolist()
        top_keywords_per_cluster[cluster_num] = top_keywords
        
        print(f"Cluster {cluster_num}: {len(cluster_indices)} logs - {', '.join(top_keywords)}")

    # Print cluster distribution summary
    print("\n=== CLUSTER DISTRIBUTION ===")
    cluster_dist = pd.Series(cluster_labels).value_counts().sort_index()
    print(cluster_dist.to_dict())
    print(f"Total clusters used: {len([s for s in cluster_sizes.values() if s > 0])}/{KMEANS_CLUSTERS}")

    # Save results
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nFailure clusters (BERT) saved → {OUTPUT_FILE}")

    # Create models directory and save models
    os.makedirs("models", exist_ok=True)
    joblib.dump(kmeans, "models/cluster_model_failure.pkl")
    joblib.dump(model, "models/bert_encoder_failure.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer_failure.pkl")  # Bonus: save TF-IDF too

    print("Clustering model saved as → models/cluster_model_failure.pkl")
    print("BERT encoder saved as    → models/bert_encoder_failure.pkl")
    print("TF-IDF vectorizer saved → models/tfidf_vectorizer_failure.pkl")

    # Save cluster summary as separate CSV for reporting
    cluster_summary = pd.DataFrame({
        'cluster': list(range(KMEANS_CLUSTERS)),
        'size': [cluster_sizes.get(i, 0) for i in range(KMEANS_CLUSTERS)],
        'top_keywords': [', '.join(top_keywords_per_cluster.get(i, [])) for i in range(KMEANS_CLUSTERS)]
    })
    cluster_summary.to_csv(os.path.join(OUTPUT_DIR, "cluster_summary.csv"), index=False)
    print("Cluster summary saved →", os.path.join(OUTPUT_DIR, "cluster_summary.csv"))

    return df, top_keywords_per_cluster, cluster_sizes


if __name__ == "__main__":
    df, keywords, sizes = cluster_failures_bert()
