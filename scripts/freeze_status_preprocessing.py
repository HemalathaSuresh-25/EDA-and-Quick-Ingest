#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# CONFIG
# =========================
TRAIN_FILE = "C:/Users/hemalatha/Desktop/attest-eda/data/splits/train.csv"
MODEL_DIR  = "C:/Users/hemalatha/Desktop/attest-eda/models_1"

TFIDF_FILE     = f"{MODEL_DIR}/tfidf_error_msg.pkl"
FEATURES_FILE  = f"{MODEL_DIR}/status_features.pkl"

os.makedirs(MODEL_DIR, exist_ok=True)


def freeze_preprocessing():
    print(" Freezing preprocessing artifacts...")

    df = pd.read_csv(TRAIN_FILE)
    print(f"✔ Rows loaded: {len(df)}")

    # -------------------------
    # TF-IDF on error_msg
    # -------------------------
    tfidf = TfidfVectorizer(
        max_features=2000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3
    )

    tfidf.fit(df["error_msg"].fillna(""))

    joblib.dump(tfidf, TFIDF_FILE)
    print(f"✔ TF-IDF vectorizer saved → {TFIDF_FILE}")

    # -------------------------
    # Save feature column order
    # -------------------------
    drop_cols = [
        "status",
        "status_label",
        "root_cause_label",
        "root_cause_fail",
        "root_cause_abort",
        "error_msg"
    ]

    feature_cols = [c for c in df.columns if c not in drop_cols]

    joblib.dump(feature_cols, FEATURES_FILE)
    print(f"✔ Feature column order saved → {FEATURES_FILE}")

    print("\n Preprocessing freeze complete")


if __name__ == "__main__":
    freeze_preprocessing()
