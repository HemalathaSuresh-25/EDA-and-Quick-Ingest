#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# =========================
# CONFIG
# =========================
TRAIN_FILE = "C:/Users/hemalatha/Desktop/attest-eda/data/splits/train.csv"
TEST_FILE  = "C:/Users/hemalatha/Desktop/attest-eda/data/splits/test.csv"

MODEL_DIR = "C:/Users/hemalatha/Desktop/attest-eda/models_1"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_OUT    = f"{MODEL_DIR}/rootcause_abort_classifier.pkl"
FEATURES_OUT = f"{MODEL_DIR}/rootcause_abort_features.pkl"
TFIDF_OUT    = f"{MODEL_DIR}/tfidf_rootcause_abort.pkl"

RANDOM_STATE = 42


def train_rootcause_abort():
    print(" Loading train/test data...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df  = pd.read_csv(TEST_FILE)

    # =========================
    # Filter ABORT only
    # =========================
    train_df = train_df[train_df["status_label"] == 2]
    test_df  = test_df[test_df["status_label"] == 2]

    # Remove NORMAL / missing labels
    train_df = train_df[train_df["root_cause_label"] != "NORMAL"]
    test_df  = test_df[test_df["root_cause_label"] != "NORMAL"]

    print(f"✔ ABORT Train rows: {len(train_df)}")
    print(f"✔ ABORT Test rows : {len(test_df)}")

    # =========================
    # Features
    # =========================
    text_col = "error_msg"

    numeric_features = [
        c for c in train_df.columns
        if c not in [
            "root_cause_label",
            "error_msg",
            "status_label"
        ]
        and train_df[c].dtype != "object"
    ]

    # =========================
    # TF-IDF (ABORT specific)
    # =========================
    tfidf = TfidfVectorizer(
        max_features=3000,
        stop_words="english"
    )

    X_train_text = tfidf.fit_transform(train_df[text_col].fillna(""))
    X_test_text  = tfidf.transform(test_df[text_col].fillna(""))

    # =========================
    # Numeric features
    # =========================
    X_train_num = train_df[numeric_features].fillna(0)
    X_test_num  = test_df[numeric_features].fillna(0)

    X_train = hstack([X_train_text, X_train_num])
    X_test  = hstack([X_test_text, X_test_num])

    y_train = train_df["root_cause_label"]
    y_test  = test_df["root_cause_label"]

    # =========================
    # Train model
    # =========================
    print("\n Training ABORT Root Cause Model...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # =========================
    # Evaluation
    # =========================
    preds = model.predict(X_test)

    print("\n Evaluation on TEST data:")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    # =========================
    # Save artifacts
    # =========================
    joblib.dump(model, MODEL_OUT)
    joblib.dump(tfidf, TFIDF_OUT)
    joblib.dump(numeric_features, FEATURES_OUT)

    print("\n ABORT Root Cause Model saved:")
    print(f" - {MODEL_OUT}")
    print(f" - {TFIDF_OUT}")
    print(f" - {FEATURES_OUT}")


if __name__ == "__main__":
    train_rootcause_abort()
