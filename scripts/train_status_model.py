#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# =========================
# CONFIG
# =========================
TRAIN_FILE = "C:/Users/hemalatha/Desktop/attest-eda/data/splits/train.csv"
TEST_FILE  = "C:/Users/hemalatha/Desktop/attest-eda/data/splits/test.csv"

MODEL_DIR = "C:/Users/hemalatha/Desktop/attest-eda/models_1"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = f"{MODEL_DIR}/status_classifier_rf.pkl"

RANDOM_STATE = 42

# =========================
# FEATURES (NO LEAKAGE)
# =========================
TEXT_FEATURES = ["error_msg"]

NUMERIC_FEATURES = [
    "execution_duration",
    "time_since_last_failure",
    "time_since_last_abort",
    "failure_freq_suite",
    "abort_freq_suite",
    "failure_freq_dut",
    "abort_freq_dut"
]

TARGET = "status_label"

# =========================
# LOAD DATA
# =========================
print(" Loading training data...")
train_df = pd.read_csv(TRAIN_FILE)
test_df  = pd.read_csv(TEST_FILE)

print(f"✔ Train rows: {len(train_df)}")
print(f"✔ Test rows : {len(test_df)}")

# Fill missing values safely
for col in TEXT_FEATURES:
    train_df[col] = train_df[col].fillna("")
    test_df[col]  = test_df[col].fillna("")

for col in NUMERIC_FEATURES:
    train_df[col] = train_df[col].fillna(0)
    test_df[col]  = test_df[col].fillna(0)

X_train = train_df[TEXT_FEATURES + NUMERIC_FEATURES]
y_train = train_df[TARGET]

X_test = test_df[TEXT_FEATURES + NUMERIC_FEATURES]
y_test = test_df[TARGET]

# =========================
# FEATURE PIPELINE
# =========================
text_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_features=15000,
        sublinear_tf=True
    ))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("text", text_pipeline, "error_msg"),
        ("num", "passthrough", NUMERIC_FEATURES)
    ]
)

# =========================
# MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

pipeline = Pipeline([
    ("features", preprocessor),
    ("classifier", model)
])

# =========================
# TRAIN
# =========================
print("\n Training RandomForest Status Model...")
pipeline.fit(X_train, y_train)

# =========================
# EVALUATE
# =========================
print("\n Evaluation on TEST data:")
y_pred = pipeline.predict(X_test)

print(classification_report(
    y_test,
    y_pred,
    target_names=["PASS", "FAIL", "ABORT"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================
joblib.dump(pipeline, MODEL_PATH)

print(f"\n Model saved to: {MODEL_PATH}")
print(" Status Model Training Complete")
