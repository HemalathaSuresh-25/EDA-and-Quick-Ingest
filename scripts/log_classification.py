"""
Goal:
Predict log status/category (PASS / FAIL / ABORT)
from textual error messages using TF-IDF + Random Forest.

Final Output:
    data/outputs/model_rf/classified_logs.csv  
    → Contains ALL LOGS (train + test = 8605 rows)

Columns:
    test_case_id
    filename
    dut
    suite
    cluster_label
    predictedlabel
    actualstatus
    confidence
    lowconfidenceflag
    errormsg
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

# ---------------------------- CONFIG ----------------------------
TRAIN_FILE = "data/train_dataset.csv"
TEST_FILE = "data/test_dataset.csv"

OUTPUT_DIR = "data/outputs/model_rf"
MODEL_DIR = "models"

TEXT_COL = "error_msg"
LABEL_COL = "label"
CONF_THRESHOLD = 0.6

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------- LOAD DATA ----------------------------
print("Loading data...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Train shape: {train_df.shape}")
print(f"Test shape:  {test_df.shape}")

# ---------------------------- MODEL INPUT ----------------------------
X_train = train_df[TEXT_COL].astype(str)
y_train = train_df[LABEL_COL].astype(str)

X_test = test_df[TEXT_COL].astype(str)
y_test = test_df[LABEL_COL].astype(str)

# ---------------------------- TF-IDF ----------------------------
print("\nVectorizing logs using TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("TF-IDF shape:", X_train_vec.shape)

# ---------------------------- MODEL TRAIN ----------------------------
print("\nTraining Random Forest Model...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_vec, y_train)

# ---------------------------- EVALUATE ----------------------------
print("\nEvaluating model...")
y_pred = rf.predict(X_test_vec)
y_prob = rf.predict_proba(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print(f"✔ Accuracy: {accuracy:.4f}")
print(f"✔ Weighted F1-score: {f1:.4f}")

# Save classification report
report = classification_report(y_test, y_pred, output_dict=True)
pd.DataFrame(report).transpose().to_csv(
    os.path.join(OUTPUT_DIR, "classification_report.csv"),
    index=True
)

# ---------------------------- CONFUSION MATRIX ----------------------------
print("Saving confusion matrix...")

cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap="Blues",
    xticklabels=rf.classes_, yticklabels=rf.classes_
)
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

# ---------------------------- BUILD TEST OUTPUT ----------------------------
print("Building test predictions...")

confidence_scores = y_prob.max(axis=1)
low_conf_flags = (confidence_scores < CONF_THRESHOLD).astype(int)

test_out = pd.DataFrame({
    "test_case_id": test_df.get("test_case_id", ["N/A"] * len(test_df)),
    "filename": test_df.get("filename", ["N/A"] * len(test_df)),
    "dut": test_df.get("dut", ["N/A"] * len(test_df)),
    "suite": test_df.get("suite", ["N/A"] * len(test_df)),
    "cluster_label": test_df.get("cluster", ["N/A"] * len(test_df)),

    "predictedlabel": y_pred,
    "actualstatus": y_test,
    "confidence": confidence_scores,
    "lowconfidenceflag": low_conf_flags,

    "errormsg": X_test
})

# ---------------------------- BUILD TRAIN OUTPUT ----------------------------
print("Building train predictions...")

train_pred = rf.predict(X_train_vec)
train_prob = rf.predict_proba(X_train_vec)
train_conf = train_prob.max(axis=1)
train_low_flag = (train_conf < CONF_THRESHOLD).astype(int)

train_out = pd.DataFrame({
    "test_case_id": train_df.get("test_case_id", ["N/A"] * len(train_df)),
    "filename": train_df.get("filename", ["N/A"] * len(train_df)),
    "dut": train_df.get("dut", ["N/A"] * len(train_df)),
    "suite": train_df.get("suite", ["N/A"] * len(train_df)),
    "cluster_label": train_df.get("cluster", ["N/A"] * len(train_df)),

    "predictedlabel": train_pred,
    "actualstatus": y_train,
    "confidence": train_conf,
    "lowconfidenceflag": train_low_flag,

    "errormsg": X_train
})

# ---------------------------- MERGE TRAIN + TEST ----------------------------
final_df = pd.concat([train_out, test_out], ignore_index=True)

print(f"\nFinal output rows: {final_df.shape[0]} (expected ~8605)")

final_df.to_csv(os.path.join(OUTPUT_DIR, "classified_logs.csv"), index=False)

# ---------------------------- SAVE MODEL ----------------------------
print("\nSaving model and vectorizer...")
joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

print("\nRandom Forest Log Classification Completed ✔")
print(f"Final classified logs saved → {OUTPUT_DIR}/classified_logs.csv")
