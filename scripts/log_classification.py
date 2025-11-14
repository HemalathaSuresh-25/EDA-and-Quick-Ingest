"""
Goal:
Predict log status/category (PASS / FAIL / ABORT)
from textual error messages using TF-IDF + Random Forest.

Input:
    data/train_dataset.csv
    data/test_dataset.csv

Output:
    data/outputs/model_rf/
        classification_report.csv
        confusion_matrix.png
        classified_logs.csv
    models/
        rf_model.pkl
        tfidf_vectorizer.pkl
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

TRAIN_FILE = "data/train_dataset.csv"
TEST_FILE = "data/test_dataset.csv"

OUTPUT_DIR = "data/outputs/model_rf"
MODEL_DIR = "models"

TEXT_COL = "error_msg"         
LABEL_COL = "label"            
CONF_THRESHOLD = 0.6           

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading data...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Train shape: {train_df.shape}")
print(f"Test shape:  {test_df.shape}")

if TEXT_COL not in train_df.columns or LABEL_COL not in train_df.columns:
    raise KeyError(f"Dataset must include '{TEXT_COL}' and '{LABEL_COL}' columns.")


X_train = train_df[TEXT_COL].astype(str)
y_train = train_df[LABEL_COL].astype(str)

X_test = test_df[TEXT_COL].astype(str)
y_test = test_df[LABEL_COL].astype(str)

print("\n Vectorizing logs using TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(" TF-IDF shape:", X_train_vec.shape)

print("\nTraining Random Forest Model...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_vec, y_train)

print("\n Evaluating model...")
y_pred = rf.predict(X_test_vec)
y_prob = rf.predict_proba(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print(f"✔ Accuracy: {accuracy:.4f}")
print(f"✔ Weighted F1-score: {f1:.4f}")

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(OUTPUT_DIR, "classification_report.csv"), index=True)
print("✔ Classification report saved.")

print(" Saving confusion matrix...")

cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap="Blues",
    xticklabels=rf.classes_,
    yticklabels=rf.classes_
)
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

print(" Saving prediction results...")

confidence_scores = y_prob.max(axis=1)
low_conf_flags = (confidence_scores < CONF_THRESHOLD).astype(int)

classified_df = pd.DataFrame({
    "TestCase": test_df.get("test_case_id", ["N/A"] * len(y_test)),
    "DUT": test_df.get("dut", ["N/A"] * len(y_test)),
    "Suite": test_df.get("suite", ["N/A"] * len(y_test)),
    "PredictedLabel": y_pred,
    "ActualStatus": y_test,
    "Confidence": confidence_scores,
    "LowConfidenceFlag": low_conf_flags,
    "ErrorMsg": X_test,
})

classified_df.to_csv(os.path.join(OUTPUT_DIR, "classified_logs.csv"), index=False)

low_conf_count = low_conf_flags.sum()

print("\n Saving model and vectorizer...")
joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

print("Random Forest Log Classification Completed")
print(f"Outputs saved to: {OUTPUT_DIR}")
print(f"Models saved to:  {MODEL_DIR}")
print(f"Low-confidence predictions (< {CONF_THRESHOLD}): {low_conf_count}/{len(y_test)}")

