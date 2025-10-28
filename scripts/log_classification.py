"""
Goal:
Predict log status/category (PASS / FAIL / ABORT)
from textual error messages using TF-IDF + Random Forest.

Input:
    data/train_dataset.csv
    data/test_dataset.csv

Output:
    data/outputs/model_rf/
        ├── classification_report.csv
        ├── confusion_matrix.png
        ├── classified_logs.csv
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

# Configuration
TRAIN_FILE = "data/train_dataset.csv"
TEST_FILE = "data/test_dataset.csv"
OUTPUT_DIR = "data/outputs/model_rf"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print("Loading data...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Prepare features and label
TEXT_COL = "error_msg"   # log message text
LABEL_COL = "label"      # derived label (PASS/FAIL/ABORT)

X_train = train_df[TEXT_COL].astype(str)
y_train = train_df[LABEL_COL].astype(str)
X_test = test_df[TEXT_COL].astype(str)
y_test = test_df[LABEL_COL].astype(str)

# TF-IDF Vectorization
print(" Vectorizing text logs with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"TF-IDF shape: {X_train_vec.shape}")

# Train Random Forest Classifier
print("\nTraining Random Forest Classifier...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_vec, y_train)

# Evaluation
print("Evaluating model...")
y_pred = rf.predict(X_test_vec)
y_prob = rf.predict_proba(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print(f" Accuracy: {accuracy:.4f}")
print(f"Weighted F1-score: {f1:.4f}")

# Detailed classification report
report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
report_df.to_csv(os.path.join(OUTPUT_DIR, "classification_report.csv"), index=True)
print("\nDetailed Classification Report:")
print(report_df)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=rf.classes_, yticklabels=rf.classes_)
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

# Save Classified Logs
classified_df = pd.DataFrame({
    "TestCase": test_df.get("test_case_id", ["N/A"] * len(y_test)),
    "PredictedLabel": y_pred,
    "ActualStatus": y_test,
    "Confidence": y_prob.max(axis=1)
})
classified_df.to_csv(os.path.join(OUTPUT_DIR, "classified_logs.csv"), index=False)

print(f"\nSaved all outputs in → {OUTPUT_DIR}")
print("Random Forest Log Classification completed successfully.")
