#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.sparse import hstack
from joblib import dump
import xgboost as xgb
import os

# =======================
# Create models directory if not exists
# =======================
os.makedirs("models", exist_ok=True)

# =======================
# Load dataset
# =======================
df = pd.read_csv("C:/Users/hemalatha/Desktop/attest-eda/data/train_dataset.csv")

# =======================
# Encode target labels
# =======================
le = LabelEncoder()
y_encoded = le.fit_transform(df['label'])  # 'ABORT','FAIL','PASS' -> 0,1,2

# =======================
# TF-IDF Vectorization
# =======================
# Word-level TF-IDF
vectorizer_word = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 2),
    max_features=10000,
    min_df=2,
    max_df=0.9
)
X_word = vectorizer_word.fit_transform(df['error_msg'].astype(str))

# Char-level TF-IDF
vectorizer_char = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3, 5),
    max_features=5000
)
X_char = vectorizer_char.fit_transform(df['error_msg'].astype(str))

# Combine TF-IDF features
X_features = hstack([X_word, X_char])

# =======================
# Train/Test Split
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# =======================
# Initialize XGBoost Classifier
# =======================
clf = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# =======================
# Cross-Validation
# =======================
cv_scores = cross_val_score(clf, X_features, y_encoded, cv=5, scoring='f1_weighted')
print("5-Fold CV Weighted F1-score: {:.4f}".format(cv_scores.mean()))

# =======================
# Train the Model
# =======================
clf.fit(X_train, y_train)

# =======================
# Evaluate
# =======================
y_pred = clf.predict(X_test)
y_pred_labels = le.inverse_transform(y_pred)
y_test_labels = le.inverse_transform(y_test)

print("\nClassification Report:\n")
print(classification_report(y_test_labels, y_pred_labels))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test_labels, y_pred_labels))

accuracy = accuracy_score(y_test_labels, y_pred_labels)
print("\nTest Accuracy: {:.4f}".format(accuracy))

# =======================
# Save Model & Vectorizers
# =======================
dump(clf, "models/status_classifier_xgb.joblib")
dump(vectorizer_word, "models/tfidf_vectorizer_word.pkl")
dump(vectorizer_char, "models/tfidf_vectorizer_char.pkl")
dump(le, "models/label_encoder.joblib")  # save LabelEncoder for decoding future predictions

print("\n✔ Saved status_classifier_xgb.joblib, TF-IDF vectorizers, and LabelEncoder")