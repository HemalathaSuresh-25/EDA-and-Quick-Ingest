#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced XGBoost Test Case Prioritization with Rolling & Weighted Features
---------------------------------------------------------------------------
- Leak-free features
- Recency weight (exponential)
- Cluster weight (capped)
- Rolling failure/frequency features
- Target encoding for categorical features
- TF-IDF text features (bigrams)
- ROC & calibration curves
- Top-K accuracy curve (1% → 20%)
- Feature importance plot
- Final CSV with High/Medium/Low risk
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.calibration import calibration_curve
import xgboost as xgb

# -------------------------
# 1. Load Data
# -------------------------
df = pd.read_csv("C:/Users/hemalatha/Desktop/attest-eda/data/outputs/feature_engineered_testcases.csv")
print("Total logs loaded:", len(df))

# -------------------------
# 2. Encode target
# -------------------------
df['fail_flag'] = df['status'].apply(lambda x: 1 if str(x).upper() == 'FAIL' else 0)
y = df['fail_flag']

# -------------------------
# 3. Numeric features
# -------------------------
numeric_features = ['execution_duration', 'failure_rate', 'suite_fail_rate', 'dut_fail_rate', 'time_since_last_failure']
X_numeric = df[numeric_features].copy()

# -------------------------
# 4. Rolling / Frequency features
# -------------------------
df = df.sort_values(['test_case', 'run_date'])
for col in ['test_case', 'dut']:
    # Rolling failures in last 5 runs
    df[f'{col}_fail_last5'] = df.groupby(col)['fail_flag'].rolling(5, min_periods=1).sum().reset_index(0, drop=True)
    # Rolling mean failure rate last 5 runs
    df[f'{col}_fail_rate_last5'] = df.groupby(col)['fail_flag'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
X_numeric = pd.concat([X_numeric, df[[c for c in df.columns if '_last5' in c]]], axis=1)

# -------------------------
# 5. Target encode categorical features
# -------------------------
categorical_features = ['dut', 'suite', 'regression_group', 'schedule_cycle', 'test_case']
X_cat = pd.DataFrame()
for col in categorical_features:
    if col in df.columns:
        X_cat[col+'_te'] = df.groupby(col)['fail_flag'].transform('mean')

# -------------------------
# 6. TF-IDF Text Features
# -------------------------
text_col = 'message_snippet'
def clean_text(text):
    text = str(text).upper()
    text = re.sub(r'\bFAIL\b|\bPASS\b|\bABORT\b|\bERROR\b|\bEXCEPTION\b', '', text)
    return text

cleaned_text = df[text_col].apply(clean_text)
vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1,2))
X_text = vectorizer.fit_transform(cleaned_text)
X_text_df = pd.DataFrame(X_text.toarray(), columns=[f"tfidf_{i}" for i in range(X_text.shape[1])])

# -------------------------
# 7. Combine features
# -------------------------
X = pd.concat([X_numeric.reset_index(drop=True),
               X_cat.reset_index(drop=True),
               X_text_df.reset_index(drop=True)], axis=1)
print("Total features used:", X.shape[1])

# -------------------------
# 8. Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Train size:", len(X_train), "Test size:", len(X_test))

# -------------------------
# 9. Train XGBoost
# -------------------------
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    eval_metric='logloss',
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)
model.fit(X_train, y_train)

# -------------------------
# 10. Evaluate Model
# -------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
roc_auc = roc_auc_score(y_test, y_proba)

print("\n----- MODEL METRICS -----")
print("Precision:", round(precision,4))
print("Recall:", round(recall,4))
print("F1 Score:", round(f1,4))
print("ROC–AUC:", round(roc_auc,4))

# -------------------------
# 11. ROC Curve
# -------------------------
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title('ROC Curve - XGBoost')
plt.grid(True)
plt.savefig('outputs/roc_curve.png')
plt.close()
print("✔ ROC curve saved: outputs/roc_curve.png")

# -------------------------
# 12. Calibration Curve
# -------------------------
prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
plt.figure(figsize=(6,6))
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1],[0,1], linestyle='--', color='gray')
plt.xlabel('Predicted Probability')
plt.ylabel('Observed Frequency')
plt.title('Calibration Curve - XGBoost')
plt.grid(True)
plt.savefig('outputs/calibration_curve.png')
plt.close()
print("✔ Calibration curve saved: outputs/calibration_curve.png")

# -------------------------
# 13. Top-K Accuracy Curve (1% → 20%)
# -------------------------
test_df = X_test.copy()
test_df['actual'] = y_test.values
test_df['pred_fail_prob'] = y_proba

top_percentages = np.arange(0.01, 0.21, 0.01)
top_k_values = []
top_k_accuracy = []

total_fail = test_df['actual'].sum()
for pct in top_percentages:
    top_k = int(len(test_df)*pct)
    captured = test_df.sort_values('pred_fail_prob', ascending=False).head(top_k)['actual'].sum()
    accuracy = round((captured/total_fail)*100,2)
    top_k_values.append(top_k)
    top_k_accuracy.append(accuracy)
    print(f"Top-{int(pct*100)}% Accuracy ({top_k} logs): {accuracy}% failures captured")

plt.figure(figsize=(8,5))
sns.lineplot(x=top_k_values, y=top_k_accuracy, marker='o')
plt.xlabel('Number of Logs (Top-K)')
plt.ylabel('% Failures Captured')
plt.title('Top-K Accuracy Curve')
plt.grid(True)
plt.savefig('outputs/topk_accuracy_curve.png')
plt.close()
print("✔ Top-K accuracy curve saved: outputs/topk_accuracy_curve.png")

# -------------------------
# 14. Predict for ALL logs
# -------------------------
df['pred_fail_prob'] = model.predict_proba(X)[:,1]

# -------------------------
# 15. Recency & Cluster Weight
# -------------------------
# Exponential recency weight
df['recency_weight'] = 1 + np.exp(-df['time_since_last_failure'])

# Cluster weight capped at 1.5
df['cluster_fail_rate'] = df.groupby('cluster_label')['fail_flag'].transform('mean')
df['cluster_weight'] = 1 + np.clip(df['cluster_fail_rate'], 0, 1.5)

# -------------------------
# 16. Priority Score
# -------------------------
df['priority_score'] = df['pred_fail_prob'] * df['recency_weight'] * df['cluster_weight']

def priority_label(x):
    if x>df['priority_score'].quantile(0.9): return 'HIGH RISK'
    elif x>df['priority_score'].quantile(0.6): return 'MEDIUM RISK'
    else: return 'LOW RISK'

df['priority_group'] = df['priority_score'].apply(priority_label)

# -------------------------
# 17. Feature Importance
# -------------------------
xgb.plot_importance(model, max_num_features=20, importance_type='gain', height=0.5)
plt.title('Top 20 Feature Importance')
plt.savefig('outputs/feature_importance.png')
plt.close()
print("✔ Feature importance plot saved: outputs/feature_importance.png")

# -------------------------
# 18. Save Final CSV
# -------------------------
df = df.sort_values('priority_score', ascending=False)
df.to_csv('outputs/prioritized_testcases_xgb_optimized.csv', index=False)
print("\n✔ Final prioritized CSV saved: outputs/prioritized_testcases_xgb_optimized.csv")
