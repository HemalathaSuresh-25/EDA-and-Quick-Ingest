#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced training + evaluation + prioritization script.

Outputs:
 - outputs/roc_curve.png
 - outputs/confusion_matrix.png
 - outputs/calibration_curve.png
 - outputs/topk_results_testset.csv
 - outputs/feature_importance.csv + .png
 - outputs/xgb_model.json
 - outputs/model_features.joblib
 - outputs/test_predictions.csv
 - outputs/prioritized_testcases_xgb.csv
"""

import os
import math
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier


INPUT_CSV = r"C:/Users/hemalatha/Desktop/attest-eda/data/feature_engineered_testcases.csv"
OUTDIR = "data/task4output"
MODELDIR = "models"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(MODELDIR, exist_ok=True)

ROC_PNG = os.path.join(OUTDIR, "roc_curve.png")
CM_PNG = os.path.join(OUTDIR, "confusion_matrix.png")
CAL_PNG = os.path.join(OUTDIR, "calibration_curve.png")
TOPK_CSV = os.path.join(OUTDIR, "topk_results_testset.csv")
FI_CSV = os.path.join(OUTDIR, "feature_importance.csv")
FI_PNG = os.path.join(OUTDIR, "feature_importance.png")
MODEL_JSON = os.path.join(MODELDIR, "xgb_model.json")
MODEL_FEATURES = os.path.join(MODELDIR, "model_features.joblib")
TEST_PRED_CSV = os.path.join(OUTDIR, "test_predictions.csv")
PRIORITIZED_CSV = os.path.join(OUTDIR, "prioritized_testcases_xgb.csv")

RANDOM_STATE = 42

df = pd.read_csv(INPUT_CSV)
print("Total rows loaded:", len(df))
print("Columns:", df.columns.tolist())

# Target -> support 'target' or fallback to 'status'/'fail_flag'
if "target" in df.columns:
    df["isFail"] = df["target"].astype(int)
elif "fail_flag" in df.columns:
    df["isFail"] = df["fail_flag"].astype(int)
elif "status" in df.columns:
    df["isFail"] = df["status"].astype(str).str.lower().eq("fail").astype(int)
else:
    raise KeyError("No target column found. Expected one of: 'target','fail_flag','status'")

print("Positive cases (failures):", int(df["isFail"].sum()))


# Column name normalization helper
def pick_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# pick numeric cols (explicit list + auto)
numeric_candidates = [
    "past_failure_rate", "failure_rate", "suite_fail_rate", "dut_fail_rate",
    "time_since_last_failure", "execution_duration", "executiontime",
    "execution_frequency", "avg_duration", "keyword_fail",
    "rolling_anomaly_rate", "anomaly_score", "mean_anomaly_score",
    "exec_day", "exec_hour"
]
numeric_features = [c for c in numeric_candidates if c in df.columns]

# categorical candidates (we'll target-encode)
cat_candidates = ["test_case", "test_case_id", "dut", "suite", "cluster_label", "clusterlabel", "regression_group", "schedule_cycle"]
cat_present = [c for c in cat_candidates if c in df.columns]

# normalize cluster name
cluster_col = pick_column(df, ["cluster_label", "clusterlabel", "cluster"])
if cluster_col and cluster_col not in cat_present:
    cat_present.append(cluster_col)

print("Numeric features detected:", numeric_features)
print("Categorical columns detected:", cat_present)

#  Target-encode categorical features (mean fail rate)
X_cat = pd.DataFrame(index=df.index)
for c in cat_present:
    te_name = f"{c}_te"
    X_cat[te_name] = df.groupby(c)["isFail"].transform("mean").fillna(0)

print("Target-encoded categorical features:", X_cat.columns.tolist())

#  Build feature matrix
X_num = df[numeric_features].copy().fillna(0)
X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
# ensure numeric dtype
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

y = df["isFail"]

print("Total features used:", X.shape[1])

#  Train / Test split (stratified)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print("Train:", len(X_train), "Test:", len(X_test))


#  Train XGBoost
scale_pos_weight = max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum()))
model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    n_jobs=4
)
model.fit(X_train, y_train)


# Evaluate on test set
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n----- MODEL METRICS -----")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))

# ROC curve (save)
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.4f}")
plt.plot([0,1],[0,1], linestyle="--", color="gray")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.savefig(ROC_PNG)
plt.close()
print("Saved ROC curve ->", ROC_PNG)

#  Confusion matrix (save)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=[0,1])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (test set)")
plt.savefig(CM_PNG)
plt.close()
print("Saved confusion matrix ->", CM_PNG)

# Calibration curve (save)
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
plt.figure(figsize=(6,6))
plt.plot(prob_pred, prob_true, marker='o', label="Calibration")
plt.plot([0,1],[0,1], linestyle='--', color='gray', label='Perfect')
plt.xlabel("Predicted probability")
plt.ylabel("Observed frequency")
plt.title("Calibration Curve")
plt.legend()
plt.grid(True)
plt.savefig(CAL_PNG)
plt.close()
print("Saved calibration curve ->", CAL_PNG)

# Top-K accuracy (1%..20%) — save CSV
test_df = pd.DataFrame(index=X_test.index)
test_df["actual"] = y_test
test_df["pred_prob"] = y_prob

total_failures = test_df["actual"].sum()
topk_rows = []
test_sorted = test_df.sort_values("pred_prob", ascending=False)
n_test = len(test_df)

for pct in np.arange(0.01, 0.21, 0.01):
    pct_label = int(round(pct*100))
    k = max(1, math.ceil(n_test * pct))
    captured = test_sorted.head(k)["actual"].sum()
    pct_captured = round((captured / total_failures) * 100, 2) if total_failures > 0 else 0.0
    topk_rows.append({"pct": pct_label, "k": k, "captured": int(captured), "captured_pct": pct_captured})
    print(f"Top-{pct_label}% ({k} rows): {pct_captured}% failures captured")

pd.DataFrame(topk_rows).to_csv(TOPK_CSV, index=False)
print("Saved Top-K results ->", TOPK_CSV)

# Feature importance -> CSV + PNG
fi = model.feature_importances_
fi_df = pd.DataFrame({"feature": X.columns, "importance": fi}).sort_values("importance", ascending=False)
fi_df.to_csv(FI_CSV, index=False)

plt.figure(figsize=(8, max(4, min(20, len(fi_df))*0.25)))
plt.barh(fi_df.head(30)["feature"][::-1], fi_df.head(30)["importance"][::-1])
plt.xlabel("Importance")
plt.title("Top features (XGBoost)")
plt.tight_layout()
plt.savefig(FI_PNG)
plt.close()
print("Saved feature importance ->", FI_CSV, "and", FI_PNG)

# Save model & features
model.save_model(MODEL_JSON)
joblib.dump(list(X.columns), MODEL_FEATURES)
print("Saved XGBoost model ->", MODEL_JSON)
print("Saved model features ->", MODEL_FEATURES)

#Save test predictions CSV
test_out = X_test.copy()
test_out["actual"] = y_test
test_out["pred_prob"] = y_prob
test_out["pred_label"] = y_pred
test_out.to_csv(TEST_PRED_CSV, index=False)
print("Saved test predictions ->", TEST_PRED_CSV)

# compute predictions for full X (coerced numeric)
df_features = X.copy()
df["pred_prob"] = model.predict_proba(df_features)[:, 1]

# safer recency weight
if "time_since_last_failure" in df.columns:
    tslf = pd.to_numeric(df["time_since_last_failure"], errors="coerce").fillna(0).clip(0, 1000)
    df["recency_weight"] = 1 + 5 * np.exp(-tslf / 30.0)
else:
    df["recency_weight"] = 1.0

# cluster weight (use whichever cluster col exists)
if cluster_col:
    df["cluster_fail_rate"] = df.groupby(cluster_col)["isFail"].transform("mean").fillna(0)
    df["cluster_weight"] = 1 + 2 * np.clip(df["cluster_fail_rate"], 0, 1.0)
else:
    df["cluster_weight"] = 1.0

df["priority_score"] = df["pred_prob"] * df["recency_weight"] * df["cluster_weight"]

q_high = df["priority_score"].quantile(0.90)
q_med = df["priority_score"].quantile(0.60)
df["priority_group"] = df["priority_score"].apply(lambda v: "HIGH RISK" if v > q_high else ("MEDIUM RISK" if v > q_med else "LOW RISK"))

df.sort_values("priority_score", ascending=False).to_csv(PRIORITIZED_CSV, index=False)
print("Saved prioritized CSV ->", PRIORITIZED_CSV)
print("\nAll done. Outputs are in the 'outputs' folder.")
