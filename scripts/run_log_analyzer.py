#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import normalize

# ================================================================
#                   LOAD TRAINED MODELS
# ================================================================

cluster_model = joblib.load("models/cluster_model.pkl")
bert_encoder = joblib.load("models/bert_encoder.pkl")
xgb_model = xgb.Booster()
xgb_model.load_model("models/xgb_model.json")
model_features = joblib.load("models/model_features.joblib")  # feature list

# ================================================================
#          LOAD FAILURE → RECOMMENDATION KNOWLEDGE BASE
# ================================================================

with open("models/recommendation_rules.json") as f:
    recommendations = json.load(f)

# ================================================================
#                 HUMAN LABELS FOR CLUSTER IDS
# ================================================================

HUMAN_LABELS = {
    0: "Interface / Port Mismatch",
    1: "Capture / ID Handling Error",
    2: "CLI / Command Execution Failure",
    3: "Test Result / Validation Issue",
    4: "PTP Transmission / PDELAY_RESP Failure",
    5: "BIT / Marker Configuration Error",
    6: "DUT Configuration Value Error",
    7: "Port State / Port Mismatch",
    8: "Announce / SIP Transmission Error",
    9: "PTP Command / Domain Configuration Error"
}
#          LOAD FEATURE CSV

features_csv_path = "C:/Users/hemalatha/Desktop/attest-eda/data/task4output/prioritized_testcases_xgb.csv"
features_df = pd.read_csv(features_csv_path)

# Create missing features if required
for f in model_features:
    if f not in features_df.columns:
        base = f.replace("_te", "")
        if base in features_df.columns:
            features_df[f] = features_df.groupby(base)["isFail"].transform("mean").fillna(0)
        else:
            features_df[f] = 0.0  # default

#   MAIN ANALYZER FUNCTION

def analyze_log(log_path):

    print(f"\nAnalyzing Log File:\n{log_path}\n")

    if not os.path.exists(log_path):
        print(" ERROR: File not found")
        return
    
    raw = open(log_path, "r", errors="ignore").read().lower()
    log_name = os.path.basename(log_path)

    if log_name not in features_df['filename'].values:
        print(" No features available for this log in CSV")
        return

    row = features_df[features_df["filename"] == log_name].iloc[0]
    xgb_input = row[model_features].to_numpy().reshape(1, -1)
    pred = float(xgb_model.predict(xgb.DMatrix(xgb_input, feature_names=model_features))[0])

    # 3-CLASS DECISION
    if any(x in raw for x in ["abort", "terminated", "stopped", "exit"]):
        status = "ABORT"
        status_conf = round(1 - pred, 3)

    else:
        status = "FAIL" if pred >= 0.50 else "PASS"
        status_conf = pred if status=="FAIL" else round(1-pred,3)

    # FAIL ONLY — Cluster + Reason + Fix
    if status == "FAIL":
        x_bert = normalize(bert_encoder.encode([raw]))
        cluster = int(cluster_model.predict(x_bert)[0])
        cluster_name = HUMAN_LABELS.get(cluster,"Unknown Cluster")

        detected = next((k for k in recommendations if k in raw), "unknown")
        fix = recommendations.get(detected,"No recommended fix available")


    print("=========== RESULT ===========")
    print(f"Status Prediction        : {status}")
    print(f"Model Confidence         : {status_conf}")

    if status == "FAIL":
        print(f"Failure Probability      : {round(pred,3)}")
        print(f"Cluster                  : {cluster_name}")
        print(f"Detected Reason          : {detected}")
        print(f"Recommendation           : {fix}")
    else:
        print("No reason analysis or fix shown for PASS / ABORT logs.")

    print("====================================\n")

if __name__ == "__main__":

    analyze_log(r"C:\Users\hemalatha\Desktop\attest-eda\data\standardized\2024-12-09\generic_dut\dtmf\tc_conf_dtmf_pvg_005_20241209-210619.log")