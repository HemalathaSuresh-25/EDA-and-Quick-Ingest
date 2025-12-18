#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import joblib
import pickle
import pandas as pd
from datetime import datetime

# =========================
# CONFIG
# =========================
MODEL_DIR = "C:/Users/hemalatha/Desktop/attest-eda/models_1"

STATUS_MODEL_FILE = os.path.join(MODEL_DIR, "status_classifier_rf.pkl")
STATUS_FEATURES_FILE = os.path.join(MODEL_DIR, "status_features.pkl")

FAIL_MODEL_FILE  = os.path.join(MODEL_DIR, "rootcause_fail_classifier.pkl")
TFIDF_FAIL_FILE  = os.path.join(MODEL_DIR, "tfidf_rootcause_fail.pkl")

ABORT_MODEL_FILE = os.path.join(MODEL_DIR, "rootcause_abort_classifier.pkl")
TFIDF_ABORT_FILE = os.path.join(MODEL_DIR, "tfidf_rootcause_abort.pkl")

# =========================
# LOAD MODELS
# =========================
print("🔄 Loading ML models...")

status_model = joblib.load(STATUS_MODEL_FILE)

with open(STATUS_FEATURES_FILE, "rb") as f:
    status_features = pickle.load(f)

fail_model  = joblib.load(FAIL_MODEL_FILE)
tfidf_fail  = joblib.load(TFIDF_FAIL_FILE)

abort_model = joblib.load(ABORT_MODEL_FILE)
tfidf_abort = joblib.load(TFIDF_ABORT_FILE)

print("✔ Models loaded\n")

# =========================
# HELPERS
# =========================
def read_log(path):
    with open(path, "r", errors="ignore") as f:
        return f.read()

def detect_pass(log):
    return bool(re.search(r"Result:\s*PASSED", log, re.IGNORECASE))

def extract_error_text(log):
    return " ".join(
        l for l in log.splitlines()
        if re.search(r'error|fail|abort|timeout|exception', l, re.IGNORECASE)
    ) or "No Error"

# =========================
# FEATURE BUILDER (MATCH TRAINING)
# =========================
def build_feature_row(log_path, log_text):
    now = datetime.now()

    return {
        "filename": os.path.basename(log_path),
        "dut": "generic_dut",
        "dut_version": "unknown",
        "os_version": "unknown",
        "config": "default",
        "test_case_id": os.path.basename(log_path).split("_")[0],
        "line_number": len(log_text.splitlines()),
        "timestamp": now.strftime("%H:%M:%S"),
        "run_date": now.strftime("%Y-%m-%d"),
        "suite": "ptp",
        "raw_line": extract_error_text(log_text),
        "row_id": 0,

        "failure_freq_suite": log_text.lower().count("fail"),
        "failure_freq_dut": log_text.lower().count("fail"),
        "abort_freq_suite": log_text.lower().count("abort"),
        "abort_freq_dut": log_text.lower().count("abort"),

        "execution_duration": 0,
        "time_since_last_failure": 9999,
        "time_since_last_abort": 9999,

        "recent_failure_flag": int("fail" in log_text.lower()),
        "recent_abort_flag": int("abort" in log_text.lower()),
        "recent_status_flag": 0,

        "config_hash": hash("default") % 100000,
        "fail_cluster": -1,
        "abort_cluster": -1
    }

def recommendation(reason):
    mapping = {
        "Timeout Error": "Check DUT connectivity and retry",
        "Protocol Failure": "Verify protocol configuration",
        "Invalid Configuration": "Validate test setup",
        "Precondition Failure": "Check DUT preconditions"
    }
    return mapping.get(reason, "Inspect log manually")

# =========================
# MAIN
# =========================
def analyze_log(log_path):
    print(f"📂 Analyzing log file: {log_path}\n")
    log_text = read_log(log_path)

    # ---------- PASS OVERRIDE ----------
    if detect_pass(log_text):
        print("✅ Analysis Result")
        print("Status        : PASS")
        print("Reason        : Result: PASSED found in log")
        print("Recommendation: No action required")
        return

    # ---------- STATUS ML ----------
    row = build_feature_row(log_path, log_text)
    df = pd.DataFrame([row])[status_features]

    status_pred = status_model.predict(df)[0]
    status_map = {0: "PASS", 1: "FAIL", 2: "ABORT"}
    status = status_map.get(status_pred, "FAIL")

    # ---------- ROOT CAUSE ----------
    error_text = row["raw_line"]
    reason = "Unknown"
    rec = "Inspect log manually"

    if status == "FAIL":
        vec = tfidf_fail.transform([error_text])
        reason = fail_model.predict(vec)[0]
        rec = recommendation(reason)

    elif status == "ABORT":
        vec = tfidf_abort.transform([error_text])
        reason = abort_model.predict(vec)[0]
        rec = recommendation(reason)

    # ---------- OUTPUT ----------
    print("✅ Analysis Result")
    print(f"Status        : {status}")
    print(f"Reason        : {reason}")
    print(f"Recommendation: {rec}")

# =========================
# CLI
# =========================
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python log_analyzer.py <log_file.log>")
        sys.exit(1)

    analyze_log(sys.argv[1])
