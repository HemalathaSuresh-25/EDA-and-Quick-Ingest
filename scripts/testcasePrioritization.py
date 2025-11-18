#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

print("\n=== Loading Files ===")

classified_path = "C:/Users/hemalatha/Desktop/attest-eda/data/outputs/model_rf/classified_logs.csv"
anomaly_path    = "C:/Users/hemalatha/Desktop/attest-eda/data/outputs/anomaly_reports/anomaly_full_scores.csv"

df_c = pd.read_csv(classified_path)
df_a = pd.read_csv(anomaly_path)

# Normalize case
df_c.columns = df_c.columns.str.lower()
df_a.columns = df_a.columns.str.lower()

df_c = df_c.rename(columns={
    "test_case_id": "test_case",
    "predictedlabel": "predicted_label",
    "actualstatus": "actual_status",
    "errormsg": "error_msg",
    "logid": "log_id",
    "messagesnippet": "message_snippet"
})

df_a = df_a.rename(columns={
    "test_case_id": "test_case",
    "logid": "log_id",
    "messagesnippet": "message_snippet"
})

print("\nClassified:", df_c.shape)
print("Anomaly:", df_a.shape)

# ============================================================
# 1️⃣ Remove duplicates
# ============================================================

merge_keys = ["filename", "dut", "suite", "test_case"]

df_c = df_c.drop_duplicates()
df_c = df_c.sort_values("confidence", ascending=False).drop_duplicates(
    subset=merge_keys, keep="first"
)

print("\nClassified after key-based dedupe:", df_c.shape)

df_a = df_a.drop_duplicates()
print("Anomaly after dedupe:", df_a.shape)

# ============================================================
# 2️⃣ Merge safely (1-to-1 merge only)
# ============================================================

df = df_a.merge(df_c, on=merge_keys, how="left", suffixes=("_anom", "_clf"))

print("\nMerged shape:", df.shape)

# ============================================================
# 3️⃣ Unify Actual Status
# ============================================================

df["actual_status"] = df["actual_status"].fillna(df.get("status", ""))
df["actual_status"] = df["actual_status"].astype(str).str.upper()

# ============================================================
# 4️⃣ FAIL FLAG
# ============================================================

df["fail_flag"] = (df["actual_status"] == "FAIL").astype(int)

# ============================================================
# 5️⃣ Basic Failure Rates
# ============================================================

def safe_rate(df, col):
    if col not in df.columns:
        return 0
    return df.groupby(col)["fail_flag"].transform("mean")

df["failure_rate"]     = safe_rate(df, "test_case")
df["suite_fail_rate"]  = safe_rate(df, "suite")
df["dut_fail_rate"]    = safe_rate(df, "dut")

# ============================================================
# 6️⃣ NEW FEATURE: Regression Group
# Grouping repeated failing test cases
# ============================================================

df["regression_group"] = (
    df.groupby("test_case")["fail_flag"].transform("sum")
)

# higher value = more historically unstable testcase

# ============================================================
# 7️⃣ NEW FEATURE: Schedule Cycle
# order of execution within suite & DUT
# ============================================================

df = df.sort_values(["dut", "suite", "run_date", "timestamp"], ascending=True)

df["schedule_cycle"] = df.groupby(["dut", "suite"]).cumcount() + 1

# ============================================================
# 8️⃣ NEW FEATURE: Past Failure Frequency
# how many times this testcase failed before current run
# ============================================================

df["past_failure_frequency"] = (
    df.groupby("test_case")["fail_flag"].cumsum() - df["fail_flag"]
)

# ============================================================
# 9️⃣ Remove unwanted columns
# ============================================================

drop_cols = ["error_msg_anom", "error_msg_clf"]

df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ============================================================
# 🔟 FINAL CLEANUP
# ============================================================

df_final = df.drop_duplicates()
print("\nFinal:", df_final.shape)

df_final.to_csv("data/outputs/feature_engineered_testcases.csv", index=False)
print("\n✔ Saved to: data/outputs/feature_engineered_testcases.csv")
