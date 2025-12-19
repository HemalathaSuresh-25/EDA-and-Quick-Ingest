#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import random

# ==========================
# PATH CONFIG
# ==========================
INPUT_FILE = "C:/Users/hemalatha/Desktop/attest-eda/data/logs_preprocessed.csv"
OUTPUT_DIR = "C:/Users/hemalatha/Desktop/attest-eda/data/features"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "failure_features.csv")


# ==========================
# FEATURE ENGINEERING
# ==========================
def generate_features():
    print("Starting feature engineering...")

    # --------------------------
    # Load data + row_id
    # --------------------------
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df["row_id"] = df.index  # 🔑 stable row identifier
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # --------------------------
    # Normalize status
    # --------------------------
    df["status"] = (
        df["status"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df = df[df["status"].isin(["PASS", "FAIL", "ABORT"])].reset_index(drop=True)

    # --------------------------
    # Parse timestamp
    # --------------------------
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Sort for temporal features
    df = df.sort_values(
        ["dut", "test_case_id", "timestamp"],
        na_position="last"
    ).reset_index(drop=True)

    # ==========================
    # FAIL frequency features
    # ==========================
    fail_df = df[df["status"] == "FAIL"]

    suite_fail_freq = (
        fail_df.groupby("suite")
        .size()
        .rename("failure_freq_suite")
    )

    dut_fail_freq = (
        fail_df.groupby("dut")
        .size()
        .rename("failure_freq_dut")
    )

    df = df.merge(suite_fail_freq, on="suite", how="left")
    df = df.merge(dut_fail_freq, on="dut", how="left")

    df["failure_freq_suite"] = df["failure_freq_suite"].fillna(0).astype(int)
    df["failure_freq_dut"] = df["failure_freq_dut"].fillna(0).astype(int)

    # ==========================
    # ABORT frequency features
    # ==========================
    abort_df = df[df["status"] == "ABORT"]

    suite_abort_freq = (
        abort_df.groupby("suite")
        .size()
        .rename("abort_freq_suite")
    )

    dut_abort_freq = (
        abort_df.groupby("dut")
        .size()
        .rename("abort_freq_dut")
    )

    df = df.merge(suite_abort_freq, on="suite", how="left")
    df = df.merge(dut_abort_freq, on="dut", how="left")

    df["abort_freq_suite"] = df["abort_freq_suite"].fillna(0).astype(int)
    df["abort_freq_dut"] = df["abort_freq_dut"].fillna(0).astype(int)

    # ==========================
    # Execution duration (synthetic)
    # ==========================
    random.seed(42)
    df["execution_duration"] = [random.randint(10, 60) for _ in range(len(df))]

    # ==========================
    # Time since last FAIL
    # ==========================
    df["time_since_last_failure"] = 0.0

    for (dut, tc), group in df.groupby(["dut", "test_case_id"], sort=False):
        last_fail_ts = None
        deltas = []

        for ts, status in zip(group["timestamp"], group["status"]):
            if status == "FAIL" and pd.notna(ts):
                if last_fail_ts is None:
                    deltas.append(0.0)
                else:
                    deltas.append((ts - last_fail_ts).total_seconds())
                last_fail_ts = ts
            else:
                deltas.append(0.0)

        df.loc[group.index, "time_since_last_failure"] = deltas

    # ==========================
    # Time since last ABORT
    # ==========================
    df["time_since_last_abort"] = 0.0

    for (dut, tc), group in df.groupby(["dut", "test_case_id"], sort=False):
        last_abort_ts = None
        deltas = []

        for ts, status in zip(group["timestamp"], group["status"]):
            if status == "ABORT" and pd.notna(ts):
                if last_abort_ts is None:
                    deltas.append(0.0)
                else:
                    deltas.append((ts - last_abort_ts).total_seconds())
                last_abort_ts = ts
            else:
                deltas.append(0.0)

        df.loc[group.index, "time_since_last_abort"] = deltas

    # ==========================
    # Recent status flags
    # ==========================
    df["recent_failure_flag"] = (df["status"] == "FAIL").astype(int)
    df["recent_abort_flag"] = (df["status"] == "ABORT").astype(int)

    def status_flag(s):
        if s == "FAIL":
            return 1
        elif s == "ABORT":
            return 2
        return 0

    df["recent_status_flag"] = df["status"].apply(status_flag)

    # ==========================
    # Config / environment encoding
    # ==========================
    if "dut_version" in df.columns:
        df["dut_version"] = df["dut_version"].fillna("Unknown").astype(str)
    else:
        df["dut_version"] = "Unknown"

    if "config" in df.columns:
        df["config_hash"] = (
            df["config"]
            .fillna("Unknown")
            .astype(str)
            .apply(lambda x: abs(hash(x)) % (10**8))
        )
    else:
        df["config_hash"] = 0

    # ==========================
    # Error message handling
    # ==========================
    if "error_msg" in df.columns:
        df["error_msg"] = df["error_msg"].fillna("No Error")
        df.loc[df["status"] == "PASS", "error_msg"] = "No Error"
    else:
        df["error_msg"] = "No Error"

    # ==========================
    # Save output
    # ==========================
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Features saved → {OUTPUT_FILE}")
    print(f"Final shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Feature engineering complete!\n")

    return df


# ==========================
# ENTRY POINT
# ==========================
if __name__ == "__main__":
    generate_features()
