import os
import pandas as pd
import random

INPUT_FILE = "C:/Users/hemalatha/Desktop/attest-eda/data/logs_preprocessed.csv"
OUTPUT_DIR = "C:/Users/hemalatha/Desktop/attest-eda/data/features"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "failure_features.csv")


# Feature Generation 
def generate_features():
    print("Starting feature engineering...")

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Normalize status column (correct Pandas string accessor .str.strip())
    df["status"] = df["status"].astype(str).str.strip().str.upper()
    df = df[df["status"].isin(["PASS", "FAIL", "ABORT"])]

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Sort so "time since last ..." is computed chronologically per DUT+test_case
    df = df.sort_values(["dut", "test_case_id", "timestamp"]).reset_index(drop=False)

    # ================================
    # FAIL-based frequency features
    # ================================
    fail_df = df[df["status"] == "FAIL"]
    suite_fail_freq = fail_df.groupby("suite")["status"].count().rename("failure_freq_suite")
    dut_fail_freq = fail_df.groupby("dut")["status"].count().rename("failure_freq_dut")

    df = df.merge(suite_fail_freq, on="suite", how="left")
    df = df.merge(dut_fail_freq, on="dut", how="left")
    df["failure_freq_suite"].fillna(0, inplace=True)
    df["failure_freq_dut"].fillna(0, inplace=True)

    # ================================
    # ABORT-based frequency features
    # ================================
    abort_df = df[df["status"] == "ABORT"]
    suite_abort_freq = abort_df.groupby("suite")["status"].count().rename("abort_freq_suite")
    dut_abort_freq = abort_df.groupby("dut")["status"].count().rename("abort_freq_dut")

    df = df.merge(suite_abort_freq, on="suite", how="left")
    df = df.merge(dut_abort_freq, on="dut", how="left")
    df["abort_freq_suite"].fillna(0, inplace=True)
    df["abort_freq_dut"].fillna(0, inplace=True)

    # Random execution duration
    df["execution_duration"] = df.apply(lambda _: random.randint(10, 60), axis=1)

    # ================================
    # TIME SINCE LAST FAILURE
    # ================================
    df["time_since_last_failure"] = 0.0
    for (dut, tc), group in df.groupby(["dut", "test_case_id"], sort=False):
        last_fail_time = None
        times = []
        for ts, status in zip(group["timestamp"], group["status"]):
            if pd.isna(ts):
                times.append(0.0)
                continue
            if status == "FAIL":
                if last_fail_time is None:
                    times.append(0.0)
                else:
                    times.append((ts - last_fail_time).total_seconds())
                last_fail_time = ts
            else:
                times.append(0.0)
        df.loc[group.index, "time_since_last_failure"] = times

    # ================================
    # TIME SINCE LAST ABORT
    # ================================
    df["time_since_last_abort"] = 0.0
    for (dut, tc), group in df.groupby(["dut", "test_case_id"], sort=False):
        last_abort_time = None
        times = []
        for ts, status in zip(group["timestamp"], group["status"]):
            if pd.isna(ts):
                times.append(0.0)
                continue
            if status == "ABORT":
                if last_abort_time is None:
                    times.append(0.0)
                else:
                    times.append((ts - last_abort_time).total_seconds())
                last_abort_time = ts
            else:
                times.append(0.0)
        df.loc[group.index, "time_since_last_abort"] = times

    # ================================
    # Recent Flags
    # FAIL → 1, ABORT → 2, PASS → 0
    # ================================
    df["recent_failure_flag"] = df["status"].apply(lambda x: 1 if x == "FAIL" else 0)
    df["recent_abort_flag"] = df["status"].apply(lambda x: 2 if x == "ABORT" else 0)

    # Combined numeric status flag (optional)
    def flag_status(s):
        if s == "FAIL":
            return 1
        elif s == "ABORT":
            return 2
        return 0

    df["recent_status_flag"] = df["status"].apply(flag_status)

    # Encoding config/environment
    # fillna before astype to avoid string "nan"
    if "dut_version" in df.columns:
        df["dut_version"] = df["dut_version"].fillna("Unknown").astype(str)
    else:
        df["dut_version"] = "Unknown"

    if "config" in df.columns:
        df["config_hash"] = df["config"].astype(str).apply(lambda x: abs(hash(x)) % (10 ** 8))
    else:
        df["config_hash"] = 0

    # Error message handling
    if "error_msg" in df.columns:
        df["error_msg"] = df.apply(
            lambda row: "No Error" if row["status"] == "PASS" else row["error_msg"],
            axis=1
        )
        df["error_msg"] = df["error_msg"].fillna("No Error")
    else:
        df["error_msg"] = "No Error"

    # Save (use original index column name 'index' if you want to keep it, or drop it)
    # remove the temporary 'index' column created by reset_index if you don't need it
    if "index" in df.columns:
        df = df.drop(columns=["index"])

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Features saved → {OUTPUT_FILE}")
    print("Feature engineering complete!\n")
    return df


if __name__ == "__main__":
    generate_features()
