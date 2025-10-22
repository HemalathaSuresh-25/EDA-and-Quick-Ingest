import os
import pandas as pd
import random
 
INPUT_FILE = "C:/Users/hemalatha/Desktop/attest-eda/data/logs_preprocessed.csv"
OUTPUT_DIR = "C:/Users/hemalatha/Desktop/attest-eda/data/features"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "failure_features.csv")


#Feature Generation 
def generate_features():
    print("Starting feature engineering...")

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    #Normalize columns 
    df["status"] = df["status"].astype(str).str.strip().str.upper()
    df = df[df["status"].isin(["PASS", "FAIL", "ABORT"])]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    #Derived Features
    fail_df = df[df["status"] == "FAIL"]
    suite_fail_freq = fail_df.groupby("suite")["status"].count().rename("failure_freq_suite")
    dut_fail_freq = fail_df.groupby("dut")["status"].count().rename("failure_freq_dut")
    df = df.merge(suite_fail_freq, on="suite", how="left")
    df = df.merge(dut_fail_freq, on="dut", how="left")
    df["failure_freq_suite"].fillna(0, inplace=True)
    df["failure_freq_dut"].fillna(0, inplace=True)

    #Random Execution Duration
    df["execution_duration"] = df.apply(lambda _: random.randint(10, 60), axis=1)

    #Time Since Last Failure
    df["time_since_last_failure"] = 0
    for (dut, tc), group in df.groupby(["dut", "test_case_id"]):
        last_fail_time = None
        times = []
        for ts, status in zip(group["timestamp"], group["status"]):
            if pd.isna(ts):
                times.append(0)
                continue
            if status == "FAIL":
                if last_fail_time is None:
                    times.append(0)
                else:
                    times.append((ts - last_fail_time).total_seconds())
                last_fail_time = ts
            else:
                times.append(0)
        df.loc[group.index, "time_since_last_failure"] = times

    #Recent Failure Flag
    df["recent_failure_flag"] = df["status"].apply(lambda x: 1 if x == "FAIL" else 0)

    #Encode Config / Environment
    df["config_hash"] = df["config"].astype(str).apply(lambda x: abs(hash(x)) % (10 ** 8))
    df["dut_version"] = df["dut_version"].astype(str).fillna("Unknown")

    #Error / Failure Message
    df["error_msg"] = df.apply(
        lambda row: "No Error" if row["status"] == "PASS" else row["error_msg"], axis=1
    )
    df["error_msg"] = df["error_msg"].fillna("No Error")

    #Save Features
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Random execution_duration generated → {OUTPUT_FILE}")
    print("Feature engineering complete!\n")
    return df

if __name__ == "__main__":
    generate_features()
