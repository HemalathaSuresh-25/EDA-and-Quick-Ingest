import pandas as pd
import os

# ==============================
# PATH CONFIG
# ==============================
INPUT_FILE = "C:/Users/hemalatha/Desktop/attest-eda/data/cluster/all_clusters_merged.csv"
OUTPUT_FILE = "C:/Users/hemalatha/Desktop/attest-eda/data/cluster/root_cause_tagged.csv"

# ==============================
# ROOT CAUSE MAPS
# ==============================

FAIL_ROOT_CAUSE_MAP = {
    0: "Configuration Mismatch",
    1: "Missing Resource / Identifier",
    2: "Packet Transmission Failure",
    3: "Port / Interface Mismatch",
    4: "Incorrect Te",
    6: "PTP Protocol Failure",
    7: "General DUT Communist Result / Assertion Failure",
    5: "CLI Command Failurecation Failure",
    8: "Bit / Field Validation Error",
    9: "Clock / Timing Type Mismatch",
}

ABORT_ROOT_CAUSE_MAP = {
    0: "Media Stream Not Established",
    1: "Profile / Standard Not Supported",
    2: "Transport / Messaging Failure",
    3: "Test Not Applicable",
    4: "Precondition / Setup Failure",
    5: "Invalid Test Configuration",
    6: "Transport Protocol Mismatch",
    7: "User / Registration Incomplete",
}

# ==============================
# MAIN
# ==============================
def tag_root_causes():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Rows loaded: {df.shape[0]}")

    # ------------------------------
    # Initialize columns
    # ------------------------------
    df["root_cause_fail"] = "Normal"
    df["root_cause_abort"] = "Normal"

    # ------------------------------
    # FAIL root cause tagging
    # ------------------------------
    fail_mask = (df["status"] == "FAIL") & (df["fail_cluster"] != -1)

    df.loc[fail_mask, "root_cause_fail"] = (
        df.loc[fail_mask, "fail_cluster"]
        .map(FAIL_ROOT_CAUSE_MAP)
        .fillna("Unknown Failure Cause")
    )

    # ------------------------------
    # ABORT root cause tagging
    # ------------------------------
    abort_mask = (df["status"] == "ABORT") & (df["abort_cluster"] != -1)

    df.loc[abort_mask, "root_cause_abort"] = (
        df.loc[abort_mask, "abort_cluster"]
        .map(ABORT_ROOT_CAUSE_MAP)
        .fillna("Unknown Abort Cause")
    )

    # ------------------------------
    # Save
    # ------------------------------
    df.to_csv(OUTPUT_FILE, index=False)

    print("Root-cause tagging completed!")
    print(f"Saved → {OUTPUT_FILE}")
    print("Column behavior:")
    print(" - FAIL rows → root_cause_fail populated")
    print(" - ABORT rows → root_cause_abort populated")
    print(" - Others → Normal")

    return df


if __name__ == "__main__":
    tag_root_causes()
