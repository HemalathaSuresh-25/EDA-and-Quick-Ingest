#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================
INPUT_FILE = "C:/Users/hemalatha/Desktop/attest-eda/data/cluster/root_cause_tagged.csv"
OUTPUT_DIR = "C:/Users/hemalatha/Desktop/attest-eda/data/splits"

TEST_SIZE = 0.20
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_train_test_split():
    print("🔄 Loading dataset...")
    df = pd.read_csv(INPUT_FILE)
    print(f"✔ Rows loaded: {len(df)}")

    # -------------------------
    # Normalize status
    # -------------------------
    df["status"] = df["status"].astype(str).str.strip().str.upper()

    # -------------------------
    # Status label
    # -------------------------
    status_map = {"PASS": 0, "FAIL": 1, "ABORT": 2}
    df["status_label"] = df["status"].map(status_map)

    # -------------------------
    # Root cause label
    # -------------------------
    def root_cause(row):
        if row["status"] == "FAIL":
            return row.get("root_cause_fail", "UNKNOWN_FAIL")
        elif row["status"] == "ABORT":
            return row.get("root_cause_abort", "UNKNOWN_ABORT")
        return "NORMAL"

    df["root_cause_label"] = df.apply(root_cause, axis=1)

    # -------------------------
    # Drop non-ML columns
    # -------------------------
    drop_cols = [
        "status",
        "root_cause_fail",
        "root_cause_abort"
    ]

    df_ml = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # -------------------------
    # Train / Test split
    # -------------------------
    train_df, test_df = train_test_split(
        df_ml,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_ml["status_label"]
    )

    # -------------------------
    # Save
    # -------------------------
    train_path = f"{OUTPUT_DIR}/train.csv"
    test_path = f"{OUTPUT_DIR}/test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n Train/Test split completed")
    print(f" Train file: {train_path} ({len(train_df)})")
    print(f" Test file : {test_path} ({len(test_df)})")
    print(f" Status distribution:\n{df['status'].value_counts()}\n")


if __name__ == "__main__":
    create_train_test_split()
