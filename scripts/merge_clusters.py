import pandas as pd

FAIL_FILE = "C:/Users/hemalatha/Desktop/attest-eda/data/cluster/failure_clusters.csv"
ABORT_FILE = "C:/Users/hemalatha/Desktop/attest-eda/data/cluster/abort_clusters.csv"
OUTPUT_FILE = "C:/Users/hemalatha/Desktop/attest-eda/data/cluster/all_clusters_merged.csv"


def merge_cluster_files():
    df_fail = pd.read_csv(FAIL_FILE)
    df_abort = pd.read_csv(ABORT_FILE)

    # Ensure row_id exists
    if "row_id" not in df_fail.columns or "row_id" not in df_abort.columns:
        raise ValueError("row_id column missing. Add row_id during feature engineering.")

    # Keep only abort_cluster from abort file
    df_abort = df_abort[["row_id", "abort_cluster"]]

    # LEFT JOIN on row_id (1-to-1 guaranteed)
    df_merged = df_fail.merge(
        df_abort,
        on="row_id",
        how="left",
        validate="one_to_one"
    )

    df_merged["abort_cluster"] = df_merged["abort_cluster"].fillna(-1).astype(int)

    print(f"Rows after merge: {len(df_merged)} (expected: {len(df_fail)})")

    df_merged.to_csv(OUTPUT_FILE, index=False)
    print(f"Merged cluster file saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    merge_cluster_files()
