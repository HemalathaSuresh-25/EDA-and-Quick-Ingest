import os
import pandas as pd
from collections import Counter

# Configuration
INPUT_FILE = "data/cluster/failure_clusters.csv"
OUTPUT_FILE = "data/failure_patterns_labeled_human.csv"
OUTPUT_EXAMPLES = "data/outputs/root_cause_examples_human.csv"

os.makedirs("data/outputs", exist_ok=True)

# Load Data
df = pd.read_csv(INPUT_FILE)
if "cluster" not in df.columns or "error_msg" not in df.columns:
    raise KeyError("Input CSV must contain 'cluster' and 'error_msg' columns")

df["error_msg"] = df["error_msg"].fillna("No Error")


# Extract keywords per cluster (ignore -1 for now)
def extract_keywords(df, top_n=5):
    cluster_keywords = {}
    for cid, subset in df[df["cluster"] != -1].groupby("cluster"):
        text = " ".join(subset["error_msg"].astype(str)).lower()
        words = [w.strip(".,:;#") for w in text.split()]
        stop_words = {"the", "and", "for", "to", "in", "of", "on", "with", "is", "a", "at", "no", "error"}
        filtered = [w for w in words if w not in stop_words and len(w) > 2]
        common = [w for w, _ in Counter(filtered).most_common(top_n)]
        cluster_keywords[cid] = common if common else ["<no_keywords>"]
    return cluster_keywords

cluster_keywords = extract_keywords(df)

print(f"Extracted keywords for {len(cluster_keywords)} clusters")


# Map clusters to human-readable root causes
HUMAN_LABELS = {
    0: "Interface / Port Mismatch",
    1: "Capture / ID Handling Error",
    2: "CLI / Command Execution Failure",
    3: "Test Result / Validation Issue",
    4: "PTP Transmission / PDELAY_RESP Failure",
    5: "Bit / Marker Configuration Error",
    6: "DUT Configuration Value Error",
    7: "DUT Port / State Mismatch",
    8: "Announce / SIP Transmission Error",
    9: "PTP Command / Domain Configuration Error"
}

df["root_cause_label"] = df["cluster"].map(HUMAN_LABELS)
df["root_cause_label"] = df["root_cause_label"].fillna("Normal")


# Assign keywords (normal logs → “No keywords”)
df["keywords"] = df["cluster"].map(lambda c: ", ".join(cluster_keywords.get(c, [])))
df.loc[df["cluster"] == -1, "keywords"] = "No keywords"

# Representative examples (for failure clusters only)
examples = []
for cid, subset in df[df["cluster"] != -1].groupby("cluster"):
    label = subset["root_cause_label"].mode().iat[0]
    example_msgs = subset["error_msg"].head(3).tolist()
    examples.append({
        "cluster": cid,
        "root_cause_label": label,
        "keywords": ", ".join(cluster_keywords.get(cid, [])),
        "example_errors": " | ".join(example_msgs)
    })

examples_df = pd.DataFrame(examples)

# Save Results
df.to_csv(OUTPUT_FILE, index=False)
examples_df.to_csv(OUTPUT_EXAMPLES, index=False)

print(f" Root cause labeling complete → {OUTPUT_FILE}")
print(f" Representative examples saved → {OUTPUT_EXAMPLES}")
print("\n Root cause label counts:")
print(df["root_cause_label"].value_counts())
