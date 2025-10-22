import os
import pandas as pd
from collections import Counter

#Configuration
INPUT_FILE = "data/cluster/failure_clusters.csv"
OUTPUT_FILE = "data/failure_patterns_labeled_human.csv"
OUTPUT_EXAMPLES = "data/outputs/root_cause_examples_human.csv"

os.makedirs("data/outputs", exist_ok=True)

#Load Data
df = pd.read_csv(INPUT_FILE)
if "cluster" not in df.columns or "error_msg" not in df.columns:
    raise KeyError("Input CSV must contain 'cluster' and 'error_msg' columns")

df = df[df["cluster"] != -1]
df["error_msg"] = df["error_msg"].fillna("No Error")

#Extract keywords per cluster
def extract_keywords(df, top_n=5):
    cluster_keywords = {}
    for cid, subset in df.groupby("cluster"):
        words = " ".join(subset["error_msg"].astype(str)).lower().split()
        stop_words = {"the","and","for","to","in","of","on","with","is","a","at"}
        filtered = [w for w in words if w not in stop_words]
        common = [w for w,_ in Counter(filtered).most_common(top_n)]
        cluster_keywords[cid] = common
    return cluster_keywords

cluster_keywords = extract_keywords(df)

#Map clusters to human-readable root causes
HUMAN_LABELS = {
    0: "Interface / Port Mismatch",               # p1, port, transmit, message, parameters
    1: "Capture / ID Handling Error",             # capture, id, exist, error, does
    2: "CLI / Command Execution Failure",         # cli, stopping, failed, ptp, 18
    3: "Test Result / Validation Issue",          # correct, result, test, wri1, 51
    4: "PTP Transmission / PDELAY_RESP Failure",  # transmit, message, dut, does, pdelay_resp
    5: "Bit / Marker Configuration Error",        # bit, marker, segments, true, set
    6: "DUT Configuration Value Error",           # configured, value, dut, priority2, priority1
    7: "DUT Port / State Mismatch",               # dut, port, state, does, message
    8: "Announce / SIP Transmission Error",       # announce, dut, sent, sip, port
    9: "PTP Command / Domain Configuration Error" # command, ptp, v2bc, domain, v2tc
}

df["root_cause_label"] = df["cluster"].map(lambda c: HUMAN_LABELS.get(c, "Uncategorized"))
df["keywords"] = df["cluster"].map(lambda c: ", ".join(cluster_keywords.get(c, [])))

#Build representative examples 
examples = []
for cid, subset in df.groupby("cluster"):
    label = subset["root_cause_label"].mode().iat[0]
    example_msgs = subset["error_msg"].head(3).tolist()
    examples.append({
        "cluster": cid,
        "root_cause_label": label,
        "keywords": ", ".join(cluster_keywords.get(cid, [])),
        "example_errors": " | ".join(example_msgs)
    })

examples_df = pd.DataFrame(examples)

#Save results 
df.to_csv(OUTPUT_FILE, index=False)
examples_df.to_csv(OUTPUT_EXAMPLES, index=False)

print(f"Human-readable root cause labeling complete → {OUTPUT_FILE}")
print(f"Representative examples saved → {OUTPUT_EXAMPLES}")
