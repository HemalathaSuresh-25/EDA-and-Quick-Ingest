import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Configuration
INPUT_FILE = "data/failure_patterns_labeled_human.csv"
OUTPUT_DIR = "data/outputs/charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#Load Data
df = pd.read_csv(INPUT_FILE)
if "root_cause_label" not in df.columns:
    raise KeyError("Input CSV must contain 'root_cause_label' column")

#Root Cause Frequency 
root_cause_counts = df["root_cause_label"].value_counts()
plt.figure(figsize=(10,6))
sns.barplot(x=root_cause_counts.values, y=root_cause_counts.index, palette="Set2")
plt.title("Root Cause Frequency")
plt.xlabel("Number of Failures")
plt.ylabel("Root Cause")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "root_cause_frequency.png"))
plt.close()

#Cluster Distribution
if "cluster" in df.columns:
    cluster_counts = df["cluster"].value_counts().sort_index()
    plt.figure(figsize=(10,5))
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="Set3")
    plt.title("Cluster Distribution")
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of Failures")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cluster_distribution.png"))
    plt.close()

#Top Keywords per Root Cause
top_keywords_summary = []
for label, subset in df.groupby("root_cause_label"):
    all_keywords = " ".join(subset["keywords"].fillna("")).split(", ")
    most_common = pd.Series(all_keywords).value_counts().head(5).index.tolist()
    top_keywords_summary.append({
        "root_cause_label": label,
        "top_keywords": ", ".join(most_common)
    })

keywords_df = pd.DataFrame(top_keywords_summary)
keywords_df.to_csv(os.path.join(OUTPUT_DIR, "top_keywords_per_root_cause.csv"), index=False)

#Print Summary
print("Charts and reports generated in:", OUTPUT_DIR)
print("\nTop 5 Root Cause Frequency:\n", root_cause_counts.head())
