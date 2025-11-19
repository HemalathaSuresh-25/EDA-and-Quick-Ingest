"""
generate_visual_dashboard.py

Reads:  data/task4output/prioritized_testcases_xgb.csv
Optional: data/task4output/feature_importance.csv

Produces (saved to same folder):
 - failure_risk_heatmap.png
 - probability_distribution.png
 - top10_highrisk.png
 - suite_failure_trend.png
 - feature_importance.png
 - dashboard.html (embeds the PNGs)
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

OUTDIR = "data/task4output"
PRIORITIZED_CSV = os.path.join(OUTDIR, "prioritized_testcases_xgb.csv")
FI_CSV = os.path.join(OUTDIR, "feature_importance.csv")

os.makedirs(OUTDIR, exist_ok=True)
sns.set(style="whitegrid")

def safe_read_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


# Load prioritized CSV
df = safe_read_csv(PRIORITIZED_CSV)
if df is None:
    raise FileNotFoundError(f"Prioritized CSV not found: {PRIORITIZED_CSV}")

print(f"Loaded {len(df)} rows from {PRIORITIZED_CSV}")

# Ensure columns exist
if "predicted_probability" not in df.columns and "pred_prob" in df.columns:
    df["predicted_probability"] = df["pred_prob"]

if "priority_score" not in df.columns and "priority" in df.columns:
    df["priority_score"] = df["priority"]

tc_col = "test_case" if "test_case" in df.columns else ("test_case_id" if "test_case_id" in df.columns else None)

#  Probability Distribution
dist_png = os.path.join(OUTDIR, "probability_distribution.png")
plt.figure(figsize=(7,4))
if "predicted_probability" in df.columns:
    sns.histplot(df["predicted_probability"].dropna(), bins=40, kde=True)
    plt.title("Distribution of Predicted Failure Probability")
    plt.xlabel("Predicted probability")
plt.tight_layout()
plt.savefig(dist_png, dpi=150)
plt.close()
print("Saved:", dist_png)

#  Top 10 High-Risk Test Cases
top10_png = os.path.join(OUTDIR, "top10_highrisk.png")
plt.figure(figsize=(8,4))
if tc_col and "priority_score" in df.columns:
    top10 = df.groupby(tc_col)["priority_score"].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=top10.values, y=top10.index, palette="Reds_d")
    plt.xlabel("Mean priority score")
    plt.title("Top 10 High-Risk Test Cases")
plt.tight_layout()
plt.savefig(top10_png, dpi=150)
plt.close()
print("Saved:", top10_png)

# Failure Risk Heatmap
heatmap_png = os.path.join(OUTDIR, "failure_risk_heatmap.png")
if tc_col and "dut" in df.columns and "priority_score" in df.columns:
    heat = df.pivot_table(index=tc_col, columns="dut", values="priority_score", aggfunc="mean", fill_value=0)
    max_rows = 40
    if heat.shape[0] > max_rows:
        top_cases = heat.max(axis=1).sort_values(ascending=False).head(max_rows).index
        heat = heat.loc[top_cases]
    plt.figure(figsize=(10, max(4, heat.shape[0]*0.25)))
    sns.heatmap(heat, cmap="Reds", linewidths=0.2)
    plt.title("Failure Risk Heatmap (TestCase × DUT)")
    plt.tight_layout()
    plt.savefig(heatmap_png, dpi=150)
    plt.close()
    print("Saved:", heatmap_png)

# Suite-wise failure risk trend
suite_png = os.path.join(OUTDIR, "suite_failure_trend.png")
if "suite" in df.columns and "run_date" in df.columns and "priority_score" in df.columns:
    try:
        df["run_date_dt"] = pd.to_datetime(df["run_date"], errors="coerce")
        agg = df.groupby([pd.Grouper(key="run_date_dt", freq="7D"), "suite"])["priority_score"].mean().reset_index()
        plt.figure(figsize=(10,5))
        sns.lineplot(data=agg, x="run_date_dt", y="priority_score", hue="suite", marker="o")
        plt.xlabel("Run date (7-day bins)")
        plt.ylabel("Mean priority score")
        plt.title("Suite-wise failure risk trend")
        plt.legend(bbox_to_anchor=(1.02,1), loc="upper left")
        plt.tight_layout()
        plt.savefig(suite_png, dpi=150)
        plt.close()
        print("Saved:", suite_png)
    except Exception as e:
        print("Suite trend generation failed:", e)

# Feature importance plot
fi_png = os.path.join(OUTDIR, "feature_importance.png")
fi_df = safe_read_csv(FI_CSV)
if fi_df is not None and "feature" in fi_df.columns and "importance" in fi_df.columns:
    topn = min(30, len(fi_df))
    fi_df_sorted = fi_df.sort_values("importance", ascending=False).head(topn)
    plt.figure(figsize=(8, max(3, topn*0.25)))
    plt.barh(fi_df_sorted["feature"][::-1], fi_df_sorted["importance"][::-1])
    plt.xlabel("Importance")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(fi_png, dpi=150)
    plt.close()
    print("Saved:", fi_png)

#  HTML Dashboard
html_path = os.path.join(OUTDIR, "dashboard.html")
imgs = {
    "Probability distribution": os.path.basename(dist_png),
    "Top 10 high-risk test cases": os.path.basename(top10_png),
    "Failure risk heatmap": os.path.basename(heatmap_png) if os.path.exists(heatmap_png) else None,
    "Suite-wise trend": os.path.basename(suite_png) if os.path.exists(suite_png) else None,
    "Feature importance": os.path.basename(fi_png) if os.path.exists(fi_png) else None
}

html_parts = [
    "<html><head><meta charset='utf-8'><title>Prioritization Dashboard</title></head><body>",
    "<h1>Prioritization Dashboard</h1>",
    f"<p>Source: {PRIORITIZED_CSV} — generated: {datetime.utcnow().isoformat()} UTC</p>"
]

for title, filename in imgs.items():
    html_parts.append(f"<h2>{title}</h2>")
    if filename and os.path.exists(os.path.join(OUTDIR, filename)):
        html_parts.append(f"<img src='{filename}' style='max-width:100%;height:auto;border:1px solid #ccc;padding:4px;margin-bottom:20px;'>")
    else:
        html_parts.append("<p><em>Not available</em></p>")

html_parts.append("</body></html>")

with open(html_path, "w", encoding="utf-8") as f:
    f.write("\n".join(html_parts))

print("Saved HTML dashboard ->", html_path)
print("\nAll visuals saved to:", OUTDIR)
