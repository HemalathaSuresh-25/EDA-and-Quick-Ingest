import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUTDIR = "data/task4output"
MODELDIR = "models"
PRIORITIZED_CSV = os.path.join(OUTDIR, "prioritized_testcases_xgb.csv")
FI_CSV = os.path.join(OUTDIR, "feature_importance.csv")
PRED_CSV = os.path.join(OUTDIR, "predicted_risk_scores.csv")
PDF_PATH = os.path.join(OUTDIR, "priority_report.pdf")
MODEL_JSON = os.path.join(MODELDIR, "xgb_model.json")

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(MODELDIR, exist_ok=True)

def safe_read(path):
    return pd.read_csv(path) if os.path.exists(path) else None

df = safe_read(PRIORITIZED_CSV)
if df is None:
    raise FileNotFoundError(f"Prioritized CSV not found at {PRIORITIZED_CSV}. Run training script first.")

if "pred_prob" in df.columns:
    df["predicted_probability"] = df["pred_prob"]
elif "predicted_probability" not in df.columns:
    raise KeyError("No predicted probability column found (expected 'pred_prob' or 'predicted_probability').")

q_high = df["priority_score"].quantile(0.90) if "priority_score" in df.columns else df["predicted_probability"].quantile(0.90)
q_med = df["priority_score"].quantile(0.60) if "priority_score" in df.columns else df["predicted_probability"].quantile(0.60)

def get_risk(p, ps=None):
    v = ps if ps is not None else p
    if v > q_high:
        return "HIGH"
    if v > q_med:
        return "MEDIUM"
    return "LOW"

fi_df = safe_read(FI_CSV)
top_feature_names = []
if fi_df is not None and {"feature","importance"}.issubset(fi_df.columns):
    top_feature_names = fi_df.sort_values("importance", ascending=False).head(10)["feature"].tolist()

def top_features_for_row(row):
    items = []
    for f in top_feature_names[:5]:
        if f in row.index:
            val = row[f]
            items.append(f"{f}:{round(val,4)}")
    return ";".join(items) if items else ""

testcase_col = "test_case" if "test_case" in df.columns else ("test_case_id" if "test_case_id" in df.columns else None)

out = pd.DataFrame()
out["test_case"] = df[testcase_col] if testcase_col else df.index.astype(str)
out["predicted_probability"] = df["predicted_probability"]
out["priority_score"] = df["priority_score"] if "priority_score" in df.columns else df["predicted_probability"]
out["risk_level"] = out.apply(lambda r: get_risk(r["predicted_probability"], r["priority_score"]), axis=1)
out["top_features"] = df.apply(top_features_for_row, axis=1) if top_feature_names else ""

out = out.sort_values(["priority_score", "predicted_probability"], ascending=False)
out.to_csv(PRED_CSV, index=False)
print("Saved predicted risk scores ->", PRED_CSV)

def create_pdf_report(df_prioritized, fi_df=None, pdf_path=PDF_PATH):
    title = "Priority Report — Test Case Prioritization"
    created_on = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    n_total = len(df_prioritized)
    n_fail = int(df_prioritized["isFail"].sum()) if "isFail" in df_prioritized.columns else None

    summary_lines = [
        f"Report generated: {created_on}",
        f"Total rows: {n_total}",
        f"Total failures (label): {n_fail}" if n_fail is not None else "",
        "",
        "Key risk drivers (top global features):",
    ]
    if fi_df is not None:
        top_feats = fi_df.sort_values("importance", ascending=False).head(10)["feature"].tolist()
        summary_lines.append(", ".join(top_feats))
    else:
        summary_lines.append("Feature importance not available.")
    summary_lines.append("")
    summary_lines.append("How to use this report:")
    summary_lines.append("- Focus first on HIGH risk testcases to catch most failures early.")
    summary_lines.append("- Use the per-test top_features to inspect specific causes.")
    summary_text = "\n".join([l for l in summary_lines if l])

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(11.7, 8.3))  # A4 landscape
        fig.suptitle(title, fontsize=18)
        ax = fig.add_axes([0.05, 0.55, 0.45, 0.4])
        ax.axis("off")
        ax.text(0, 1, summary_text, fontsize=10, va="top")
        ax2 = fig.add_axes([0.52, 0.12, 0.46, 0.83])
        ax2.axis("off")
        table_df = df_prioritized[[testcase_col if testcase_col else df_prioritized.index.name, "predicted_probability", "priority_score"]].head(10).copy()
        table_df.columns = ["test_case", "pred_prob", "priority_score"]
        table_df["pred_prob"] = table_df["pred_prob"].map(lambda x: f"{x:.3f}")
        ax2.table(cellText=table_df.values, colLabels=table_df.columns, cellLoc="center", colLoc="center", loc="center")
        ax2.set_title("Top 10 prioritized testcases")
        pdf.savefig(fig)
        plt.close(fig)

create_pdf_report(df, fi_df=fi_df, pdf_path=PDF_PATH)
print("Saved PDF report ->", PDF_PATH)

if not os.path.exists(MODEL_JSON):
    print("Warning: model file not found at", MODEL_JSON)
else:
    print("Model present:", MODEL_JSON)

print("\nAll exports complete.")
