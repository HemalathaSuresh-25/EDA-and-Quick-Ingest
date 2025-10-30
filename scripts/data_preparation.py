"""
Prepare training and testing datasets from the combined labels file.
Creates training labels based on status or root_cause_label,
and keeps all metadata columns.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Configuration
INPUT_FILE = "data/failure_patterns_labeled_human.csv"    
OUTPUT_DIR = "data"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Loading dataset...")

# Load the combined dataset
df = pd.read_csv(INPUT_FILE)
print(f"Loaded dataset shape: {df.shape}")

# Standardize column names
df.columns = df.columns.str.strip().str.lower()

# Create final label for model training
# Priority: Use status (PASS / FAIL / ABORT) if available
if 'status' in df.columns and df['status'].notna().any():
    df['label'] = df['status'].str.upper()
elif 'root_cause_label' in df.columns:
    df['label'] = df['root_cause_label']
else:
    raise KeyError("No 'status' or 'root_cause_label' column found to create labels.")

# Select important columns for modeling
meta_cols = [
    'filename', 'dut', 'dut_version', 'os_version', 'config', 'test_case_id',
    'suite', 'run_date', 'execution_duration', 'failure_freq_suite',
    'failure_freq_dut', 'recent_failure_flag'
]

# Only keep columns that exist in the dataset
meta_cols = [c for c in meta_cols if c in df.columns]

feature_cols = ['error_msg', 'cluster', 'root_cause_label', 'keywords']
feature_cols = [c for c in feature_cols if c in df.columns]

df = df[meta_cols + feature_cols + ['label']]

# Handle missing values
df.fillna({'cluster': -1, 'root_cause_label': 'Normal'}, inplace=True)
df.fillna('Unknown', inplace=True)

# Split dataset into training and testing
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# Save output files
train_path = os.path.join(OUTPUT_DIR, "train_dataset.csv")
test_path = os.path.join(OUTPUT_DIR, "test_dataset.csv")

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"Saved training data → {train_path}")
print(f"Saved testing data → {test_path}")
print("Data preparation completed successfully.")