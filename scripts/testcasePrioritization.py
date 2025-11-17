#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature Engineering Script for Test Case Prioritization
--------------------------------------------------------
- Loads logs, classified, cluster, and anomaly CSVs
- Safely merges all data
- Computes test case, keyword, cluster, anomaly, environment, metadata, and target features
- Automatically detects timestamp and status columns
- Sets target: Fail=1, Pass/Abort=0
- Cluster label: -1 for Pass/Abort, actual cluster for Fail logs
- Extracts keywords only from Fail logs
- Keeps categorical fields as strings
- Saves final CSV with feature-engineered data
"""

import os
import pandas as pd
import numpy as np

# -----------------------------
# File paths
# -----------------------------
LOGS_CSV       = 'C:/Users/hemalatha/Desktop/attest-eda/data/logs_preprocessed.csv'
CLASSIFIED_CSV = 'C:/Users/hemalatha/Desktop/attest-eda/data/outputs/model_rf/classified_logs.csv'
CLUSTER_CSV    = 'C:/Users/hemalatha/Desktop/attest-eda/data/cluster/failure_clusters.csv'
ANOMALY_CSV    = 'C:/Users/hemalatha/Desktop/attest-eda/data/outputs/anomaly_reports/anomaly_full_scores.csv'

output_dir = 'C:/Users/hemalatha/Desktop/attest-eda/outputs'
os.makedirs(output_dir, exist_ok=True)
OUTPUT_CSV = os.path.join(output_dir, 'feature_engineered_testcases.csv')

# -----------------------------
# Load CSVs
# -----------------------------
df_logs       = pd.read_csv(LOGS_CSV)
df_classified = pd.read_csv(CLASSIFIED_CSV)
df_cluster    = pd.read_csv(CLUSTER_CSV)
df_anomaly    = pd.read_csv(ANOMALY_CSV)

# -----------------------------
# Standardize column names
# -----------------------------
for df_tmp in [df_logs, df_classified, df_cluster, df_anomaly]:
    df_tmp.columns = df_tmp.columns.str.strip().str.lower()

# -----------------------------
# Rename classified columns to match logs
# -----------------------------
df_classified = df_classified.rename(columns={'testcase': 'test_case_id'})

# -----------------------------
# Merge logs + classified
# -----------------------------
merge_keys_clf = ['test_case_id', 'suite']
df_classified_unique = df_classified.drop_duplicates(subset=merge_keys_clf)
df = df_logs.merge(df_classified_unique,
                   on=merge_keys_clf,
                   how='left',
                   validate='many_to_one')

# -----------------------------
# Merge cluster
# -----------------------------
merge_keys_clu = ['test_case_id', 'suite']
df_cluster_unique = df_cluster.drop_duplicates(subset=merge_keys_clu)
df = df.merge(df_cluster_unique,
              on=merge_keys_clu,
              how='left',
              validate='many_to_one')

# -----------------------------
# Merge anomaly
# -----------------------------
merge_keys_anom = [c for c in ['filename','test_case_id','suite'] if c in df.columns and c in df_anomaly.columns]
if merge_keys_anom:
    df_anomaly_unique = df_anomaly.drop_duplicates(subset=merge_keys_anom)
    df = df.merge(df_anomaly_unique[merge_keys_anom + ['anomaly_score','is_anomaly','messagesnippet']],
                  on=merge_keys_anom, how='left', validate='many_to_one')
else:
    df['anomaly_score'] = df_anomaly.get('anomaly_score', 0).values[:len(df)]
    df['is_anomaly'] = df_anomaly.get('is_anomaly', 0).values[:len(df)]
    df['messagesnippet'] = df_anomaly.get('messagesnippet', "").values[:len(df)]

# -----------------------------
# Fill missing anomaly values
# -----------------------------
df['anomaly_score'] = df.get('anomaly_score', 0).fillna(0)
df['is_anomaly'] = df.get('is_anomaly', 0).fillna(0).astype(int)
df['messagesnippet'] = df.get('messagesnippet', "").fillna("")

# -----------------------------
# Detect timestamp column automatically
# -----------------------------
timestamp_candidates = [c for c in df.columns if 'time' in c or 'date' in c]
if not timestamp_candidates:
    raise KeyError("No timestamp column found!")
timestamp_col = timestamp_candidates[0]
df['executiontime'] = pd.to_datetime(df[timestamp_col], errors='coerce')

# -----------------------------
# Detect status column automatically
# -----------------------------
status_candidates = [c for c in df.columns if 'status' in c or 'result' in c or 'outcome' in c]
if not status_candidates:
    raise KeyError("No Pass/Fail/Abort column found!")
status_col = status_candidates[0]

# -----------------------------
# Target & test case features
# -----------------------------
df['target'] = df[status_col].apply(lambda x: 1 if str(x).lower() == 'fail' else 0)
df['is_fail'] = df['target']

df['past_failure_rate'] = df.groupby('test_case_id')['is_fail']\
                            .transform(lambda x: x.shift().expanding().mean()).fillna(0)
df['last_fail_time'] = df[df['is_fail']==1].groupby('test_case_id')['executiontime'].transform('last')
df['time_since_last_failure'] = (df['executiontime'] - df['last_fail_time']).dt.total_seconds().fillna(0)
df['execution_frequency'] = df.groupby('test_case_id').cumcount()
df['avg_duration'] = df.groupby('test_case_id')['execution_duration'].transform(lambda x: x.expanding().mean()) \
                      if 'execution_duration' in df.columns else 0

# -----------------------------
# Keyword features (only Fail logs)
# -----------------------------
df['keyword_fail'] = 0
fail_mask = df['is_fail'] == 1
df.loc[fail_mask, 'keyword_fail'] = df.loc[fail_mask, 'messagesnippet'] \
    .str.contains(r'fail|error|exception|abort', case=False, regex=True).astype(int)

# -----------------------------
# Cluster label
# -----------------------------
df['clusterlabel'] = '-1'  # default for Pass/Abort
df.loc[fail_mask, 'clusterlabel'] = df.loc[fail_mask, 'cluster'].fillna('Unknown').astype(str)

# -----------------------------
# Anomaly features
# -----------------------------
df['rolling_anomaly_rate'] = df.groupby('test_case_id')['is_anomaly']\
                               .transform(lambda x: x.shift().expanding().mean()).fillna(0)
df['mean_anomaly_score'] = df.groupby('test_case_id')['anomaly_score'].transform(lambda x: x.expanding().mean())

# -----------------------------
# Environment / DUT features (keep as string)
# -----------------------------
env_cols = ['dut_version','dut','config','regression_group','schedule_cycle']
for col in env_cols:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown').astype(str)

# -----------------------------
# Metadata features
# -----------------------------
df['exec_day'] = df['executiontime'].dt.dayofweek
df['exec_hour'] = df['executiontime'].dt.hour
for col in ['buildversion','suite']:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown').astype(str)

# -----------------------------
# Final feature columns
# -----------------------------
feature_cols = [
    'test_case_id','executiontime',
    'past_failure_rate','time_since_last_failure','execution_frequency','avg_duration',
    'keyword_fail','clusterlabel','rolling_anomaly_rate','mean_anomaly_score',
    'exec_day','exec_hour','dut_version','dut','config',
    'regression_group','schedule_cycle','buildversion','suite',
    'target'
]
feature_cols = [c for c in feature_cols if c in df.columns]
df_features = df[feature_cols]

# -----------------------------
# Save CSV
# -----------------------------
df_features.to_csv(OUTPUT_CSV, index=False)
print("Feature Engineering Completed ✔")
print(f"Saved: {OUTPUT_CSV}")
print("Final row count:", len(df_features))
