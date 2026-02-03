#!/usr/bin/env python3
import os
import pandas as pd

base_clean = r"C:\Users\Leonia Li\Downloads\9417Groupproj_preprocess-main\CleanData"

train_clean = pd.read_csv(os.path.join(base_clean, "anomaly_free_train.csv"))
val_clean   = pd.read_csv(os.path.join(base_clean, "anomaly_free_val.csv"))
test_clean  = pd.read_csv(os.path.join(base_clean, "anomaly_free_test.csv"))

# Concatenate and sort by timestamp so it matches the style of feature_store.parquet
df_clean = (
    pd.concat([train_clean, val_clean, test_clean], ignore_index=True)
      .dropna(subset=["timestamp"])
)

df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"], errors="coerce")
df_clean = df_clean.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

print("Clean feature_store size:", df_clean.shape)

out_path = r"C:\Users\Leonia Li\Downloads\air+quality\data\features\feature_store_clean.parquet"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df_clean.to_parquet(out_path, index=False)

print("Wrote:", out_path)
