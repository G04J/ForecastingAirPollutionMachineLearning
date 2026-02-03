#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

# -----------------------------------
# 1. Config
# -----------------------------------
targets = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
sensors = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)"]
wea_var = ["T", "RH", "AH"]

feature_cols = sensors + wea_var      # used for IsolationForest
all_check_cols = targets + sensors + wea_var  # (kept for possible future checks)

base_dir = os.path.join(
    os.path.expanduser("~"),
    "Downloads",
    "air+quality",
    "data",
    "splits",
)

train_path = os.path.join(base_dir, "feature_store_train.csv")
val_path   = os.path.join(base_dir, "feature_store_val.csv")
test_path  = os.path.join(base_dir, "feature_store_test.csv")

print("[INFO] Reading CSVs:")
print("  train:", train_path)
print("  val  :", val_path)
print("  test :", test_path)

df_train = pd.read_csv(train_path)
df_val   = pd.read_csv(val_path)
df_test  = pd.read_csv(test_path)

# -----------------------------------
# 2. Ensure timestamp index + basic time features
# -----------------------------------
for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
    if "timestamp" not in df.columns:
        raise ValueError(f"[ERROR] 'timestamp' column missing in {name} dataframe.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.set_index("timestamp", inplace=True)

    df["hour"] = df.index.hour
    df["weekday"] = df.index.dayofweek
    df["month"] = df.index.month

    print(f"[INFO] {name}: {len(df)} rows after sorting + timestamp cleaning.")

# -----------------------------------
# 3. Rolling window anomaly flags on targets (24h, 3σ rule)
#    This is done separately for each split.
# -----------------------------------
for col in targets:
    print(f"[INFO] Rolling anomalies for target: {col}")
    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        # time-based 24h rolling window
        roll_mean = df[col].rolling("24h", min_periods=12).mean()
        roll_std  = df[col].rolling("24h", min_periods=12).std()

        resid = df[col] - roll_mean

        # avoid problems when std == 0
        thresh = 3 * roll_std
        an_resid = ((resid.abs() > thresh) & roll_std.notna()).astype(int)

        flag_col = f"an_resid_{col}"
        df[flag_col] = an_resid

        print(f"  [{name}] {flag_col}: marked {an_resid.sum()} points as anomalies")

# -----------------------------------
# 4. IsolationForest on features (TRAIN-FIT ONLY, then apply to all)
# -----------------------------------
# We fit on train using an imputer + scaler + isolation forest
print("\n[INFO] Fitting IsolationForest on train features...")

# restrict to rows where we at least have some sensor/weather data
X_train = df_train[feature_cols].copy()

# build pipeline: impute -> scale -> isolation forest
iso_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("iso", IsolationForest(
        contamination=0.02,  # 2% anomalies (tunable)
        random_state=42
    )),
])

iso_pipeline.fit(X_train)

def predict_iso(name, df):
    X = df[feature_cols].copy()
    # pipeline will impute + scale inside
    pred = iso_pipeline.predict(X)   # +1 = normal, -1 = anomaly
    df["an_iso"] = (pred == -1).astype(int)
    print(f"[INFO] IsolationForest [{name}]: flagged {df['an_iso'].sum()} anomalies out of {len(df)} rows.")

for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
    predict_iso(name, df)

# -----------------------------------
# 5. Combine all anomaly flags
#    an_any = 1 if ANY individual anomaly indicator is 1.
# -----------------------------------
for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
    anomaly_cols = [c for c in df.columns if c.startswith("an_")]  # an_resid_*, an_iso
    df["an_any"] = df[anomaly_cols].max(axis=1)
    print(f"[INFO] {name}: total rows = {len(df)}, rows with an_any=1 → {int(df['an_any'].sum())}")

# -----------------------------------
# 6. Filter: keep only rows with an_any == 0
# -----------------------------------
train_clean = df_train[df_train["an_any"] == 0].copy()
val_clean   = df_val[df_val["an_any"] == 0].copy()
test_clean  = df_test[df_test["an_any"] == 0].copy()

print("\n[INFO] Clean set sizes:")
print(f"  train_clean: {len(train_clean)} rows (dropped {len(df_train) - len(train_clean)})")
print(f"  val_clean  : {len(val_clean)} rows (dropped {len(df_val) - len(val_clean)})")
print(f"  test_clean : {len(test_clean)} rows (dropped {len(df_test) - len(test_clean)})")

# -----------------------------------
# 7. Save to disk
# -----------------------------------
output_dir = os.path.expanduser("~/Downloads/9417Groupproj_preprocess-main/CleanData")
os.makedirs(output_dir, exist_ok=True)

train_out = os.path.join(output_dir, "anomaly_free_train.csv")
val_out   = os.path.join(output_dir, "anomaly_free_val.csv")
test_out  = os.path.join(output_dir, "anomaly_free_test.csv")

# reset_index() so timestamp becomes a column again, like original
train_clean.reset_index().to_csv(train_out, index=False)
val_clean.reset_index().to_csv(val_out, index=False)
test_clean.reset_index().to_csv(test_out, index=False)

print("\n[OK] Anomaly detection completed. Clean files saved to:")
print(" ", train_out)
print(" ", val_out)
print(" ", test_out)
