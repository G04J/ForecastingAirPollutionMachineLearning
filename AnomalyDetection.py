import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import os

targets = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
sensors = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)"]
wea_var = ["T", "RH", "AH"]

feature_cols = sensors + wea_var
all_check_cols = targets + sensors + wea_var

train_path = "~/Downloads/9417Groupproj_preprocess-main/data/splits/airq_test.csv"
val_path   = "~/Downloads/9417Groupproj_preprocess-main/data/splits/airq_val.csv"
test_path  = "~/Downloads/9417Groupproj_preprocess-main/data/splits/airq_test.csv"

df_train = pd.read_csv(train_path)
df_val   = pd.read_csv(val_path)
df_test  = pd.read_csv(test_path)

df_train = df_train.sort_values("timestamp").reset_index(drop=True)
df_val   = df_val.sort_values("timestamp").reset_index(drop=True)
df_test  = df_test.sort_values("timestamp").reset_index(drop=True)

for df in [df_train, df_val, df_test]:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.set_index("timestamp", inplace=True)

for df in [df_train, df_val, df_test]:
    df["hour"] = df.index.hour
    df["weekday"] = df.index.dayofweek
    df["month"] = df.index.month

for col in targets:
    for df in [df_train, df_val, df_test]:
        roll_mean = df[col].rolling("24h").mean()
        roll_std  = df[col].rolling("24h").std()
        resid = df[col] - roll_mean

        an_resid = (resid.abs() > 3 * roll_std).astype(int)
        df[f"an_resid_{col}"] = an_resid

X_train = df_train[feature_cols].dropna()

iso_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("iso", IsolationForest(contamination=0.02, random_state=42))
])

iso_pipeline.fit(X_train)

pred_train = iso_pipeline.predict(df_train[feature_cols])
pred_val   = iso_pipeline.predict(df_val[feature_cols])
pred_test  = iso_pipeline.predict(df_test[feature_cols])

df_train["an_iso"] = (pred_train == -1).astype(int)
df_val["an_iso"]   = (pred_val == -1).astype(int)
df_test["an_iso"]  = (pred_test == -1).astype(int)

for df in [df_train, df_val, df_test]:
    anomaly_cols = [c for c in df.columns if c.startswith("an_")]
    df["an_any"] = df[anomaly_cols].max(axis=1)

train_clean = df_train[df_train["an_any"] == 0].copy()
val_clean   = df_val[df_val["an_any"] == 0].copy()
test_clean  = df_test[df_test["an_any"] == 0].copy()

output_dir = os.path.expanduser("~/Downloads/9417Groupproj_preprocess-main/CleanData")
os.makedirs(output_dir, exist_ok=True)

train_clean.to_csv("~/Downloads/9417Groupproj_preprocess-main/CleanData/anomaly_free_train.csv")
val_clean.to_csv("~/Downloads/9417Groupproj_preprocess-main/CleanData/anomaly_free_val.csv")
test_clean.to_csv("~/Downloads/9417Groupproj_preprocess-main/CleanData/anomaly_free_test.csv")

print("Anomaly detection completed. Clean files saved.")
