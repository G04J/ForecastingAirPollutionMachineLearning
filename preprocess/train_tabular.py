#!/usr/bin/env python3
"""
train_tabular.py — Neural models for UCI Air Quality (single/multi-target + classification)

Supports:
  - regression          : one target (e.g., NO2(GT))
  - regression_multi    : multiple targets at once (default: CO, C6H6, NOx, NO2)
  - classification      : 3-class CO(GT) bins (<1.5, 1.5–2.5, ≥2.5) at t+h

Trains a separate model for each forecast horizon h ∈ --horizons (default: 1 6 12 24).
Respects temporal splits and purge window from splits.json. Computes naïve baseline (y(t+h) ≈ y(t)).
Saves model, scaler, feature column order, and metrics. Can print per-epoch logs and save CSV logs.

Dependencies:
  tensorflow (or tensorflow-cpu), pandas, numpy, scikit-learn, pyarrow (if reading parquet)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

# Reduce TF log noise
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score

REG_ALLOWED = ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]

# ------------------------------ CLI ------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train MLPs for regression (single/multi) and classification with temporal splits.")
    p.add_argument("--features_path", required=True, type=Path, help="Feature CSV/Parquet (must have 'timestamp').")
    p.add_argument("--splits_json", required=True, type=Path, help="splits.json from make_splits.py")
    p.add_argument("--task", choices=["regression", "regression_multi", "classification"], required=True)
    p.add_argument("--target", default="NO2(GT)", help="For regression (single target).")
    p.add_argument("--targets", nargs="+", default=REG_ALLOWED, help="For regression_multi.")
    p.add_argument("--horizons", nargs="+", type=int, default=[1,6,12,24])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out_dir", type=Path, default=Path("artifacts"))

    # Logging controls
    p.add_argument("--verbose", type=int, choices=[0,1,2], default=1,
                   help="Keras verbosity: 0=silent, 1=bar, 2=one line per epoch")
    p.add_argument("--save_csv_logs", action="store_true",
                   help="Write per-epoch metrics to log.csv in each run folder.")
    return p.parse_args()

# ------------------------------ IO ------------------------------
def read_features(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("features file must contain 'timestamp'")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return df

def load_splits(splits_path: Path) -> dict:
    with open(splits_path, "r", encoding="utf-8") as f:
        js = json.load(f)
    for k in ["train","val","test"]:
        js[k] = [pd.to_datetime(js[k][0]), pd.to_datetime(js[k][1])]
    js["purge_hours"] = int(js.get("purge_hours", 0))
    return js

def rmse_safe(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(mean_squared_error(a[mask], b[mask])))


def apply_ranges(df: pd.DataFrame, rng):
    start, end = rng
    return df.loc[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()

def exclude_purge_from_train(df_train: pd.DataFrame, val_start: pd.Timestamp, test_start: pd.Timestamp, purge_hours: int):
    if purge_hours <= 0:
        return df_train
    purge = pd.Timedelta(hours=purge_hours)
    m_purge_val  = (df_train["timestamp"] >= (val_start  - purge)) & (df_train["timestamp"] < val_start)
    m_purge_test = (df_train["timestamp"] >= (test_start - purge)) & (df_train["timestamp"] < test_start)
    return df_train.loc[~(m_purge_val | m_purge_test)].copy()

# ------------------------------ Labels ------------------------------
def make_reg_labels(df: pd.DataFrame, target: str, horizon_h: int):
    y = df[target].shift(-horizon_h)
    naive = df[target]
    out = df.copy()
    out["y"] = y
    out["y_naive"] = naive
    # Drop rows where either future or current target is NaN
    out = out.dropna(subset=["y", "y_naive"]).reset_index(drop=True)
    return out


def make_reg_multi_labels(df: pd.DataFrame, targets: List[str], horizon_h: int):
    out = df.copy()
    Y = []
    Y_naive = []
    for t in targets:
        out[f"y__{t}"] = df[t].shift(-horizon_h)
        out[f"y_naive__{t}"] = df[t]
        Y.append(f"y__{t}")
        Y_naive.append(f"y_naive__{t}")
    out = out.dropna(subset=Y).reset_index(drop=True)
    out = out.dropna(subset=Y_naive).reset_index(drop=True)

    return out, Y, Y_naive

def make_cls_labels_CO_bins(df: pd.DataFrame, horizon_h: int):
    co_future = df["CO(GT)"].shift(-horizon_h)
    co_now = df["CO(GT)"]

    bins = [-np.inf, 1.5, 2.5, np.inf]

    # no need to cast to float here; we'll drop NAs and then cast to int
    y = pd.cut(co_future, bins=bins, labels=[0, 1, 2], right=False)
    naive = pd.cut(co_now, bins=bins, labels=[0, 1, 2], right=False)

    out = df.copy()
    out["y"] = y
    out["y_naive"] = naive

    # IMPORTANT: drop rows where either future OR current bin is NA
    out = out.dropna(subset=["y", "y_naive"]).reset_index(drop=True)

    # now safe to cast to int
    out["y"] = out["y"].astype(int)
    out["y_naive"] = out["y_naive"].astype(int)
    print("[DEBUG] make_cls_labels_CO_bins: kept rows:", len(out))

    return out


# ------------------------------ Models ------------------------------
from tensorflow.keras import regularizers

def build_mlp(input_dim: int, output_dim: int|None, lr: float, is_class: bool):
    inp = keras.Input(shape=(input_dim,))

    if is_class:
        x = layers.Dense(
            32,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4),
        )(inp)

        out = layers.Dense(output_dim, activation="softmax")(x)
        model = keras.Model(inp, out)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    else:
        x = layers.Dense(
            64, activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        )(inp)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(
            32, activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        )(x)

        out_dim = 1 if output_dim is None or output_dim == 1 else output_dim
        out = layers.Dense(out_dim)(x)
        model = keras.Model(inp, out)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.Huber(delta=1.0),  # a bit more robust than pure MSE
            metrics=["mae"],
        )

    return model



def rmse(a, b) -> float:
    return float(np.sqrt(mean_squared_error(a, b)))

# ------------------------------ Logging helpers ------------------------------
def print_training_summary(hist, is_class: bool):
    if not hasattr(hist, "history") or not hist.history:
        return
    dfh = pd.DataFrame(hist.history)

    # robust best-epoch selection
    if "val_loss" in dfh and dfh["val_loss"].notna().any():
        best_idx = int(dfh["val_loss"].idxmin())
    else:
        best_idx = len(dfh) - 1

    print("\n=== Training history (last 5 epochs) ===")
    print(dfh.tail(5).to_string(index=False))
    print("\nBest epoch by val_loss:", best_idx + 1)
    keys = ["loss", "val_loss"]
    if is_class:
        keys += ["accuracy", "val_accuracy"]
    print("Best values:")
    for k in keys:
        if k in dfh.columns:
            val = dfh.iloc[best_idx][k]
            if isinstance(val, (int, float)):
                print(f"  {k}: {val:.6f}")
            else:
                print(f"  {k}: {val}")
    print("========================================\n")


# ------------------------------ Training routines ------------------------------
def run_reg_single(df_full, splits, target, horizon, epochs, batch_size, lr, out_dir, verbose, save_csv_logs):
    # ----- split by time & purge -----
    df_tr = exclude_purge_from_train(
        apply_ranges(df_full, splits["train"]),
        splits["val"][0], splits["test"][0], splits["purge_hours"]
    )
    df_vl = apply_ranges(df_full, splits["val"])
    df_te = apply_ranges(df_full, splits["test"])

    # ----- labels (ensure both y and y_naive are non-NaN) -----
    tr = make_reg_labels(df_tr, target, horizon)
    vl = make_reg_labels(df_vl, target, horizon)
    te = make_reg_labels(df_te, target, horizon)

    print("\n[DEBUG] Train label sample (first 5 rows):")
    print(tr[["timestamp", target, "y", "y_naive"]].head())
    print("\n[DEBUG] Val label sample (first 5 rows):")
    print(vl[["timestamp", target, "y", "y_naive"]].head())
    print("\n[DEBUG] Test label sample (first 5 rows):")
    print(te[["timestamp", target, "y", "y_naive"]].head())

    # ----- feature columns: everything except timestamp + labels -----
    drop_cols = ["timestamp", "y", "y_naive"]
    all_feat = [c for c in tr.columns if c not in drop_cols]

    # Heuristic: keep columns that mention key pollutants or met vars
    KEEP_TOKENS = ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)", "O3", "T", "RH", "AH"]

    feature_cols = [
        c for c in all_feat
        if any(tok in c for tok in KEEP_TOKENS)
    ]

    # Fallback: if we somehow filtered everything out, revert to all features
    if not feature_cols:
        feature_cols = all_feat

    print("[DEBUG] using", len(feature_cols), "features out of", len(all_feat))


    X_tr_df = tr[feature_cols].copy()
    X_vl_df = vl[feature_cols].copy()
    X_te_df = te[feature_cols].copy()

    y_tr = tr["y"].values.astype("float32")
    y_vl = vl["y"].values.astype("float32")
    y_te = te["y"].values.astype("float32")

    print("[DEBUG] X_tr shape:", X_tr_df.shape, "| X_vl:", X_vl_df.shape, "| X_te:", X_te_df.shape)
    print("[DEBUG] y_tr range:", float(y_tr.min()), "→", float(y_tr.max()))

    # ----- target normalisation on TRAIN only -----
    y_mean = float(y_tr.mean())
    y_std = float(y_tr.std())
    if y_std <= 1e-6:
        y_std = 1.0

    y_tr_n = (y_tr - y_mean) / y_std
    y_vl_n = (y_vl - y_mean) / y_std
    y_te_n = (y_te - y_mean) / y_std  # for debugging only

    # ----- feature NaN handling: train-mean imputation -----
    feat_means = X_tr_df.mean()
    X_tr_df = X_tr_df.fillna(feat_means)
    X_vl_df = X_vl_df.fillna(feat_means)
    X_te_df = X_te_df.fillna(feat_means)

    # ----- feature scaling -----
    scaler = StandardScaler().fit(X_tr_df.values)
    X_tr = scaler.transform(X_tr_df.values).astype("float32")
    X_vl = scaler.transform(X_vl_df.values).astype("float32")
    X_te = scaler.transform(X_te_df.values).astype("float32")

    print("[DEBUG] nan in X_tr:", np.isnan(X_tr).sum(), "inf in X_tr:", np.isinf(X_tr).sum())
    print("[DEBUG] nan in y_tr_n:", np.isnan(y_tr_n).sum(), "inf in y_tr_n:", np.isinf(y_tr_n).sum())
    print("[DEBUG] y_mean:", y_mean, "y_std:", y_std)

    # ----- build & train model -----
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_mlp(X_tr.shape[1], output_dim=1, lr=lr, is_class=False)

    cbs = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=100,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    if save_csv_logs:
        cbs.append(keras.callbacks.CSVLogger(str(out_dir / "log.csv"), append=False))

    hist = model.fit(
        X_tr, y_tr_n,
        validation_data=(X_vl, y_vl_n),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cbs,
        verbose=verbose,
    )
    print_training_summary(hist, is_class=False)

    # ----- predictions (de-normalised) -----
    pred_tr_n = model.predict(X_tr, verbose=0).ravel()
    pred_vl_n = model.predict(X_vl, verbose=0).ravel()
    pred_te_n = model.predict(X_te, verbose=0).ravel()

    pred_tr = pred_tr_n * y_std + y_mean
    pred_vl = pred_vl_n * y_std + y_mean
    pred_te = pred_te_n * y_std + y_mean

    metrics = {
        "model": {
            "train_rmse": rmse_safe(y_tr, pred_tr),
            "val_rmse":   rmse_safe(y_vl, pred_vl),
            "test_rmse":  rmse_safe(y_te, pred_te),
        },
        "naive": {
            "train_rmse": rmse_safe(y_tr, tr["y_naive"].values[:len(y_tr)]),
            "val_rmse":   rmse_safe(y_vl, vl["y_naive"].values[:len(y_vl)]),
            "test_rmse":  rmse_safe(y_te, te["y_naive"].values[:len(y_te)]),
        },
    }

    print(
        f"[h={horizon}] FINAL — train RMSE: {metrics['model']['train_rmse']:.4f} | "
        f"val RMSE: {metrics['model']['val_rmse']:.4f} | test RMSE: {metrics['model']['test_rmse']:.4f}"
    )
    print(
        f"[h={horizon}] NAIVE — train RMSE: {metrics['naive']['train_rmse']:.4f} | "
        f"val RMSE: {metrics['naive']['val_rmse']:.4f} | test RMSE: {metrics['naive']['test_rmse']:.4f}"
    )
    print("DEBUG: y_tr stats:", np.min(y_tr), np.max(y_tr), np.mean(y_tr))
    print("DEBUG: pred_tr_n stats:", np.min(pred_tr_n), np.max(pred_tr_n), np.mean(pred_tr_n))

    # ----- save artifacts -----
    model.save(out_dir / "model.keras")
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # feature list used by this model
    with open(out_dir / "features.txt", "w", encoding="utf-8") as f:
        for c in feature_cols:
            f.write(str(c) + "\n")

    # target scaler (for eval to de-normalise)
    with open(out_dir / "target_scaler.json", "w", encoding="utf-8") as f:
        json.dump({"y_mean": y_mean, "y_std": y_std}, f, indent=2)

    # feature means for NaN imputation at eval time
    with open(out_dir / "feature_means.json", "w", encoding="utf-8") as f:
        json.dump(feat_means.to_dict(), f, indent=2)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)



def run_reg_multi(df_full, splits, targets: List[str], horizon, epochs, batch_size, lr, out_dir, verbose, save_csv_logs):
    # ----- split by time & purge -----
    df_tr = exclude_purge_from_train(
        apply_ranges(df_full, splits["train"]),
        splits["val"][0], splits["test"][0], splits["purge_hours"]
    )
    df_vl = apply_ranges(df_full, splits["val"])
    df_te = apply_ranges(df_full, splits["test"])

    # ----- make labels -----
    tr, Y_tr_cols, YN_tr_cols = make_reg_multi_labels(df_tr, targets, horizon)
    vl, Y_vl_cols, YN_vl_cols = make_reg_multi_labels(df_vl, targets, horizon)
    te, Y_te_cols, YN_te_cols = make_reg_multi_labels(df_te, targets, horizon)

    assert Y_tr_cols == Y_vl_cols == Y_te_cols, "Label columns mismatch across splits."

    Y_cols = Y_tr_cols      # e.g. ["y__CO(GT)", "y__C6H6(GT)", ...]
    YN_cols = YN_tr_cols    # e.g. ["y_naive__CO(GT)", ...]

    # ----- feature columns (all non-label, non-timestamp) -----
    drop_cols = ["timestamp"] + Y_cols + YN_cols
    all_feat = [c for c in tr.columns if c not in drop_cols]

    feature_cols = all_feat  # later you can restrict this if you want
    print("[DEBUG] multi-reg: using", len(feature_cols), "features")

    X_tr_df = tr[feature_cols].copy()
    X_vl_df = vl[feature_cols].copy()
    X_te_df = te[feature_cols].copy()

    Y_tr = tr[Y_cols].values.astype("float32")
    Y_vl = vl[Y_cols].values.astype("float32")
    Y_te = te[Y_cols].values.astype("float32")

    print("[DEBUG] multi-reg X_tr shape:", X_tr_df.shape, "| Y_tr shape:", Y_tr.shape)

    # ----- per-target target normalisation on TRAIN only -----
    Y_mean = Y_tr.mean(axis=0)
    Y_std = Y_tr.std(axis=0)
    Y_std[Y_std <= 1e-6] = 1.0  # avoid divide-by-zero

    Y_tr_n = (Y_tr - Y_mean) / Y_std
    Y_vl_n = (Y_vl - Y_mean) / Y_std
    Y_te_n = (Y_te - Y_mean) / Y_std  # only for debugging

    # ----- feature NaN handling: train-mean imputation -----
    feat_means = X_tr_df.mean()
    X_tr_df = X_tr_df.fillna(feat_means)
    X_vl_df = X_vl_df.fillna(feat_means)
    X_te_df = X_te_df.fillna(feat_means)

    # ----- feature scaling -----
    scaler = StandardScaler().fit(X_tr_df.values)
    X_tr = scaler.transform(X_tr_df.values).astype("float32")
    X_vl = scaler.transform(X_vl_df.values).astype("float32")
    X_te = scaler.transform(X_te_df.values).astype("float32")

    print("[DEBUG] multi-reg nan in X_tr:", np.isnan(X_tr).sum(), "inf:", np.isinf(X_tr).sum())
    print("[DEBUG] multi-reg nan in Y_tr_n:", np.isnan(Y_tr_n).sum(), "inf:", np.isinf(Y_tr_n).sum())

    # ----- build and train model -----
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_mlp(X_tr.shape[1], output_dim=len(targets), lr=lr, is_class=False)

    cbs = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=50,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-5,
            verbose=1,
        ),
    ]
    if save_csv_logs:
        cbs.append(keras.callbacks.CSVLogger(str(out_dir / "log.csv"), append=False))

    hist = model.fit(
        X_tr, Y_tr_n,
        validation_data=(X_vl, Y_vl_n),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cbs,
        verbose=verbose,
    )
    print_training_summary(hist, is_class=False)

    # ----- predictions (de-normalised) -----
    P_tr_n = model.predict(X_tr, verbose=0)
    P_vl_n = model.predict(X_vl, verbose=0)
    P_te_n = model.predict(X_te, verbose=0)

    P_tr = P_tr_n * Y_std + Y_mean
    P_vl = P_vl_n * Y_std + Y_mean
    P_te = P_te_n * Y_std + Y_mean

    # naive predictions (no normalisation)
    YN_tr = tr[YN_cols].values.astype("float32")
    YN_vl = vl[YN_cols].values.astype("float32")
    YN_te = te[YN_cols].values.astype("float32")

    def per_target_rmse(Y_true, Y_pred):
        d = {}
        for i, t in enumerate(targets):
            col_name = Y_cols[i]
            d[f"{t}_rmse"] = rmse_safe(Y_true[:, i], Y_pred[:, i])
        d["macro_rmse"] = float(np.nanmean([d[f"{t}_rmse"] for t in targets]))
        return d

    metrics = {
        "model": {
            "train": per_target_rmse(Y_tr, P_tr),
            "val":   per_target_rmse(Y_vl, P_vl),
            "test":  per_target_rmse(Y_te, P_te),
        },
        "naive": {
            "train": per_target_rmse(Y_tr, YN_tr),
            "val":   per_target_rmse(Y_vl, YN_vl),
            "test":  per_target_rmse(Y_te, YN_te),
        },
    }

    print(f"[h={horizon}] FINAL — macro RMSE: "
          f"train {metrics['model']['train']['macro_rmse']:.4f} | "
          f"val {metrics['model']['val']['macro_rmse']:.4f} | "
          f"test {metrics['model']['test']['macro_rmse']:.4f}")
    print("Per-target test RMSE:")
    for t in targets:
        print(f"  {t}: {metrics['model']['test'][t + '_rmse']:.4f}")

    # ----- save artifacts -----
    model.save(out_dir / "model.keras")
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # feature list used by this model
    with open(out_dir / "features.txt", "w", encoding="utf-8") as f:
        for c in feature_cols:
            f.write(str(c) + "\n")

    # per-target scaler (for eval)
    target_scaler_multi = {
        "Y_cols": Y_cols,
        "means": {col: float(m) for col, m in zip(Y_cols, Y_mean)},
        "stds":  {col: float(s) for col, s in zip(Y_cols, Y_std)},
    }
    with open(out_dir / "target_scaler_multi.json", "w", encoding="utf-8") as f:
        json.dump(target_scaler_multi, f, indent=2)

    # feature means for NaN imputation at eval time
    with open(out_dir / "feature_means.json", "w", encoding="utf-8") as f:
        json.dump(feat_means.to_dict(), f, indent=2)

    with open(out_dir / "targets.json", "w", encoding="utf-8") as f:
        json.dump({"targets": targets}, f, indent=2)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def run_cls(df_full, splits, horizon, epochs, batch_size, lr, out_dir, verbose, save_csv_logs):
    df_tr = exclude_purge_from_train(
        apply_ranges(df_full, splits["train"]),
        splits["val"][0],
        splits["test"][0],
        splits["purge_hours"],
    )
    df_vl = apply_ranges(df_full, splits["val"])
    df_te = apply_ranges(df_full, splits["test"])

    tr = make_cls_labels_CO_bins(df_tr, horizon)
    vl = make_cls_labels_CO_bins(df_vl, horizon)
    te = make_cls_labels_CO_bins(df_te, horizon)

    print("\n[DEBUG] Train cls sample (first 5 rows):")
    print(tr[["timestamp", "CO(GT)", "y", "y_naive"]].head())
    print("\n[DEBUG] Val cls sample (first 5 rows):")
    print(vl[["timestamp", "CO(GT)", "y", "y_naive"]].head())
    print("\n[DEBUG] Test cls sample (first 5 rows):")
    print(te[["timestamp", "CO(GT)", "y", "y_naive"]].head())

    drop_cols = ["timestamp", "y", "y_naive"]
    X_tr_df = tr.drop(columns=drop_cols)
    X_vl_df = vl.drop(columns=drop_cols)
    X_te_df = te.drop(columns=drop_cols)

    y_tr = tr["y"].values
    y_vl = vl["y"].values
    y_te = te["y"].values
    # ---- NEW: class weights for imbalanced classes ----
    num_classes = 3
    class_counts = np.bincount(y_tr, minlength=num_classes)
    total = len(y_tr)

    class_weights = {
        i: float(total / (num_classes * class_counts[i]))
        for i in range(num_classes)
        if class_counts[i] > 0
    }
    print("[DEBUG] class_counts:", class_counts, "class_weights:", class_weights)

    # ---- NEW: check & impute NaNs in features ----
    print(f"[DEBUG] raw X_tr shape: {X_tr_df.shape}")
    print(f"[DEBUG] NaNs in raw X_tr: {X_tr_df.isna().sum().sum()}")
    print(f"[DEBUG] NaNs in raw X_vl: {X_vl_df.isna().sum().sum()}")
    print(f"[DEBUG] NaNs in raw X_te: {X_te_df.isna().sum().sum()}")

    feat_means = X_tr_df.mean()
    X_tr_df = X_tr_df.fillna(feat_means)
    X_vl_df = X_vl_df.fillna(feat_means)
    X_te_df = X_te_df.fillna(feat_means)

    print(f"[DEBUG] NaNs AFTER fillna — X_tr: {X_tr_df.isna().sum().sum()}, "
          f"X_vl: {X_vl_df.isna().sum().sum()}, X_te: {X_te_df.isna().sum().sum()}")

    scaler = StandardScaler().fit(X_tr_df.values)
    X_tr = scaler.transform(X_tr_df.values).astype("float32")
    X_vl = scaler.transform(X_vl_df.values).astype("float32")
    X_te = scaler.transform(X_te_df.values).astype("float32")

    # tiny sanity check
    print(f"[DEBUG] any NaN in scaled X_tr? {np.isnan(X_tr).any()}")

    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_mlp(X_tr.shape[1], output_dim=3, lr=lr, is_class=True)

    cbs = [
        keras.callbacks.EarlyStopping(
            patience=50,
            restore_best_weights=True,
            monitor="val_loss",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1,
        ),
    ]
    if save_csv_logs:
        cbs.append(keras.callbacks.CSVLogger(str(out_dir / "log.csv"), append=False))

    if save_csv_logs:
        cbs.append(keras.callbacks.CSVLogger(str(out_dir / "log.csv"), append=False))

    hist = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_vl, y_vl),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cbs,
        verbose=verbose,
        class_weight=class_weights,   # <-- add this
    )

    print_training_summary(hist, is_class=True)

    # predictions etc. (unchanged)
    pred_tr = np.argmax(model.predict(X_tr, verbose=0), axis=1)
    pred_vl = np.argmax(model.predict(X_vl, verbose=0), axis=1)
    pred_te = np.argmax(model.predict(X_te, verbose=0), axis=1)

    naive_tr = tr["y_naive"].values
    naive_vl = vl["y_naive"].values
    naive_te = te["y_naive"].values

    metrics = {
        "model": {
            "train_acc": float(accuracy_score(y_tr, pred_tr)),
            "val_acc":   float(accuracy_score(y_vl, pred_vl)),
            "test_acc":  float(accuracy_score(y_te, pred_te)),
        },
        "naive": {
            "train_acc": float(accuracy_score(y_tr, naive_tr)),
            "val_acc":   float(accuracy_score(y_vl, naive_vl)),
            "test_acc":  float(accuracy_score(y_te, naive_te)),
        },
    }

    model.save(out_dir / "model.keras")
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(out_dir / "features.txt", "w", encoding="utf-8") as f:
        for c in X_tr_df.columns:
            f.write(str(c) + "\n")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"[h={horizon}] FINAL — train Acc: {metrics['model']['train_acc']:.4f} | "
        f"val Acc: {metrics['model']['val_acc']:.4f} | "
        f"test Acc: {metrics['model']['test_acc']:.4f}"
    )
    print(
        f"[h={horizon}] NAIVE — train Acc: {metrics['naive']['train_acc']:.4f} | "
        f"val Acc: {metrics['naive']['val_acc']:.4f} | "
        f"test Acc: {metrics['naive']['test_acc']:.4f}"
    )



# ------------------------------ Main ------------------------------
def main():
    a = parse_args()
    df = read_features(a.features_path)
    splits = load_splits(a.splits_json)

    if a.task == "regression":
        if a.target not in REG_ALLOWED:
            raise ValueError(f"--target must be one of {REG_ALLOWED}")
        base = a.out_dir / "regression" / a.target
        base.mkdir(parents=True, exist_ok=True)
        for h in a.horizons:
            run_reg_single(df, splits, a.target, h, a.epochs, a.batch_size, a.lr, base / f"h{h}",
                           verbose=a.verbose, save_csv_logs=a.save_csv_logs)

    elif a.task == "regression_multi":
        for t in a.targets:
            if t not in REG_ALLOWED:
                raise ValueError(f"--targets contains invalid item: {t}")
        base = a.out_dir / "regression_multi" / ("+".join(a.targets))
        base.mkdir(parents=True, exist_ok=True)
        for h in a.horizons:
            run_reg_multi(df, splits, a.targets, h, a.epochs, a.batch_size, a.lr, base / f"h{h}",
                          verbose=a.verbose, save_csv_logs=a.save_csv_logs)

    else:  # classification
        base = a.out_dir / "classification" / "CO_bins"
        base.mkdir(parents=True, exist_ok=True)
        for h in a.horizons:
            run_cls(df, splits, h, a.epochs, a.batch_size, a.lr, base / f"h{h}",
                    verbose=a.verbose, save_csv_logs=a.save_csv_logs)

if __name__ == "__main__":
    main()
