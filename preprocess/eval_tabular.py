#!/usr/bin/env python3
"""
eval_tabular.py — Evaluate saved models on the TEST split (verbose patched)

Supports:
  - regression          (single target)
  - regression_multi    (multiple targets)
  - classification      (3-class CO bins)

Reads:
  - features file (CSV/Parquet, must include 'timestamp')
  - splits.json (from make_splits.py)
  - run_dir (folder that contains model.keras, scaler.pkl, features.txt)

Outputs (under --out_dir):
  - metrics.json
  - predictions CSV(s): timestamp, y_true, y_pred, y_naive, residual
  - optional plots (residual histogram / scatter or confusion matrix)
  - run_log.txt (basic run trace)
"""

from __future__ import annotations
import argparse, json, pickle, sys, traceback
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, recall_score
from tensorflow import keras

REG_ALLOWED = ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate saved Air-Quality models on TEST split.")
    p.add_argument("--features_path", required=True, type=Path)
    p.add_argument("--splits_json", required=True, type=Path)
    p.add_argument("--task", choices=["regression", "regression_multi", "classification"], required=True)
    p.add_argument("--target", default="NO2(GT)")
    p.add_argument("--targets", nargs="+", default=REG_ALLOWED)
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--run_dir", required=True, type=Path, help="Folder containing model.keras, scaler.pkl, features.txt")
    p.add_argument("--out_dir", type=Path, default=Path("eval_out"))
    p.add_argument("--export_csv", action="store_true")
    p.add_argument("--plots", action="store_true")
    p.add_argument("--verbose", action="store_true", help="Print step-by-step progress")
    return p.parse_args()

# -------------- utils ---------------
def vprint(flag, *args):
    if flag: print(*args)


def read_features(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix.lower()==".parquet" else pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("features file must contain 'timestamp'")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return df

def load_splits(path: Path) -> dict:
    js = json.loads(Path(path).read_text(encoding="utf-8"))
    for k in ["train","val","test"]:
        js[k] = [pd.to_datetime(js[k][0]), pd.to_datetime(js[k][1])]
    js["purge_hours"] = int(js.get("purge_hours", 0))
    return js

def slice_range(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()

def rmse_safe(a, b) -> float:
    aa = np.asarray(a, float); bb = np.asarray(b, float)
    m = np.isfinite(aa) & np.isfinite(bb)
    if not np.any(m): return float("nan")
    return float(np.sqrt(mean_squared_error(aa[m], bb[m])))

# ---- label builders (match training) ----
def make_reg_labels(df: pd.DataFrame, target: str, horizon_h: int):
    out = df.copy()
    out["y"] = df[target].shift(-horizon_h)
    out["y_naive"] = df[target]
    out = out.dropna(subset=["y", "y_naive"]).reset_index(drop=True)
    return out


def make_reg_multi_labels(df: pd.DataFrame, targets: List[str], horizon_h: int):
    out = df.copy()
    Y_cols, YN_cols = [], []
    for t in targets:
        out[f"y__{t}"] = df[t].shift(-horizon_h)
        out[f"y_naive__{t}"] = df[t]
        Y_cols.append(f"y__{t}")
        YN_cols.append(f"y_naive__{t}")
    out = out.dropna(subset=Y_cols).reset_index(drop=True)
    return out, Y_cols, YN_cols

def make_cls_labels_CO_bins(df: pd.DataFrame, horizon_h: int):
    out = df.copy()
    future = df["CO(GT)"].shift(-horizon_h)
    now = df["CO(GT)"]

    bins = [-np.inf, 1.5, 2.5, np.inf]
    out["y"] = pd.cut(future, bins=bins, labels=[0, 1, 2], right=False)
    out["y_naive"] = pd.cut(now, bins=bins, labels=[0, 1, 2], right=False)

    out = out.dropna(subset=["y", "y_naive"]).reset_index(drop=True)
    out["y"] = out["y"].astype(int)
    out["y_naive"] = out["y_naive"].astype(int)
    return out


# -------------- plotting --------------
def plot_residual_hist(resid: np.ndarray, out_path: Path, title: str):
    plt.figure()
    plt.hist(resid[~np.isnan(resid)], bins=50, alpha=0.9)
    plt.title(title); plt.xlabel("Residual"); plt.ylabel("Count"); plt.grid(True, ls="--", alpha=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight"); plt.close()

def plot_scatter_true_pred(y_true, y_pred, out_path: Path, title: str):
    plt.figure()
    plt.scatter(y_true, y_pred, s=8, alpha=0.6)
    mn = np.nanmin([np.nanmin(y_true), np.nanmin(y_pred)])
    mx = np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)])
    plt.plot([mn, mx], [mn, mx], "k--", linewidth=1)
    plt.title(title); plt.xlabel("True"); plt.ylabel("Pred"); plt.grid(True, ls="--", alpha=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight"); plt.close()

def plot_confusion(y_true, y_pred, out_path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks([0,1,2], ["low","mid","high"])
    plt.yticks([0,1,2], ["low","mid","high"])
    plt.xlabel("Pred"); plt.ylabel("True"); plt.title(title)
    for i in range(3):
        for j in range(3):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight"); plt.close()

# -------------- main eval --------------
def main():
    a = parse_args()
    # create out_dir early and log basic info
    a.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = a.out_dir / "run_log.txt"
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write("=== eval_tabular run log ===\n")

    def log(msg):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(str(msg) + "\n")

    try:
        log(f"[eval] features_path = {a.features_path}")
        log(f"[eval] splits_json   = {a.splits_json}")
        log(f"[eval] run_dir       = {a.run_dir}")
        log(f"[eval] out_dir       = {a.out_dir}")
        log(f"[eval] task={a.task} horizon={a.horizon}")

        # Paths sanity
        if not a.features_path.exists():
            raise FileNotFoundError(f"features_path not found: {a.features_path}")
        if not a.splits_json.exists():
            raise FileNotFoundError(f"splits_json not found: {a.splits_json}")
        if not a.run_dir.exists():
            raise FileNotFoundError(f"run_dir not found: {a.run_dir}")

        # Load inputs
        df = read_features(a.features_path)
        splits = load_splits(a.splits_json)
        log(f"[eval] features rows = {len(df)} (min ts={df['timestamp'].min()}, max ts={df['timestamp'].max()})")
        log(f"[eval] test window   = {splits['test'][0]} → {splits['test'][1]}")

        model_path = a.run_dir / "model.keras"
        scaler_path = a.run_dir / "scaler.pkl"
        feat_path = a.run_dir / "features.txt"
        target_scaler_path = a.run_dir / "target_scaler.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Missing scaler file: {scaler_path}")
        if not feat_path.exists():
            raise FileNotFoundError(f"Missing features list: {feat_path}")

        log("[eval] Loading model/scaler/feature list …")
        model = keras.models.load_model(model_path)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        feat_list = [ln.strip() for ln in feat_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        log(f"[eval] features used by model: {len(feat_list)} columns")

        feat_means_path = a.run_dir / "feature_means.json"
        feat_means = None
        if feat_means_path.exists():
            feat_means = pd.Series(json.loads(feat_means_path.read_text(encoding="utf-8")))
            log(f"[eval] loaded feature_means for {len(feat_means)} columns")
                # NEW: load target scaler (for normalized regression targets)
        if target_scaler_path.exists():
            ts = json.loads(target_scaler_path.read_text(encoding="utf-8"))
            y_mean = float(ts.get("y_mean", 0.0))
            y_std  = float(ts.get("y_std", 1.0)) or 1.0
            log(f"[eval] loaded target_scaler: mean={y_mean:.4f}, std={y_std:.4f}")
        else:
            # backwards-compatible fallback if you run eval on old models
            y_mean, y_std = 0.0, 1.0
            log("[eval] target_scaler.json not found; assuming outputs are in original units.")


        # Slice test range
        te = slice_range(df, splits["test"][0], splits["test"][1])
        log(f"[eval] test rows after slice = {len(te)}")
        if len(te) == 0:
            log("[WARN] Test slice has 0 rows. Check that your splits.json 'test' window overlaps your feature timestamps.")
            log("[WARN] No metrics will be written. Exiting cleanly.")
            return 0

        if a.task == "regression":
            if a.target not in REG_ALLOWED:
                raise ValueError(f"--target must be one of {REG_ALLOWED}")
            te_lbl = make_reg_labels(te, a.target, a.horizon)
            log(f"[eval] regression: rows with label = {len(te_lbl)}")

            # optional safety: ensure all model features exist
            missing = [c for c in feat_list if c not in te_lbl.columns]
            if missing:
                raise ValueError(
                    f"Feature mismatch: {len(missing)} missing features in features file. Example: {missing[:5]}"
                )

            X_df = te_lbl[feat_list].copy()

            # NEW: apply the same train-mean imputation as in training
            if feat_means is not None:
                # align indices just in case
                common = [c for c in feat_list if c in feat_means.index]
                X_df[common] = X_df[common].fillna(feat_means[common])

            X = X_df.values
            y_true = te_lbl["y"].values
            y_naive = te_lbl["y_naive"].values

            Xs = scaler.transform(X).astype("float32")

            # model outputs normalized targets; de-normalise
            y_pred_n = model.predict(Xs, verbose=0).ravel()
            y_pred   = y_pred_n * y_std + y_mean
            log(f"[DEBUG] NaNs in X_df after imputation: {np.isnan(X_df.values).sum()}")

            m = {
                "model": {"test_rmse": rmse_safe(y_true, y_pred)},
                "naive": {"test_rmse": rmse_safe(y_true, y_naive)}
            }

            (a.out_dir / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
            log(f"[eval] wrote: {a.out_dir / 'metrics.json'}")

            if a.export_csv:
                out = pd.DataFrame({
                    "timestamp": te_lbl["timestamp"],
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "y_naive": y_naive,
                    "residual": y_true - y_pred
                })
                out.to_csv(a.out_dir / "predictions_regression.csv", index=False)
                log(f"[eval] wrote: {a.out_dir / 'predictions_regression.csv'}")
            if a.plots:
                plot_residual_hist(y_true - y_pred, a.out_dir / "plots/resid_hist.png", "Residual Histogram")
                plot_scatter_true_pred(y_true, y_pred, a.out_dir / "plots/true_vs_pred.png", "True vs Pred")
                log(f"[eval] wrote plots under: {a.out_dir / 'plots'}")

            print("Test RMSE (model):", m["model"]["test_rmse"], "| Naive:", m["naive"]["test_rmse"])

        elif a.task == "regression_multi":
            for t in a.targets:
                if t not in REG_ALLOWED:
                    raise ValueError(f"--targets contains invalid item: {t}")

            te_lbl, Y_cols, YN_cols = make_reg_multi_labels(te, a.targets, a.horizon)
            log(f"[eval] regression_multi: rows with labels = {len(te_lbl)}")
            if len(te_lbl) == 0:
                log("[WARN] After shifting labels for horizon, no rows remain. Try a smaller horizon or adjust test window.")
                return 0

            # ensure all model features exist
            missing = [c for c in feat_list if c not in te_lbl.columns]
            if missing:
                raise ValueError(f"Feature mismatch: {len(missing)} missing features in features file. Example: {missing[:5]}")

            # feature dataframe + same imputation as training
            X_df = te_lbl[feat_list].copy()
            if feat_means is not None:
                common = [c for c in feat_list if c in feat_means.index]
                X_df[common] = X_df[common].fillna(feat_means[common])

            X = X_df.values
            Xs = scaler.transform(X).astype("float32")

            Y_true = te_lbl[Y_cols].values
            Y_naive = te_lbl[YN_cols].values

            # load per-target scaler for de-normalisation if available
            ts_multi_path = a.run_dir / "target_scaler_multi.json"
            if ts_multi_path.exists():
                ts_multi = json.loads(ts_multi_path.read_text(encoding="utf-8"))
                stored_cols = ts_multi.get("Y_cols", [])
                if stored_cols != Y_cols:
                    log("[WARN] target_scaler_multi Y_cols mismatch; using raw outputs.")
                    Y_mean = np.zeros(len(Y_cols), dtype="float32")
                    Y_std = np.ones(len(Y_cols), dtype="float32")
                else:
                    means_dict = ts_multi["means"]
                    stds_dict = ts_multi["stds"]
                    Y_mean = np.array([means_dict[c] for c in Y_cols], dtype="float32")
                    Y_std = np.array([max(stds_dict[c], 1e-6) for c in Y_cols], dtype="float32")
                log("[eval] loaded target_scaler_multi.")
            else:
                Y_mean = np.zeros(len(Y_cols), dtype="float32")
                Y_std = np.ones(len(Y_cols), dtype="float32")
                log("[eval] target_scaler_multi.json not found; assuming outputs are in original units.")

            Y_pred_n = model.predict(Xs, verbose=0)
            Y_pred = Y_pred_n * Y_std + Y_mean

            per_t = {}
            for i, t in enumerate(a.targets):
                per_t[t] = {
                    "rmse_model": rmse_safe(Y_true[:, i], Y_pred[:, i]),
                    "rmse_naive": rmse_safe(Y_true[:, i], Y_naive[:, i]),
                }
            macro_model = float(np.nanmean([per_t[t]["rmse_model"] for t in a.targets]))
            macro_naive = float(np.nanmean([per_t[t]["rmse_naive"] for t in a.targets]))
            m = {"model_macro_rmse": macro_model, "naive_macro_rmse": macro_naive, "per_target": per_t}
            (a.out_dir / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
            log(f"[eval] wrote: {a.out_dir / 'metrics.json'}")

            if a.export_csv:
                ts_arr = te_lbl["timestamp"].values
                long_rows = []
                for i, t in enumerate(a.targets):
                    long_rows.append(pd.DataFrame({
                        "timestamp": ts_arr,
                        "target": t,
                        "y_true": Y_true[:, i],
                        "y_pred": Y_pred[:, i],
                        "y_naive": Y_naive[:, i],
                        "residual": Y_true[:, i] - Y_pred[:, i],
                    }))
                csv_path = a.out_dir / "predictions_reg_multi.csv"
                pd.concat(long_rows, ignore_index=True).to_csv(csv_path, index=False)
                log(f"[eval] wrote: {csv_path}")

            if a.plots:
                for i, t in enumerate(a.targets):
                    plot_residual_hist(
                        Y_true[:, i] - Y_pred[:, i],
                        a.out_dir / f"plots/{t}_resid_hist.png",
                        f"Residual Histogram: {t}",
                    )
                    plot_scatter_true_pred(
                        Y_true[:, i], Y_pred[:, i],
                        a.out_dir / f"plots/{t}_true_vs_pred.png",
                        f"True vs Pred: {t}",
                    )
                log(f"[eval] wrote plots under: {a.out_dir / 'plots'}")

            print(f"Macro RMSE — model: {macro_model:.4f} | naive: {macro_naive:.4f}")
            for t in a.targets:
                print(f"  {t}: model {per_t[t]['rmse_model']:.4f} | naive {per_t[t]['rmse_naive']:.4f}")


        else:  # classification
            te_lbl = make_cls_labels_CO_bins(te, a.horizon)
            log(f"[eval] classification: rows with label = {len(te_lbl)}")
            if len(te_lbl) == 0:
                log("[WARN] After shifting labels for horizon, no rows remain. Try a smaller horizon or adjust test window.")
                return 0

            # Ensure features exist
            missing = [c for c in feat_list if c not in te_lbl.columns]
            if missing:
                raise ValueError(f"Feature mismatch: {len(missing)} missing features in features file. Example: {missing[:5]}")

            X = te_lbl[feat_list].fillna(0.0).values
            y_true = te_lbl["y"].values
            y_naive = te_lbl["y_naive"].values
            Xs = scaler.transform(X).astype("float32")
            y_pred = np.argmax(model.predict(Xs, verbose=0), axis=1)

            def macro_recall(y_t, y_p):
                return float(recall_score(y_t, y_p, average="macro", zero_division=0))

            m = {
                "model": {
                    "test_acc": float(accuracy_score(y_true, y_pred)),
                    "test_macro_recall": macro_recall(y_true, y_pred),
                },
                "naive": {
                    "test_acc": float(accuracy_score(y_true, y_naive)),
                    "test_macro_recall": macro_recall(y_true, y_naive),
                },
            }
            (a.out_dir / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
            log(f"[eval] wrote: {a.out_dir / 'metrics.json'}")

            if a.export_csv:
                out = pd.DataFrame({
                    "timestamp": te_lbl["timestamp"],
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "y_naive": y_naive
                })
                csv_path = a.out_dir / "predictions_classification.csv"
                out.to_csv(csv_path, index=False)
                log(f"[eval] wrote: {csv_path}")

            if a.plots:
                plot_confusion(y_true, y_pred, a.out_dir / "plots/confusion.png", f"Confusion Matrix (h={a.horizon})")
                log(f"[eval] wrote plots under: {a.out_dir / 'plots'}")

            print("Test Acc (model):", m["model"]["test_acc"], "| Naive:", m["naive"]["test_acc"])
            print("Test macro recall (model):", m["model"]["test_macro_recall"], "| Naive:", m["naive"]["test_macro_recall"])


        log("[eval] Done.")
        return 0

    except Exception as e:
        print("[ERROR]", e)
        traceback.print_exc()
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write("\n[ERROR] " + str(e) + "\n")
            lf.write(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
