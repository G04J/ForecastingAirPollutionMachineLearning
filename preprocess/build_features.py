#!/usr/bin/env python3
"""
Build derived features for the UCI Air Quality dataset (leakage-safe).

Inputs:
  - Cleaned data from preprocess_air_quality.py (must have 'timestamp' column)

Features (past-only):
  - Lags: 1..L per column (default L=48 hours)
  - Rolling means/stds over windows (3,6,12,24h) using past values only
  - Exponentially-weighted means (spans 6,12,24h) using past values only
  - Deltas: first differences for sensors & met variables
  - Calendar: hour_of_day, day_of_week, month + sin/cos encodings
  - Weekend flag
  - Simple anomaly flags (per pollutant): |z|>3 based on 24h rolling mean/std of past values

Usage:
  python build_features.py \
    --in_path data/clean/air_quality_clean.csv \
    --out_path data/features/feature_store.parquet \
    --format both \
    --max_lag 48 \
    --roll_windows 3 6 12 24 \
    --ewm_spans 6 12 24
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

POLLUTANTS = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
SENSORS = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)"]
MET_VARS = ["T", "RH", "AH"]

ESSENTIAL = POLLUTANTS + SENSORS + MET_VARS

def _parse_args():
    p = argparse.ArgumentParser(description="Build leakage-safe derived features.")
    p.add_argument("--in_path", required=True, type=Path)
    p.add_argument("--out_path", required=True, type=Path)
    p.add_argument("--format", choices=["csv", "parquet", "both"], default="parquet")
    p.add_argument("--max_lag", type=int, default=48, help="Max lag (hours) to create")
    p.add_argument("--roll_windows", nargs="+", type=int, default=[3,6,12,24],
                   help="Rolling windows (hours) for mean/std")
    p.add_argument("--ewm_spans", nargs="+", type=int, default=[6,12,24],
                   help="EWM spans (hours) for means")
    return p.parse_args()

def _read_clean(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    # ensure timestamp dtype
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    return df.set_index("timestamp")

def _make_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["hour"] = df.index.hour
    out["dow"] = df.index.dayofweek
    out["month"] = df.index.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    # Cyclical encodings
    out["hour_sin"] = np.sin(2*np.pi*out["hour"]/24)
    out["hour_cos"] = np.cos(2*np.pi*out["hour"]/24)
    out["dow_sin"]  = np.sin(2*np.pi*out["dow"]/7)
    out["dow_cos"]  = np.cos(2*np.pi*out["dow"]/7)
    out["month_sin"]= np.sin(2*np.pi*(out["month"]-1)/12)
    out["month_cos"]= np.cos(2*np.pi*(out["month"]-1)/12)
    return out

def _lag_features(df: pd.DataFrame, cols: List[str], L: int) -> pd.DataFrame:
    feats = {}
    for c in cols:
        for l in range(1, L+1):
            feats[f"{c}_lag_{l}"] = df[c].shift(l)
    return pd.DataFrame(feats, index=df.index)

def _rolling_features(df: pd.DataFrame, cols: List[str], windows: List[int]) -> pd.DataFrame:
    """
    Past-only rolling stats: shift by 1 step BEFORE rolling so current row
    never sees its own value.
    """
    feats = {}
    for c in cols:
        s = df[c].shift(1)  # exclude current
        for w in windows:
            r = s.rolling(window=w, min_periods=1)
            feats[f"{c}_roll{w}_mean"] = r.mean()
            feats[f"{c}_roll{w}_std"]  = r.std(ddof=0)
    return pd.DataFrame(feats, index=df.index)

def _ewm_features(df: pd.DataFrame, cols: List[str], spans: List[int]) -> pd.DataFrame:
    feats = {}
    for c in cols:
        s = df[c].shift(1)  # exclude current
        for span in spans:
            e = s.ewm(span=span, adjust=False, min_periods=1)
            feats[f"{c}_ewm{span}_mean"] = e.mean()
    return pd.DataFrame(feats, index=df.index)

def _delta_features(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    feats = {}
    for c in cols:
        feats[f"{c}_diff_1"] = df[c].diff(1)  # difference uses past implicitly
    return pd.DataFrame(feats, index=df.index)

def _simple_anomaly_flags(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Z-score vs past 24h mean/std (current excluded).
    Flag |z| > 3 as anomaly. Works even with NaNs.
    """
    flags = {}
    for c in cols:
        s = df[c]
        past = s.shift(1)
        m = past.rolling(24, min_periods=12).mean()
        sd = past.rolling(24, min_periods=12).std(ddof=0)
        z = (s - m) / sd
        flags[f"{c}_is_anom"] = (np.abs(z) > 3).astype("float").where(~z.isna())
    return pd.DataFrame(flags, index=df.index)

def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return True
        except Exception:
            return False

def main():
    args = _parse_args()
    df = _read_clean(args.in_path)

    # restrict to essential columns if present
    present = [c for c in ESSENTIAL if c in df.columns]
    base = df[present].copy()

    # Derived features
    cal = _make_calendar(base)
    lags = _lag_features(base, present, args.max_lag)
    rolls = _rolling_features(base, present, args.roll_windows)
    ewms = _ewm_features(base, present, args.ewm_spans)
    deltas = _delta_features(base, SENSORS + MET_VARS)  # deltas mostly useful for sensors & weather
    anom = _simple_anomaly_flags(base, POLLUTANTS)

    # Concatenate (align on index)
    feat = pd.concat([base, cal, lags, rolls, ewms, deltas, anom], axis=1)

    # Save
    out_base = Path(args.out_path).with_suffix("")  # strip extension
    out_dir = out_base.parent
    out_dir.mkdir(parents=True, exist_ok=True)      # <-- create folders

    csv_path = out_base.with_suffix(".csv")
    parquet_path = out_base.with_suffix(".parquet")

    wrote = []
    if args.format in ("csv", "both"):
        feat.reset_index().to_csv(csv_path, index=False)
        wrote.append(str(csv_path))
    if args.format in ("parquet", "both"):
        if _parquet_available():
            feat.reset_index().to_parquet(parquet_path, index=False)
            wrote.append(str(parquet_path))
        else:
            if args.format == "parquet":
                raise ImportError("Parquet engine not found. Install 'pyarrow' or 'fastparquet', or use --format csv.")
            print("Warning: Parquet engine not found; wrote CSV only.")

    print("FEATURES: done")
    print(f"Rows: {len(feat):,}  Cols: {feat.shape[1]:,}")
    print("Wrote:")
    for w in wrote:
        print(" -", w)


if __name__ == "__main__":
    main()
