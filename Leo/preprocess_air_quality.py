#!/usr/bin/env python3
"""
Preprocess the UCI Air Quality dataset and print a detailed modification report.

- Handles semicolon separator + comma decimals.
- Builds a 'timestamp' column from Date+Time (HH.MM.SS → HH:MM:SS).
- Drops "trailing empty" columns (columns whose non-NA values are all empty strings).
- Converts numerics; replaces sentinel -200 with NaN.
- Optional feature-only imputation (ffill / ffill+rolling).
- Saves CSV, Parquet, or both. Parquet is skipped with a warning if no engine is installed.
- Prints a comprehensive report; optionally writes it to JSON via --report_path.

Usage (both formats + report):
  python preprocess_air_quality.py \
    --in_path data/raw/AirQualityUCI.csv \
    --out_path data/clean/air_quality_clean.parquet \
    --format both \
    --impute_features ffill+rolling \
    --rolling_window 6 \
    --drop_rows_with_all_targets_missing \
    --report_path data/clean/preprocess_report.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

POLLUTANTS = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
SENSOR_RESPONSES = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)"]
MET_VARS = ["T", "RH", "AH"]

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess UCI Air Quality with detailed reporting.")
    p.add_argument("--in_path", required=True, type=Path, help="Path to raw AirQualityUCI.csv")
    p.add_argument(
        "--out_path", required=True, type=Path,
        help="Output base path. If --format=both, writes .csv and .parquet using this base name."
    )
    p.add_argument("--format", choices=["parquet", "csv", "both"], default="parquet",
                   help="Which format(s) to write.")
    p.add_argument("--impute_features", default="none",
                   choices=["none", "ffill", "ffill+rolling"],
                   help="Impute strategy for *feature* columns only (not targets).")
    p.add_argument("--rolling_window", type=int, default=6,
                   help="Rolling window (hours) for mean when using 'ffill+rolling'.")
    p.add_argument("--drop_rows_with_all_targets_missing", action="store_true",
                   help="Drop rows where all pollutant targets are NaN after sentinel replacement.")
    p.add_argument("--report_path", type=Path, default=None,
                   help="Optional JSON file to save the modification report.")
    return p.parse_args()

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

def _read_raw_with_report(in_path: Path) -> Tuple[pd.DataFrame, Dict]:
    rep: Dict = {"phase": "read_raw", "dropped_empty_columns": [], "raw_shape_before": None, "raw_shape_after": None}
    df = pd.read_csv(
        in_path, sep=";", decimal=",", na_values=["NA", "NaN"], low_memory=False
    )
    rep["raw_shape_before"] = list(df.shape)

    # Identify “trailing empty” columns: all non-NA values are empty strings
    empty_cols = [c for c in df.columns if df[c].dropna().astype(str).str.strip().eq("").all()]
    if empty_cols:
        df = df.drop(columns=empty_cols, errors="ignore")
        rep["dropped_empty_columns"] = empty_cols

    # Normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]
    rep["raw_shape_after"] = list(df.shape)
    return df, rep

def _parse_timestamp_with_report(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    rep = {"phase": "timestamp", "rows_before": len(df), "rows_bad_timestamp": 0, "duplicates_removed": 0}
    # Parse timestamp
    if "Date" not in df.columns or "Time" not in df.columns:
        raise ValueError("Raw file must contain 'Date' and 'Time' columns.")

    time_fixed = df["Time"].astype(str).str.replace(".", ":", regex=False)
    ts = pd.to_datetime(
        df["Date"].astype(str).str.strip() + " " + time_fixed.str.strip(),
        dayfirst=True, format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    df = df.assign(timestamp=ts).drop(columns=["Date", "Time"])
    bad_ts = df["timestamp"].isna().sum()
    rep["rows_bad_timestamp"] = int(bad_ts)
    df = df.dropna(subset=["timestamp"])

    # Sort & de-duplicate by timestamp (keep first)
    df = df.sort_values("timestamp")
    dup_count = int(df["timestamp"].duplicated().sum())
    rep["duplicates_removed"] = dup_count
    df = df.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
    rep["rows_after"] = len(df)
    return df, rep

def _coerce_numeric_with_report(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    rep = {"phase": "numeric_coercion", "coerced_columns": [], "non_numeric_columns": []}
    for c in df.columns:
        if c == "timestamp":
            continue
        # Attempt numeric coercion with comma-decimal safety
        before_dtype = df[c].dtype
        s = df[c]
        if not np.issubdtype(s.dropna().dtype, np.number):
            s_conv = pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False).str.strip(), errors="coerce")
            if s_conv.notna().sum() > 0:
                df[c] = s_conv
                rep["coerced_columns"].append({"column": c, "from": str(before_dtype), "to": str(df[c].dtype)})
            else:
                rep["non_numeric_columns"].append(c)
        # If already numeric, leave it
    return df, rep

def _sentinel_to_nan_with_report(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    rep = {"phase": "sentinel", "replacements_per_column": {}}
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        mask = df[c] == -200.0
        cnt = int(mask.sum())
        if cnt > 0:
            df.loc[mask, c] = np.nan
        rep["replacements_per_column"][c] = cnt
    return df, rep

def _drop_all_targets_missing_with_report(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    rep = {"phase": "drop_all_targets_missing", "rows_before": len(df), "rows_dropped": 0}
    mask_all_missing = df[POLLUTANTS].isna().all(axis=1)
    dropped = int(mask_all_missing.sum())
    if dropped > 0:
        df = df.loc[~mask_all_missing].copy()
    rep["rows_dropped"] = dropped
    rep["rows_after"] = len(df)
    return df, rep

def _impute_features_with_report(
    df: pd.DataFrame, strategy: str, window: int, feature_cols: List[str]
) -> Tuple[pd.DataFrame, Dict]:
    rep = {"phase": "imputation", "strategy": strategy, "window": window, "filled_counts_per_column": {}}
    if strategy == "none":
        return df, rep

    df = df.sort_values("timestamp").set_index("timestamp")
    # Track NaN counts before
    before_nan = df[feature_cols].isna().sum()

    if strategy in {"ffill", "ffill+rolling"}:
        df[feature_cols] = df[feature_cols].ffill()

    if strategy == "ffill+rolling":
        roll = df[feature_cols].rolling(window=window, min_periods=1).mean()
        df[feature_cols] = df[feature_cols].fillna(roll)

    after_nan = df[feature_cols].isna().sum()
    filled = (before_nan - after_nan).clip(lower=0)
    rep["filled_counts_per_column"] = filled.astype(int).to_dict()

    df = df.reset_index()
    return df, rep

def _order_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = ["timestamp"] + POLLUTANTS + SENSOR_RESPONSES + MET_VARS
    ordered = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[ordered]

def preprocess(
    in_path: Path,
    out_path: Path,
    out_format: str = "parquet",
    impute_features: str = "none",
    rolling_window: int = 6,
    drop_all_targets_missing: bool = False,
    report_path: Path | None = None,
) -> None:
    # Read
    df_raw, rep_read = _read_raw_with_report(in_path)
    # Timestamp parsing, sorting, de-duping
    df_ts, rep_ts = _parse_timestamp_with_report(df_raw)
    # Numeric coercion
    df_num, rep_num = _coerce_numeric_with_report(df_ts)
    # Sentinel replacement
    df_nan, rep_sen = _sentinel_to_nan_with_report(df_num)

    # Optional drop rows where all targets missing
    if drop_all_targets_missing:
        df_nan, rep_drop = _drop_all_targets_missing_with_report(df_nan)
    else:
        rep_drop = {"phase": "drop_all_targets_missing", "skipped": True}

    # Impute features (not targets)
    feature_cols = [c for c in df_nan.columns if c not in POLLUTANTS + ["timestamp"]]
    df_imp, rep_imp = _impute_features_with_report(df_nan, impute_features, rolling_window, feature_cols)

    # Final ordering
    df_final = _order_columns(df_imp)

    # Save
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = out_path.with_suffix("")  # strip extension
    parquet_path = base.with_suffix(".parquet")
    csv_path = base.with_suffix(".csv")

    wrote = []
    if out_format in ("parquet", "both"):
        if _parquet_available():
            df_final.to_parquet(parquet_path, index=False)
            wrote.append(str(parquet_path))
        else:
            if out_format == "parquet":
                raise ImportError("Parquet engine not found. Install 'pyarrow' or 'fastparquet', or use --format csv.")
            logging.warning("Parquet engine not found. Skipping Parquet.")
    if out_format in ("csv", "both"):
        df_final.to_csv(csv_path, index=False)
        wrote.append(str(csv_path))

    # Build and print report
    report = {
        "in_path": str(in_path),
        "outputs": wrote,
        "summary": {
            "rows_final": int(len(df_final)),
            "cols_final": int(df_final.shape[1]),
        },
        "steps": {
            "read_raw": rep_read,
            "timestamp": rep_ts,
            "numeric_coercion": rep_num,
            "sentinel": rep_sen,
            "drop_all_targets_missing": rep_drop,
            "imputation": rep_imp,
        },
        "columns_present": list(df_final.columns),
        "targets_non_null_counts": {c: int(df_final[c].notna().sum()) for c in POLLUTANTS if c in df_final.columns},
        "targets_null_counts": {c: int(df_final[c].isna().sum()) for c in POLLUTANTS if c in df_final.columns},
    }

    print("\n=== PREPROCESS REPORT ===")
    print(f"Input: {report['in_path']}")
    print("Wrote:")
    for o in wrote:
        print(f"  - {o}")

    # Read/raw phase
    rr = report["steps"]["read_raw"]
    print("\n[read_raw]")
    print(f"  raw shape before: {tuple(rr['raw_shape_before'])}")
    print(f"  raw shape after : {tuple(rr['raw_shape_after'])}")
    if rr["dropped_empty_columns"]:
        print(f"  dropped empty columns: {rr['dropped_empty_columns']}")
    else:
        print("  dropped empty columns: none")

    # Timestamp
    tsr = report["steps"]["timestamp"]
    print("\n[timestamp]")
    print(f"  rows before            : {tsr['rows_before']}")
    print(f"  rows with bad timestamp: {tsr['rows_bad_timestamp']}")
    print(f"  duplicate timestamps   : {tsr['duplicates_removed']}")
    print(f"  rows after             : {tsr['rows_after']}")

    # Numeric coercion
    ncr = report["steps"]["numeric_coercion"]
    print("\n[numeric_coercion]")
    print(f"  coerced columns ({len(ncr['coerced_columns'])}): {ncr['coerced_columns']}")
    if ncr["non_numeric_columns"]:
        print(f"  non-numeric columns kept as object: {ncr['non_numeric_columns']}")

    # Sentinel
    sen = report["steps"]["sentinel"]
    print("\n[sentinel]")
    replaced_any = {c: v for c, v in sen["replacements_per_column"].items() if v > 0}
    print(f"  -200 → NaN replacements (nonzero only): {replaced_any if replaced_any else 'none'}")

    # Drop rows with all targets missing
    dr = report["steps"]["drop_all_targets_missing"]
    print("\n[drop_all_targets_missing]")
    if "skipped" in dr:
        print("  skipped: True")
    else:
        print(f"  rows before: {dr['rows_before']}")
        print(f"  rows dropped: {dr['rows_dropped']}")
        print(f"  rows after : {dr['rows_after']}")

    # Imputation
    imp = report["steps"]["imputation"]
    print("\n[imputation]")
    print(f"  strategy: {imp['strategy']}")
    if imp["strategy"] != "none":
        # only show columns where we actually filled something
        filled = {c: n for c, n in imp["filled_counts_per_column"].items() if n > 0}
        print(f"  filled counts per column (nonzero only): {filled if filled else 'none'}")
        print(f"  window: {imp['window']}")

    # Summary
    print("\n[summary]")
    print(f"  final shape: {report['summary']['rows_final']} rows × {report['summary']['cols_final']} cols")
    for c in POLLUTANTS:
        if c in report["targets_non_null_counts"]:
            nn = report["targets_non_null_counts"][c]
            na = report["targets_null_counts"][c]
            print(f"  {c}: non-NaN={nn}, NaN={na}")

    # Optional JSON report
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport JSON written to: {report_path}")

def main():
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    preprocess(
        in_path=args.in_path,
        out_path=args.out_path,
        out_format=args.format,
        impute_features=args.impute_features,
        rolling_window=args.rolling_window,
        drop_all_targets_missing=args.drop_rows_with_all_targets_missing,
        report_path=args.report_path,
    )

if __name__ == "__main__":
    main()
