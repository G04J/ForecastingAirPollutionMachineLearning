#!/usr/bin/env python3
"""
Validate the cleaned Air Quality dataset against the raw UCI CSV.

Checks:
  - timestamp creation (parsed, sorted, unique)
  - essential columns present
  - sentinel handling: -200 in raw -> NaN in clean
  - no unintended value changes for TARGET columns (within tolerance)
  - optional: feature equality unless imputation allowed
  - optional: allow dropping rows where ALL targets are missing

Exit code: 0 on success, 1 on failure.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

POLLUTANTS = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
SENSORS = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)"]
MET_VARS = ["T", "RH", "AH"]
ESSENTIAL = POLLUTANTS + SENSORS + MET_VARS

def read_raw_and_prepare_timestamp(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        raw_path, sep=";", decimal=",", na_values=["NA", "NaN"], low_memory=False
    )
    empty_cols = [c for c in df.columns if df[c].dropna().eq("").all()]
    if empty_cols:
        df = df.drop(columns=empty_cols, errors="ignore")
    df.columns = [c.strip() for c in df.columns]
    if "Date" not in df.columns or "Time" not in df.columns:
        raise AssertionError("Raw file must contain 'Date' and 'Time' columns.")
    time_fixed = df["Time"].astype(str).str.replace(".", ":", regex=False)
    ts = pd.to_datetime(
        df["Date"].astype(str).str.strip() + " " + time_fixed.str.strip(),
        dayfirst=True, format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    df = df.assign(timestamp=ts).drop(columns=["Date", "Time"])
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
    return df

def read_clean(clean_path: Path) -> pd.DataFrame:
    if clean_path.suffix.lower() == ".parquet":
        return pd.read_parquet(clean_path)
    if clean_path.suffix.lower() == ".csv":
        return pd.read_csv(clean_path)
    # try both if extensionless/wrong
    if clean_path.with_suffix(".parquet").exists():
        return pd.read_parquet(clean_path.with_suffix(".parquet"))
    if clean_path.with_suffix(".csv").exists():
        return pd.read_csv(clean_path.with_suffix(".csv"))
    raise FileNotFoundError(f"Could not find {clean_path} (.csv or .parquet).")

def numeric_like(s: pd.Series) -> pd.Series:
    """Coerce to float, handling comma decimals if needed."""
    if np.issubdtype(s.dropna().dtype, np.number):
        return s.astype(float)
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_path", required=True, type=Path)
    ap.add_argument("--clean_path", required=True, type=Path)
    ap.add_argument("--allow_feature_imputation", action="store_true")
    ap.add_argument("--allow_drop_all_targets_missing", action="store_true")
    ap.add_argument("--rtol", type=float, default=0.0)
    ap.add_argument("--atol", type=float, default=1e-12)
    args = ap.parse_args()

    ok = True
    errors: list[str] = []

    raw = read_raw_and_prepare_timestamp(args.raw_path)
    clean = read_clean(args.clean_path)

    if "timestamp" not in clean.columns:
        errors.append("Clean file missing 'timestamp' column.")
    else:
        # If CSV saved timestamps as strings, parse them:
        if not np.issubdtype(pd.Series(clean["timestamp"]).dtype, np.datetime64):
            try:
                clean["timestamp"] = pd.to_datetime(clean["timestamp"])
            except Exception:
                errors.append("Clean 'timestamp' column is not parseable as datetime.")

    for col in ESSENTIAL:
        if col not in clean.columns:
            errors.append(f"Clean file missing column: {col}")

    if errors:
        print("VALIDATION: ❌ FAILED")
        for e in errors: print(" -", e)
        sys.exit(1)

    if not pd.Index(clean["timestamp"]).is_monotonic_increasing:
        errors.append("Clean timestamps are not sorted ascending.")
    if clean["timestamp"].duplicated().any():
        errors.append("Clean timestamps contain duplicates.")

    # Index by timestamp
    raw_idx = raw.set_index("timestamp")
    clean_idx = clean.set_index("timestamp")

    # Build expected-raw (after replacing sentinel)
    raw_exp = raw_idx.copy()
    # Replace -200 with NaN on all numeric columns
    for c in raw_exp.columns:
        s = numeric_like(raw_exp[c])
        raw_exp[c] = s.replace(-200.0, np.nan)

    if args.allow_drop_all_targets_missing:
        mask_all_missing_targets = raw_exp[POLLUTANTS].isna().all(axis=1)
        raw_exp = raw_exp.loc[~mask_all_missing_targets]

    # Work only on overlapping timestamps
    common_idx = raw_exp.index.intersection(clean_idx.index).sort_values()
    if len(common_idx) == 0:
        errors.append("No overlapping timestamps between expected-raw and clean.")

    # Sentinel check: -200 in raw (original) must be NaN in clean at the same stamps
    raw_orig = raw_idx  # before sentinel replacement
    for col in ESSENTIAL:
        if col not in raw_orig.columns or col not in clean_idx.columns: 
            continue
        rcol = raw_orig.loc[common_idx, col]
        ccol = clean_idx.loc[common_idx, col]

        # Build sentinel mask robustly
        if np.issubdtype(rcol.dropna().dtype, np.number):
            sent_mask = rcol == -200.0
        else:
            sent_mask = rcol.astype(str).str.strip().eq("-200")

        raw_sentinel_count = int(sent_mask.sum())
        clean_nan_on_sentinel = int(pd.isna(ccol[sent_mask]).sum())
        if clean_nan_on_sentinel != raw_sentinel_count:
            errors.append(
                f"Sentinel replacement mismatch for {col}: "
                f"-200 in raw={raw_sentinel_count}, NaN in clean={clean_nan_on_sentinel}"
            )

    # Compare targets where raw had valid numeric values (not NaN, not -200)
    for col in POLLUTANTS:
        if col not in raw_orig.columns or col not in clean_idx.columns:
            continue
        r_all = raw_orig.loc[common_idx, col]
        c_all = clean_idx.loc[common_idx, col]

        r_vals = numeric_like(r_all)
        c_vals = numeric_like(c_all)

        valid_mask = r_vals.notna() & (r_vals != -200.0)
        if valid_mask.any():
            r_ok = r_vals[valid_mask].to_numpy(dtype=float)
            c_ok = c_vals[valid_mask].to_numpy(dtype=float)

            diffs = np.abs(c_ok - r_ok)
            tol = args.atol + args.rtol * np.abs(r_ok)
            bad = diffs > tol
            if bad.any():
                errors.append(
                    f"Target value drift in {col}: {int(bad.sum())} points exceed tolerance "
                    f"(rtol={args.rtol}, atol={args.atol})"
                )

    # Feature equality (unless imputation allowed)
    if not args.allow_feature_imputation:
        feature_cols = [c for c in ESSENTIAL if c not in POLLUTANTS]
        for col in feature_cols:
            if col not in raw_orig.columns or col not in clean_idx.columns:
                continue
            r_all = raw_orig.loc[common_idx, col]
            c_all = clean_idx.loc[common_idx, col]

            r_vals = numeric_like(r_all)
            c_vals = numeric_like(c_all)

            valid_mask = r_vals.notna() & (r_vals != -200.0)
            if valid_mask.any():
                r_ok = r_vals[valid_mask].to_numpy(dtype=float)
                c_ok = c_vals[valid_mask].to_numpy(dtype=float)

                diffs = np.abs(c_ok - r_ok)
                tol = args.atol + args.rtol * np.abs(r_ok)
                bad = diffs > tol
                if bad.any():
                    errors.append(
                        f"Feature value drift in {col}: {int(bad.sum())} points exceed tolerance "
                        f"(rtol={args.rtol}, atol={args.atol}). "
                        f"Use --allow_feature_imputation if that was intentional."
                    )

    if errors:
        print("VALIDATION: ❌ FAILED")
        for e in errors:
            print(" -", e)
        sys.exit(1)

    # Success summary
    print("VALIDATION: ✅ PASSED")
    print(f"Rows (clean): {len(clean_idx)}")
    print(f"Columns (clean): {len(clean.columns)}")
    present = ", ".join(sorted(set(ESSENTIAL) & set(clean.columns)))
    print("Essential columns present:", present)
    for col in POLLUTANTS:
        if col in clean.columns:
            s = numeric_like(clean[col])
            print(f"{col}: non-NaN={int(s.notna().sum())}, NaN={int(s.isna().sum())}, min={s.min()}, max={s.max()}")

if __name__ == "__main__":
    main()
