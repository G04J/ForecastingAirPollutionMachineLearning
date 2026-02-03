#!/usr/bin/env python3
"""
make_splits.py — Create leakage-safe temporal splits for Air Quality

Modes:
  A) DATE MODE  — you supply explicit start/end dates for train/val/test
  B) RATIO MODE — you supply train/val/test ratios; script computes split dates by row counts

The script:
  - Reads cleaned or feature data (CSV or Parquet) with a 'timestamp' column
  - Sorts & deduplicates timestamps
  - Creates Train/Val/Test ranges, with a purge gap between sets
  - Snaps requested dates to the nearest existing timestamps
  - Computes rows & percentages per split
  - Optionally emits rolling-origin CV folds
  - Writes splits.json (always)
  - Optional extras:
      * Summary CSV (3 rows)
      * Per-timestamp assignments CSV
      * Export filtered datasets (train/val/test) as CSV/Parquet
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# ----------------- JSON safety helpers -----------------
def _to_native(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(o).strftime("%Y-%m-%d %H:%M:%S")
    return str(o)

# ----------------- CLI -----------------
def parse_args():
    p = argparse.ArgumentParser(description="Create temporal Train/Val/Test + optional rolling CV splits.")
    p.add_argument("--in_path", required=True, type=Path, help="Clean/feature CSV or Parquet with 'timestamp'")
    p.add_argument("--out_path", required=True, type=Path, help="Where to write splits.json")

    # DATE MODE (optional)
    p.add_argument("--train_start")
    p.add_argument("--train_end")
    p.add_argument("--val_start")
    p.add_argument("--val_end")
    p.add_argument("--test_start")
    p.add_argument("--test_end")

    # RATIO MODE (optional)
    p.add_argument("--train_ratio", type=float)
    p.add_argument("--val_ratio", type=float)
    p.add_argument("--test_ratio", type=float)

    # Common options
    p.add_argument("--purge_hours", type=int, default=48, help="Gap between sets to avoid leakage (hours)")
    p.add_argument("--horizons", nargs="+", type=int, default=[1,6,12,24], help="Forecast horizons to record")
    p.add_argument("--make_rolling_cv", action="store_true", help="Emit 2 rolling monthly CV folds (Nov/Dec)")

    # Optional CSV outputs
    p.add_argument("--csv_summary_path", type=Path, default=None,
                   help="Optional CSV with one row per split: start/end/count/percent.")
    p.add_argument("--assignments_csv_path", type=Path, default=None,
                   help="Optional CSV labeling each timestamp with split: train/val/test[/purge].")
    p.add_argument("--apply_purge_labels", action="store_true",
                   help="If set, rows within purge_hours before Val/Test are labeled 'purge' instead of 'train'.")

    # NEW: Export filtered datasets
    p.add_argument("--export_dir", type=Path, default=None,
                   help="If set, write train/val/test datasets here as CSV/Parquet.")
    p.add_argument("--export_format", choices=["csv","parquet","both"], default="both",
                   help="File format(s) to export when --export_dir is set.")
    p.add_argument("--export_prefix", type=str, default=None,
                   help="Filename prefix for exported datasets (default: input filename stem).")
    p.add_argument("--exclude_purge_from_train", action="store_true",
                   help="When exporting, drop 'purge' rows from the TRAIN file.")
    return p.parse_args()

# ----------------- IO helpers -----------------
def read_full_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("Input file must contain 'timestamp' column.")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    df = df.reset_index(drop=True)
    return df

def read_timestamps(path: Path) -> pd.Series:
    df = read_full_df(path)
    return df["timestamp"]  # RangeIndex series

def _as_dtindex(ts: pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(ts)

def snap_to_existing(ts: pd.Series, target: pd.Timestamp, side: str) -> pd.Timestamp:
    """
    Snap target datetime to the nearest existing timestamp in ts.
    side='left'  → previous or equal
    side='right' → next or equal
    """
    tdx = _as_dtindex(ts)
    pos = tdx.searchsorted(target, side="left")
    if side == "left":
        pos = min(max(pos - (0 if (pos < len(tdx) and tdx[pos] == target) else 1), 0), len(tdx)-1)
        return tdx[pos]
    else:
        if pos >= len(tdx):
            return tdx[-1]
        return tdx[pos]

def clamp_range_to_data(ts: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    s = snap_to_existing(ts, start, side="right")
    e = snap_to_existing(ts, end, side="left")
    if s > e:
        raise ValueError(f"Requested range {start}..{end} has no coverage in data after snapping.")
    return s, e

def ratio_mode_to_dates(ts: pd.Series, tr: float, vr: float, te: float):
    n = len(ts)
    if any(x is None for x in [tr, vr, te]):
        raise ValueError("In ratio mode, you must supply --train_ratio, --val_ratio, --test_ratio.")
    if tr <= 0 or vr <= 0 or te <= 0:
        raise ValueError("Ratios must be positive.")
    if not np.isclose(tr + vr + te, 1.0, atol=1e-6):
        raise ValueError("Ratios must sum to 1.0.")

    n_tr = max(1, int(round(n * tr)))
    n_vl = max(1, int(round(n * vr)))
    # keep at least 1 row for test
    if n_tr + n_vl >= n:
        n_tr = int(n * tr)
        n_vl = max(1, min(int(n * vr), n - n_tr - 1))

    tr_start = ts.iloc[0]
    tr_end   = ts.iloc[n_tr - 1]
    vl_start = ts.iloc[n_tr]
    vl_end   = ts.iloc[n_tr + n_vl - 1]
    te_start = ts.iloc[n_tr + n_vl]
    te_end   = ts.iloc[-1]
    return tr_start, tr_end, vl_start, vl_end, te_start, te_end

def count_rows_in_range(ts: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> int:
    tdx = _as_dtindex(ts)
    left = tdx.searchsorted(start, side="left")
    right = tdx.searchsorted(end, side="right")
    return max(0, right - left)

def make_rolling_cv(ts: pd.Series) -> List[Dict]:
    """Two simple monthly folds for Nov and Dec 2004 (adjusts to data coverage)."""
    folds = []
    nov_s, nov_e = pd.Timestamp("2004-11-01 00:00:00"), pd.Timestamp("2004-11-30 23:59:59")
    dec_s, dec_e = pd.Timestamp("2004-12-01 00:00:00"), pd.Timestamp("2004-12-31 23:59:59")
    # Fold 1: train up to Oct, validate Nov
    tr1_s, tr1_e = pd.Timestamp("2004-05-01 00:00:00"), pd.Timestamp("2004-10-31 23:59:59")
    v1_s, v1_e = nov_s, nov_e
    # Fold 2: train up to Nov, validate Dec
    tr2_s, tr2_e = pd.Timestamp("2004-06-01 00:00:00"), nov_e
    v2_s, v2_e = dec_s, dec_e

    for (trs, tre, vs, ve) in [(tr1_s, tr1_e, v1_s, v1_e), (tr2_s, tr2_e, v2_s, v2_e)]:
        try:
            trs_, tre_ = clamp_range_to_data(ts, trs, tre)
            vs_, ve_ = clamp_range_to_data(ts, vs, ve)
            folds.append({"train": [str(trs_), str(tre_)], "val": [str(vs_), str(ve_)]})
        except Exception:
            continue
    return folds

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

# ----------------- main -----------------
def main():
    args = parse_args()

    # Read timestamps and compute ranges
    ts = read_timestamps(args.in_path)
    total_rows = len(ts)

    date_flags = [args.train_start, args.train_end, args.val_start, args.val_end, args.test_start, args.test_end]
    ratio_flags = [args.train_ratio, args.val_ratio, args.test_ratio]
    using_dates = any(x is not None for x in date_flags)
    using_ratios = any(x is not None for x in ratio_flags)
    if using_dates and using_ratios:
        raise ValueError("Provide either explicit dates OR ratios, not both.")

    if using_dates:
        required = ["train_start","train_end","val_start","val_end","test_start","test_end"]
        missing = [k for k,v in zip(required, date_flags) if v is None]
        if missing:
            raise ValueError(f"DATE MODE: missing flags: {missing}")

        tr_s_raw, tr_e_raw = pd.to_datetime(args.train_start), pd.to_datetime(args.train_end)
        vl_s_raw, vl_e_raw = pd.to_datetime(args.val_start), pd.to_datetime(args.val_end)
        te_s_raw, te_e_raw = pd.to_datetime(args.test_start), pd.to_datetime(args.test_end)

        tr_s, tr_e = clamp_range_to_data(ts, tr_s_raw, tr_e_raw)
        vl_s, vl_e = clamp_range_to_data(ts, vl_s_raw, vl_e_raw)
        te_s, te_e = clamp_range_to_data(ts, te_s_raw, te_e_raw)
    else:
        tr = args.train_ratio if args.train_ratio is not None else 0.7
        vr = args.val_ratio   if args.val_ratio   is not None else 0.15
        te = args.test_ratio  if args.test_ratio  is not None else 0.15
        tr_s, tr_e, vl_s, vl_e, te_s, te_e = ratio_mode_to_dates(ts, tr, vr, te)

    purge_hours = int(args.purge_hours)

    # Row counts
    tr_n = count_rows_in_range(ts, tr_s, tr_e)
    vl_n = count_rows_in_range(ts, vl_s, vl_e)
    te_n = count_rows_in_range(ts, te_s, te_e)

    def pct(n): return round(100.0 * n / total_rows, 2)
    summary = {
        "total_rows": int(total_rows),
        "train_rows": int(tr_n), "train_pct": pct(tr_n),
        "val_rows":   int(vl_n), "val_pct":   pct(vl_n),
        "test_rows":  int(te_n), "test_pct":  pct(te_n),
    }

    rolling = make_rolling_cv(ts) if args.make_rolling_cv else None

    out = {
        "horizons": args.horizons,
        "purge_hours": purge_hours,
        "train": [str(tr_s), str(tr_e)],
        "val":   [str(vl_s), str(vl_e)],
        "test":  [str(te_s), str(te_e)],
        "summary": summary,
        "mode": "dates" if using_dates else "ratios",
    }
    if using_dates:
        out["requested_dates"] = {
            "train": [str(pd.to_datetime(args.train_start)), str(pd.to_datetime(args.train_end))],
            "val":   [str(pd.to_datetime(args.val_start)),   str(pd.to_datetime(args.val_end))],
            "test":  [str(pd.to_datetime(args.test_start)),  str(pd.to_datetime(args.test_end))]
        }
    else:
        out["requested_ratios"] = {
            "train_ratio": float(args.train_ratio if args.train_ratio is not None else 0.7),
            "val_ratio":   float(args.val_ratio   if args.val_ratio   is not None else 0.15),
            "test_ratio":  float(args.test_ratio  if args.test_ratio  is not None else 0.15),
        }
    if rolling:
        out["rolling_cv"] = rolling

    # Overlap warning
    if not (tr_e < vl_s and vl_e < te_s):
        out["warning"] = "Snapped ranges are not strictly non-overlapping; please review your inputs."

    # Write JSON
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=_to_native)

    # --- OPTIONAL CSV OUTPUTS: summary + per-timestamp labels ---
    if args.csv_summary_path is not None:
        args.csv_summary_path.parent.mkdir(parents=True, exist_ok=True)
        df_sum = pd.DataFrame([
            {"split": "train", "start": out["train"][0], "end": out["train"][1],
             "rows": summary["train_rows"], "percent": summary["train_pct"]},
            {"split": "val",   "start": out["val"][0],   "end": out["val"][1],
             "rows": summary["val_rows"],   "percent": summary["val_pct"]},
            {"split": "test",  "start": out["test"][0],  "end": out["test"][1],
             "rows": summary["test_rows"],  "percent": summary["test_pct"]},
        ])
        df_sum.to_csv(args.csv_summary_path, index=False)
        print(f"Summary CSV written to: {args.csv_summary_path}")

    # Per-timestamp labels
    purge = pd.Timedelta(hours=purge_hours)
    if args.assignments_csv_path is not None:
        args.assignments_csv_path.parent.mkdir(parents=True, exist_ok=True)
        tr_s_dt, tr_e_dt = pd.to_datetime(out["train"][0]), pd.to_datetime(out["train"][1])
        vl_s_dt, vl_e_dt = pd.to_datetime(out["val"][0]),   pd.to_datetime(out["val"][1])
        te_s_dt, te_e_dt = pd.to_datetime(out["test"][0]),  pd.to_datetime(out["test"][1])

        df_lab = pd.DataFrame({"timestamp": ts})
        lab = pd.Series("other", index=df_lab.index)
        lab[(df_lab["timestamp"] >= tr_s_dt) & (df_lab["timestamp"] <= tr_e_dt)] = "train"
        lab[(df_lab["timestamp"] >= vl_s_dt) & (df_lab["timestamp"] <= vl_e_dt)] = "val"
        lab[(df_lab["timestamp"] >= te_s_dt) & (df_lab["timestamp"] <= te_e_dt)] = "test"
        if args.apply_purge_labels:
            purge_to_val  = (df_lab["timestamp"] >= (vl_s_dt - purge)) & (df_lab["timestamp"] < vl_s_dt)
            purge_to_test = (df_lab["timestamp"] >= (te_s_dt - purge)) & (df_lab["timestamp"] < te_s_dt)
            lab[purge_to_val | purge_to_test] = "purge"
        df_lab["split"] = lab
        df_lab.to_csv(args.assignments_csv_path, index=False)
        print(f"Assignments CSV written to: {args.assignments_csv_path}")

    # --- OPTIONAL: EXPORT FILTERED DATASETS (train/val/test) ---
    if args.export_dir is not None:
        args.export_dir.mkdir(parents=True, exist_ok=True)
        df_full = read_full_df(args.in_path)

        tr_s_dt, tr_e_dt = pd.to_datetime(out["train"][0]), pd.to_datetime(out["train"][1])
        vl_s_dt, vl_e_dt = pd.to_datetime(out["val"][0]),   pd.to_datetime(out["val"][1])
        te_s_dt, te_e_dt = pd.to_datetime(out["test"][0]),  pd.to_datetime(out["test"][1])

        # Base masks by range
        m_train = (df_full["timestamp"] >= tr_s_dt) & (df_full["timestamp"] <= tr_e_dt)
        m_val   = (df_full["timestamp"] >= vl_s_dt) & (df_full["timestamp"] <= vl_e_dt)
        m_test  = (df_full["timestamp"] >= te_s_dt) & (df_full["timestamp"] <= te_e_dt)

        # Optional: exclude purge rows from TRAIN export
        if args.exclude_purge_from_train and purge_hours > 0:
            m_purge_val  = (df_full["timestamp"] >= (vl_s_dt - purge)) & (df_full["timestamp"] < vl_s_dt)
            m_purge_test = (df_full["timestamp"] >= (te_s_dt - purge)) & (df_full["timestamp"] < te_s_dt)
            m_train = m_train & ~(m_purge_val | m_purge_test)

        df_tr, df_vl, df_te = df_full[m_train], df_full[m_val], df_full[m_test]

        prefix = args.export_prefix or args.in_path.stem
        def save_split(df: pd.DataFrame, name: str):
            base = args.export_dir / f"{prefix}_{name}"
            if args.export_format in ("csv", "both"):
                df.to_csv(base.with_suffix(".csv"), index=False)
            if args.export_format in ("parquet", "both"):
                if _parquet_available():
                    df.to_parquet(base.with_suffix(".parquet"), index=False)
                else:
                    print("Warning: Parquet engine not found; wrote CSV only for", name)

        save_split(df_tr, "train")
        save_split(df_vl, "val")
        save_split(df_te, "test")
        print(f"Exported filtered datasets to: {args.export_dir} (format={args.export_format})")

    # Human summary
    print(f"Splits written to: {args.out_path}")
    print("Mode:", out["mode"])
    print("Train:", out["train"][0], "→", out["train"][1], f"({summary['train_rows']} rows, {summary['train_pct']}%)")
    print("Val  :", out["val"][0],   "→", out["val"][1],   f"({summary['val_rows']} rows, {summary['val_pct']}%)")
    print("Test :", out["test"][0],  "→", out["test"][1],  f"({summary['test_rows']} rows, {summary['test_pct']}%)")
    print("Purge gap (hours):", args.purge_hours)
    if "warning" in out:
        print("WARNING:", out["warning"])
    if rolling:
        print(f"Rolling CV folds: {len(rolling)}")

if __name__ == "__main__":
    main()
