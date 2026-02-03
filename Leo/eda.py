#!/usr/bin/env python3
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns  # if not installed, comment these lines and use matplotlib only

IN_PATHS = [
    "data/clean/air_quality_clean.csv",          # fallback
    "data/features/feature_store.parquet"        # preferred if exists
]
OUT_DIR = Path("reports/eda")

TARGETS = ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]

def load_df():
    for p in IN_PATHS:
        path = Path(p)
        if path.exists():
            if path.suffix.lower()==".parquet":
                return pd.read_parquet(path)
            else:
                return pd.read_csv(path)
    raise FileNotFoundError("No input found. Expected one of: " + ", ".join(IN_PATHS))

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_df().copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        df = df.drop_duplicates(subset=["timestamp"])

    # Summary stats
    desc = df.describe(include="all").T
    miss = df.isna().mean().rename("missing_rate").to_frame()

    with open(OUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Rows: {len(df)} | Columns: {df.shape[1]}\n")
        if "timestamp" in df.columns:
            f.write(f"Time range: {df['timestamp'].min()} → {df['timestamp'].max()}\n")
        f.write("\nTop missing columns:\n")
        f.write(miss.sort_values("missing_rate", ascending=False).head(10).to_string())
        f.write("\n")

    desc.to_csv(OUT_DIR / "summary_stats.csv")
    miss.to_csv(OUT_DIR / "missing_rates.csv")

    # Histograms for targets
    for t in TARGETS:
        if t in df.columns:
            plt.figure(figsize=(6,4))
            df[t].dropna().hist(bins=50)
            plt.title(f"Histogram: {t}")
            plt.xlabel(t); plt.ylabel("Count"); plt.grid(True, ls="--", alpha=0.4)
            plt.tight_layout(); plt.savefig(OUT_DIR / f"hist_{t}.png"); plt.close()

    # Diurnal/weekly patterns (if timestamp available)
    if "timestamp" in df.columns:
        df["hour"] = df["timestamp"].dt.hour
        df["dow"]  = df["timestamp"].dt.dayofweek
        for t in TARGETS:
            if t in df.columns:
                gp_h = df.groupby("hour")[t].mean()
                gp_d = df.groupby("dow")[t].mean()

                plt.figure(); gp_h.plot()
                plt.title(f"Mean by Hour: {t}"); plt.xlabel("Hour"); plt.ylabel(t)
                plt.grid(True, ls="--", alpha=0.4); plt.tight_layout()
                plt.savefig(OUT_DIR / f"hourly_{t}.png"); plt.close()

                plt.figure(); gp_d.plot()
                plt.title(f"Mean by Day-of-Week: {t} (0=Mon)"); plt.xlabel("DOW"); plt.ylabel(t)
                plt.grid(True, ls="--", alpha=0.4); plt.tight_layout()
                plt.savefig(OUT_DIR / f"dow_{t}.png"); plt.close()

    # Correlation heatmap (numeric only)
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] > 1:
        corr = num.corr()
        corr.to_csv(OUT_DIR / "correlations.csv")
        plt.figure(figsize=(10,8))
        sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0, cbar_kws={"shrink": .6})
        plt.title("Figure 1 Correlation Heatmap")
        plt.tight_layout(); plt.savefig(OUT_DIR / "correlation_heatmap.png"); plt.close()

    # Rolling mean/variance for volatility insight (example: NO2)
    for t in [c for c in TARGETS if c in df.columns][:1]:
        if "timestamp" in df.columns:
            s = df.set_index("timestamp")[t].sort_index()
            roll = pd.DataFrame({
                "mean_24h": s.rolling(24, min_periods=8).mean(),
                "std_24h":  s.rolling(24, min_periods=8).std()
            }).dropna()
            roll.to_csv(OUT_DIR / f"{t}_rolling24.csv")
            roll.plot(subplots=False, figsize=(10,4), grid=True, alpha=0.9)
            plt.title(f"{t} Rolling 24h Mean/Std")
            plt.tight_layout(); plt.savefig(OUT_DIR / f"{t}_rolling24.png"); plt.close()

    print(f"[EDA] Wrote tables/plots to: {OUT_DIR}")

if __name__ == "__main__":
    main()
