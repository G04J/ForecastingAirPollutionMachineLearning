# COMP9417 Group Project - Group Numpy 

## Feed-forward neural network (MLP) for regression + classification
 Haocheng Li (Leo z5571326)
## Air Quality Forecasting Pipeline

This repository contains an end-to-end pipeline for air-quality forecasting on the UCI Air Quality dataset. It covers:

- Cleaning and standardising the raw dataset
- Building leakage-safe tabular features
- Creating time-based train/val/test splits
- Detecting and removing anomalies to create a “clean” dataset
- Training neural models (regression, multi-target, classification)
- Evaluating trained models against naïve baselines
- Basic exploratory data analysis (EDA)
- Validating that cleaned/split data is consistent with the original raw file

The code is organised as a set of standalone scripts that can be run individually or chained into a full workflow.

---

## 1. Overall Workflow

A typical end-to-end run looks like this:

1. **Preprocess the raw UCI CSV**  
   `preprocess_air_quality.py`  

2. **Validate the cleaned dataset**  
   `check.py`  

3. **Build a feature store from the cleaned data**  
   `build_features.py`  

4. **Create temporal train/val/test splits**  
   `make_splits.py`  

5. **(Optional) Detect anomalies and build a “clean” feature store**  
   `detect_anomalies.py`, `make_clean.py`  

6. **Train tabular neural models**  
   `train_tabular.py`  

7. **Evaluate trained models on the test split**  
   `eval_tabular.py`  

8. **Run exploratory data analysis**  
   `eda.py`  

Downstream, you can plug the feature store + splits into any training/evaluation code you like (neural nets, tree-based models, linear baselines, etc.), but `train_tabular.py` + `eval_tabular.py` already give a complete baseline.

---

## 2. `preprocess_air_quality.py` — Clean & Standardise the Raw CSV

This script ingests the original UCI `AirQualityUCI.csv` file and produces a clean dataset plus a detailed processing report.

### Main responsibilities

- Read the raw file with:
  - Semicolon (`;`) separator
  - Comma-as-decimal handling
- Build a single `timestamp` column from `Date` + `Time`:
  - Fixes `HH.MM.SS` into `HH:MM:SS`
  - Parses as day-first `%d/%m/%Y %H:%M:%S`
- Drop “trailing empty” columns (all non-NA values are empty strings).
- Strip whitespace from column names.
- Coerce numeric-looking columns to real numeric dtype (handling comma decimals).
- Replace sentinel `-200` values with `NaN` in all numeric columns.
- **Optionally**:
  - Drop rows where **all pollutant targets** are missing.
  - Impute *feature* columns (not targets) via:
    - `none`
    - `ffill`
    - `ffill+rolling` (forward fill then rolling-mean fill)
- Save the result to CSV, Parquet, or both.
- Print a human-readable report and optionally save JSON.

### Key concepts

- **Target / pollutant columns**  
  The script treats these as targets and does **not** impute them:
  - `CO(GT)`, `NMHC(GT)`, `C6H6(GT)`, `NOx(GT)`, `NO2(GT)`

- **Feature columns**  
  Everything that is not a pollutant or `timestamp` is treated as a feature for the purposes of optional imputation.

### CLI arguments (high level)

- `--in_path` (required): path to `AirQualityUCI.csv`
- `--out_path` (required): base path for outputs (extension is handled by `--format`)
- `--format`: `parquet` | `csv` | `both` (default `parquet`)
- `--impute_features`: `none` | `ffill` | `ffill+rolling` (default `none`)
- `--rolling_window`: window size (in hours) for rolling-mean imputation (default `6`)
- `--drop_rows_with_all_targets_missing`: if set, drop rows where all pollutants are `NaN`
- `--report_path`: optional JSON path to write the detailed report

### Example usage

```bash
python preprocess_air_quality.py   --in_path data/raw/AirQualityUCI.csv   --out_path data/clean/air_quality_clean.parquet   --format both   --impute_features ffill+rolling   --rolling_window 6   --drop_rows_with_all_targets_missing   --report_path data/clean/preprocess_report.json
```

### Outputs

- Cleaned dataset:
  - `data/clean/air_quality_clean.parquet`
  - `data/clean/air_quality_clean.csv`
- JSON report (optional), with:
  - Shapes before/after each step
  - Which columns were dropped/coerced
  - Sentinel replacement counts
  - Imputation counts per feature column
  - Non-null / null counts for each pollutant

---

## 3. `check.py` — Validate the Cleaned Dataset

This script is a **sanity checker** that compares the cleaned dataset against the raw UCI file. It is designed to be used after `preprocess_air_quality.py` and can be wired into CI (exit code `0` on success, `1` on failure).

### What it checks

1. **Timestamp sanity**
   - `timestamp` column exists in the clean file.
   - Timestamps are parseable, sorted ascending, and unique.

2. **Essential columns present**
   - Pollutants:
     - `CO(GT)`, `NMHC(GT)`, `C6H6(GT)`, `NOx(GT)`, `NO2(GT)`
   - Sensor responses:
     - `PT08.S1(CO)`, `PT08.S2(NMHC)`, `PT08.S3(NOx)`, `PT08.S4(NO2)`, `PT08.S5(O3)`
   - Meteorology:
     - `T`, `RH`, `AH`

3. **Sentinel handling**
   - For all essential columns, every `-200` in the raw file must correspond to a `NaN` in the clean file at the same timestamp.

4. **Target value drift**
   - For pollutant columns, where raw values are valid (numeric and not `-200`), the clean data must match **within tolerance** (`rtol`, `atol`).

5. **Feature value drift (optional)**
   - For non-target essential features, you can ensure values didn’t change unless you explicitly allow imputation:
     - If `--allow_feature_imputation` is **not** given, any drift beyond tolerance is flagged as an error.

6. **Dropping rows with all targets missing (optional)**
   - If `--allow_drop_all_targets_missing` is set, the validator accepts that the clean data may have fewer rows than the raw data because rows with all targets missing were removed.

### CLI arguments (high level)

- `--raw_path` (required): path to original `AirQualityUCI.csv`
- `--clean_path` (required): path to cleaned file (CSV or Parquet)
- `--allow_feature_imputation`: allow drift in feature columns
- `--allow_drop_all_targets_missing`: allow rows with all targets missing to be dropped
- `--rtol`, `--atol`: relative & absolute tolerances for numeric comparison

### Example usage

```bash
python check.py   --raw_path data/raw/AirQualityUCI.csv   --clean_path data/clean/air_quality_clean.parquet   --allow_feature_imputation   --allow_drop_all_targets_missing   --rtol 0.0   --atol 1e-12
```

On success, it prints a summary and exits with code `0`. On failure, it lists all issues and exits with `1`.

---

## 4. `build_features.py` — Leakage-Safe Feature Store

This script takes the cleaned dataset and builds a **feature store** with only past information (no target leakage).

### Inputs

- Clean data from `preprocess_air_quality.py` (CSV or Parquet).
- Must contain a `timestamp` column and as many of the following as possible:
  - Pollutants:
    - `CO(GT)`, `NMHC(GT)`, `C6H6(GT)`, `NOx(GT)`, `NO2(GT)`
  - Sensor responses:
    - `PT08.S1(CO)`, `PT08.S2(NMHC)`, `PT08.S3(NOx)`, `PT08.S4(NO2)`, `PT08.S5(O3)`
  - Meteorology:
    - `T`, `RH`, `AH`

The script will restrict to the subset of these columns that are actually present.

### Feature groups

All features are **past-only**; the current row never uses its own value for rolling/EWM stats.

1. **Base “essential” features**
   - The raw columns for the pollutants, sensors, and meteorological variables (as available).

2. **Calendar features**
   - Hour of day, day of week, month.
   - Weekend flag.
   - Cyclical encodings:
     - `hour_sin`, `hour_cos`
     - `dow_sin`, `dow_cos`
     - `month_sin`, `month_cos`.

3. **Lag features**
   - For each essential column, lags from `1` to `max_lag` hours:
     - e.g. `NO2(GT)_lag_1`, `NO2(GT)_lag_2`, ..., `NO2(GT)_lag_48`.

4. **Rolling statistics (past-only)**
   - For each essential column and each window `w` in `roll_windows`:
     - Rolling mean and std of **past** values:
       - `c_roll{w}_mean`, `c_roll{w}_std`
     - Implemented as: shift by 1 time step, then apply rolling window.

5. **Exponentially weighted means (EWM)**
   - For each essential column and each `span` in `ewm_spans`:
     - `c_ewm{span}_mean` on past values (shifted by 1 before EWM).

6. **Deltas (first differences)**
   - For sensors + met variables:
     - `c_diff_1 = df[c].diff(1)`.

7. **Simple anomaly flags (per pollutant)**
   - For each pollutant:
     - Past-only 24h mean and std (with a minimum of 12 points).
     - Z-score of current value vs past window.
     - Flag `|z| > 3` as anomaly → `c_is_anom`.

### CLI arguments (high level)

- `--in_path` (required): cleaned dataset (CSV or Parquet)
- `--out_path` (required): base output path (extension determined by `--format`)
- `--format`: `csv` | `parquet` | `both` (default `parquet`)
- `--max_lag`: maximum lag in hours (default `48`)
- `--roll_windows`: list of window sizes in hours (default `3 6 12 24`)
- `--ewm_spans`: list of spans for EWM means (default `6 12 24`)

### Example usage

```bash
python build_features.py   --in_path data/clean/air_quality_clean.parquet   --out_path data/features/feature_store.parquet   --format both   --max_lag 48   --roll_windows 3 6 12 24   --ewm_spans 6 12 24
```

### Outputs

- Feature store as:
  - `data/features/feature_store.parquet`
  - `data/features/feature_store.csv`
- Printed summary of:
  - Number of rows
  - Number of columns
  - Paths written

---

## 5. `make_splits.py` — Temporal Train/Val/Test Splits

This script creates **leakage-safe temporal splits** for training and evaluation, operating directly on your cleaned or feature data.

It supports two modes:

- **Date mode** — you specify explicit start/end datetimes for train, val, and test.
- **Ratio mode** — you specify row-count ratios; the script derives the split dates from the ordered timestamps.

In both cases it:

- Sorts and deduplicates timestamps.
- Ensures all requested ranges are snapped to **existing** timestamps.
- Records the number of rows and percentage per split.
- Applies a **purge gap** (in hours) between splits to reduce leakage around boundaries.
- Optionally:
  - Writes CSV summaries.
  - Labels each timestamp with its split.
  - Exports filtered train/val/test datasets.
  - Generates simple rolling CV folds (for November / December 2004).

### Basic inputs

- `--in_path` (required): cleaned or feature CSV/Parquet with a `timestamp` column.
- `--out_path` (required): where `splits.json` will be written.

### Mode selection

You must use **either** date mode or ratio mode, not both.

#### Date mode

Provide:

- `--train_start`, `--train_end`
- `--val_start`, `--val_end`
- `--test_start`, `--test_end`

Each is parsed to a datetime and then **clamped** to the nearest timestamps in the data.

#### Ratio mode

Provide:

- `--train_ratio`
- `--val_ratio`
- `--test_ratio`

These must be positive and (approximately) sum to 1.0. The script uses row counts to determine the boundaries between splits and then converts those rows back to timestamp ranges. If ratios are not provided, defaults are:

- `train_ratio = 0.7`
- `val_ratio = 0.15`
- `test_ratio = 0.15`

### Purge gap and horizons

- `--purge_hours`: integer number of hours between splits used to avoid leakage (default `48`).
- `--horizons`: list of forecast horizons to record in the JSON metadata (default `[1, 6, 12, 24]`).  
  These are not used to filter data, but downstream training scripts can read them to know which horizons to model.

### Optional rolling CV folds

- `--make_rolling_cv`: if set, adds a small set of rolling monthly folds to `splits.json`.  
  These use (clamped to data coverage):
  - Training up to October → validate in November.
  - Training up to November → validate in December.  

### CSV and label outputs

- `--csv_summary_path`: path to a CSV summarising each split (start, end, row count, percentage).
- `--assignments_csv_path`: path to a CSV labeling every timestamp with a split:
  - `train`, `val`, `test`, and optionally `purge` or `other`.
- `--apply_purge_labels`:
  - If set, rows within `purge_hours` of the start of val/test are labeled `purge` (instead of `train`) in the assignments CSV.

### Exporting filtered datasets

- `--export_dir`: directory in which to export train/val/test datasets.
- `--export_format`: `csv` | `parquet` | `both` (default `both`).
- `--export_prefix`: optional prefix for output filenames (default: input filename stem).
- `--exclude_purge_from_train`:
  - If set, rows within the purge window before val/test are excluded from the exported train file.

Example exports will look like:

- `<export_dir>/<prefix>_train.csv` / `.parquet`
- `<export_dir>/<prefix>_val.csv` / `.parquet`
- `<export_dir>/<prefix>_test.csv` / `.parquet`

### Example (ratio mode, common case)

```bash
python make_splits.py   --in_path data/features/feature_store.parquet   --out_path data/splits/splits.json   --train_ratio 0.6   --val_ratio 0.2   --test_ratio 0.2   --purge_hours 48   --horizons 1 6 12 24   --csv_summary_path data/splits/split_summary.csv   --assignments_csv_path data/splits/split_assignments.csv   --apply_purge_labels   --export_dir data/splits/exported   --export_format both   --export_prefix feature_store   --exclude_purge_from_train
```

### Example (date mode)

```bash
python make_splits.py   --in_path data/features/feature_store.parquet   --out_path data/splits/splits.json   --train_start 2004-03-10   --train_end   2004-09-30   --val_start   2004-10-01   --val_end     2004-11-30   --test_start  2004-12-01   --test_end    2005-04-04   --purge_hours 48
```

### `splits.json` contents (conceptual)

The JSON contains:

- `horizons`
- `purge_hours`
- Split ranges:
  - `"train": ["YYYY-MM-DD HH:MM:SS", "YYYY-MM-DD HH:MM:SS"]`
  - `"val":   [...]`
  - `"test":  [...]`
- A `summary` block:
  - Total rows
  - Rows and percentages per split
- Either:
  - `requested_dates` (date mode), or
  - `requested_ratios` (ratio mode)
- Optionally:
  - `rolling_cv`: list of rolling-fold train/val ranges
  - `warning` if snapped ranges overlap and need attention

---

## 6. `detect_anomalies.py` — Rolling Residuals + Isolation Forest

This script flags anomalies separately on the train/val/test splits and writes **anomaly-free** CSVs for each split. It is intended to be run after `make_splits.py` has exported split-specific feature CSVs.

### Inputs

- `feature_store_train.csv`, `feature_store_val.csv`, `feature_store_test.csv`, typically under something like:
  - `~/Downloads/air+quality/data/splits/feature_store_*.csv`

It assumes each CSV has a `timestamp` column and standard target/sensor/weather columns.

### Steps

1. **Load and clean timestamps**
   - Ensure `timestamp` exists.
   - Convert to datetime, drop invalid timestamps, sort chronologically.
   - Set `timestamp` as index.
   - Add basic time features: `hour`, `weekday`, `month`.

2. **Rolling 24h residual anomalies (per target)**
   - For each target pollutant (`CO(GT)`, `NMHC(GT)`, `C6H6(GT)`, `NOx(GT)`, `NO2(GT)`):
     - Compute 24-hour rolling mean and std on that split (train/val/test separately, time-based windows).
     - Compute residuals = value − rolling_mean.
     - Flag anomalies where `|residual| > 3 * rolling_std` (and std is non-null).
     - Store indicator in `an_resid_<target>`.

3. **Isolation Forest on sensor + weather features**
   - Use `feature_cols = sensors + weather`, i.e. PT08.* plus `T`, `RH`, `AH`.
   - Build a scikit-learn `Pipeline`:
     - `SimpleImputer(strategy="median")`
     - `StandardScaler`
     - `IsolationForest(contamination=0.02, random_state=42)`
   - **Fit only on train** features, then apply to train/val/test.
   - Add `an_iso` column: 1 if anomaly, 0 otherwise.

4. **Combine anomaly flags**
   - For each split, collect all columns starting with `an_` and define:
     - `an_any = max(an_* across row)`
   - This indicates if any of the residual-based or IsolationForest detectors triggered.

5. **Filter clean rows**
   - For each split:
     - `*_clean = df[df["an_any"] == 0].copy()`
   - Prints row counts and how many were dropped.

6. **Write anomaly-free split CSVs**
   - Writes to:
     - e.g. `~/Downloads/9417Groupproj_preprocess-main/CleanData/anomaly_free_train.csv`
     - plus `anomaly_free_val.csv`, `anomaly_free_test.csv`
   - Resets index so `timestamp` is back as a column.

> **Note:** The paths are hard-coded to a specific `Downloads/...` layout. You will almost certainly want to update `base_dir` and `output_dir` to match your project structure.

---

## 7. `make_clean.py` — Assemble a “Clean” Feature Store

This script combines the anomaly-free train/val/test CSVs and writes a single, time-sorted Parquet file (a “clean” feature store).

### Steps

1. Read:
   - `anomaly_free_train.csv`
   - `anomaly_free_val.csv`
   - `anomaly_free_test.csv`
   from a configured `base_clean` directory.

2. Concatenate and clean timestamps
   - Concatenate the three into one DataFrame.
   - Drop rows with missing `timestamp`.
   - Parse `timestamp` to datetime, drop unparseable rows.
   - Sort by `timestamp` and reset the index.

3. Write clean feature store
   - Save as a Parquet file to a configured path (e.g. `feature_store_clean.parquet`).
   - This file matches the style of the original `feature_store.parquet` but with anomaly rows removed.

Again, paths are currently hard-coded to a local Windows directory and should be adapted to your own `data/` structure.

---

## 8. `train_tabular.py` — Train Neural Tabular Models

This script trains Keras MLPs for three task types:

- `regression` — single-target regression (e.g. NO2(GT) at t+h)
- `regression_multi` — multi-target regression (default: CO, C6H6, NOx, NO2)
- `classification` — 3-class CO(GT) bins (<1.5, 1.5–2.5, ≥2.5) at t+h

Models are trained separately for each forecast horizon `h` in `--horizons` and compared against a strong **naïve baseline**:

- Regression: persistence `y_naive(t+h) = y(t)`
- Classification: “copy current bin” baseline

### Inputs

- `--features_path`: feature store CSV/Parquet with a `timestamp` column.
- `--splits_json`: produced by `make_splits.py`, including:
  - `train`, `val`, `test` time ranges
  - `purge_hours` (optional)

### Tasks and labels

1. **Single-target regression**
   - `--task regression`
   - `--target` one of `[CO(GT), C6H6(GT), NOx(GT), NO2(GT)]`
   - For each horizon `h`:
     - Label `y = target.shift(-h)` (future value).
     - Naive `y_naive = target` (current value).
     - Rows where either is NaN are dropped.

2. **Multi-target regression**
   - `--task regression_multi`
   - `--targets` multiple targets (default `[CO(GT), C6H6(GT), NOx(GT), NO2(GT)]`).
   - For each horizon `h`:
     - For each target `t`:
       - `y__t = t.shift(-h)`
       - `y_naive__t = t`
     - Drop rows where any future/naive target is NaN.

3. **3-class CO classification**
   - `--task classification`
   - Future and current CO(GT) are binned into:
     - <1.5 → class 0
     - 1.5–2.5 → class 1
     - ≥2.5 → class 2
   - Drop rows where current or future bin is undefined.

### Split handling & purge

For each task and horizon:

- Slice full feature store into train/val/test ranges using `splits.json`.
- Optionally remove rows in the purge window right before val/test from the training data (based on `purge_hours`).

### Features, scaling & normalisation

- **Single-target regression**
  - Build labels, then use all non-label, non-`timestamp` columns as candidates.
  - Apply a heuristic filter to favour columns containing tokens such as:
    - `CO(GT)`, `C6H6(GT)`, `NOx(GT)`, `NO2(GT)`, `O3`, `T`, `RH`, `AH`.
  - Fallback to all features if the filter removes everything.
  - Impute features using train means, then scale with `StandardScaler`.
  - Normalise the target `y` using train mean and std; model predicts normalised values, which are de-normalised for metrics and saving.

- **Multi-target regression**
  - Use all non-label, non-`timestamp` columns as features.
  - Train-mean imputation + `StandardScaler`.
  - Per-target normalisation (vector of means/stds).

- **Classification**
  - Use all non-`timestamp`, non-label columns as features.
  - Train-mean imputation + `StandardScaler`.
  - Compute class weights from training labels to handle class imbalance and pass them to Keras via `class_weight`.

### Model architectures

- **Regression (single & multi)**  
  MLP:
  - Dense(64, ReLU, L2) → Dropout(0.4) → Dense(32, ReLU, L2) → Dense(output_dim)  
  Compiled with:
  - Optimiser: Adam(lr=`--lr`)
  - Loss: Huber (delta=1.0)
  - Metric: MAE

- **Classification**
  - Dense(32, ReLU, L2) → Dense(3, softmax)  
  Compiled with:
  - Loss: sparse categorical cross-entropy
  - Metrics: accuracy

### Training details

- Early stopping on `val_loss` with patience (100 epochs for regression, 50 for multi and classification).
- `ReduceLROnPlateau` on `val_loss` to adapt learning rate.
- Optional CSV logging (`--save_csv_logs`).

### CLI arguments (high level)

- `--task`: `regression` | `regression_multi` | `classification`
- `--target`: single target (regression only)
- `--targets`: list of targets (regression_multi)
- `--horizons`: one or more horizons (default `1 6 12 24`)
- `--epochs`: max epochs (default `200`)
- `--batch_size`: default `512`
- `--lr`: learning rate (default `1e-3`)
- `--out_dir`: base output directory (default `artifacts`)
- `--verbose`: Keras verbosity (0, 1, or 2)
- `--save_csv_logs`: if set, writes `log.csv` per run

### Example usage

```bash
## Single-target NO2 regression, horizons 6 and 24
python train_tabular.py   --features_path data/features/feature_store.parquet   --splits_json data/splits/splits.json   --task regression   --target "NO2(GT)"   --horizons 6 24   --out_dir artifacts   --epochs 200   --batch_size 512   --lr 1e-3   --save_csv_logs
```

```bash
## Multi-target regression for CO, C6H6, NOx, NO2 at horizon 24
python train_tabular.py   --features_path data/features/feature_store_clean.parquet   --splits_json data/splits/splits.json   --task regression_multi   --targets "CO(GT)" "C6H6(GT)" "NOx(GT)" "NO2(GT)"   --horizons 24   --out_dir artifacts_clean
```

```bash
## 3-class CO classification at h=2 and h=24
python train_tabular.py   --features_path data/features/feature_store.parquet   --splits_json data/splits/splits.json   --task classification   --horizons 2 24   --out_dir artifacts
```

### Outputs per run

For each task/horizon, under a directory like:

- `artifacts/regression/NO2(GT)/h24/`
- `artifacts/regression_multi/CO(GT)+C6H6(GT)+NOx(GT)+NO2(GT)/h24/`
- `artifacts/classification/CO_bins/h24/`

you get (some subsets depending on the task):

- `model.keras` — Keras model file
- `scaler.pkl` — scikit-learn `StandardScaler` for features
- `features.txt` — feature column names in the order used by the model
- `feature_means.json` — train-time feature means used for NaN imputation (regression & multi)
- `target_scaler.json` — mean/std for single-target regression (for de-normalisation)
- `target_scaler_multi.json` — means/stds for multi-target regression
- `targets.json` — list of targets (multi-target only)
- `metrics.json` — train/val/test RMSE or accuracy vs naïve baseline
- `log.csv` — optional per-epoch logs (if `--save_csv_logs`)

---

## 9. `eval_tabular.py` — Evaluate Models on the Test Split

This script evaluates saved models on the **test** split for the same three task types:

- `regression`
- `regression_multi`
- `classification`

It expects the same `features_path` and `splits_json` as `train_tabular.py`, plus a **run directory** containing the trained artifacts (e.g. `model.keras`, `scaler.pkl`, `features.txt`, and any target scaler JSON).

### Inputs

- `--features_path`: feature store CSV/Parquet.
- `--splits_json`: from `make_splits.py`.
- `--task`: same choices as in training.
- `--target` / `--targets`: for regression / multi-regression.
- `--horizon`: **single** horizon (int).
- `--run_dir`: path to a specific training run (e.g. `artifacts/regression/NO2(GT)/h24`).
- `--out_dir`: where evaluation outputs will be written.
- `--export_csv`: if set, write prediction CSVs.
- `--plots`: if set, generate diagnostic plots.
- `--verbose`: if set, print extra debug and write a more detailed `run_log.txt`.

### Behaviour

1. **Sanity checks**
   - Confirms that `features_path`, `splits_json`, and `run_dir` exist.
   - Checks that `model.keras`, `scaler.pkl`, and `features.txt` exist in `run_dir`.
   - Loads a (optional) `target_scaler.json` and/or `target_scaler_multi.json` to de-normalise model outputs.

2. **Slice test range**
   - Uses `splits["test"][0:1]` to select rows from the full features DataFrame.
   - Rebuilds labels at the requested horizon using the same label functions as in training.

3. **Feature alignment & imputation**
   - Ensures all features listed in `features.txt` exist in the test DataFrame.
   - Applies train-time feature means (`feature_means.json` if present) to impute any NaNs in those features.
   - Transforms using the saved `StandardScaler` (`scaler.pkl`).

4. **Prediction and de-normalisation**
   - Runs `model.predict` for all test rows.
   - For regression and multi-regression, de-normalises outputs using the stored mean/std scalers.
   - Gathers `y_true` and `y_naive` (persistence).

5. **Metrics & outputs**

   - **Regression (single target)**
     - Computes test RMSE for model and naïve baseline.
     - Writes `metrics.json` with these values.
     - If `--export_csv`, writes `predictions_regression.csv` with:
       - `timestamp`, `y_true`, `y_pred`, `y_naive`, `residual`.
     - If `--plots`, creates:
       - `plots/resid_hist.png` — residual histogram.
       - `plots/true_vs_pred.png` — true vs predicted scatter.

   - **Regression multi**
     - De-normalises each target.
     - Computes per-target RMSE for model and naïve, plus macro (average) RMSE.
     - Writes `metrics.json` with:
       - `model_macro_rmse`, `naive_macro_rmse`, and a `per_target` block.
     - If `--export_csv`, writes a long-format `predictions_reg_multi.csv` with:
       - `timestamp`, `target`, `y_true`, `y_pred`, `y_naive`, `residual`.
     - If `--plots`, generates per-target residual histograms and true-vs-pred plots.

   - **Classification**
     - Computes:
       - Accuracy for model and naïve baseline.
       - Macro recall for model and naïve baseline.
     - Writes `metrics.json`.
     - If `--export_csv`, writes `predictions_classification.csv` with:
       - `timestamp`, `y_true`, `y_pred`, `y_naive`.
     - If `--plots`, writes a confusion matrix plot:
       - `plots/confusion.png` (labels “low/mid/high”).

6. **Run log**
   - Writes a text file `run_log.txt` in `--out_dir` recording paths, key decisions, warnings, and errors.

### Example usage

```bash
## Evaluate NO2 single-target regression model at h=24
python eval_tabular.py   --features_path data/features/feature_store.parquet   --splits_json data/splits/splits.json   --task regression   --target "NO2(GT)"   --horizon 24   --run_dir artifacts/regression/NO2(GT)/h24   --out_dir eval_out/no2_h24   --export_csv   --plots   --verbose
```

---

## 10. `eda.py` — Exploratory Data Analysis

This script performs basic EDA on either the cleaned dataset or the feature store and writes summary tables and plots under `reports/eda/`.

### Inputs

It tries paths in order:

1. `data/features/feature_store.parquet` (preferred)
2. `data/clean/air_quality_clean.csv` (fallback)

If neither exists, it raises an error.

### Steps

1. **Load and clean**
   - Read CSV/Parquet.
   - If `timestamp` exists:
     - Parse to datetime.
     - Drop invalid timestamps.
     - Sort by time and drop duplicates.

2. **Summary statistics & missingness**
   - Compute `.describe()` for all columns and write `summary_stats.csv`.
   - Compute per-column missing rate and write `missing_rates.csv`.
   - Write `summary.txt` with:
     - Row/column counts.
     - Time range (if timestamp present).
     - Top 10 columns by missing rate.

3. **Histograms for key targets**
   - For each of:
     - `CO(GT)`, `C6H6(GT)`, `NOx(GT)`, `NO2(GT)` (if present)
   - Plot a histogram with 50 bins and save as `hist_<target>.png`.

4. **Diurnal and weekly patterns**
   - If `timestamp` exists:
     - Create `hour` and `dow` (day-of-week) columns.
     - For each target:
       - Plot mean by hour: `hourly_<target>.png`.
       - Plot mean by day-of-week: `dow_<target>.png` (0=Monday).

5. **Correlation heatmap**
   - Select numeric columns only.
   - Compute correlation matrix and write `correlations.csv`.
   - Plot a heatmap (using seaborn) as `correlation_heatmap.png`.

6. **Rolling mean / variance example**
   - For the first available target in the target list:
     - Create a 24-hour rolling mean and std on a time-indexed series.
     - Write values to `<target>_rolling24.csv`.
     - Plot both as `<target>_rolling24.png`.

All outputs are written to `reports/eda/`.

### Example usage

```bash
python eda.py
```

---

## 11. Requirements & Setup

These scripts assume:

- Python 3.x
- Common data-science libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn` (for EDA heatmaps; optional)
- Modelling / ML libraries:
  - `scikit-learn`
  - `tensorflow` or `tensorflow-cpu` (for Keras models)
- Parquet support (optional but recommended):
  - `pyarrow` or `fastparquet`

A minimal setup might be:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow pyarrow
```

(Adjust to your environment and whether you use GPU or CPU builds of TensorFlow.)

---

## 12. Suggested Directory Layout

You can use any layout you like; one convenient convention is:

```text
project_root/
  data/
    raw/
      AirQualityUCI.csv
    clean/
      air_quality_clean.csv
      air_quality_clean.parquet
    features/
      feature_store.parquet
      feature_store.csv
      feature_store_clean.parquet      # optional: anomaly-cleaned features
    splits/
      splits.json
      split_summary.csv
      split_assignments.csv
      exported/
        feature_store_train.csv
        feature_store_val.csv
        feature_store_test.csv
  reports/
    eda/
      summary.txt
      summary_stats.csv
      missing_rates.csv
      hist_*.png
      hourly_*.png
      dow_*.png
      correlation_heatmap.png
  artifacts/
    regression/
      NO2(GT)/
        h24/
          model.keras
          scaler.pkl
          features.txt
          target_scaler.json
          feature_means.json
          metrics.json
          log.csv
    regression_multi/
      CO(GT)+C6H6(GT)+NOx(GT)+NO2(GT)/
        h24/
          ...
    classification/
      CO_bins/
        h24/
          ...
  preprocess_air_quality.py
  check.py
  build_features.py
  make_splits.py
  detect_anomalies.py
  make_clean.py
  train_tabular.py
  eval_tabular.py
  eda.py
  README.md
```

## REGRESSION MODELS 

## 1. Random Forest Regression for Air Quality Forecasting

This project implements a Random Forest regression pipeline to forecast pollutant concentrations using the UCI Air Quality dataset. Model performance is evaluated against a persistence baseline and visualised across multiple prediction horizons.

## Overview

The pipeline performs the following steps:

- Loads pre-split training, validation, and test datasets.
- Sorts timestamps to ensure proper temporal ordering.
- Generates horizon-specific target variables for CO, NMHC, C6H6, NOx, and NO2.
- Imputes missing values using mean imputation.
- Trains Random Forest regressors for each pollutant and horizon.
- Evaluates models on validation and test sets using RMSE against the naive persistence baseline.
- Stores trained models, imputers, and feature sets for each pollutant and horizon.

This module implements **Random Forest regressors** to forecast pollutant concentrations using the UCI Air Quality dataset. Models are evaluated across multiple horizons and compared against a persistence baseline.

---

## 1. Setup

The script expects pre-split train, validation, and test CSV files:

- `airq_train.csv`
- `airq_val.csv`
- `airq_test.csv`

Preprocessing includes:

- Converting and sorting timestamps for temporal consistency.
- Adding **time features**: `hour`, `weekday`, `month`.
- Generating **lag features** and **moving averages** for each pollutant and sensor variable.
- Imputing missing values using **mean imputation**.
- Creating horizon-specific target columns (`t+1h`, `t+6h`, `t+12h`, `t+24h`).

---

## 2. The Model

A separate **RandomForestRegressor** is trained for each pollutant and each forecast horizon. Random Forest is chosen because it:

- Handles nonlinear relationships between features and targets,
- Provides stable, low-variance predictions,
- Produces interpretable feature importance values,
- Performs reliably on small, tabular datasets like AirQualityUCI.

Each model is evaluated against the **persistence baseline** using RMSE on validation and test sets.

---

## 3. Hyperparameters

The model uses the following configuration:

```python
RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    random_state=0
)
```
---
## Feature Handling

- All features except timestamps and target columns are used for prediction.
- Missing values are handled with `SimpleImputer`.
- Targets are shifted according to forecast horizons (1, 6, 12, 24 hours).

## Evaluation and Visualisation

- Computes RMSE for validation and test sets.
- Compares Random Forest performance against the naive baseline.
- Generates line plots, heatmaps, and bar plots to visualise RMSE across pollutants and horizons.

## Output

- `rf_regression_results.csv` summarising RMSE metrics for each pollutant and horizon.
- Plots for validation vs test RMSE, baseline comparisons, and improvements over baseline.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn


## 2. Gradient Boosting Regression — Air Quality Forecasting

This module trains **Gradient Boosting Regressors** using both the **vanilla dataset** and the **anomaly-cleaned dataset**.

---

## 1. Setup

The script expects the following train/val/test splits:

- **Vanilla data:**
  - `airq_train.csv`
  - `airq_val.csv`
  - `airq_test.csv`

- **Cleaned data (after anomaly removal):**
  - `anomaly_free_train.csv`
  - `anomaly_free_val.csv`
  - `anomaly_free_test.csv`

Before modelling, both datasets pass through an identical preprocessing pipeline:

- Convert and sort timestamps.
- Add **time features**: `hour`, `weekday`, `month`.
- Create **lag features** for each pollutant (1-hour lag).
- Interpolate missing values in time.
- Apply **median imputation**.
- Generate horizon-specific targets: `t+1h`, `t+6h`, `t+12h`, `t+24h`.

This ensures a fair comparison between vanilla and cleaned data.

---

## 2. The Model

The script trains a separate **GradientBoostingRegressor** for each pollutant and each forecasting horizon.

Gradient Boosting is chosen because it:

- Captures nonlinear relationships,
- Works well on small tabular datasets like AirQualityUCI,
- Is stable with noisy sensor readings,
- Provides interpretable feature importances.

Each model is evaluated against:

- **Persistence baseline** (predict current value),
- **24-hour seasonal baseline**,
- **Validation RMSE**,
- **Test RMSE**.

Results for both datasets (vanilla + cleaned) are stored and compared directly.

---

## 3. Hyperparameters

The model uses a moderately regularised configuration:

```python
GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.9,
    min_samples_split=4,
    random_state=42
)
```
---

## CLASSIFICATION MODELS

## 1. Feature Engineering

Feature engineering is performed **separately** on the three provided datasets (`airq_train.csv`, `airq_val.csv`, `airq_test.csv`) to avoid data leakage.

The script generates:
- `airq_train_fe.csv`
- `airq_val_fe.csv`
- `airq_test_fe.csv`

The following temporal features are added:

- **Calendar features**
  - Hour of day
  - Day of week

- **Lag features**  
  For each pollutant / sensor variable:  
  `lag = 1, 3, 6, 12, 24 hours`

- **Moving averages**  
  Rolling windows:  
  `MA = 3, 6, 12, 24 hours`

These feature-engineered datasets are used by all “advanced” model versions.

---

## 2. Logistic Regression

Two notebooks are provided:

- `logistic.ipynb` — uses original preprocessed datasets  
- `adv_logistic_regression.ipynb` — uses feature-engineered datasets  

Pipeline summary:

1. Convert CO(GT) into 3 classes (low / mid / high).  
2. Create forecast targets using time-shifting for horizons 1/6/12/24.  
3. Remove rows with missing target.  
4. Impute missing values (mean).  
5. (Advanced version) Apply `StandardScaler`.  
6. Train multinomial Logistic Regression (`lbfgs`, `max_iter=2000`).  
7. Evaluate:
   - Accuracy
   - Macro-F1
   - Confusion Matrix
   - Persistence baseline (predict CO_class(t) → CO_class(t+H))


---

## 3. Random Forest

Two notebooks are provided:

- `random_forest.ipynb`  
- `adv_random_forest.ipynb`  

Pipeline summary:

1. Same target construction and masking as Logistic Regression.  
2. Mean imputation for features.  
3. Train Random Forest classifier  
   (`n_estimators ≈ 300–400`, `n_jobs=-1`).  
4. Evaluate using:
   - Accuracy
   - Macro-F1
   - Confusion Matrix
   - Persistence baseline

Feature engineering improves performance substantially at **6h and 12h horizons**.

## 4. Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn

## 2. Classification – Decision Tree (CO(GT))

This folder contains three Jupyter notebooks for the CO(GT) classification task:

1. **Classification_decision_tree_lag_ma_final.ipynb** (main notebook)  
2. **Classification_decision_tree_tuned.ipynb**  
3. **Classification_decision_tree_lag_ma_interactions_tuned.ipynb**

---

## Requirements

**Python 3.x**

Packages:
- pandas  
- numpy  
- scikit-learn  
- matplotlib  

Install with:

```
pip install pandas numpy scikit-learn matplotlib
```

---

## Data

Place the following CSV files in `data/splits`:

- `airq_train.csv`  
- `airq_val.csv`  
- `airq_test.csv`

No extra preprocessing is needed.  
The notebooks handle timestamp parsing, feature construction (lags, moving averages), and discretisation.

---

## Notebook Overview

### 1. **Classification_decision_tree_lag_ma_final.ipynb** (main)

- Uses temporal features (hour, weekday, month)  
- CO lag features: t−1, t−6, t−12, t−24  
- Moving averages: 3h, 6h, 12h, 24h  

A Decision Tree with:

- `max_depth = 7`
- `min_samples_leaf = 30`

is trained for each horizon (1h, 6h, 12h, 24h).

Outputs include:

- Validation and test accuracy  
- Baseline comparison  
- Confusion matrices  
- Tree visualisations  
- Accuracy-vs-horizon plots  

**How to run:**

1. Place the three CSV files in the same directory  
2. Open the notebook in Jupyter  
3. Run all cells from top to bottom  

---

### 2. **Classification_decision_tree_tuned.ipynb**

- Uses the same feature set  
- Performs grid search over `max_depth` and `min_samples_leaf`  
- Shows tuned models do not generalise better due to temporal drift between years  

**How to run:**  
Open the notebook and run all cells.

---

### 3. **Classification_decision_tree_lag_ma_interactions_tuned.ipynb**

Adds interaction features such as:

- CO × NOx  
- CO × NO2  
- CO × RH  
- NOx × hour  

Runs the same tuning procedure as the tuned notebook.  
Demonstrates that interaction features reduce generalisation performance.

**How to run:**  
Open the notebook and execute all cells.

---

## Reproducing Results

To reproduce results in the report:

1. Open `Classification_decision_tree_lag_ma_final.ipynb`  
2. Run all cells  
3. Use the printed accuracy values, baseline comparisons, confusion matrices,  
   tree diagrams, and accuracy-vs-horizon plots.

The other two notebooks provide tuning and interaction experiments included for comparison.

All notebooks use **random_state = 42** for reproducibility.

## Classification – Decision Tree (CO(GT))

This folder contains three Jupyter notebooks for the CO(GT) classification task:

1. **Classification_decision_tree_lag_ma_final.ipynb** (main notebook)  
2. **Classification_decision_tree_tuned.ipynb**  
3. **Classification_decision_tree_lag_ma_interactions_tuned.ipynb**

---

## Requirements

**Python 3.x**

Packages:
- pandas  
- numpy  
- scikit-learn  
- matplotlib  

Install with:

```
pip install pandas numpy scikit-learn matplotlib
```

---

## Data

Place the following CSV files in `data/splits`:

- `airq_train.csv`  
- `airq_val.csv`  
- `airq_test.csv`

No extra preprocessing is needed.  
The notebooks handle timestamp parsing, feature construction (lags, moving averages), and discretisation.

---

## Notebook Overview

### 1. **Classification_decision_tree_lag_ma_final.ipynb** (main)

- Uses temporal features (hour, weekday, month)  
- CO lag features: t−1, t−6, t−12, t−24  
- Moving averages: 3h, 6h, 12h, 24h  

A Decision Tree with:

- `max_depth = 7`
- `min_samples_leaf = 30`

is trained for each horizon (1h, 6h, 12h, 24h).

Outputs include:

- Validation and test accuracy  
- Baseline comparison  
- Confusion matrices  
- Tree visualisations  
- Accuracy-vs-horizon plots  

**How to run:**

1. Place the three CSV files in the same directory  
2. Open the notebook in Jupyter  
3. Run all cells from top to bottom  

---

### 2. **Classification_decision_tree_tuned.ipynb**

- Uses the same feature set  
- Performs grid search over `max_depth` and `min_samples_leaf`  
- Shows tuned models do not generalise better due to temporal drift between years  

**How to run:**  
Open the notebook and run all cells.

---

### 3. **Classification_decision_tree_lag_ma_interactions_tuned.ipynb**

Adds interaction features such as:

- CO × NOx  
- CO × NO2  
- CO × RH  
- NOx × hour  

Runs the same tuning procedure as the tuned notebook.  
Demonstrates that interaction features reduce generalisation performance.

**How to run:**  
Open the notebook and execute all cells.

---

## Reproducing Results

To reproduce results in the report:

1. Open `Classification_decision_tree_lag_ma_final.ipynb`  
2. Run all cells  
3. Use the printed accuracy values, baseline comparisons, confusion matrices,  
   tree diagrams, and accuracy-vs-horizon plots.

The other two notebooks provide tuning and interaction experiments included for comparison.

All notebooks use **random_state = 42** for reproducibility.


## Gradient Boosting Regression — Air Quality Forecasting

This module trains **Gradient Boosting Regressors** using both the **vanilla dataset** and the **anomaly-cleaned dataset**.

---

## 1. Setup

The script expects the following train/val/test splits:

- **Vanilla data:**
  - `airq_train.csv`
  - `airq_val.csv`
  - `airq_test.csv`

- **Cleaned data (after anomaly removal):**
  - `anomaly_free_train.csv`
  - `anomaly_free_val.csv`
  - `anomaly_free_test.csv`

Before modelling, both datasets pass through an identical preprocessing pipeline:

- Convert and sort timestamps.
- Add **time features**: `hour`, `weekday`, `month`.
- Create **lag features** for each pollutant (1-hour lag).
- Interpolate missing values in time.
- Apply **median imputation**.
- Generate horizon-specific targets: `t+1h`, `t+6h`, `t+12h`, `t+24h`.

This ensures a fair comparison between vanilla and cleaned data.

---

## 2. The Model

The script trains a separate **GradientBoostingRegressor** for each pollutant and each forecasting horizon.

Gradient Boosting is chosen because it:

- Captures nonlinear relationships,
- Works well on small tabular datasets like AirQualityUCI,
- Is stable with noisy sensor readings,
- Provides interpretable feature importances.

Each model is evaluated against:

- **Persistence baseline** (predict current value),
- **24-hour seasonal baseline**,
- **Validation RMSE**,
- **Test RMSE**.

Results for both datasets (vanilla + cleaned) are stored and compared directly.

---

## 3. Hyperparameters

The model uses a moderately regularised configuration:

```python
GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.9,
    min_samples_split=4,
    random_state=42
)
```
