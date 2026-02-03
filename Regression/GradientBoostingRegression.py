import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

van_train_path = "~/Downloads/9417Groupproj_preprocess-main/data/splits/airq_train.csv"
van_val_path   = "~/Downloads/9417Groupproj_preprocess-main/data/splits/airq_val.csv"
van_test_path  = "~/Downloads/9417Groupproj_preprocess-main/data/splits/airq_test.csv"

clean_train_path = "~/Downloads/9417Groupproj_preprocess-main/CleanData/anomaly_free_train.csv"
clean_val_path   = "~/Downloads/9417Groupproj_preprocess-main/CleanData/anomaly_free_val.csv"
clean_test_path  = "~/Downloads/9417Groupproj_preprocess-main/CleanData/anomaly_free_test.csv"

targets = ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
horizons = [1, 6, 12, 24]

sensor_features = [
    "PT08.S1(CO)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
]

time_features = ["hour", "weekday", "month"]

results_dict = {}
#*******************************************************************
def my_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def baseline_persist(df, tgt, h):
    col_true = f"{tgt}_tplus_{h}"
    df2 = df.dropna(subset=[col_true, tgt])
    if len(df2) == 0:
        return np.nan
    y_true = df2[col_true]
    y_pred = df2[tgt]
    return my_rmse(y_true, y_pred)

def baseline_season(df, tgt, h):
    col_true = f"{tgt}_tplus_{h}"
    tmp = df.copy()
    tmp["season_pred"] = tmp[tgt].shift(24)
    tmp2 = tmp.dropna(subset=[col_true, "season_pred"])

    if len(tmp2) == 0:
        return np.nan
    return my_rmse(tmp2[col_true], tmp2["season_pred"])
#*******************************************************************
def preprocess_data(train_path, val_path, test_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    for df in [train_df, val_df, test_df]:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        if "hour" not in df.columns:
            df["hour"] = df["timestamp"].dt.hour
        if "weekday" not in df.columns:

            df["weekday"] = df["timestamp"].dt.weekday

        if "month" not in df.columns:
            df["month"] = df["timestamp"].dt.month

    lag_features = []
    for t in targets:
        lag_col = t + "_lag1"
        lag_features.append(lag_col)
        train_df[lag_col] = train_df[t].shift(1)
        val_df[lag_col] = val_df[t].shift(1)
        test_df[lag_col] = test_df[t].shift(1)

    all_features = sensor_features + time_features + lag_features

    for df in [train_df, val_df, test_df]:
        df.set_index("timestamp", inplace=True)
        for col in all_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].interpolate(method="time").ffill().bfill()
        df.reset_index(inplace=True)

    imputer = SimpleImputer(strategy="median")
    imputer.fit(train_df[all_features])

    train_df[all_features] = imputer.transform(train_df[all_features])
    val_df[all_features] = imputer.transform(val_df[all_features])
    test_df[all_features] = imputer.transform(test_df[all_features])

    for t in targets:
        for h in horizons:
            target_col = f"{t}_tplus_{h}"
            if target_col not in train_df.columns:
                train_df[target_col] = train_df[t].shift(-h)
            if target_col not in val_df.columns:
                val_df[target_col] = val_df[t].shift(-h)
            if target_col not in test_df.columns:
                test_df[target_col] = test_df[t].shift(-h)

    return train_df, val_df, test_df, all_features
#*******************************************************************
def train_and_eval(train_df, val_df, test_df, all_features, dataset_tag):
    results = []

    for tgt in targets:
        for h in horizons:
            y_col = f"{tgt}_tplus_{h}"

            train_subset = train_df.dropna(subset=[y_col])
            val_subset = val_df.dropna(subset=[y_col])
            test_subset = test_df.dropna(subset=[y_col])

            if len(train_subset) == 0 or len(val_subset) == 0 or len(test_subset) == 0:
                continue

            X_train = train_subset[all_features]
            y_train = train_subset[y_col]

            X_val = val_subset[all_features]
            y_val = val_subset[y_col]

            X_test = test_subset[all_features]
            y_test = test_subset[y_col]

            model = gbr = GradientBoostingRegressor(
                            random_state=42,
                            n_estimators=300,
                            learning_rate=0.05,
                            max_depth=3,
                            subsample=0.9,
                            min_samples_split=4)

            model.fit(X_train, y_train)

            val_preds = model.predict(X_val)
            test_preds = model.predict(X_test)

            val_rmse = my_rmse(y_val, val_preds)
            test_rmse = my_rmse(y_test, test_preds)

            persist_rmse = baseline_persist(test_df, tgt, h)
            seasonal_rmse = baseline_season(test_df, tgt, h)

            results.append([
                dataset_tag,
                tgt,
                h,
                persist_rmse,
                seasonal_rmse,
                val_rmse,
                test_rmse,
            ])
            
            results_dict[(dataset_tag, tgt, h)] = {
                "model": model,
                "features": all_features,
                "y_test": y_test,
                "preds": test_preds
            }

    results_df = pd.DataFrame(results,
                              columns=["dataset", "target", "horizon",
                                       "rmse_persist", "rmse_season",
                                       "rmse_val", "rmse_test"])
    return results_df
#*******************************************************************
if __name__ == "__main__":
    van_train_df, van_val_df, van_test_df, van_features = preprocess_data(van_train_path, van_val_path, van_test_path)
    vanilla_results = train_and_eval(van_train_df, van_val_df, van_test_df, van_features, "vanilla")

    clean_train_df, clean_val_df, clean_test_df, clean_features = preprocess_data(clean_train_path, clean_val_path, clean_test_path)
    cleaned_results = train_and_eval(clean_train_df, clean_val_df, clean_test_df, clean_features, "clean")

    print("\nVanilla dataset results:")
    print(vanilla_results)

    print("\nCleaned dataset results:")
    print(cleaned_results)

    sample_res = pd.concat([vanilla_results, cleaned_results], ignore_index=True)
    sample_res = pd.concat([vanilla_results, cleaned_results], ignore_index=True)

    ana_targets = {
        "CO(GT)": {"label": "CO", "color": "blue"},
        "C6H6(GT)": {"label": "C6H6", "color": "green"}
    }
#*******************************************************************
    for target, config in ana_targets.items():
        m = results_dict[("clean", target, 1)]
        
        model = m["model"]
        y_test = m["y_test"]
        test_preds = m["preds"]
        features = m["features"]
        
        importances = pd.Series(model.feature_importances_, 
                              index=features).sort_values(ascending=False)
        
        fig = plt.figure(figsize=(7, 4))
        importances.head(10).plot(kind="bar")
        plt.title(f"top 10 Features - {target} +1h")
        plt.tight_layout()
        plt.show()
        
        fig = plt.figure(figsize=(5, 4))
        plt.scatter(y_test, test_preds, alpha=0.4)
        min_val = min(y_test.min(), test_preds.min())
        max_val = max(y_test.max(), test_preds.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--")
        plt.xlabel(f"actual {config['label']}")
        plt.ylabel(f"predicted {config['label']}")
        plt.title(f"predicted vs actual - {target} +1h")
        plt.show()
        
        vanilla_data = sample_res[(sample_res["dataset"] == "vanilla") & 
                                        (sample_res["target"] == target)]
        cleaned_data = sample_res[(sample_res["dataset"] == "clean") & 
                                        (sample_res["target"] == target)]
        
        fig = plt.figure(figsize=(6, 4))
        plt.plot(vanilla_data["horizon"], vanilla_data["rmse_test"], 
                 marker="o", label="Vanilla")
        plt.plot(cleaned_data["horizon"], cleaned_data["rmse_test"], 
                 marker="o", label="Cleaned")
        plt.xlabel("forecast horizon (hours)")
        plt.ylabel("test RMSE")
        plt.title(f"{target} test: vanilla vs cleaned data")
        plt.legend()
        plt.grid(True)
        plt.show()
