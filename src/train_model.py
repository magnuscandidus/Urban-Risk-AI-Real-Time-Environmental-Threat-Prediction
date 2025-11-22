import os
import json
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# -------------------------
# Config
# -------------------------
INPUT_CSV = "data_processed/final_dataset.csv"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
RANDOM_STATE = 42
N_SPLITS = 5  # TimeSeriesSplit folds
N_ITER_SEARCH = 8  # keep small for speed
TEST_BLOCK_SIZE = None  # leave None to let TimeSeriesSplit decide
# -------------------------


def ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def load_dataset(path=INPUT_CSV):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed dataset not found at: {path}")
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def create_lag_features(df, lags=(1, 2, 3), roll_windows=(3, 6)):
    """
    Create lag and rolling features that use only past values.
    Returns a new DataFrame with new features (no target created here).
    """
    d = df.copy()
    # ensure expected columns exist
    for c in ["pm25", "temperature", "humidity", "precipitation", "traffic_speed"]:
        if c not in d.columns:
            d[c] = np.nan

    # time features
    d["hour"] = d["time"].dt.hour
    d["dayofweek"] = d["time"].dt.dayofweek
    d["is_weekend"] = d["dayofweek"].isin([5, 6]).astype(int)

    # create lags for numeric sensors
    lag_cols = ["pm25", "temperature", "humidity", "precipitation", "traffic_speed"]
    for lag in lags:
        for col in lag_cols:
            d[f"{col}_lag{lag}"] = d[col].shift(lag)

    # rolling means (use only past values via shift(1) then rolling)
    for w in roll_windows:
        d[f"pm25_roll{w}_mean"] = d["pm25"].shift(1).rolling(window=w, min_periods=1).mean()

    return d


def prepare_training_data(df):
    """
    Build X, y for next-hour prediction (pm25 at t+1).
    All features at time t can use data <= t only (ensured by create_lag_features).
    """
    d = create_lag_features(df)

    # target: next hour pm25
    d["pm25_next1"] = d["pm25"].shift(-1)

    # drop rows where target is NaN (last row)
    d = d.dropna(subset=["pm25_next1"]).reset_index(drop=True)

    # choose candidate features (only those that exist)
    candidate_features = [
        "pm25", "pm25_lag1", "pm25_lag2", "pm25_lag3",
        "pm25_roll3_mean", "pm25_roll6_mean",
        "temperature", "temperature_lag1", "temperature_lag2",
        "humidity", "humidity_lag1", "humidity_lag2",
        "precipitation", "precipitation_lag1",
        "traffic_speed", "traffic_speed_lag1",
        "free_flow_speed", "congestion_index",
        "hour", "dayofweek", "is_weekend",
    ]
    features = [c for c in candidate_features if c in d.columns]

    # conservative: drop rows with NaN in core lag features (to keep data quality)
    required = ["pm25", "pm25_lag1", "hour"]
    required = [r for r in required if r in d.columns]
    d = d.dropna(subset=required).reset_index(drop=True)

    X = d[features].copy()
    y = d["pm25_next1"].copy()
    times = d["time"].copy()

    # impute any remaining small gaps by forward/backfill
    X = X.fillna(method="ffill").fillna(method="bfill").fillna(0)

    return X, y, features, times


def normalized_r2(y_true, y_pred):
    """
    Normalized R2 (NR2) robust for low-variance targets:
      NR2 = 1 - RMSE / std(y_true)
    This yields a meaningful score if std(y_true) > 0.
    """
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    std = np.std(y_true)
    if std == 0:
        return float("nan")
    return 1.0 - (rmse / std)


def train_with_time_series_cv(X, y, n_splits=N_SPLITS):
    """
    Perform TimeSeriesSplit cross-validation. For each fold:
      - perform a RandomizedSearchCV (small iter) to tune hyperparams on the training part
      - evaluate on the fold's test set
    Returns: best_estimator (trained on entire data after cv), per-fold metrics DataFrame
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []
    fold = 0

    # param grid for RF inside pipeline
    param_dist = {
        "rf__n_estimators": [100, 200, 400],
        "rf__max_depth": [6, 10, 20, None],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4],
    }

    # pipeline
    base_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))
    ])

    # We'll keep the best pipeline found across folds
    best_models = []
    fold_preds = []

    for train_idx, test_idx in tscv.split(X):
        fold += 1
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Randomized search
        search = RandomizedSearchCV(
            base_pipeline,
            param_distributions=param_dist,
            n_iter=N_ITER_SEARCH,
            scoring="neg_mean_absolute_error",
            cv=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )
        print(f"[INFO] Fold {fold}: search on train len={len(train_idx)} test len={len(test_idx)} ...")
        search.fit(X_train, y_train)

        best = search.best_estimator_
        best_models.append(best)

        # evaluate
        y_pred = best.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        nr2 = normalized_r2(y_test, y_pred)

        fold_metrics.append({
            "fold": fold,
            "train_start": X.index[train_idx[0]],
            "train_end": X.index[train_idx[-1]],
            "test_start": X.index[test_idx[0]],
            "test_end": X.index[test_idx[-1]],
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "nr2": nr2,
            "best_params": search.best_params_,
        })

        # store predictions for analysis
        fold_preds.append((test_idx, y_test.values, y_pred))

    # Choose the single best model by average CV score (lowest mae across folds)
    fold_df = pd.DataFrame(fold_metrics)
    fold_df_sorted = fold_df.sort_values("mae")
    best_fold_idx = fold_df_sorted.index[0]
    # index to best_models uses order appended (0..)
    best_model = best_models[best_fold_idx]

    # Refit best model on full dataset for final use
    print("[INFO] Re-fitting best pipeline on entire dataset...")
    best_model.fit(X, y)

    return best_model, pd.DataFrame(fold_metrics), fold_preds


def plot_feature_importance(model, feature_names, out_path):
    # retrieve RF from pipeline
    rf = None
    if hasattr(model, "named_steps") and "rf" in model.named_steps:
        rf = model.named_steps["rf"]
    elif hasattr(model, "feature_importances_"):
        rf = model
    else:
        print("[WARN] Could not find RandomForest to plot importances.")
        return

    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1][:30]
    names = [feature_names[i] for i in idx]
    values = importances[idx]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(values))[::-1], values, align="center")
    plt.yticks(range(len(values))[::-1], names)
    plt.xlabel("Feature importance")
    plt.title("RandomForest Feature Importances")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved feature importance plot: {out_path}")


def save_artifacts(model, feature_cols, cv_df):
    ensure_dirs()
    model_path = os.path.join(MODELS_DIR, "rf_pm25_model.joblib")
    joblib.dump(model, model_path)

    meta = {"features": feature_cols}
    with open(os.path.join(MODELS_DIR, "feature_list.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    cv_df.to_csv(os.path.join(OUTPUTS_DIR, "cv_metrics.csv"), index=False)
    print(f"[INFO] Saved model to: {model_path}")
    print(f"[INFO] Saved CV metrics to: {os.path.join(OUTPUTS_DIR, 'cv_metrics.csv')}")


def save_sample_predictions(fold_preds, times, X, out_path):
    """
    Save a small sample of true vs pred across folds for inspection.
    """
    rows = []
    for test_idx, y_true, y_pred in fold_preds:
        for i, j in enumerate(test_idx):
            rows.append({
                "index": int(j),
                "time": times.iloc[j] if times is not None else np.nan,
                "y_true": float(y_true[i]),
                "y_pred": float(y_pred[i])
            })
    if not rows:
        return
    dfp = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)
    dfp.to_csv(out_path, index=False)
    print(f"[INFO] Saved fold predictions sample to: {out_path}")


def main():
    ensure_dirs()
    df = load_dataset(INPUT_CSV)
    X, y, features, times = prepare_training_data(df)
    print(f"[INFO] Dataset after feature engineering: X={X.shape}, y={y.shape}")

    # CV training
    model, cv_df, fold_preds = train_with_time_series_cv(X, y, n_splits=N_SPLITS)

    # aggregate metrics
    agg = {
        "mae_mean": cv_df["mae"].mean(),
        "mae_std": cv_df["mae"].std(),
        "rmse_mean": cv_df["rmse"].mean(),
        "rmse_std": cv_df["rmse"].std(),
        "r2_mean": cv_df["r2"].mean(),
        "nr2_mean": cv_df["nr2"].mean()
    }
    print("[RESULTS] Cross-validated metrics (mean ± std):")
    print(f"  MAE:  {agg['mae_mean']:.4f} ± {agg['mae_std']:.4f}")
    print(f"  RMSE: {agg['rmse_mean']:.4f} ± {agg['rmse_std']:.4f}")
    print(f"  R2:   {agg['r2_mean']:.4f}")
    print(f"  NR2:  {agg['nr2_mean']:.4f}")

    # Save artifacts
    save_artifacts(model, features, cv_df)

    # feature importance
    plot_feature_importance(model, features, os.path.join(OUTPUTS_DIR, "feature_importance.png"))

    # sample preds
    save_sample_predictions(fold_preds, times, X, os.path.join(OUTPUTS_DIR, "predictions_sample.csv"))


if __name__ == "__main__":
    main()
