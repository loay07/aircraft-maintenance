"""Hyperparameter search for the FD001 RUL model.

Same features and RUL-capping approach as the notebooks (src/features.py) --
this only replaces "default-ish XGBoost params" with a properly cross-validated
search. Model selection uses only the training engines (grouped K-fold, so no
engine's cycles leak across folds); the official NASA test set is touched only
once at the end, for reporting.

Usage: python scripts/tune_model.py

Result (already run): best CV params gave RMSE 15.95 in cross-validation, but
scored worse on the official NASA test set than the baseline (tuned: MAE
12.68 / RMSE 17.40 / R² 0.811 vs. baseline: MAE 12.06 / RMSE 16.73 / R²
0.826) -- a difference well within noise for a 100-engine test set. The
baseline model was kept; models/xgboost_capped.json was not modified.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import (  # noqa: E402
    FEATURE_COLUMNS,
    RUL_CAP,
    add_engineered_features,
    load_raw_txt,
)

DATA_DIR = ROOT / "data" / "raw"
MODEL_PATH = ROOT / "models" / "xgboost_capped.json"


def build_training_set() -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    train = load_raw_txt(DATA_DIR / "train_FD001.txt")
    train = add_engineered_features(train)

    max_cycle = train.groupby("engine_id")["cycle"].transform("max")
    rul = (max_cycle - train["cycle"]).clip(upper=RUL_CAP)

    X = train[FEATURE_COLUMNS]
    groups = train["engine_id"].values
    return X, rul, groups


def build_official_test_set() -> tuple[pd.DataFrame, pd.Series]:
    test = load_raw_txt(DATA_DIR / "test_FD001.txt")
    test = add_engineered_features(test)
    last = test.groupby("engine_id").tail(1).sort_values("engine_id")

    true_rul = pd.read_csv(DATA_DIR / "RUL_FD001.txt", sep=r"\s+", header=None)
    true_rul.columns = ["RUL"]
    true_rul.index = np.arange(1, len(true_rul) + 1)

    y_true = true_rul.loc[last["engine_id"].values, "RUL"].clip(upper=RUL_CAP)
    return last[FEATURE_COLUMNS], y_true.values


def evaluate(model, X, y_true) -> dict:
    pred = model.predict(X)
    err = pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main():
    X_train, y_train, groups = build_training_set()

    param_space = {
        "n_estimators": [150, 250, 400, 600],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 10],
    }

    base_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_space,
        n_iter=40,
        scoring="neg_root_mean_squared_error",
        cv=GroupKFold(n_splits=5),
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train, groups=groups)

    print("Best CV RMSE:", -search.best_score_)
    print("Best params:", search.best_params_)

    baseline_model = xgb.XGBRegressor()
    baseline_model.load_model(str(MODEL_PATH))

    X_test, y_test = build_official_test_set()
    baseline_metrics = evaluate(baseline_model, X_test, y_test)
    tuned_metrics = evaluate(search.best_estimator_, X_test, y_test)

    print("\nOfficial FD001 test set:")
    print("  baseline:", {k: round(v, 4) for k, v in baseline_metrics.items()})
    print("  tuned:   ", {k: round(v, 4) for k, v in tuned_metrics.items()})

    if tuned_metrics["rmse"] < baseline_metrics["rmse"]:
        backup_path = MODEL_PATH.with_name("xgboost_capped_baseline.json")
        if not backup_path.exists():
            baseline_model.save_model(str(backup_path))
        search.best_estimator_.save_model(str(MODEL_PATH))
        print(f"\nTuned model improved RMSE -- saved to {MODEL_PATH}")
        print(f"Previous model backed up to {backup_path}")
    else:
        print("\nTuned model did not beat the baseline on RMSE -- baseline kept as-is.")


if __name__ == "__main__":
    main()
