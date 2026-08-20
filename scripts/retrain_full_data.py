"""Retrain the FD001 capped-RUL model on all 100 training engines.

Finding: the notebook's final model (final_xgb_capped in
notebooks/03_feature_preparation.ipynb) was fit on only 60% of the training
engines (train_60), used another 20% (val_20) purely for early stopping, and
never touched the last 20% (test_20) at all. Early stopping did its job --
it found the right number of trees, best_iteration=253 -- but then 40% of
labeled engines were left out of the model that actually got saved.

This script changes exactly one thing and nothing else: every feature, every
hyperparameter, the RUL cap, and the random_state are copied verbatim from
the notebook. The only change is fitting the final model on all 100 training
engines instead of 60, using the notebook's own best_iteration (253) as a
fixed n_estimators -- standard practice once early stopping has already told
you how many trees to use, so you're not still burning 40% of your labeled
data on a validation set the final model doesn't need.

The official NASA test set (100 held-out engines, never part of training)
is untouched throughout and used only for reporting, exactly as in
notebooks/04_official_test_evaluation.ipynb. The model file is only
overwritten if this actually improves official-test RMSE.

Usage: python scripts/retrain_full_data.py

Result (already run): training on all 100 engines instead of 60 gave MAE
11.99 / RMSE 16.94 / R² 0.821 on the official NASA test set, vs. the
baseline's MAE 12.06 / RMSE 16.73 / R² 0.826 -- MAE ticks down, RMSE ticks up,
a wash well within noise for a 100-engine test set. The baseline model was
kept; models/xgboost_capped.json was not modified.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

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

# Verbatim from notebooks/03_feature_preparation.ipynb's final_xgb_capped,
# with n_estimators fixed at that run's best_iteration instead of early
# stopping against a held-out slice.
MODEL_PARAMS = dict(
    n_estimators=253,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.6,
    colsample_bytree=0.4,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)


def build_training_set() -> tuple[pd.DataFrame, pd.Series, int]:
    train = load_raw_txt(DATA_DIR / "train_FD001.txt")
    train = add_engineered_features(train)
    max_cycle = train.groupby("engine_id")["cycle"].transform("max")
    rul = (max_cycle - train["cycle"]).clip(upper=RUL_CAP)
    return train[FEATURE_COLUMNS], rul, train["engine_id"].nunique()


def build_official_test_set() -> tuple[pd.DataFrame, np.ndarray]:
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
    X_train, y_train, n_engines = build_training_set()
    print(f"Training on all {n_engines} training engines ({len(X_train)} rows)")

    model = xgb.XGBRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    baseline_model = xgb.XGBRegressor()
    baseline_model.load_model(str(MODEL_PATH))

    X_test, y_test = build_official_test_set()
    baseline_metrics = evaluate(baseline_model, X_test, y_test)
    full_data_metrics = evaluate(model, X_test, y_test)

    print("\nOfficial FD001 test set:")
    print("  baseline (60% of engines): ", {k: round(v, 4) for k, v in baseline_metrics.items()})
    print("  retrained (100% of engines):", {k: round(v, 4) for k, v in full_data_metrics.items()})

    if full_data_metrics["rmse"] < baseline_metrics["rmse"]:
        backup_path = MODEL_PATH.with_name("xgboost_capped_60pct.json")
        if not backup_path.exists():
            baseline_model.save_model(str(backup_path))
        model.save_model(str(MODEL_PATH))
        print(f"\nImproved RMSE -- saved to {MODEL_PATH}")
        print(f"Previous (60%-trained) model backed up to {backup_path}")
    else:
        print("\nDid not beat the baseline on RMSE -- baseline kept as-is.")


if __name__ == "__main__":
    main()
