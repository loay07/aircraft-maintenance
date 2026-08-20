"""Loads the trained XGBoost RUL model and runs predictions."""
from pathlib import Path

import pandas as pd
import xgboost as xgb

from src.features import FEATURE_COLUMNS

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "xgboost_capped.json"

_model = None


def get_model() -> xgb.XGBRegressor:
    global _model
    if _model is None:
        model = xgb.XGBRegressor()
        model.load_model(str(MODEL_PATH))
        _model = model
    return _model


def predict_rul(features_df: pd.DataFrame):
    model = get_model()
    return model.predict(features_df[FEATURE_COLUMNS])
