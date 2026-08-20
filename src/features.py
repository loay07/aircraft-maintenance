"""Feature engineering for CMAPSS FD001 RUL prediction.

Mirrors the pipeline built in notebooks/04_official_test_evaluation.ipynb so
that predictions served by the API match the model's official evaluation.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS = (
    ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

SELECTED_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_17", "sensor_20",
]

STD_SENSORS = ["sensor_4", "sensor_11", "sensor_2", "sensor_15", "sensor_9"]

RUL_CAP = 125

FEATURE_COLUMNS = (
    ["cycle"]
    + SELECTED_SENSORS
    + [f"{s}_roll5" for s in SELECTED_SENSORS]
    + [f"{s}_roll10" for s in SELECTED_SENSORS]
    + [f"{s}_slope10" for s in SELECTED_SENSORS]
    + [f"{s}_std10" for s in STD_SENSORS]
)


def load_raw_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.columns = RAW_COLUMNS
    return df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)


def _slope(values: np.ndarray) -> float:
    x = np.arange(len(values))
    return np.polyfit(x, values, 1)[0]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean / slope / std features per engine.

    Must run on each engine's full time series (not a single row) so the
    rolling windows are correct; safe to call across many engines at once as
    long as the frame is sorted by engine_id, cycle.
    """
    df = df.sort_values(["engine_id", "cycle"]).copy()

    for sensor in SELECTED_SENSORS:
        df[f"{sensor}_roll5"] = (
            df.groupby("engine_id")[sensor]
            .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        )

    for sensor in SELECTED_SENSORS:
        df[f"{sensor}_roll10"] = (
            df.groupby("engine_id")[sensor]
            .transform(lambda x: x.rolling(window=10, min_periods=1).mean())
        )

    for sensor in SELECTED_SENSORS:
        df[f"{sensor}_slope10"] = (
            df.groupby("engine_id")[sensor]
            .transform(
                lambda x: x.rolling(window=10, min_periods=2).apply(_slope, raw=True)
            )
            .fillna(0)
        )

    for sensor in STD_SENSORS:
        df[f"{sensor}_std10"] = (
            df.groupby("engine_id")[sensor]
            .transform(lambda x: x.rolling(window=10, min_periods=2).std())
            .fillna(0)
        )

    return df
