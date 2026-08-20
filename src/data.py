"""Loads NASA's official FD001 test set, engineers features, and precomputes
predictions + evaluation metrics once at process startup so API requests are
just dictionary/array lookups.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import RUL_CAP, add_engineered_features, load_raw_txt
from src.model import predict_rul

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Sensors shown in the demo's degradation trend chart (clear monotonic drift).
DISPLAY_SENSORS = ["sensor_11", "sensor_4", "sensor_9"]


class Store:
    def __init__(self):
        test = load_raw_txt(DATA_DIR / "test_FD001.txt")

        true_rul = pd.read_csv(DATA_DIR / "RUL_FD001.txt", sep=r"\s+", header=None)
        true_rul.columns = ["RUL"]
        true_rul.index = np.arange(1, len(true_rul) + 1)

        featured = add_engineered_features(test)
        featured["predicted_rul"] = predict_rul(featured)

        self.featured = featured
        self.true_rul = true_rul["RUL"]

        last = featured.groupby("engine_id").tail(1).sort_values("engine_id")
        last = last.set_index("engine_id")
        actual_capped = self.true_rul.clip(upper=RUL_CAP)

        self.metrics_df = pd.DataFrame({
            "engine_id": last.index,
            "predicted": last["predicted_rul"].values,
            "actual": actual_capped.loc[last.index].values,
        })

        err = self.metrics_df["predicted"] - self.metrics_df["actual"]
        self.mae = float(np.mean(np.abs(err)))
        self.rmse = float(np.sqrt(np.mean(err ** 2)))
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((self.metrics_df["actual"] - self.metrics_df["actual"].mean()) ** 2))
        self.r2 = 1 - ss_res / ss_tot

    def engine_ids(self):
        return sorted(int(e) for e in self.featured["engine_id"].unique())

    def engine_summary(self, engine_id: int) -> dict:
        sub = self.featured[self.featured.engine_id == engine_id]
        last_row = sub.iloc[-1]
        true_rul = float(self.true_rul.loc[engine_id])
        return {
            "engine_id": int(engine_id),
            "max_cycle": int(last_row["cycle"]),
            "true_rul": true_rul,
            "true_rul_capped": float(min(true_rul, RUL_CAP)),
            "predicted_rul_last_cycle": round(float(last_row["predicted_rul"]), 1),
        }

    def engine_trend(self, engine_id: int) -> dict:
        sub = self.featured[self.featured.engine_id == engine_id].sort_values("cycle")
        max_cycle = int(sub["cycle"].max())
        true_rul_at_end = float(self.true_rul.loc[engine_id])
        cycles = sub["cycle"].tolist()
        true_ref = [min(true_rul_at_end + (max_cycle - c), RUL_CAP) for c in cycles]

        return {
            "engine_id": int(engine_id),
            "cycles": cycles,
            "predicted_rul": [round(v, 1) for v in sub["predicted_rul"].tolist()],
            "true_rul_reference": [round(v, 1) for v in true_ref],
            "sensors": {
                s: [round(v, 2) for v in sub[s].tolist()] for s in DISPLAY_SENSORS
            },
        }

    def metrics(self) -> dict:
        return {
            "mae": round(self.mae, 2),
            "rmse": round(self.rmse, 2),
            "r2": round(self.r2, 4),
            "rul_cap": RUL_CAP,
            "n_engines": int(len(self.metrics_df)),
            "scatter": [
                {
                    "engine_id": int(row.engine_id),
                    "actual": float(row.actual),
                    "predicted": round(float(row.predicted), 1),
                }
                for row in self.metrics_df.itertuples()
            ],
        }


store = Store()
