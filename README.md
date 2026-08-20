# Aircraft Engine Predictive Maintenance

Predicts Remaining Useful Life (RUL) of turbofan engines from sensor telemetry,
using an XGBoost model trained on NASA's C-MAPSS FD001 dataset (official test
MAE 12.06 cycles, RMSE 16.73, R² 0.826).

- `notebooks/` — EDA, feature engineering, and model training/evaluation
- `src/` — reusable feature engineering + model inference pipeline (mirrors the notebooks) used by the web app
- `app/` — FastAPI backend + static frontend (the website)
- `models/xgboost_capped.json` — the trained model

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000

## Deploy to Render

1. Push this repo to GitHub.
2. In Render, click **New > Blueprint**, point it at the repo — `render.yaml`
   configures the build/start commands automatically.
3. Render builds and serves the FastAPI app (frontend + API) on one URL.
