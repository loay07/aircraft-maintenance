# Aircraft Engine Predictive Maintenance

Predicts Remaining Useful Life (RUL) of turbofan engines from sensor telemetry,
using an XGBoost model trained on NASA's C-MAPSS FD001 dataset (official test
MAE 12.06 cycles, RMSE 16.73, R² 0.826).

- `notebooks/` — EDA, feature engineering, and model training/evaluation
- `src/` — reusable feature engineering + model inference pipeline (mirrors the notebooks)
- `app/` — FastAPI backend + static frontend (the website)
- `app/static/data/*.json` — the model's output for all 100 official NASA test engines,
  precomputed by `scripts/export_static_data.py` so the site can run with zero backend
- `models/xgboost_capped.json` — the trained model

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000

## Deploy — GitHub Pages (free, no card, no account beyond GitHub)

The frontend reads its data from static JSON files (`app/static/data/`), so it needs no
server at all. `.github/workflows/deploy-pages.yml` deploys `app/static/` automatically
on every push to `main`.

1. Push this repo to GitHub (already done if you're reading this from the repo).
2. In the repo, go to **Settings > Pages** and set **Source** to **GitHub Actions**.
3. Push to `main` (or run the workflow manually from the **Actions** tab) — the site
   deploys to `https://<username>.github.io/<repo>/`.

To regenerate the static data after retraining or changing the model:

```bash
python scripts/export_static_data.py
```

## Deploy — Render (optional, if you want a real live backend later)

1. In Render, click **New > Blueprint**, point it at the repo — `render.yaml`
   configures the build/start commands automatically.
2. Render builds and serves the FastAPI app (frontend + `/api/*` endpoints) on one URL.
   Free tier available, but spins down after 15 min idle (30-60s cold start on next visit).
