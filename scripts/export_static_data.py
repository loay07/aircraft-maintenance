"""Precompute the model's output for all 100 official FD001 test engines and
dump it as static JSON, so the site can run on GitHub Pages (or any static
host) with zero backend -- no server, no account, no card needed.

This doesn't change what the demo shows: the predictions are the same
xgboost_capped.json model, run over the same real NASA test engines the
FastAPI /api endpoints would have served. The only difference is *when* the
model runs -- once, here, instead of per HTTP request.

Usage: python scripts/export_static_data.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import store  # noqa: E402

OUT_DIR = ROOT / "app" / "static" / "data"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    engines = [store.engine_summary(eid) for eid in store.engine_ids()]
    (OUT_DIR / "engines.json").write_text(json.dumps(engines), encoding="utf-8")

    metrics = store.metrics()
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    trends = {str(eid): store.engine_trend(eid) for eid in store.engine_ids()}
    (OUT_DIR / "trends.json").write_text(json.dumps(trends), encoding="utf-8")

    print(f"Wrote engines.json, metrics.json, trends.json to {OUT_DIR}")


if __name__ == "__main__":
    main()
