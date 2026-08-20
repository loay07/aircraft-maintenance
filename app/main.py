from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from src.data import store

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Aircraft Engine RUL Predictor")


@app.get("/api/engines")
def list_engines():
    return [store.engine_summary(engine_id) for engine_id in store.engine_ids()]


@app.get("/api/engines/{engine_id}/trend")
def engine_trend(engine_id: int):
    if engine_id not in store.engine_ids():
        raise HTTPException(status_code=404, detail=f"Unknown engine_id {engine_id}")
    return store.engine_trend(engine_id)


@app.get("/api/metrics")
def metrics():
    return store.metrics()


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
